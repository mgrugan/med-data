"""Model 3 - FT-Transformer (Gorishniy et al., 2021) in PyTorch.

Architecture
* Feature tokenizer: each numeric feature x_j -> x_j * W_j + b_j; each categorical
  feature -> learned embedding + per-feature bias. A learned [CLS] token is prepended.
* N pre-norm transformer blocks: x + MHA(LN(x)), then x + FFN(LN(x)), with residual,
  attention and FFN dropout.
* Head: LN -> ReLU -> Linear on the [CLS] token.

Training safeguards against overfitting
* AdamW (decoupled weight decay), linear warm-up + cosine learning-rate schedule,
  gradient-norm clipping, dropout.
* Validation monitoring on an early-stopping subset carved from the *training* rows,
  early stopping with patience, and restoration of the best checkpoint.

Attention weights are NOT used as feature importance: attention is not a faithful
attribution method. Global importance is estimated by permutation importance instead.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.ml.features import FEATURE_SUBSETS_A, TabularEncoder, feature_columns
from .base import LOSModel, early_stop_split


class Attention(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        n, t, d = x.shape
        dh = d // self.h
        split = lambda z: z.view(n, t, self.h, dh).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        att = self.drop(torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1))
        return self.o((att @ v).transpose(1, 2).reshape(n, t, d))


class Block(nn.Module):
    def __init__(self, d, heads, ffn_mult, attn_dropout, ffn_dropout, residual_dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads, attn_dropout)
        hidden = int(d * ffn_mult)
        self.ffn1, self.ffn2 = nn.Linear(d, hidden), nn.Linear(hidden, d)
        self.ffn_drop = nn.Dropout(ffn_dropout)
        self.res_drop = nn.Dropout(residual_dropout)

    def forward(self, x):
        x = x + self.res_drop(self.attn(self.ln1(x)))
        h = self.ffn2(self.ffn_drop(F.gelu(self.ffn1(self.ln2(x)))))
        return x + self.res_drop(h)


class FTTransformer(nn.Module):
    def __init__(self, cardinalities, n_num, d=32, n_blocks=2, heads=4, ffn_mult=4 / 3,
                 attn_dropout=0.1, ffn_dropout=0.1, residual_dropout=0.0):
        super().__init__()
        self.n_num = n_num
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.num_weight = nn.Parameter(torch.randn(n_num, d) * (1 / math.sqrt(d)))
        self.num_bias = nn.Parameter(torch.zeros(n_num, d))
        self.cat_emb = nn.ModuleList(nn.Embedding(c, d) for c in cardinalities)
        for e in self.cat_emb:
            nn.init.normal_(e.weight, std=1 / math.sqrt(d))
        self.cat_bias = nn.Parameter(torch.zeros(len(cardinalities), d))
        self.blocks = nn.ModuleList(Block(d, heads, ffn_mult, attn_dropout, ffn_dropout, residual_dropout)
                                    for _ in range(n_blocks))
        self.head_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x_cat, x_num):
        n = x_cat.shape[0]
        toks = [self.cls.expand(n, -1, -1)]
        if self.n_num:
            toks.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        for j, emb in enumerate(self.cat_emb):
            toks.append((emb(x_cat[:, j]) + self.cat_bias[j]).unsqueeze(1))
        x = torch.cat(toks, dim=1)
        for b in self.blocks:
            x = b(x)
        return self.head(F.relu(self.head_ln(x[:, 0]))).squeeze(-1)


class FTTransformerLOS(LOSModel):
    key = "ft_transformer"
    display_name = "FT-Transformer"
    family = "neural"
    complexity = 5
    uses_early_stopping = True
    is_neural = True

    def default_params(self):
        return {"feature_subset": "A_full", "d_token": 32, "n_blocks": 2, "n_heads": 4,
                "attn_dropout": 0.1, "ffn_dropout": 0.1, "residual_dropout": 0.0,
                "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 512, "loss": "l1"}

    def suggest(self, trial):
        return {
            "feature_subset": trial.suggest_categorical("feature_subset", list(FEATURE_SUBSETS_A)),
            # Search space sized to the CPU budget (see training/config.py). With ~8 feature tokens,
            # wider/deeper models were 4-6x slower without any validation gain in pilot runs.
            "d_token": trial.suggest_categorical("d_token", [16, 32]),
            "n_blocks": trial.suggest_int("n_blocks", 1, 2),
            "n_heads": 4,
            "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.4),
            "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.4),
            "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.2),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": 512,
            "loss": trial.suggest_categorical("loss", ["l1", "mse"]),
        }

    def _arrays(self, df):
        return (torch.as_tensor(self.encoder.codes(df)), torch.as_tensor(self.encoder.numeric_matrix(df)))

    def fit(self, df, params=None):
        self.params = {**self.default_params(), **(params or {})}
        p = self.params
        max_epochs = int(self.extra.get("max_epochs", 40))
        patience = int(self.extra.get("patience", 6))
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        df = self._prep_fit(df)
        cats, nums = feature_columns(self.feature_set, p["feature_subset"])
        self.encoder = TabularEncoder(cats, nums).fit(df)
        y = self.y(df)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)
        yt = torch.as_tensor((y - self.y_mean) / self.y_std, dtype=torch.float32)
        xc, xn = self._arrays(df)
        tr, es = early_stop_split(len(df), self.seed)
        tr_t, es_t = torch.as_tensor(tr), torch.as_tensor(es)
        self.net = FTTransformer(self.encoder.cardinalities(), len(nums), d=p["d_token"], n_blocks=p["n_blocks"],
                                 heads=p["n_heads"], attn_dropout=p["attn_dropout"], ffn_dropout=p["ffn_dropout"],
                                 residual_dropout=p["residual_dropout"])
        # no weight decay on biases / norms / embeddings' bias params
        decay, no_decay = [], []
        for name, prm in self.net.named_parameters():
            (no_decay if (prm.ndim < 2 or "ln" in name or "bias" in name) else decay).append(prm)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": p["weight_decay"]},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=p["lr"])
        bs = int(p["batch_size"])
        steps_per_epoch = math.ceil(len(tr) / bs)
        total, warm = max_epochs * steps_per_epoch, steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        loss_fn = nn.L1Loss() if p["loss"] == "l1" else nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = float("inf"), None, 0
        self.history = []
        for epoch in range(max_epochs):
            self.net.train()
            order = tr_t[torch.randperm(len(tr_t), generator=gen)]
            tr_loss = 0.0
            for i in range(0, len(order), bs):
                b = order[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(self.net(xc[b], xn[b]), yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
                tr_loss += loss.item() * len(b)
            self.net.eval()
            with torch.no_grad():
                pred = self.net(xc[es_t], xn[es_t]).numpy() * self.y_std + self.y_mean
            val_mae = float(np.mean(np.abs(pred - y[es])))
            self.history.append({"epoch": epoch + 1, "train_loss": tr_loss / len(tr), "es_mae": val_mae})
            if val_mae < best - 1e-4:
                best, bad, self.best_epoch_ = val_mae, 0, epoch + 1
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)  # checkpoint restoration
        self.net.eval()
        return self

    def predict(self, df):
        xc, xn = self._arrays(self._prep(df))
        out = []
        with torch.no_grad():
            for i in range(0, len(xc), 4096):
                out.append(self.net(xc[i:i + 4096], xn[i:i + 4096]).numpy())
        return np.concatenate(out) * self.y_std + self.y_mean

    def best_iteration(self):
        return getattr(self, "best_epoch_", None)

    def export_numpy(self, path) -> dict:
        sd = {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        np.savez(path, **sd)
        return {"n_blocks": self.params["n_blocks"], "n_heads": self.params["n_heads"],
                "n_cat": len(self.encoder.categorical), "n_num": len(self.encoder.numeric),
                "y_mean": self.y_mean, "y_std": self.y_std, "encoder": self.encoder.to_dict()}
