"""Pure-NumPy forward pass of the trained FT-Transformer.

The deployed API does not need PyTorch: weights are exported to ``.npz`` and this module
re-implements inference exactly (tested against the PyTorch model in
``tests/test_ft_transformer.py``).
"""
from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)


def _erf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf  # type: ignore
        return erf(x)
    except Exception:  # pragma: no cover - fallback
        return np.vectorize(math.erf)(x)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / _SQRT2))


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W.T + b


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class FTTransformerNumpy:
    def __init__(self, weights: dict[str, np.ndarray], n_blocks: int, n_heads: int,
                 n_cat: int, n_num: int):
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_cat = n_cat
        self.n_num = n_num

    @classmethod
    def load(cls, path: str, meta: dict) -> "FTTransformerNumpy":
        data = np.load(path)
        return cls({k: data[k] for k in data.files}, meta["n_blocks"], meta["n_heads"],
                   meta["n_cat"], meta["n_num"])

    def tokens(self, x_cat: np.ndarray, x_num: np.ndarray) -> np.ndarray:
        w = self.w
        n = x_cat.shape[0] if self.n_cat else x_num.shape[0]
        toks = [np.broadcast_to(w["cls"], (n, 1, w["cls"].shape[-1]))]
        if self.n_num:
            toks.append(x_num[:, :, None] * w["num_weight"][None] + w["num_bias"][None])
        for j in range(self.n_cat):
            emb = w[f"cat_emb.{j}.weight"][x_cat[:, j]] + w["cat_bias"][j]
            toks.append(emb[:, None, :])
        return np.concatenate(toks, axis=1)

    def _attention(self, x: np.ndarray, i: int) -> np.ndarray:
        w = self.w
        p = f"blocks.{i}.attn."
        n, t, d = x.shape
        h = self.n_heads
        dh = d // h
        q = _linear(x, w[p + "q.weight"], w[p + "q.bias"]).reshape(n, t, h, dh).transpose(0, 2, 1, 3)
        k = _linear(x, w[p + "k.weight"], w[p + "k.bias"]).reshape(n, t, h, dh).transpose(0, 2, 1, 3)
        v = _linear(x, w[p + "v.weight"], w[p + "v.bias"]).reshape(n, t, h, dh).transpose(0, 2, 1, 3)
        att = _softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh))
        out = (att @ v).transpose(0, 2, 1, 3).reshape(n, t, d)
        return _linear(out, w[p + "o.weight"], w[p + "o.bias"])

    def predict_standardized(self, x_cat: np.ndarray, x_num: np.ndarray) -> np.ndarray:
        w = self.w
        x = self.tokens(x_cat, x_num)
        for i in range(self.n_blocks):
            p = f"blocks.{i}."
            x = x + self._attention(_layer_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"]), i)
            hdn = _layer_norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            hdn = _linear(_gelu(_linear(hdn, w[p + "ffn1.weight"], w[p + "ffn1.bias"])),
                          w[p + "ffn2.weight"], w[p + "ffn2.bias"])
            x = x + hdn
        cls = _layer_norm(x[:, 0], w["head_ln.weight"], w["head_ln.bias"])
        cls = np.maximum(cls, 0.0)
        return _linear(cls, w["head.weight"], w["head.bias"])[:, 0]
