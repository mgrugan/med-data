import numpy as np

from backend.app.ml.ft_numpy import FTTransformerNumpy
from training.train_ft_transformer import FTTransformerLOS


def test_numpy_inference_matches_pytorch(signal_df, tmp_path):
    m = FTTransformerLOS(seed=0, max_epochs=3).fit(signal_df.iloc[:800])
    meta = m.export_numpy(tmp_path / "ft.npz")
    net = FTTransformerNumpy.load(str(tmp_path / "ft.npz"), meta)
    te = signal_df.iloc[800:900]
    z = net.predict_standardized(m.encoder.codes(te), m.encoder.numeric_matrix(te).astype(np.float64))
    np.testing.assert_allclose(z * meta["y_std"] + meta["y_mean"], m.predict(te), rtol=1e-4, atol=1e-4)


def test_early_stopping_restores_best_checkpoint(signal_df):
    m = FTTransformerLOS(seed=0, max_epochs=8, patience=2).fit(signal_df.iloc[:800])
    best = min(h["es_mae"] for h in m.history)
    assert m.history[m.best_epoch_ - 1]["es_mae"] == best
