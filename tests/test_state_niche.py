import numpy as np
import pandas as pd

from sat4sc import state_niche as sn


class DummyAnnData:
    def __init__(self, obs, obsm):
        self.obs = obs
        self.obsm = obsm
        self.uns = {}
        self.n_obs = len(obs)


def make_data(seed=1):
    rng = np.random.default_rng(seed)
    n = 60
    obs = pd.DataFrame({
        "sample_name": ["A"] * n + ["B"] * n,
        "cell_type": rng.choice(["Tumor", "Myeloid"], 2 * n),
    })
    spatial = np.vstack([
        rng.normal([0, 0], 2, (n, 2)),
        rng.normal([50, 50], 2, (n, 2)),
    ]).astype(np.float32)
    latent = rng.normal(size=(2 * n, 4)).astype(np.float32)
    return DummyAnnData(obs, {"spatial": spatial, "X_scVI": latent})


def test_feature_shape_and_no_cross_sample_neighbors():
    adata = make_data()
    result = sn.build_features(adata, n_neighbors=5, chunk_size=20)
    assert result.raw.shape == (adata.n_obs, 8)
    assert result.scaled.shape == (adata.n_obs, 8)
    samples = adata.obs["sample_name"].to_numpy()
    nbr_samples = samples[result.neighbors.indices]
    assert np.all(nbr_samples == samples[:, None])


def test_autok_and_prediction():
    adata = make_data()
    sn.build_features(adata, n_neighbors=5)
    autok = sn.ClusterAutoK((2, 3), max_runs=2, covariance_type="diag")
    autok.fit(
        adata,
        use_rep="X_state_niche",
        sample_key="sample_name",
        max_cells_per_sample=30,
    )
    assert set(autok.stability_["k"]) == {2, 3}
    labels = autok.predict(adata, key_added="state_niche")
    assert labels.shape == (adata.n_obs,)
    assert "state_niche_max_prob" in adata.obs
