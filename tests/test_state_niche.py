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


def test_autok_pairwise_ari_nmi_and_summary(tmp_path):
    adata = make_data(seed=2)
    sn.build_features(adata, n_neighbors=5)
    autok = sn.ClusterAutoK(
        (2, 4),
        max_runs=3,
        covariance_type="diag",
        primary_metric="ari",
        silhouette_mode="representative",
        silhouette_sample_size=50,
        random_state=7,
    )
    autok.fit(
        adata,
        use_rep="X_state_niche",
        sample_key="sample_name",
        max_cells_per_sample=30,
    )

    # 3 runs -> C(3,2)=3 pairs for every K; 3 K values -> 9 rows.
    assert autok.pairwise_metrics_.shape[0] == 9
    assert {"fmi", "ari", "nmi", "seed_i", "seed_j"}.issubset(
        autok.pairwise_metrics_.columns
    )
    assert {
        "ari_mean", "ari_median", "ari_sd", "ari_iqr",
        "nmi_mean", "nmi_median", "nmi_sd", "nmi_iqr",
        "fmi_mean", "fmi_median", "fmi_sd", "fmi_iqr",
        "silhouette", "primary_metric",
    }.issubset(autok.stability_.columns)
    assert set(autok.stability_["primary_metric"]) == {"ari"}
    assert np.allclose(
        autok.stability_["stability"],
        autok.stability_["ari_mean"],
        equal_nan=True,
    )

    paths = autok.save_results(tmp_path, prefix="demo")
    assert all(path.exists() for path in paths.values())


def test_predict_all_k_and_scan_convenience(tmp_path):
    adata = make_data(seed=3)
    sn.build_features(adata, n_neighbors=5)
    autok = sn.scan_cluster_stability(
        adata,
        use_rep="X_state_niche",
        n_clusters=(2, 3),
        max_runs=2,
        primary_metric="ari",
        silhouette_mode=None,
        covariance_type="diag",
        sample_key="sample_name",
        max_cells_per_sample=30,
        predict_all_k=True,
        output_dir=tmp_path,
        output_prefix="scan",
        random_state=11,
    )
    assert autok.best_k_ in {2, 3}
    assert "state_niche_k2" in adata.obs
    assert "state_niche_k3" in adata.obs
    assert (tmp_path / "scan_stability_summary.csv").exists()
    assert (tmp_path / "scan_pairwise_stability.csv").exists()
    assert (tmp_path / "scan_run_metrics.csv").exists()
