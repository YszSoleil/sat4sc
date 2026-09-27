"""Transcriptional-state spatial niche analysis for :mod:`sat4sc`.

This module builds *extrinsic* spatial niche representations from a latent
single-cell embedding (for example ``adata.obsm['X_scVI']``). For every cell,
its nearest spatial neighbours are found **within the same sample**, the
neighbour latent coordinates are summarized by permutation-invariant
statistics (currently mean and standard deviation), and the resulting feature
matrix is clustered with a Gaussian mixture model (GMM).

The main workflow is::

    spatial kNN (within sample)
        -> neighbour latent aggregation (mean + std)
        -> feature standardization
        -> GMM clustering
        -> repeated-GMM stability scan across candidate cluster numbers

The centre cell is excluded from the default representation. This deliberately
emphasizes the transcriptional state of the surrounding microenvironment
rather than re-clustering the centre cell by its own identity.

The automatic cluster-number scan is CellCharter-inspired rather than a direct
reimplementation. It reports repeated-run Fowlkes-Mallows stability at each K
and identifies local stability peaks. The exact score definition is documented
in :class:`ClusterAutoK` and stored in ``stability_``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from sklearn.metrics import fowlkes_mallows_score
from sklearn.preprocessing import StandardScaler

try:
    from anndata import AnnData
except Exception:  # pragma: no cover - optional import for type checking
    AnnData = object  # type: ignore


DEFAULT_SAMPLE_KEY = "sample_name"
DEFAULT_SPATIAL_KEY = "spatial"
DEFAULT_COORD_COLS = ("x_centroid", "y_centroid")
DEFAULT_REP_KEY = "X_scVI"
DEFAULT_FEATURE_KEY = "X_state_niche"


@dataclass
class SpatialKNNResult:
    """Spatial nearest-neighbour result aligned to the full AnnData row order."""

    indices: np.ndarray
    distances: np.ndarray
    sample_key: str
    n_neighbors: int
    spatial_key: str | None
    coord_cols: tuple[str, str] | None
    include_self: bool = False

    @property
    def kth_distance(self) -> np.ndarray:
        """Distance to the farthest retained neighbour for each cell."""

        return self.distances[:, -1]

    @property
    def mean_distance(self) -> np.ndarray:
        """Mean distance to retained neighbours for each cell."""

        return self.distances.mean(axis=1)


@dataclass
class StateNicheFeatures:
    """Container returned by :func:`build_features`."""

    raw: np.ndarray
    scaled: np.ndarray
    mean: np.ndarray | None
    std: np.ndarray | None
    scaler: StandardScaler | None
    neighbors: SpatialKNNResult
    use_rep: str
    aggregations: tuple[str, ...]
    include_center: bool
    output_key: str
    settings: dict = field(default_factory=dict)


@dataclass
class ClusterResult:
    """Fixed-K GMM clustering result."""

    labels: np.ndarray
    probabilities: np.ndarray
    model: GaussianMixture
    n_clusters: int
    use_rep: str
    key_added: str | None = None
    fit_indices: np.ndarray | None = None
    settings: dict = field(default_factory=dict)



def _as_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {x.shape}.")
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("Representation must be numeric.")
    x = x.astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("Representation contains NaN or infinite values.")
    return x


def _get_spatial(
    adata: AnnData,
    spatial_key: str = DEFAULT_SPATIAL_KEY,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
) -> tuple[np.ndarray, str | None, tuple[str, str] | None]:
    """Return spatial coordinates, preferring ``obsm[spatial_key]``."""

    if hasattr(adata, "obsm") and spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"adata.obsm[{spatial_key!r}] must have shape (n_cells, >=2)."
            )
        coords = coords[:, :2]
        key, cols = spatial_key, None
    else:
        if len(coord_cols) != 2:
            raise ValueError("coord_cols must contain exactly two column names.")
        missing = [c for c in coord_cols if c not in adata.obs.columns]
        if missing:
            raise KeyError(
                f"Spatial coordinates not found in adata.obsm[{spatial_key!r}] "
                f"and obs columns are missing: {missing}."
            )
        coords = adata.obs[list(coord_cols)].to_numpy(dtype=np.float64)
        key, cols = None, (str(coord_cols[0]), str(coord_cols[1]))

    if coords.shape[0] != adata.n_obs:
        raise ValueError("Spatial coordinates are not aligned with adata.n_obs.")
    if not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates contain NaN or infinite values.")
    return coords, key, cols


def _validate_sample_key(adata: AnnData, sample_key: str) -> np.ndarray:
    if sample_key not in adata.obs.columns:
        raise KeyError(f"{sample_key!r} not found in adata.obs.")
    samples = adata.obs[sample_key]
    if samples.isna().any():
        raise ValueError(f"adata.obs[{sample_key!r}] contains missing values.")
    return samples.astype(str).to_numpy()


def build_spatial_neighbors(
    adata: AnnData,
    *,
    sample_key: str = DEFAULT_SAMPLE_KEY,
    spatial_key: str = DEFAULT_SPATIAL_KEY,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    n_neighbors: int = 25,
    include_self: bool = False,
    workers: int = -1,
    annotate_obs: bool = True,
    obs_prefix: str = "state_niche",
) -> SpatialKNNResult:
    """Build spatial kNN independently inside each sample.

    Parameters
    ----------
    adata
        AnnData-like object.
    sample_key
        Column in ``adata.obs`` identifying independent tissue sections/samples.
    spatial_key
        Preferred coordinate key in ``adata.obsm``.
    coord_cols
        Fallback coordinate columns in ``adata.obs``.
    n_neighbors
        Number of retained neighbours per cell.
    include_self
        If ``False`` (default), the centre cell is removed and ``n_neighbors``
        *other* cells are retained. If ``True``, the centre cell can be part of
        the returned neighbour set.
    workers
        Number of workers passed to ``scipy.spatial.cKDTree.query``.
    annotate_obs
        Store kNN radius summaries in ``adata.obs``.
    obs_prefix
        Prefix for the radius-summary columns.
    """

    if int(n_neighbors) != n_neighbors or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    n_neighbors = int(n_neighbors)

    samples = _validate_sample_key(adata, sample_key)
    coords, used_spatial_key, used_coord_cols = _get_spatial(
        adata, spatial_key=spatial_key, coord_cols=coord_cols
    )

    n_obs = int(adata.n_obs)
    neighbor_indices = np.empty((n_obs, n_neighbors), dtype=np.int64)
    neighbor_distances = np.empty((n_obs, n_neighbors), dtype=np.float32)

    # Preserve first-appearance order rather than sorting sample labels.
    unique_samples = pd.unique(samples)
    for sample in unique_samples:
        global_idx = np.flatnonzero(samples == sample)
        n_sample = global_idx.size
        min_needed = n_neighbors if include_self else n_neighbors + 1
        if n_sample < min_needed:
            raise ValueError(
                f"Sample {sample!r} has {n_sample} cells, but at least "
                f"{min_needed} are required for n_neighbors={n_neighbors} "
                f"with include_self={include_self}."
            )

        tree = cKDTree(coords[global_idx])
        query_k = n_neighbors if include_self else n_neighbors + 1
        distances, local_idx = tree.query(
            coords[global_idx], k=query_k, workers=workers
        )

        if query_k == 1:
            distances = distances[:, None]
            local_idx = local_idx[:, None]

        if include_self:
            keep_dist = distances[:, :n_neighbors]
            keep_local = local_idx[:, :n_neighbors]
        else:
            # The query point is normally column 0. To be robust to tied
            # coordinates, explicitly remove one self occurrence per row.
            keep_local = np.empty((n_sample, n_neighbors), dtype=np.int64)
            keep_dist = np.empty((n_sample, n_neighbors), dtype=np.float64)
            for r in range(n_sample):
                row_idx = np.asarray(local_idx[r])
                row_dist = np.asarray(distances[r])
                self_hits = np.flatnonzero(row_idx == r)
                if self_hits.size:
                    drop = int(self_hits[0])
                    row_idx = np.delete(row_idx, drop)
                    row_dist = np.delete(row_dist, drop)
                else:
                    # Extremely unusual, but do not silently keep fewer than K.
                    row_idx = row_idx[:n_neighbors]
                    row_dist = row_dist[:n_neighbors]
                if row_idx.size < n_neighbors:
                    raise RuntimeError(
                        f"Could not obtain {n_neighbors} non-self neighbors for "
                        f"cell {r} in sample {sample!r}."
                    )
                keep_local[r] = row_idx[:n_neighbors]
                keep_dist[r] = row_dist[:n_neighbors]

        neighbor_indices[global_idx] = global_idx[keep_local]
        neighbor_distances[global_idx] = keep_dist.astype(np.float32)

    result = SpatialKNNResult(
        indices=neighbor_indices,
        distances=neighbor_distances,
        sample_key=sample_key,
        n_neighbors=n_neighbors,
        spatial_key=used_spatial_key,
        coord_cols=used_coord_cols,
        include_self=include_self,
    )

    if annotate_obs:
        adata.obs[f"{obs_prefix}_knn{n_neighbors}_rmax"] = result.kth_distance
        adata.obs[f"{obs_prefix}_knn{n_neighbors}_rmean"] = result.mean_distance

    return result


def _aggregate_neighbors_chunked(
    representation: np.ndarray,
    neighbor_indices: np.ndarray,
    aggregations: tuple[str, ...],
    *,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    n_obs = neighbor_indices.shape[0]
    latent_dim = representation.shape[1]
    outputs = {
        agg: np.empty((n_obs, latent_dim), dtype=np.float32) for agg in aggregations
    }

    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        z = representation[neighbor_indices[start:stop]]
        if "mean" in outputs:
            outputs["mean"][start:stop] = z.mean(axis=1, dtype=np.float64).astype(
                np.float32
            )
        if "std" in outputs:
            outputs["std"][start:stop] = z.std(axis=1, ddof=0).astype(np.float32)
        if "median" in outputs:
            outputs["median"][start:stop] = np.median(z, axis=1).astype(np.float32)

    return outputs


def build_features(
    adata: AnnData,
    *,
    use_rep: str = DEFAULT_REP_KEY,
    sample_key: str = DEFAULT_SAMPLE_KEY,
    spatial_key: str = DEFAULT_SPATIAL_KEY,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    n_neighbors: int = 25,
    aggregations: Sequence[str] = ("mean", "std"),
    include_center: bool = False,
    scale: bool = True,
    output_key: str = DEFAULT_FEATURE_KEY,
    chunk_size: int = 20_000,
    workers: int = -1,
    neighbors: SpatialKNNResult | None = None,
    copy: bool = False,
) -> StateNicheFeatures | tuple[AnnData, StateNicheFeatures]:
    """Construct neighbour-only transcriptional-state niche features.

    The default representation is ``[mean(neighbour latent),
    std(neighbour latent)]``. When ``include_center=True``, the centre-cell
    latent representation is prepended as an optional sensitivity analysis.

    Results are written to ``adata.obsm`` using ``output_key`` and companion
    keys such as ``X_state_niche_mean`` and ``X_state_niche_std``.
    """

    if copy:
        adata = adata.copy()
    if use_rep not in adata.obsm:
        raise KeyError(f"{use_rep!r} not found in adata.obsm.")
    z = _as_float32(adata.obsm[use_rep])
    if z.shape[0] != adata.n_obs:
        raise ValueError(f"adata.obsm[{use_rep!r}] is not aligned with adata.n_obs.")

    aggregations = tuple(str(a).lower() for a in aggregations)
    valid_aggs = {"mean", "std", "median"}
    if not aggregations:
        raise ValueError("aggregations must contain at least one aggregation.")
    unknown = sorted(set(aggregations) - valid_aggs)
    if unknown:
        raise ValueError(f"Unsupported aggregations: {unknown}. Supported: {sorted(valid_aggs)}")
    if len(set(aggregations)) != len(aggregations):
        raise ValueError("aggregations contains duplicate entries.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    if neighbors is None:
        neighbors = build_spatial_neighbors(
            adata,
            sample_key=sample_key,
            spatial_key=spatial_key,
            coord_cols=coord_cols,
            n_neighbors=n_neighbors,
            include_self=False,
            workers=workers,
            annotate_obs=True,
        )
    else:
        if neighbors.indices.shape != (adata.n_obs, n_neighbors):
            raise ValueError(
                "Provided neighbors are incompatible with adata.n_obs or n_neighbors."
            )
        if neighbors.include_self:
            warnings.warn(
                "Provided SpatialKNNResult includes the centre cell. This differs "
                "from the default extrinsic-neighbour definition.",
                RuntimeWarning,
                stacklevel=2,
            )

    agg_values = _aggregate_neighbors_chunked(
        z,
        neighbors.indices,
        aggregations,
        chunk_size=int(chunk_size),
    )

    blocks: list[np.ndarray] = []
    if include_center:
        blocks.append(z)
    blocks.extend(agg_values[a] for a in aggregations)
    raw = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    scaler: StandardScaler | None = None
    if scale:
        scaler = StandardScaler(copy=True)
        scaled = scaler.fit_transform(raw).astype(np.float32)
    else:
        scaled = raw.copy()

    base = output_key
    if base.startswith("X_"):
        prefix = base
    else:
        prefix = f"X_{base}"

    for agg, arr in agg_values.items():
        adata.obsm[f"{prefix}_{agg}"] = arr
    adata.obsm[f"{prefix}_raw"] = raw
    adata.obsm[prefix] = scaled

    settings = {
        "use_rep": use_rep,
        "sample_key": sample_key,
        "spatial_key": neighbors.spatial_key,
        "coord_cols": neighbors.coord_cols,
        "n_neighbors": n_neighbors,
        "aggregations": aggregations,
        "include_center": include_center,
        "scale": scale,
        "chunk_size": int(chunk_size),
    }
    if hasattr(adata, "uns"):
        adata.uns[f"{prefix}_settings"] = settings

    result = StateNicheFeatures(
        raw=raw,
        scaled=scaled,
        mean=agg_values.get("mean"),
        std=agg_values.get("std"),
        scaler=scaler,
        neighbors=neighbors,
        use_rep=use_rep,
        aggregations=aggregations,
        include_center=include_center,
        output_key=prefix,
        settings=settings,
    )
    if copy:
        return adata, result
    return result


def balanced_fit_indices(
    adata: AnnData,
    *,
    sample_key: str = DEFAULT_SAMPLE_KEY,
    max_cells_per_sample: int | None = 10_000,
    random_state: int = 123,
    annotate_obs: bool = False,
    obs_key: str = "state_niche_fit_set",
) -> np.ndarray:
    """Return sample-balanced row indices for fitting GMM models.

    Samples with fewer than ``max_cells_per_sample`` cells contribute all cells;
    larger samples are randomly downsampled without replacement.
    """

    samples = _validate_sample_key(adata, sample_key)
    if max_cells_per_sample is None:
        idx = np.arange(adata.n_obs, dtype=np.int64)
    else:
        if max_cells_per_sample < 1:
            raise ValueError("max_cells_per_sample must be >=1 or None.")
        rng = np.random.default_rng(random_state)
        chunks: list[np.ndarray] = []
        for sample in pd.unique(samples):
            sidx = np.flatnonzero(samples == sample)
            if sidx.size > max_cells_per_sample:
                sidx = np.sort(
                    rng.choice(sidx, size=max_cells_per_sample, replace=False)
                )
            chunks.append(sidx)
        idx = np.concatenate(chunks).astype(np.int64, copy=False)

    if annotate_obs:
        flag = np.zeros(adata.n_obs, dtype=bool)
        flag[idx] = True
        adata.obs[obs_key] = flag
    return idx


def _get_rep_matrix(adata: AnnData, use_rep: str) -> np.ndarray:
    if use_rep not in adata.obsm:
        raise KeyError(f"{use_rep!r} not found in adata.obsm.")
    return _as_float32(adata.obsm[use_rep])


def fit(
    adata: AnnData,
    *,
    n_clusters: int,
    use_rep: str = DEFAULT_FEATURE_KEY,
    key_added: str = "state_niche",
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    n_init: int = 1,
    random_state: int = 123,
    fit_indices: np.ndarray | Sequence[int] | None = None,
    sample_key: str | None = None,
    max_cells_per_sample: int | None = None,
    store_probabilities: bool = True,
) -> ClusterResult:
    """Fit a fixed-K Gaussian mixture model and annotate all cells."""

    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2.")
    x = _get_rep_matrix(adata, use_rep)

    if fit_indices is None and max_cells_per_sample is not None:
        if sample_key is None:
            raise ValueError(
                "sample_key is required when max_cells_per_sample is provided."
            )
        fit_indices = balanced_fit_indices(
            adata,
            sample_key=sample_key,
            max_cells_per_sample=max_cells_per_sample,
            random_state=random_state,
            annotate_obs=True,
        )
    if fit_indices is None:
        fit_idx = np.arange(adata.n_obs, dtype=np.int64)
    else:
        fit_idx = np.asarray(fit_indices, dtype=np.int64)
        if fit_idx.ndim != 1 or fit_idx.size == 0:
            raise ValueError("fit_indices must be a non-empty 1D integer array.")
        if fit_idx.min() < 0 or fit_idx.max() >= adata.n_obs:
            raise IndexError("fit_indices contains out-of-range cell indices.")

    model = GaussianMixture(
        n_components=int(n_clusters),
        covariance_type=covariance_type,
        reg_covar=float(reg_covar),
        max_iter=int(max_iter),
        n_init=int(n_init),
        random_state=int(random_state),
    )
    model.fit(x[fit_idx])
    labels = model.predict(x).astype(np.int32)
    probabilities = model.predict_proba(x).astype(np.float32)

    adata.obs[key_added] = pd.Categorical(labels.astype(str))
    adata.obs[f"{key_added}_max_prob"] = probabilities.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(
            np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0),
            axis=1,
        )
    if n_clusters > 1:
        entropy = entropy / np.log(n_clusters)
    adata.obs[f"{key_added}_entropy"] = entropy.astype(np.float32)
    if store_probabilities:
        adata.obsm[f"X_{key_added}_prob"] = probabilities

    settings = {
        "n_clusters": int(n_clusters),
        "use_rep": use_rep,
        "covariance_type": covariance_type,
        "reg_covar": float(reg_covar),
        "max_iter": int(max_iter),
        "n_init": int(n_init),
        "random_state": int(random_state),
        "n_fit_cells": int(fit_idx.size),
    }
    adata.uns[f"{key_added}_settings"] = settings

    return ClusterResult(
        labels=labels,
        probabilities=probabilities,
        model=model,
        n_clusters=int(n_clusters),
        use_rep=use_rep,
        key_added=key_added,
        fit_indices=fit_idx,
        settings=settings,
    )


class ClusterAutoK:
    """Repeated-GMM cluster-number stability scan.

    This class is **CellCharter-inspired**. For every candidate number of
    clusters ``K``, multiple GMMs are fitted with different random seeds.
    Stability is quantified with the Fowlkes-Mallows index (FMI), which is
    label-permutation invariant.

    The table ``stability_`` contains:

    ``within_stability``
        Mean pairwise FMI between repeated runs at the same K.

    ``prev_similarity`` / ``next_similarity``
        Mean best-match FMI between runs at K and runs at K-1 / K+1. These are
        descriptive continuity diagnostics, not subtracted from the primary
        within-K score.

    ``stability``
        Alias of ``within_stability``; used for peak detection.

    ``is_peak``
        Local maxima of the stability curve. Edge K values are not declared
        peaks unless only one K is supplied.

    ``best_k_`` is the peak with the highest stability. If no strict local peak
    exists, the globally most stable K is returned. Biological interpretability,
    spatial coherence and cross-sample recurrence should still be considered
    before selecting a final K.
    """

    def __init__(
        self,
        n_clusters: Iterable[int] | tuple[int, int] = (2, 20),
        *,
        max_runs: int = 10,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        random_state: int = 123,
        n_init: int = 1,
    ) -> None:
        if isinstance(n_clusters, tuple) and len(n_clusters) == 2:
            lo, hi = map(int, n_clusters)
            if lo > hi:
                raise ValueError("n_clusters lower bound must be <= upper bound.")
            ks = list(range(lo, hi + 1))
        else:
            ks = sorted({int(k) for k in n_clusters})
        if not ks or min(ks) < 2:
            raise ValueError("All candidate cluster numbers must be >=2.")
        if max_runs < 2:
            raise ValueError("max_runs must be >=2 to estimate repeated-run stability.")

        self.n_clusters = ks
        self.max_runs = int(max_runs)
        self.covariance_type = covariance_type
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.n_init = int(n_init)

        self.models_: dict[int, list[GaussianMixture]] = {}
        self.labels_: dict[int, list[np.ndarray]] = {}
        self.stability_: pd.DataFrame | None = None
        self.best_k_: int | None = None
        self.peaks_: list[int] = []
        self.fit_indices_: np.ndarray | None = None
        self.use_rep_: str | None = None
        self._x_fit: np.ndarray | None = None

    def _fit_matrix(self, x_fit: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        self.models_.clear()
        self.labels_.clear()

        for k in self.n_clusters:
            models: list[GaussianMixture] = []
            labels: list[np.ndarray] = []
            seeds = rng.integers(0, np.iinfo(np.int32).max, size=self.max_runs)
            for seed in seeds:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type=self.covariance_type,
                    reg_covar=self.reg_covar,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    random_state=int(seed),
                )
                model.fit(x_fit)
                models.append(model)
                labels.append(model.predict(x_fit).astype(np.int32))
            self.models_[k] = models
            self.labels_[k] = labels

    @staticmethod
    def _mean_pairwise_fmi(label_runs: Sequence[np.ndarray]) -> float:
        vals = [
            fowlkes_mallows_score(label_runs[i], label_runs[j])
            for i, j in combinations(range(len(label_runs)), 2)
        ]
        return float(np.mean(vals)) if vals else np.nan

    @staticmethod
    def _mean_best_cross_fmi(
        labels_a: Sequence[np.ndarray], labels_b: Sequence[np.ndarray]
    ) -> float:
        if not labels_a or not labels_b:
            return np.nan
        vals = []
        for a in labels_a:
            vals.append(max(fowlkes_mallows_score(a, b) for b in labels_b))
        for b in labels_b:
            vals.append(max(fowlkes_mallows_score(b, a) for a in labels_a))
        return float(np.mean(vals))

    def _summarize_stability(self) -> pd.DataFrame:
        rows = []
        kset = set(self.n_clusters)
        for k in self.n_clusters:
            within = self._mean_pairwise_fmi(self.labels_[k])
            prev = (
                self._mean_best_cross_fmi(self.labels_[k], self.labels_[k - 1])
                if k - 1 in kset
                else np.nan
            )
            nxt = (
                self._mean_best_cross_fmi(self.labels_[k], self.labels_[k + 1])
                if k + 1 in kset
                else np.nan
            )
            rows.append(
                {
                    "k": k,
                    "stability": within,
                    "within_stability": within,
                    "prev_similarity": prev,
                    "next_similarity": nxt,
                }
            )
        out = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

        is_peak = np.zeros(len(out), dtype=bool)
        if len(out) == 1:
            is_peak[0] = True
        elif len(out) >= 3:
            s = out["stability"].to_numpy()
            for i in range(1, len(out) - 1):
                if s[i] >= s[i - 1] and s[i] >= s[i + 1] and (
                    s[i] > s[i - 1] or s[i] > s[i + 1]
                ):
                    is_peak[i] = True
        out["is_peak"] = is_peak
        return out

    def fit(
        self,
        adata: AnnData,
        *,
        use_rep: str = DEFAULT_FEATURE_KEY,
        fit_indices: np.ndarray | Sequence[int] | None = None,
        sample_key: str | None = None,
        max_cells_per_sample: int | None = None,
        annotate_fit_set: bool = True,
    ) -> "ClusterAutoK":
        """Fit all candidate K values on either all cells or a balanced subset."""

        x = _get_rep_matrix(adata, use_rep)
        if fit_indices is None and max_cells_per_sample is not None:
            if sample_key is None:
                raise ValueError(
                    "sample_key is required when max_cells_per_sample is provided."
                )
            fit_idx = balanced_fit_indices(
                adata,
                sample_key=sample_key,
                max_cells_per_sample=max_cells_per_sample,
                random_state=self.random_state,
                annotate_obs=annotate_fit_set,
            )
        elif fit_indices is None:
            fit_idx = np.arange(adata.n_obs, dtype=np.int64)
        else:
            fit_idx = np.asarray(fit_indices, dtype=np.int64)
            if fit_idx.ndim != 1 or fit_idx.size == 0:
                raise ValueError("fit_indices must be a non-empty 1D integer array.")

        x_fit = x[fit_idx]
        if x_fit.shape[0] <= max(self.n_clusters):
            raise ValueError("Too few fit cells for the requested candidate K values.")

        self.fit_indices_ = fit_idx
        self.use_rep_ = use_rep
        self._x_fit = x_fit
        self._fit_matrix(x_fit)
        self.stability_ = self._summarize_stability()
        self.peaks_ = self.stability_.loc[
            self.stability_["is_peak"], "k"
        ].astype(int).tolist()

        candidates = self.stability_[self.stability_["is_peak"]]
        if candidates.empty:
            candidates = self.stability_
        best_row = candidates.sort_values(
            ["stability", "k"], ascending=[False, True]
        ).iloc[0]
        self.best_k_ = int(best_row["k"])
        return self

    def get_model(self, k: int | None = None) -> GaussianMixture:
        """Return the most internally representative run for a candidate K."""

        if self.stability_ is None:
            raise RuntimeError("Call fit() before get_model().")
        if k is None:
            if self.best_k_ is None:
                raise RuntimeError("best_k_ is unavailable.")
            k = self.best_k_
        k = int(k)
        if k not in self.models_:
            raise KeyError(f"K={k} was not fitted.")

        label_runs = self.labels_[k]
        if len(label_runs) == 1:
            return self.models_[k][0]
        scores = np.zeros(len(label_runs), dtype=float)
        for i in range(len(label_runs)):
            others = [
                fowlkes_mallows_score(label_runs[i], label_runs[j])
                for j in range(len(label_runs))
                if i != j
            ]
            scores[i] = np.mean(others)
        return self.models_[k][int(np.argmax(scores))]

    def predict(
        self,
        adata: AnnData,
        *,
        k: int | None = None,
        use_rep: str | None = None,
        key_added: str | None = None,
        store_probabilities: bool = True,
    ) -> np.ndarray:
        """Predict all cells using a representative fitted GMM for K."""

        if self.stability_ is None:
            raise RuntimeError("Call fit() before predict().")
        if use_rep is None:
            if self.use_rep_ is None:
                raise RuntimeError("use_rep is unavailable.")
            use_rep = self.use_rep_
        x = _get_rep_matrix(adata, use_rep)
        model = self.get_model(k)
        labels = model.predict(x).astype(np.int32)
        probabilities = model.predict_proba(x).astype(np.float32)

        if key_added is not None:
            adata.obs[key_added] = pd.Categorical(labels.astype(str))
            adata.obs[f"{key_added}_max_prob"] = probabilities.max(axis=1)
            n_clusters = model.n_components
            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = -np.sum(
                    np.where(
                        probabilities > 0,
                        probabilities * np.log(probabilities),
                        0.0,
                    ),
                    axis=1,
                )
            entropy = entropy / np.log(n_clusters)
            adata.obs[f"{key_added}_entropy"] = entropy.astype(np.float32)
            if store_probabilities:
                adata.obsm[f"X_{key_added}_prob"] = probabilities
            adata.uns[f"{key_added}_autok"] = {
                "selected_k": int(model.n_components),
                "best_k": int(self.best_k_) if self.best_k_ is not None else None,
                "peaks": [int(x) for x in self.peaks_],
                "use_rep": use_rep,
                "max_runs": self.max_runs,
                "covariance_type": self.covariance_type,
            }
        return labels


def state_niche(
    adata: AnnData,
    *,
    use_rep: str = DEFAULT_REP_KEY,
    sample_key: str = DEFAULT_SAMPLE_KEY,
    spatial_key: str = DEFAULT_SPATIAL_KEY,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    n_neighbors: int = 25,
    aggregations: Sequence[str] = ("mean", "std"),
    include_center: bool = False,
    scale: bool = True,
    feature_key: str = DEFAULT_FEATURE_KEY,
    n_clusters: int = 12,
    key_added: str = "state_niche",
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    random_state: int = 123,
    max_cells_per_sample: int | None = 10_000,
    chunk_size: int = 20_000,
    workers: int = -1,
) -> ClusterResult:
    """Convenience wrapper: build state-niche features and fit a fixed-K GMM."""

    build_features(
        adata,
        use_rep=use_rep,
        sample_key=sample_key,
        spatial_key=spatial_key,
        coord_cols=coord_cols,
        n_neighbors=n_neighbors,
        aggregations=aggregations,
        include_center=include_center,
        scale=scale,
        output_key=feature_key,
        chunk_size=chunk_size,
        workers=workers,
    )
    return fit(
        adata,
        n_clusters=n_clusters,
        use_rep=feature_key,
        key_added=key_added,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=random_state,
        sample_key=sample_key,
        max_cells_per_sample=max_cells_per_sample,
    )


__all__ = [
    "SpatialKNNResult",
    "StateNicheFeatures",
    "ClusterResult",
    "build_spatial_neighbors",
    "build_features",
    "balanced_fit_indices",
    "fit",
    "ClusterAutoK",
    "state_niche",
]
