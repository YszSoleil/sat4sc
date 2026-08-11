"""sat4sc.pysphere

Python/AnnData reimplementation and single-cell extension of the SPHERE spatial
displacement framework.

Two interchangeable backends are provided:

``backend="grid"``
    Continuous cell centroids are rasterized to equal-area regular bins, then
    the original eight-direction SPHERE displacement/Jaccard logic is applied.
    This is the most faithful adaptation when spatial *area* should carry equal
    weight and is also suitable for Visium-like lattices.

``backend="kdtree"``
    Continuous cell coordinates are retained. Positive feature cells are
    converted into radius-defined occupancy domains on the real cell anchors
    using :class:`scipy.spatial.cKDTree`. Object B is then virtually displaced
    in eight directions and re-projected onto the fixed anchors before computing
    Jaccard and delta-Jaccard. This is a cell-resolved extension of SPHERE.

The common downstream quantities are unchanged: minimum/maximum delta-Jaccard
at each step, final-step projected score, and vector-path magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.ndimage import label as ndi_label

try:
    from anndata import AnnData
except Exception:  # pragma: no cover
    AnnData = object  # type: ignore


DEFAULT_STEPS = (2, 4, 6, 8, 10, 12)
DEFAULT_KDTREE_STEPS = (25.0, 50.0, 75.0, 100.0, 125.0, 150.0)
DEFAULT_COORD_COLS = ("x_centroid", "y_centroid")
DIRECTIONS = (
    (1, 0, "add_X"),
    (-1, 0, "minus_X"),
    (0, 1, "add_Y"),
    (0, -1, "minus_Y"),
    (1, 1, "add_X_add_Y"),
    (1, -1, "add_X_minus_Y"),
    (-1, 1, "minus_X_add_Y"),
    (-1, -1, "minus_X_minus_Y"),
)


@dataclass
class SpatialAdjusted:
    """Coordinate/feature container analogous to SPHERE::spatial_adjust."""

    coords: np.ndarray
    values: np.ndarray | None
    feature: str | None
    sample: str | None = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=float))


@dataclass
class BinStatResult:
    """Result analogous to the non-exported SPHERE::spatial_binstat helper."""

    stat_mtx_pct: np.ndarray
    stat_mtx_cnt: np.ndarray
    stat_mtx_msk: np.ndarray
    stat_df: pd.DataFrame
    stat_summary: dict | None
    x_edges: np.ndarray
    y_edges: np.ndarray
    min_cutoff: float
    min_count: int
    pct_cutoff: float
    feature: str | None = None


@dataclass
class CordStatResult:
    """Result analogous to SPHERE::spatial_cordstat."""

    mov_stat: pd.DataFrame
    mov_summary: pd.Series
    feature: str
    cutoff_target: float
    cutoff_feature: float
    grid_size: float


@dataclass
class SpatialVectorResult:
    """Single- or multi-sample SPHERE vector result."""

    vectors: pd.DataFrame
    target: str
    features: list[str]
    projected_score: pd.Series
    vector_len: pd.Series
    pool_raw: pd.DataFrame | None = None
    sample_projected_score: pd.DataFrame | None = None
    sample_vector_len: pd.DataFrame | None = None
    settings: dict = field(default_factory=dict)


@dataclass
class PairwiseResult:
    """Pairwise projected-score result, useful for Figure-3A-like heatmaps."""

    matrix: pd.DataFrame
    sample_scores: pd.DataFrame
    features: list[str]
    settings: dict = field(default_factory=dict)


@dataclass
class GridFeatureMap:
    """One rasterized gene/signature map for spatial visualization."""

    grid: np.ndarray
    occupied: np.ndarray
    counts: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    feature: str
    sample: str | None
    grid_size: float
    agg: str
    cutoff: float


@dataclass
class KDTreeDomainResult:
    """Radius-defined target/feature domains on fixed cell anchors."""

    coords: np.ndarray
    target_domain: np.ndarray
    feature_domain: np.ndarray
    target_positive: np.ndarray
    feature_positive: np.ndarray
    target: str
    feature: str
    sample: str | None
    radius: float
    cutoff_target: float
    cutoff_feature: float
    shift: tuple[float, float] = (0.0, 0.0)
    coverage_fraction: float = 1.0


CUTOFF_METHODS = (
    "median_of_sample_medians",
    "mean_of_sample_means",
    "balanced_global_median",
    "balanced_global_mean",
)


@dataclass
class CutoffResult:
    """Cohort-level feature cutoffs for comparable positive cell/grid calls.

    ``cutoffs`` contains one cohort-wide threshold per feature. ``sample_stats``
    records per-sample means/medians and unit counts. For balanced methods,
    ``replicate_cutoffs`` stores the cutoff from every repeated equal-size
    subsampling iteration.
    """

    cutoffs: pd.Series
    sample_stats: pd.DataFrame
    replicate_cutoffs: pd.DataFrame | None
    settings: dict = field(default_factory=dict)


@dataclass
class NicheSampleResult:
    """Per-sample grid-domain representation of a spatial niche.

    A niche is defined as a connected component of positive grids with at least
    ``min_connected_grids`` members. Cell-level membership is inherited from the
    grid containing each cell/spot.
    """

    sample: str
    coords: np.ndarray
    cell_indices: np.ndarray
    grid: np.ndarray
    occupied: np.ndarray
    positive_grid: np.ndarray
    niche_grid: np.ndarray
    component_labels: np.ndarray
    counts: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    cell_positive_grid: np.ndarray
    cell_niche: np.ndarray
    cell_component: np.ndarray
    component_sizes: dict[int, int]


@dataclass
class NicheResult:
    """Cohort-aware spatial niche result for one feature/signature.

    ``positive_grid`` is determined by the selected cutoff strategy. A grid is
    promoted from positive to niche only when it belongs to a spatially
    connected component containing at least ``min_connected_grids`` positive
    grids.
    """

    feature: str
    cutoff: float
    sample_results: dict[str, NicheSampleResult]
    summary: pd.DataFrame
    settings: dict = field(default_factory=dict)
    cutoff_result: CutoffResult | None = None


def _normalize_cutoff_level(level: str) -> str:
    key = str(level).lower().replace("-", "_")
    aliases = {
        "cell": "cell", "cells": "cell", "cell_level": "cell",
        "grid": "grid", "grids": "grid", "grid_level": "grid",
    }
    if key not in aliases:
        raise ValueError("level must be 'cell'/'cell_level' or 'grid'/'grid_level'.")
    return aliases[key]


def _normalize_cutoff_method(method: str) -> str:
    key = str(method).lower().replace("-", "_")
    aliases = {
        "median_of_sample_level_median_scores": "median_of_sample_medians",
        "median_of_sample_level_medians": "median_of_sample_medians",
        "median_of_sample_median": "median_of_sample_medians",
        "median_of_sample_medians": "median_of_sample_medians",
        "mean_of_sample_level_mean_scores": "mean_of_sample_means",
        "mean_of_sample_level_means": "mean_of_sample_means",
        "mean_of_sample_mean": "mean_of_sample_means",
        "mean_of_sample_means": "mean_of_sample_means",
        "balanced_global_median": "balanced_global_median",
        "balanced_global_mean": "balanced_global_mean",
    }
    if key not in aliases:
        raise ValueError(f"Unknown cutoff method {method!r}. Supported methods: {CUTOFF_METHODS}.")
    return aliases[key]


def _coerce_cutoff_mapping(cutoffs) -> dict[str, float]:
    if cutoffs is None:
        return {}
    if isinstance(cutoffs, CutoffResult):
        return {str(k): float(v) for k, v in cutoffs.cutoffs.items() if np.isfinite(v)}
    if isinstance(cutoffs, pd.Series):
        return {str(k): float(v) for k, v in cutoffs.items() if np.isfinite(v)}
    if isinstance(cutoffs, Mapping):
        return {str(k): float(v) for k, v in cutoffs.items() if np.isfinite(v)}
    raise TypeError("cutoffs must be a mapping, pandas Series, CutoffResult, or None.")


def _resolve_feature_cutoff_input(cutoff, feature: str | None, default: float | None = None) -> float | None:
    if isinstance(cutoff, CutoffResult):
        mapping = _coerce_cutoff_mapping(cutoff)
        if feature is None or feature not in mapping:
            raise KeyError(f"Cannot resolve cutoff for feature {feature!r} from CutoffResult.")
        return mapping[feature]
    if isinstance(cutoff, Mapping) or isinstance(cutoff, pd.Series):
        mapping = _coerce_cutoff_mapping(cutoff)
        if feature is None or feature not in mapping:
            raise KeyError(f"Cannot resolve cutoff for feature {feature!r} from cutoff mapping.")
        return mapping[feature]
    if cutoff is None:
        return default
    return float(cutoff)


def _resolve_cutoff_samples(adata: AnnData, sample_key: str, samples: Sequence[str] | None) -> list[str]:
    if sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key {sample_key!r} not found in adata.obs.")
    available = adata.obs[sample_key].astype(str).to_numpy()
    if samples is None:
        return list(pd.unique(available))
    out = list(map(str, samples))
    missing = [s for s in out if not np.any(available == s)]
    if missing:
        raise ValueError(f"No observations found for samples: {missing}.")
    return out


def _balanced_draw_size(min_count: int, balance_round_to: int = 1000, balance_n: int | None = None) -> int:
    if min_count <= 0:
        raise ValueError("At least one sample has zero finite units for cutoff calculation.")
    if balance_n is not None:
        n = int(balance_n)
        if n <= 0:
            raise ValueError("balance_n must be > 0.")
        if n > min_count:
            raise ValueError(f"balance_n={n} exceeds the smallest sample size ({min_count}).")
        return n
    round_to = int(balance_round_to)
    if round_to <= 0:
        raise ValueError("balance_round_to must be > 0.")
    # Example requested by the package design: 12,345 -> 12,000. If the
    # smallest sample has <1,000 units, use the exact minimum rather than zero.
    rounded = (int(min_count) // round_to) * round_to
    return int(min_count) if rounded == 0 else int(rounded)


def calculate_cutoffs(
    adata: AnnData,
    features: Sequence[str] | str,
    sample_key: str = "sample_name",
    samples: Sequence[str] | None = None,
    level: str = "cell",
    method: str = "median_of_sample_medians",
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
    layer: str | None = None,
    grid_size: float = 20.0,
    agg: str = "mean",
    min_cells_per_bin: int = 1,
    n_repeats: int = 100,
    balance_round_to: int = 1000,
    balance_n: int | None = None,
    random_state: int = 666,
) -> CutoffResult:
    """Calculate one cohort-wide cutoff per feature at cell or grid level.

    Supported methods
    -----------------
    ``median_of_sample_medians``
        Compute each sample's median score, then take the median across samples.
    ``mean_of_sample_means``
        Compute each sample's mean score, then take the mean across samples.
    ``balanced_global_median``
        For every repeat, draw an equal number of units from every sample,
        pool them, and calculate the pooled median. The final cutoff is the
        median of the repeat-specific pooled medians.
    ``balanced_global_mean``
        Same equal-size repeated sampling, but calculate the pooled mean. The
        final cutoff is the mean of the repeat-specific pooled means.

    ``level='cell'`` uses finite cell/spot feature values. ``level='grid'`` first
    rasterizes every sample independently with the same grid settings used by
    grid-SPHERE, then uses occupied finite grid scores as the units.

    For balanced methods the default draw size is the smallest sample's finite
    unit count rounded down to a multiple of ``balance_round_to`` (default
    1,000; e.g. 12,345 -> 12,000). If the smallest sample has fewer than 1,000
    units, its exact count is used. Sampling is without replacement and repeated
    ``n_repeats=100`` times by default.
    """
    level = _normalize_cutoff_level(level)
    method = _normalize_cutoff_method(method)
    features = [str(features)] if isinstance(features, str) else list(map(str, features))
    if not features:
        raise ValueError("features must contain at least one feature.")
    samples = _resolve_cutoff_samples(adata, sample_key, samples)
    if int(n_repeats) < 1:
        raise ValueError("n_repeats must be >= 1.")
    sample_values_all = adata.obs[sample_key].astype(str).to_numpy()
    sample_indices = {s: np.flatnonzero(sample_values_all == s) for s in samples}
    grid_geometry = {}
    if level == "grid":
        for sample in samples:
            idx = sample_indices[sample]
            coords = _resolve_coords(adata, idx, coord_cols=coord_cols, spatial_key=spatial_key)
            grid_geometry[sample] = _make_grid(coords, grid_size, min_cells_per_bin)

    cutoff_values = {}
    sample_rows = []
    repeat_rows = []
    balance_n_by_feature = {}
    for feature_i, feature in enumerate(features):
        arrays = []
        for sample in samples:
            idx = sample_indices[sample]
            values = _resolve_feature_vector(adata, feature, idx, layer=layer)
            if level == "cell":
                arr = np.asarray(values, dtype=float)
                arr = arr[np.isfinite(arr)]
            else:
                flat, counts, occupied, shape, _ = grid_geometry[sample]
                grid = _aggregate_grid(values, flat, counts, shape, agg=agg)
                arr = grid[occupied & np.isfinite(grid)].astype(float, copy=False)
            if arr.size == 0:
                raise ValueError(f"Feature {feature!r} has no finite {level} values in sample {sample!r}.")
            arrays.append(arr)
            sample_rows.append({
                "feature": feature,
                "sample": sample,
                "level": level,
                "n_units": int(arr.size),
                "sample_mean": float(np.mean(arr)),
                "sample_median": float(np.median(arr)),
            })

        if method == "median_of_sample_medians":
            cutoff = float(np.median([np.median(x) for x in arrays]))
        elif method == "mean_of_sample_means":
            cutoff = float(np.mean([np.mean(x) for x in arrays]))
        else:
            min_count = min(len(x) for x in arrays)
            n_draw = _balanced_draw_size(min_count, balance_round_to=balance_round_to, balance_n=balance_n)
            balance_n_by_feature[feature] = int(n_draw)
            # A feature-specific RNG makes results reproducible while avoiding
            # identical draw streams for every feature.
            feature_seed = int(zlib.crc32(feature.encode("utf-8")))
            rng = np.random.default_rng(np.random.SeedSequence([int(random_state), feature_seed]))
            rep_vals = []
            for rep in range(int(n_repeats)):
                drawn = []
                for arr in arrays:
                    if len(arr) == n_draw:
                        sub = arr
                    else:
                        take = rng.choice(len(arr), size=n_draw, replace=False)
                        sub = arr[take]
                    drawn.append(sub)
                pooled = np.concatenate(drawn)
                rep_cutoff = float(np.median(pooled) if method == "balanced_global_median" else np.mean(pooled))
                rep_vals.append(rep_cutoff)
                repeat_rows.append({
                    "feature": feature,
                    "repeat": rep + 1,
                    "cutoff": rep_cutoff,
                    "n_per_sample": int(n_draw),
                    "n_samples": int(len(samples)),
                    "level": level,
                })
            cutoff = float(np.median(rep_vals) if method == "balanced_global_median" else np.mean(rep_vals))
        cutoff_values[feature] = cutoff

    return CutoffResult(
        cutoffs=pd.Series(cutoff_values, dtype=float, name="cutoff"),
        sample_stats=pd.DataFrame(sample_rows),
        replicate_cutoffs=pd.DataFrame(repeat_rows) if repeat_rows else None,
        settings={
            "level": level,
            "method": method,
            "sample_key": sample_key,
            "samples": samples,
            "grid_size": float(grid_size) if level == "grid" else None,
            "agg": str(agg) if level == "grid" else None,
            "min_cells_per_bin": int(min_cells_per_bin) if level == "grid" else None,
            "n_repeats": int(n_repeats) if method.startswith("balanced_global_") else None,
            "balance_round_to": int(balance_round_to) if method.startswith("balanced_global_") else None,
            "balance_n": balance_n,
            "balance_n_by_feature": balance_n_by_feature,
            "random_state": int(random_state) if method.startswith("balanced_global_") else None,
            "coord_cols": tuple(coord_cols),
            "spatial_key": spatial_key,
            "layer": layer,
        },
    )


def calculate_global_cutoffs(*args, **kwargs) -> CutoffResult:
    """Alias for :func:`calculate_cutoffs`."""
    return calculate_cutoffs(*args, **kwargs)


def _connected_component_structure(connectivity: int) -> np.ndarray:
    """Return a 2-D 4- or 8-neighbor connectivity structure."""
    connectivity = int(connectivity)
    if connectivity == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    if connectivity == 8:
        return np.ones((3, 3), dtype=np.int8)
    raise ValueError("connectivity must be 4 or 8.")


def _filter_positive_components(
    positive_grid: np.ndarray,
    min_connected_grids: int,
    connectivity: int,
):
    """Convert positive grids into retained niche components.

    Returns
    -------
    niche_grid
        Boolean mask containing only retained components.
    component_labels
        Integer labels for retained components; 0 denotes non-niche grids.
    component_sizes
        Mapping from retained component id to number of grids.
    raw_component_sizes
        Mapping for all positive components before minimum-size filtering.
    """
    min_connected_grids = int(min_connected_grids)
    if min_connected_grids < 1:
        raise ValueError("min_connected_grids must be >= 1.")
    positive_grid = np.asarray(positive_grid, dtype=bool)
    labels_raw, n_raw = ndi_label(positive_grid, structure=_connected_component_structure(connectivity))
    raw_sizes = {}
    kept_old_ids = []
    for old_id in range(1, int(n_raw) + 1):
        size = int(np.sum(labels_raw == old_id))
        raw_sizes[old_id] = size
        if size >= min_connected_grids:
            kept_old_ids.append(old_id)
    niche_grid = np.isin(labels_raw, kept_old_ids)
    component_labels = np.zeros_like(labels_raw, dtype=np.int32)
    component_sizes = {}
    for new_id, old_id in enumerate(kept_old_ids, start=1):
        mask = labels_raw == old_id
        component_labels[mask] = new_id
        component_sizes[new_id] = int(mask.sum())
    return niche_grid, component_labels, component_sizes, raw_sizes


def define_niche(
    adata: AnnData,
    feature: str,
    sample_key: str = "sample_name",
    samples: Sequence[str] | None = None,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
    layer: str | None = None,
    grid_size: float = 20.0,
    agg: str = "mean",
    min_cells_per_bin: int = 1,
    cutoff: float | None = None,
    cutoff_method: str | None = "balanced_global_mean",
    cohort_cutoffs=None,
    cutoff_samples: Sequence[str] | None = None,
    cutoff_n_repeats: int = 100,
    cutoff_balance_round_to: int = 1000,
    cutoff_balance_n: int | None = None,
    cutoff_random_state: int = 666,
    min_connected_grids: int = 3,
    connectivity: int = 8,
    annotate_obs: bool = False,
    obs_prefix: str | None = None,
) -> NicheResult:
    """Define a spatial niche as ``positive grid + spatial continuity``.

    The feature is rasterized independently within each sample using the same
    regular-grid geometry as ``backend='grid'``. Grids with score greater than
    the selected cutoff are called positive. Positive grids are then labeled by
    4- or 8-neighbor connected components, and only components containing at
    least ``min_connected_grids`` grids are retained as niches.

    By default, the cutoff is ``balanced_global_mean`` calculated at grid level,
    so every sample is evaluated using the same cohort-wide threshold while
    samples contribute equally to threshold estimation. Pass ``cutoff`` for an
    explicit threshold, or ``cohort_cutoffs`` to reuse a precomputed mapping or
    :class:`CutoffResult`.

    Cell/spot niche membership is inherited from the retained niche grid that
    contains each cell/spot. When ``annotate_obs=True``, three columns are added:
    ``<prefix>_positive_grid``, ``<prefix>_niche`` and ``<prefix>_niche_id``.
    """
    feature = str(feature)
    samples_resolved = _resolve_cutoff_samples(adata, sample_key, samples)
    cutoff_cohort = list(map(str, cutoff_samples)) if cutoff_samples is not None else list(samples_resolved)
    cutoff_result = None
    mapping = _coerce_cutoff_mapping(cohort_cutoffs)
    if cutoff is not None:
        chosen_cutoff = float(cutoff)
    elif feature in mapping:
        chosen_cutoff = float(mapping[feature])
    elif cutoff_method is not None:
        cutoff_result = calculate_cutoffs(
            adata,
            features=[feature],
            sample_key=sample_key,
            samples=cutoff_cohort,
            level="grid",
            method=cutoff_method,
            coord_cols=coord_cols,
            spatial_key=spatial_key,
            layer=layer,
            grid_size=grid_size,
            agg=agg,
            min_cells_per_bin=min_cells_per_bin,
            n_repeats=cutoff_n_repeats,
            balance_round_to=cutoff_balance_round_to,
            balance_n=cutoff_balance_n,
            random_state=cutoff_random_state,
        )
        chosen_cutoff = float(cutoff_result.cutoffs[feature])
    else:
        raise ValueError(
            "A shared niche cutoff is required. Provide cutoff, cohort_cutoffs, "
            "or cutoff_method."
        )
    if not np.isfinite(chosen_cutoff):
        raise ValueError(f"Resolved cutoff for {feature!r} is not finite.")

    sample_values_all = adata.obs[sample_key].astype(str).to_numpy()
    sample_results = {}
    summary_rows = []
    obs_positive = np.zeros(int(adata.n_obs), dtype=bool)
    obs_niche = np.zeros(int(adata.n_obs), dtype=bool)
    obs_component = np.zeros(int(adata.n_obs), dtype=np.int32)

    for sample in samples_resolved:
        idx = np.flatnonzero(sample_values_all == str(sample))
        coords = _resolve_coords(adata, idx, coord_cols=coord_cols, spatial_key=spatial_key)
        values = _resolve_feature_vector(adata, feature, idx, layer=layer)
        flat, counts_flat, occupied, shape, xy_min = _make_grid(coords, grid_size, min_cells_per_bin)
        grid = _aggregate_grid(values, flat, counts_flat, shape, agg=agg)
        positive_grid = occupied & np.isfinite(grid) & (grid > chosen_cutoff)
        niche_grid, component_labels, component_sizes, raw_component_sizes = _filter_positive_components(
            positive_grid,
            min_connected_grids=min_connected_grids,
            connectivity=connectivity,
        )
        ny, nx = shape
        x_edges = xy_min[0] + np.arange(nx + 1, dtype=float) * float(grid_size)
        y_edges = xy_min[1] + np.arange(ny + 1, dtype=float) * float(grid_size)
        positive_flat = positive_grid.ravel()
        niche_flat = niche_grid.ravel()
        component_flat = component_labels.ravel()
        cell_positive_grid = positive_flat[flat]
        cell_niche = niche_flat[flat]
        cell_component = component_flat[flat].astype(np.int32, copy=False)
        obs_positive[idx] = cell_positive_grid
        obs_niche[idx] = cell_niche
        obs_component[idx] = cell_component
        n_occupied = int(occupied.sum())
        n_positive = int(positive_grid.sum())
        n_niche_grid = int(niche_grid.sum())
        n_niches = int(len(component_sizes))
        largest_niche = int(max(component_sizes.values())) if component_sizes else 0
        n_cells = int(len(idx))
        n_positive_cells = int(cell_positive_grid.sum())
        n_niche_cells = int(cell_niche.sum())
        sample_results[str(sample)] = NicheSampleResult(
            sample=str(sample),
            coords=np.asarray(coords, dtype=float),
            cell_indices=idx.astype(np.int64, copy=False),
            grid=np.asarray(grid, dtype=float),
            occupied=np.asarray(occupied, dtype=bool),
            positive_grid=np.asarray(positive_grid, dtype=bool),
            niche_grid=np.asarray(niche_grid, dtype=bool),
            component_labels=np.asarray(component_labels, dtype=np.int32),
            counts=np.asarray(counts_flat, dtype=np.int32).reshape(shape),
            x_edges=np.asarray(x_edges, dtype=float),
            y_edges=np.asarray(y_edges, dtype=float),
            cell_positive_grid=np.asarray(cell_positive_grid, dtype=bool),
            cell_niche=np.asarray(cell_niche, dtype=bool),
            cell_component=np.asarray(cell_component, dtype=np.int32),
            component_sizes=component_sizes,
        )
        summary_rows.append({
            "sample": str(sample),
            "feature": feature,
            "cutoff": chosen_cutoff,
            "n_occupied_grids": n_occupied,
            "n_positive_grids": n_positive,
            "positive_grid_fraction": np.nan if n_occupied == 0 else n_positive / n_occupied,
            "n_niche_grids": n_niche_grid,
            "niche_grid_fraction": np.nan if n_occupied == 0 else n_niche_grid / n_occupied,
            "n_positive_components": int(len(raw_component_sizes)),
            "n_niches": n_niches,
            "largest_niche_grids": largest_niche,
            "n_cells": n_cells,
            "n_cells_in_positive_grid": n_positive_cells,
            "positive_grid_cell_fraction": np.nan if n_cells == 0 else n_positive_cells / n_cells,
            "n_niche_cells": n_niche_cells,
            "niche_cell_fraction": np.nan if n_cells == 0 else n_niche_cells / n_cells,
            "niche_area": float(n_niche_grid * float(grid_size) ** 2),
        })

    prefix = obs_prefix
    if prefix is None:
        prefix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in feature).strip("_") or "feature"
    if annotate_obs:
        # Samples excluded by ``samples`` remain False/0, making the scope explicit.
        adata.obs[f"{prefix}_positive_grid"] = obs_positive
        adata.obs[f"{prefix}_niche"] = obs_niche
        adata.obs[f"{prefix}_niche_id"] = obs_component

    return NicheResult(
        feature=feature,
        cutoff=chosen_cutoff,
        sample_results=sample_results,
        summary=pd.DataFrame(summary_rows),
        settings={
            "definition": "positive_grid_plus_connected_component",
            "sample_key": sample_key,
            "samples": list(samples_resolved),
            "coord_cols": tuple(coord_cols),
            "spatial_key": spatial_key,
            "layer": layer,
            "grid_size": float(grid_size),
            "agg": str(agg),
            "min_cells_per_bin": int(min_cells_per_bin),
            "cutoff_method": cutoff_method if cutoff is None and feature not in mapping else "precomputed_or_manual",
            "cutoff_samples": cutoff_cohort,
            "cutoff_n_repeats": int(cutoff_n_repeats),
            "cutoff_balance_round_to": int(cutoff_balance_round_to),
            "cutoff_balance_n": cutoff_balance_n,
            "cutoff_random_state": int(cutoff_random_state),
            "min_connected_grids": int(min_connected_grids),
            "connectivity": int(connectivity),
            "annotate_obs": bool(annotate_obs),
            "obs_prefix": prefix if annotate_obs else None,
        },
        cutoff_result=cutoff_result,
    )


def define_niches(
    adata: AnnData,
    features: Sequence[str],
    **kwargs,
) -> dict[str, NicheResult]:
    """Define multiple grid-based niches using one shared cutoff calculation.

    This is a convenience wrapper around :func:`define_niche`. When no explicit
    ``cutoff``/``cohort_cutoffs`` are supplied, cohort-wide grid cutoffs for all
    requested features are calculated once and reused.
    """
    features = list(map(str, features))
    if not features:
        raise ValueError("features must contain at least one feature.")
    if "cutoff" in kwargs and kwargs["cutoff"] is not None:
        raise ValueError("define_niches() does not accept one scalar cutoff for multiple features; use cohort_cutoffs.")
    cohort_cutoffs = kwargs.pop("cohort_cutoffs", None)
    cutoff_method = kwargs.get("cutoff_method", "balanced_global_mean")
    mapping = _coerce_cutoff_mapping(cohort_cutoffs)
    missing = [f for f in features if f not in mapping]
    shared_cutoff_result = None
    if missing and cutoff_method is not None:
        sample_key = kwargs.get("sample_key", "sample_name")
        samples = kwargs.get("samples", None)
        cutoff_samples = kwargs.get("cutoff_samples", None)
        samples_resolved = _resolve_cutoff_samples(adata, sample_key, samples)
        cutoff_cohort = list(map(str, cutoff_samples)) if cutoff_samples is not None else list(samples_resolved)
        shared_cutoff_result = calculate_cutoffs(
            adata,
            features=missing,
            sample_key=sample_key,
            samples=cutoff_cohort,
            level="grid",
            method=cutoff_method,
            coord_cols=kwargs.get("coord_cols", DEFAULT_COORD_COLS),
            spatial_key=kwargs.get("spatial_key", "spatial"),
            layer=kwargs.get("layer", None),
            grid_size=kwargs.get("grid_size", 20.0),
            agg=kwargs.get("agg", "mean"),
            min_cells_per_bin=kwargs.get("min_cells_per_bin", 1),
            n_repeats=kwargs.get("cutoff_n_repeats", 100),
            balance_round_to=kwargs.get("cutoff_balance_round_to", 1000),
            balance_n=kwargs.get("cutoff_balance_n", None),
            random_state=kwargs.get("cutoff_random_state", 666),
        )
        mapping.update(shared_cutoff_result.cutoffs.to_dict())
    if missing and cutoff_method is None:
        raise ValueError("Missing cutoffs for one or more features and cutoff_method=None.")

    out = {}
    for feature in features:
        one_kwargs = dict(kwargs)
        one_kwargs["cutoff_method"] = None
        one_kwargs["cohort_cutoffs"] = mapping
        result = define_niche(adata, feature=feature, **one_kwargs)
        if shared_cutoff_result is not None and feature in shared_cutoff_result.cutoffs.index:
            result.cutoff_result = shared_cutoff_result
            result.settings["cutoff_method"] = cutoff_method
        out[feature] = result
    return out


def _merge_calculated_cutoffs(
    adata: AnnData,
    features: Sequence[str],
    cutoffs,
    cutoff_method: str | None,
    cutoff_level: str,
    sample_key: str,
    cutoff_samples: Sequence[str] | None,
    coord_cols: Sequence[str],
    spatial_key: str,
    layer: str | None,
    grid_size: float,
    agg: str,
    min_cells_per_bin: int,
    cutoff_n_repeats: int,
    cutoff_balance_round_to: int,
    cutoff_balance_n: int | None,
    cutoff_random_state: int,
):
    explicit = _coerce_cutoff_mapping(cutoffs)
    if cutoff_method is None:
        return explicit, None
    needed = [str(f) for f in features if str(f) not in explicit]
    if not needed:
        return explicit, None
    result = calculate_cutoffs(
        adata,
        features=needed,
        sample_key=sample_key,
        samples=cutoff_samples,
        level=cutoff_level,
        method=cutoff_method,
        coord_cols=coord_cols,
        spatial_key=spatial_key,
        layer=layer,
        grid_size=grid_size,
        agg=agg,
        min_cells_per_bin=min_cells_per_bin,
        n_repeats=cutoff_n_repeats,
        balance_round_to=cutoff_balance_round_to,
        balance_n=cutoff_balance_n,
        random_state=cutoff_random_state,
    )
    merged = result.cutoffs.to_dict()
    merged.update(explicit)  # explicit values always override calculated values
    return {str(k): float(v) for k, v in merged.items()}, result


def positive_proportions(
    adata: AnnData,
    features: Sequence[str] | str,
    sample_key: str = "sample_name",
    samples: Sequence[str] | None = None,
    level: str = "cell",
    cutoff_method: str = "median_of_sample_medians",
    cutoffs=None,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
    layer: str | None = None,
    grid_size: float = 20.0,
    agg: str = "mean",
    min_cells_per_bin: int = 1,
    cutoff_n_repeats: int = 100,
    cutoff_balance_round_to: int = 1000,
    cutoff_balance_n: int | None = None,
    cutoff_random_state: int = 666,
) -> pd.DataFrame:
    """Summarize positive cell/grid proportions per sample using shared cutoffs."""
    level = _normalize_cutoff_level(level)
    features = [str(features)] if isinstance(features, str) else list(map(str, features))
    samples = _resolve_cutoff_samples(adata, sample_key, samples)
    resolved, _ = _merge_calculated_cutoffs(
        adata, features, cutoffs, cutoff_method, level, sample_key, samples,
        coord_cols, spatial_key, layer, grid_size, agg, min_cells_per_bin,
        cutoff_n_repeats, cutoff_balance_round_to, cutoff_balance_n, cutoff_random_state,
    )
    sample_values_all = adata.obs[sample_key].astype(str).to_numpy()
    rows = []
    for sample in samples:
        idx = np.flatnonzero(sample_values_all == sample)
        coords = None
        geometry = None
        if level == "grid":
            coords = _resolve_coords(adata, idx, coord_cols=coord_cols, spatial_key=spatial_key)
            geometry = _make_grid(coords, grid_size, min_cells_per_bin)
        for feature in features:
            values = _resolve_feature_vector(adata, feature, idx, layer=layer)
            if level == "cell":
                vals = np.asarray(values, dtype=float)
                finite = np.isfinite(vals)
                unit_values = vals[finite]
            else:
                flat, counts, occupied, shape, _ = geometry
                grid = _aggregate_grid(values, flat, counts, shape, agg=agg)
                unit_values = grid[occupied & np.isfinite(grid)]
            cutoff = float(resolved[feature])
            n = int(unit_values.size)
            n_pos = int(np.count_nonzero(unit_values > cutoff))
            rows.append({
                "sample": sample,
                "feature": feature,
                "level": level,
                "cutoff": cutoff,
                "n_units": n,
                "n_positive": n_pos,
                "positive_fraction": np.nan if n == 0 else n_pos / n,
            })
    return pd.DataFrame(rows)


def _as_index_array(mask_or_index, n_obs: int) -> np.ndarray:
    if mask_or_index is None:
        return np.arange(n_obs, dtype=np.int64)
    arr = np.asarray(mask_or_index)
    if arr.dtype == bool:
        if arr.size != n_obs:
            raise ValueError("Boolean mask length does not match adata.n_obs.")
        return np.flatnonzero(arr)
    return arr.astype(np.int64, copy=False)


def _resolve_coords(
    adata: AnnData,
    obs_idx: np.ndarray,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
) -> np.ndarray:
    if all(col in adata.obs.columns for col in coord_cols):
        coords = adata.obs.iloc[obs_idx][list(coord_cols)].to_numpy(dtype=np.float64, copy=True)
    elif spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key][obs_idx, :2], dtype=np.float64)
    else:
        raise KeyError(
            f"Cannot find coordinates. Expected obs columns {tuple(coord_cols)} "
            f"or adata.obsm[{spatial_key!r}]."
        )
    ok = np.isfinite(coords).all(axis=1)
    if not ok.all():
        raise ValueError(f"Found {(~ok).sum()} cells/spots with non-finite spatial coordinates.")
    return coords


def _get_matrix(adata: AnnData, layer: str | None = None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer {layer!r} not found in adata.layers.")
    return adata.layers[layer]


def _resolve_feature_vector(
    adata: AnnData,
    feature: str,
    obs_idx: np.ndarray,
    layer: str | None = None,
) -> np.ndarray:
    if feature in adata.obs.columns:
        s = adata.obs.iloc[obs_idx][feature]
        if pd.api.types.is_bool_dtype(s.dtype):
            return s.to_numpy(dtype=np.float64)
        if not pd.api.types.is_numeric_dtype(s.dtype):
            raise TypeError(
                f"adata.obs[{feature!r}] is not numeric. Convert a category to an indicator "
                "column first, e.g. adata.obs['BMDM'] = (adata.obs['cell_type'] == 'BMDM').astype(float)."
            )
        return s.to_numpy(dtype=np.float64)

    if feature in adata.var_names:
        j = int(adata.var_names.get_loc(feature))
        X = _get_matrix(adata, layer=layer)
        col = X[obs_idx, j]
        if sparse.issparse(col):
            return np.asarray(col.toarray()).ravel().astype(np.float64, copy=False)
        return np.asarray(col).ravel().astype(np.float64, copy=False)

    raise KeyError(f"Feature {feature!r} not found in adata.obs or adata.var_names.")


def spatial_adjust(
    adata: AnnData,
    feature: str | None = None,
    sample: str | None = None,
    sample_key: str = "sample_name",
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
    layer: str | None = None,
    rotate: int = 0,
    mirror: str = "N",
    operator: np.ndarray | None = None,
) -> SpatialAdjusted:
    """Extract coordinates and one feature from AnnData.

    Parameters mirror: 'N', 'X', or 'Y'. Rotation is clockwise by 0/90/180/270.
    The transform does not affect SPHERE scores when all eight directions are used,
    but is kept for parity with the R package and for spatial plotting.
    """
    if rotate not in (0, 90, 180, 270):
        raise ValueError("rotate must be one of 0, 90, 180, 270.")
    if mirror not in ("N", "X", "Y"):
        raise ValueError("mirror must be one of 'N', 'X', 'Y'.")

    if sample is None:
        obs_idx = np.arange(adata.n_obs, dtype=np.int64)
    else:
        if sample_key not in adata.obs.columns:
            raise KeyError(f"sample_key {sample_key!r} not found in adata.obs.")
        obs_idx = np.flatnonzero(adata.obs[sample_key].astype(str).to_numpy() == str(sample))
        if obs_idx.size == 0:
            raise ValueError(f"No observations found for {sample_key}={sample!r}.")

    coords = _resolve_coords(adata, obs_idx, coord_cols=coord_cols, spatial_key=spatial_key)
    if operator is not None:
        transform = np.asarray(operator, dtype=float)
        if transform.shape != (2, 2):
            raise ValueError("operator must be a 2x2 matrix.")
    else:
        angle = np.deg2rad(-rotate)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float)
        mir = np.eye(2)
        if mirror == "X":
            mir[0, 0] = -1
        elif mirror == "Y":
            mir[1, 1] = -1
        transform = rot @ mir
    coords = coords @ transform.T
    values = None if feature is None else _resolve_feature_vector(adata, feature, obs_idx, layer=layer)
    return SpatialAdjusted(coords=coords, values=values, feature=feature, sample=sample, transform=transform)


def spatial_binstat(
    obj: SpatialAdjusted,
    bins: tuple[int, int] = (8, 8),
    min_cutoff=0.0,
    min_count: int = 10,
    pct_cutoff: float = 0.5,
    mask: np.ndarray | None = None,
) -> BinStatResult:
    """Bin a spatial feature and summarize the fraction above a cutoff.

    ``min_cutoff`` can be a numeric value, a mapping keyed by feature name, or a
    :class:`CutoffResult`. ``None`` retains the legacy behavior of using the mean
    of this object's finite feature values.
    """
    if obj.values is None:
        raise ValueError("SpatialAdjusted object must contain feature values.")
    ny, nx = map(int, bins)
    if ny < 1 or nx < 1:
        raise ValueError("bins must contain positive integers.")
    values = np.asarray(obj.values, dtype=float)
    coords = np.asarray(obj.coords, dtype=float)
    if min_cutoff is None:
        resolved_cutoff = float(np.nanmean(values))
    else:
        resolved_cutoff = _resolve_feature_cutoff_input(min_cutoff, obj.feature)
        if resolved_cutoff is None:
            resolved_cutoff = float(np.nanmean(values))
    x_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max(), nx + 1)
    y_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max(), ny + 1)
    xb = np.clip(np.digitize(coords[:, 0], x_edges[1:-1], right=False), 0, nx - 1)
    yb = np.clip(np.digitize(coords[:, 1], y_edges[1:-1], right=False), 0, ny - 1)
    row = ny - 1 - yb
    stat_pct = np.full((ny, nx), np.nan, dtype=float)
    stat_cnt = np.zeros((ny, nx), dtype=int)
    for r in range(ny):
        for c in range(nx):
            take = (row == r) & (xb == c) & np.isfinite(values)
            n = int(take.sum())
            stat_cnt[r, c] = n
            if n > int(min_count):
                stat_pct[r, c] = float(np.mean(values[take] > float(resolved_cutoff)))
    if mask is None:
        stat_msk = (np.isfinite(stat_pct) & (stat_pct > float(pct_cutoff))).astype(int)
    else:
        stat_msk = np.asarray(mask, dtype=int)
        if stat_msk.shape != stat_pct.shape:
            raise ValueError("mask shape must match bins.")
    stat_df = pd.DataFrame({"feature": stat_pct.ravel(), "group": stat_msk.ravel()})
    stat_summary = None
    g0 = stat_df.loc[stat_df["group"] == 0, "feature"].dropna()
    g1 = stat_df.loc[stat_df["group"] == 1, "feature"].dropna()
    if len(g0) > 1 and len(g1) > 1:
        from scipy.stats import mannwhitneyu
        test = mannwhitneyu(g1, g0, alternative="two-sided")
        stat_summary = {
            "group_1": float(g1.mean()),
            "group_0": float(g0.mean()),
            "statistic": float(test.statistic),
            "pvalue": float(test.pvalue),
        }
    return BinStatResult(
        stat_mtx_pct=stat_pct, stat_mtx_cnt=stat_cnt, stat_mtx_msk=stat_msk,
        stat_df=stat_df, stat_summary=stat_summary, x_edges=x_edges, y_edges=y_edges,
        min_cutoff=float(resolved_cutoff), min_count=int(min_count), pct_cutoff=float(pct_cutoff),
        feature=obj.feature,
    )



def _make_grid(coords: np.ndarray, grid_size: float, min_cells_per_bin: int = 1):
    if grid_size <= 0:
        raise ValueError("grid_size must be > 0.")
    xy_min = coords.min(axis=0)
    gx = np.floor((coords[:, 0] - xy_min[0]) / grid_size + 1e-12).astype(np.int64)
    gy = np.floor((coords[:, 1] - xy_min[1]) / grid_size + 1e-12).astype(np.int64)
    nx = int(gx.max()) + 1
    ny = int(gy.max()) + 1
    flat = gy * nx + gx
    counts = np.bincount(flat, minlength=nx * ny).astype(np.int32, copy=False)
    occupied = (counts >= int(min_cells_per_bin)).reshape(ny, nx)
    return flat, counts, occupied, (ny, nx), xy_min


def _aggregate_grid(
    values: np.ndarray,
    flat: np.ndarray,
    counts: np.ndarray,
    shape: tuple[int, int],
    agg: str = "mean",
) -> np.ndarray:
    n = counts.size
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        flat_use = flat[finite]
        val_use = values[finite]
        valid_counts = np.bincount(flat_use, minlength=n)
    else:
        flat_use = flat
        val_use = values
        valid_counts = counts

    if agg == "mean":
        sums = np.bincount(flat_use, weights=val_use, minlength=n)
        out = np.full(n, np.nan, dtype=np.float64)
        nz = valid_counts > 0
        out[nz] = sums[nz] / valid_counts[nz]
    elif agg == "sum":
        out = np.bincount(flat_use, weights=val_use, minlength=n).astype(np.float64, copy=False)
        out[valid_counts == 0] = np.nan
    elif agg == "max":
        out = np.full(n, -np.inf, dtype=np.float64)
        np.maximum.at(out, flat_use, val_use)
        out[~np.isfinite(out)] = np.nan
    else:
        raise ValueError("agg must be one of {'mean', 'sum', 'max'}.")
    return out.reshape(shape)



def grid_feature_map(
    adata: AnnData,
    feature: str,
    sample: str | None = None,
    sample_key: str = "sample_name",
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
    layer: str | None = None,
    grid_size: float = 20.0,
    agg: str = "mean",
    cutoff: float | None = None,
    quantile_cutoff: float | None = None,
    min_cells_per_bin: int = 1,
    cutoff_method: str | None = None,
    cutoff_samples: Sequence[str] | None = None,
    cutoff_n_repeats: int = 100,
    cutoff_balance_round_to: int = 1000,
    cutoff_balance_n: int | None = None,
    cutoff_random_state: int = 666,
) -> GridFeatureMap:
    """Rasterize one gene/signature to the same grid used by ``backend='grid'``.

    By default, ``cutoff`` is the mean of occupied finite grids in the displayed
    sample, matching the legacy SPHERE-style behavior. Set ``cutoff_method`` to
    one of the cohort-level methods returned by :func:`calculate_cutoffs` to use
    one shared grid-level cutoff across samples.
    """
    if sample is None:
        idx = np.arange(adata.n_obs, dtype=np.int64)
        sample_name = None
    else:
        if sample_key not in adata.obs.columns:
            raise KeyError(f"sample_key {sample_key!r} not found in adata.obs.")
        idx = np.flatnonzero(adata.obs[sample_key].astype(str).to_numpy() == str(sample))
        if idx.size == 0:
            raise ValueError(f"No observations found for {sample_key}={sample!r}.")
        sample_name = str(sample)
    coords = _resolve_coords(adata, idx, coord_cols=coord_cols, spatial_key=spatial_key)
    values = _resolve_feature_vector(adata, feature, idx, layer=layer)
    flat, counts, occupied, shape, xy_min = _make_grid(coords, grid_size, min_cells_per_bin)
    grid = _aggregate_grid(values, flat, counts, shape, agg=agg)
    vals = grid[occupied & np.isfinite(grid)]
    if vals.size == 0:
        chosen = np.nan
    elif quantile_cutoff is not None:
        if cutoff_method is not None:
            raise ValueError("quantile_cutoff and cutoff_method cannot be used together.")
        if not 0 <= float(quantile_cutoff) <= 1:
            raise ValueError("quantile_cutoff must be in [0, 1].")
        chosen = float(np.quantile(vals, float(quantile_cutoff)))
    elif cutoff is not None:
        chosen = float(cutoff)
    elif cutoff_method is not None:
        cr = calculate_cutoffs(
            adata,
            features=[feature],
            sample_key=sample_key,
            samples=cutoff_samples,
            level="grid",
            method=cutoff_method,
            coord_cols=coord_cols,
            spatial_key=spatial_key,
            layer=layer,
            grid_size=grid_size,
            agg=agg,
            min_cells_per_bin=min_cells_per_bin,
            n_repeats=cutoff_n_repeats,
            balance_round_to=cutoff_balance_round_to,
            balance_n=cutoff_balance_n,
            random_state=cutoff_random_state,
        )
        chosen = float(cr.cutoffs[feature])
    else:
        chosen = float(np.mean(vals))
    ny, nx = shape
    x_edges = xy_min[0] + np.arange(nx + 1, dtype=float) * float(grid_size)
    y_edges = xy_min[1] + np.arange(ny + 1, dtype=float) * float(grid_size)
    return GridFeatureMap(
        grid=grid,
        occupied=occupied,
        counts=counts.reshape(shape),
        x_edges=x_edges,
        y_edges=y_edges,
        feature=str(feature),
        sample=sample_name,
        grid_size=float(grid_size),
        agg=str(agg),
        cutoff=chosen,
    )


def _choose_cutoff(
    feature: str,
    grid: np.ndarray,
    occupied: np.ndarray,
    cutoffs=None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
) -> float:
    vals = grid[occupied & np.isfinite(grid)]
    if vals.size == 0:
        return np.nan
    if quantile_cutoffs is not None:
        q = quantile_cutoffs.get(feature, None) if isinstance(quantile_cutoffs, Mapping) else quantile_cutoffs
        if q is not None:
            if not 0 <= float(q) <= 1:
                raise ValueError("Quantile cutoffs must be in [0, 1].")
            return float(np.quantile(vals, float(q)))
    mapping = _coerce_cutoff_mapping(cutoffs)
    if feature in mapping:
        return float(mapping[feature])
    return float(np.mean(vals))



def _aligned_slices(n: int, delta: int):
    if abs(delta) >= n:
        return None
    if delta >= 0:
        src = slice(0, n - delta)
        dst = slice(delta, n)
    else:
        src = slice(-delta, n)
        dst = slice(0, n + delta)
    return src, dst


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return np.nan
    return inter / union


def _shifted_jaccard(
    target_pos: np.ndarray,
    feature_pos: np.ndarray,
    occupied: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[float, int, int, int, int]:
    ny, nx = occupied.shape
    xs = _aligned_slices(nx, dx)
    ys = _aligned_slices(ny, dy)
    if xs is None or ys is None:
        return np.nan, 0, 0, 0, 0
    src_x, dst_x = xs
    src_y, dst_y = ys

    occ_dst = occupied[dst_y, dst_x]
    occ_src = occupied[src_y, src_x]
    domain = occ_dst & occ_src
    if not domain.any():
        return np.nan, 0, 0, 0, 0

    a = target_pos[dst_y, dst_x][domain]
    b = feature_pos[src_y, src_x][domain]
    inter = int(np.count_nonzero(a & b))
    n_a = int(np.count_nonzero(a))
    n_b = int(np.count_nonzero(b))
    union = n_a + n_b - inter
    jac = np.nan if union == 0 else inter / union
    return jac, inter, n_a, n_b, int(domain.sum())


def _cordstat_from_grids(
    target_grid: np.ndarray,
    feature_grid: np.ndarray,
    occupied: np.ndarray,
    cutoff_target: float,
    cutoff_feature: float,
    step: int | None = None,
    round_jaccard: int | None = 4,
    operator_steps: Sequence[int] | None = None,
) -> pd.DataFrame:
    target_pos = occupied & np.isfinite(target_grid) & (target_grid > cutoff_target)
    feature_pos = occupied & np.isfinite(feature_grid) & (feature_grid > cutoff_feature)
    j0 = _jaccard(target_pos[occupied], feature_pos[occupied])
    if round_jaccard is not None and np.isfinite(j0):
        j0 = round(j0, round_jaccard)

    if operator_steps is None:
        if step is None:
            raise ValueError("Either step or operator_steps must be supplied.")
        px = nx_ = py = ny_ = int(step)
    else:
        if len(operator_steps) != 4:
            raise ValueError("operator_steps must contain four values: +X, -X, +Y, -Y.")
        px, nx_, py, ny_ = map(int, operator_steps)
    deltas = (
        (px, 0, "add_X"), (-nx_, 0, "minus_X"),
        (0, py, "add_Y"), (0, -ny_, "minus_Y"),
        (px, py, "add_X_add_Y"), (px, -ny_, "add_X_minus_Y"),
        (-nx_, py, "minus_X_add_Y"), (-nx_, -ny_, "minus_X_minus_Y"),
    )

    orig_inter = int(np.count_nonzero(target_pos[occupied] & feature_pos[occupied]))
    orig_a = int(np.count_nonzero(target_pos[occupied]))
    orig_b = int(np.count_nonzero(feature_pos[occupied]))
    orig_pmin = np.nan if max(orig_a, orig_b) == 0 else orig_inter / max(orig_a, orig_b)
    orig_pmax = np.nan if min(orig_a, orig_b) == 0 else orig_inter / min(orig_a, orig_b)
    if round_jaccard is not None:
        if np.isfinite(orig_pmin): orig_pmin = round(orig_pmin, round_jaccard)
        if np.isfinite(orig_pmax): orig_pmax = round(orig_pmax, round_jaccard)

    rows = []
    for dx, dy, name in deltas:
        jac, inter, n_a, n_b, n_domain = _shifted_jaccard(
            target_pos, feature_pos, occupied, dx, dy
        )
        if round_jaccard is not None and np.isfinite(jac):
            jac = round(jac, round_jaccard)
        diff = jac - j0 if np.isfinite(jac) and np.isfinite(j0) else np.nan
        pmin = np.nan if max(n_a, n_b) == 0 else inter / max(n_a, n_b)
        pmax = np.nan if min(n_a, n_b) == 0 else inter / min(n_a, n_b)
        if round_jaccard is not None:
            if np.isfinite(pmin): pmin = round(pmin, round_jaccard)
            if np.isfinite(pmax): pmax = round(pmax, round_jaccard)
        rows.append(
            {
                "direction": name,
                "pct_p": jac,
                "pct_pmin": pmin,
                "pct_pmax": pmax,
                "pct_op": j0,
                "pct_opmin": orig_pmin,
                "pct_opmax": orig_pmax,
                "diff_pct": diff,
                "cnt_p": inter,
                "cnt_p1": n_a,
                "cnt_p2": n_b,
                "cnt_px": n_domain,
                "cnt_all": int(occupied.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("direction")


def spatial_cordstat(
    obj1: SpatialAdjusted,
    obj2: SpatialAdjusted,
    min_cutoffs=None,
    min_pct_cutoffs: Sequence[float] | None = None,
    operator_steps: Sequence[int] = (4, 4, 4, 4),
    grid_size: float = 1.0,
    agg: str = "mean",
    min_cells_per_bin: int = 1,
    round_jaccard: int | None = 4,
) -> CordStatResult:
    """Compute displacement Jaccard statistics for two SpatialAdjusted objects.

    ``min_cutoffs`` may be the legacy two-value sequence, a feature-keyed
    mapping, or a :class:`CutoffResult`. This lets cohort-wide cutoffs calculated
    with :func:`calculate_cutoffs` be reused in this lower-level API.
    """
    if obj1.values is None or obj2.values is None:
        raise ValueError("Both SpatialAdjusted objects must contain feature values.")
    if obj1.coords.shape != obj2.coords.shape or not np.allclose(obj1.coords, obj2.coords):
        raise ValueError("obj1 and obj2 must refer to the same observations and coordinates.")
    if len(operator_steps) != 4:
        raise ValueError("operator_steps must contain four values: +X, -X, +Y, -Y.")

    flat, counts, occupied, shape, _ = _make_grid(obj1.coords, grid_size, min_cells_per_bin)
    g1 = _aggregate_grid(obj1.values, flat, counts, shape, agg=agg)
    g2 = _aggregate_grid(obj2.values, flat, counts, shape, agg=agg)

    if min_pct_cutoffs is not None:
        c1 = float(np.quantile(g1[occupied & np.isfinite(g1)], min_pct_cutoffs[0]))
        c2 = float(np.quantile(g2[occupied & np.isfinite(g2)], min_pct_cutoffs[1]))
    elif isinstance(min_cutoffs, (CutoffResult, Mapping, pd.Series)):
        mapping = _coerce_cutoff_mapping(min_cutoffs)
        if obj1.feature not in mapping or obj2.feature not in mapping:
            raise KeyError("min_cutoffs mapping/CutoffResult must contain both object feature names.")
        c1, c2 = float(mapping[obj1.feature]), float(mapping[obj2.feature])
    elif min_cutoffs is not None:
        c1, c2 = map(float, min_cutoffs)
    else:
        c1 = float(np.mean(g1[occupied & np.isfinite(g1)]))
        c2 = float(np.mean(g2[occupied & np.isfinite(g2)]))

    stat = _cordstat_from_grids(
        g1, g2, occupied, c1, c2, step=None, round_jaccard=round_jaccard,
        operator_steps=operator_steps,
    )
    return CordStatResult(
        mov_stat=stat,
        mov_summary=stat.mean(numeric_only=True),
        feature=f"{obj1.feature}:{obj2.feature}",
        cutoff_target=c1,
        cutoff_feature=c2,
        grid_size=float(grid_size),
    )



def spatial_vec_proj(vectors: pd.DataFrame) -> pd.Series:
    """Projected score = (min ΔJaccard + max ΔJaccard) / 2 at the final step."""
    final_step = vectors["step"].max()
    df = vectors.loc[vectors["step"] == final_step, ["feature", "min_djaccard", "max_djaccard"]].copy()
    # R implementation uses sum(x/2, na.rm=TRUE); nansum reproduces that behavior.
    scores = np.nansum(df[["min_djaccard", "max_djaccard"]].to_numpy(dtype=float), axis=1) / 2.0
    return pd.Series(scores, index=df["feature"].to_numpy(), name="projected_score")


def spatial_vec_magnitude(vectors: pd.DataFrame, features: Sequence[str] | None = None) -> pd.Series:
    """Path length from origin through successive SPHERE vectors."""
    if features is None:
        features = list(pd.unique(vectors["feature"]))
    out = {}
    for feature in features:
        tmp = vectors.loc[vectors["feature"] == feature, ["step", "min_djaccard", "max_djaccard"]].sort_values("step")
        xy = tmp[["min_djaccard", "max_djaccard"]].to_numpy(dtype=float)
        if xy.size == 0:
            out[feature] = np.nan
            continue
        xy = np.nan_to_num(xy, nan=0.0)
        prev = np.vstack([np.zeros((1, 2)), xy[:-1]])
        out[feature] = float(np.sqrt(((xy - prev) ** 2).sum(axis=1)).sum())
    return pd.Series(out, name="vector_len")


def _choose_cutoff_values(
    feature: str,
    values: np.ndarray,
    cutoffs=None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    if quantile_cutoffs is not None:
        q = quantile_cutoffs.get(feature, None) if isinstance(quantile_cutoffs, Mapping) else quantile_cutoffs
        if q is not None:
            if not 0 <= float(q) <= 1:
                raise ValueError("Quantile cutoffs must be in [0, 1].")
            return float(np.quantile(vals, float(q)))
    mapping = _coerce_cutoff_mapping(cutoffs)
    if feature in mapping:
        return float(mapping[feature])
    return float(np.mean(vals))



def _normalize_backend_steps(backend: str, steps: Sequence[float] | None) -> tuple[float, ...]:
    backend = str(backend).lower()
    if backend not in {"grid", "kdtree"}:
        raise ValueError("backend must be 'grid' or 'kdtree'.")
    if steps is None:
        steps = DEFAULT_STEPS if backend == "grid" else DEFAULT_KDTREE_STEPS
    if backend == "grid":
        out = []
        for x in steps:
            xf = float(x)
            if xf <= 0:
                continue
            if not np.isclose(xf, round(xf)):
                raise ValueError("grid backend requires integer steps measured in grid cells.")
            out.append(float(int(round(xf))))
    else:
        out = [float(x) for x in steps if float(x) > 0]
    out = sorted(set(out))
    if not out:
        raise ValueError("steps must contain at least one positive value.")
    return tuple(out)


def _direction_shifts(step: float, direction_mode: str = "sphere"):
    direction_mode = str(direction_mode).lower()
    if direction_mode not in {"sphere", "euclidean"}:
        raise ValueError("direction_mode must be 'sphere' or 'euclidean'.")
    d = float(step)
    diag = d if direction_mode == "sphere" else d / np.sqrt(2.0)
    return (
        (d, 0.0, "add_X"),
        (-d, 0.0, "minus_X"),
        (0.0, d, "add_Y"),
        (0.0, -d, "minus_Y"),
        (diag, diag, "add_X_add_Y"),
        (diag, -diag, "add_X_minus_Y"),
        (-diag, diag, "minus_X_add_Y"),
        (-diag, -diag, "minus_X_minus_Y"),
    )


def _domain_from_positive_tree(
    anchor_coords: np.ndarray,
    positive_tree: cKDTree | None,
    radius: float,
    query_shift: tuple[float, float] = (0.0, 0.0),
    workers: int = 1,
) -> np.ndarray:
    if positive_tree is None:
        return np.zeros(anchor_coords.shape[0], dtype=bool)
    shift = np.asarray(query_shift, dtype=float)
    # x lies within radius of (b + shift) iff (x - shift) lies within radius of b.
    dist, _ = positive_tree.query(
        anchor_coords - shift,
        k=1,
        distance_upper_bound=float(radius),
        workers=int(workers),
    )
    return np.isfinite(dist)


def _kdtree_prepare_feature(
    coords: np.ndarray,
    values: np.ndarray,
    feature: str,
    radius: float,
    cutoffs: Mapping[str, float] | None,
    quantile_cutoffs: Mapping[str, float] | float | None,
    workers: int,
):
    cutoff = _choose_cutoff_values(feature, values, cutoffs, quantile_cutoffs)
    positive = np.isfinite(values) & (np.asarray(values, dtype=float) > cutoff)
    positive_coords = coords[positive]
    tree = cKDTree(positive_coords) if positive_coords.shape[0] else None
    domain = _domain_from_positive_tree(coords, tree, radius, workers=workers)
    return cutoff, positive, positive_coords, tree, domain


def _shift_coverage(
    anchor_tree: cKDTree,
    positive_coords: np.ndarray,
    shift: tuple[float, float],
    radius: float,
    workers: int = 1,
) -> float:
    if positive_coords.shape[0] == 0:
        return np.nan
    moved = positive_coords + np.asarray(shift, dtype=float)
    dist, _ = anchor_tree.query(
        moved,
        k=1,
        distance_upper_bound=float(radius),
        workers=int(workers),
    )
    return float(np.mean(np.isfinite(dist)))


def _cordstat_from_kdtree(
    coords: np.ndarray,
    anchor_tree: cKDTree,
    target_domain: np.ndarray,
    feature_domain: np.ndarray,
    feature_positive_coords: np.ndarray,
    feature_tree: cKDTree | None,
    radius: float,
    step: float,
    direction_mode: str = "sphere",
    min_coverage: float = 0.0,
    workers: int = 1,
    round_jaccard: int | None = 4,
) -> pd.DataFrame:
    if radius <= 0:
        raise ValueError("radius must be > 0 for backend='kdtree'.")
    if not 0 <= float(min_coverage) <= 1:
        raise ValueError("min_coverage must be in [0, 1].")
    j0 = _jaccard(target_domain, feature_domain)
    if round_jaccard is not None and np.isfinite(j0):
        j0 = round(j0, round_jaccard)
    orig_inter = int(np.count_nonzero(target_domain & feature_domain))
    orig_a = int(np.count_nonzero(target_domain))
    orig_b = int(np.count_nonzero(feature_domain))
    rows = []
    for dx, dy, name in _direction_shifts(step, direction_mode=direction_mode):
        shift = (float(dx), float(dy))
        coverage = _shift_coverage(anchor_tree, feature_positive_coords, shift, radius, workers=workers)
        valid = np.isnan(coverage) or coverage >= float(min_coverage)
        if valid:
            shifted_domain = _domain_from_positive_tree(
                coords, feature_tree, radius, query_shift=shift, workers=workers
            )
            jac = _jaccard(target_domain, shifted_domain)
            inter = int(np.count_nonzero(target_domain & shifted_domain))
            n_b = int(np.count_nonzero(shifted_domain))
        else:
            jac = np.nan
            inter = 0
            n_b = 0
        if round_jaccard is not None and np.isfinite(jac):
            jac = round(jac, round_jaccard)
        diff = jac - j0 if np.isfinite(jac) and np.isfinite(j0) else np.nan
        rows.append(
            {
                "direction": name,
                "pct_p": jac,
                "pct_op": j0,
                "diff_pct": diff,
                "cnt_p": inter,
                "cnt_p1": orig_a,
                "cnt_p2": n_b,
                "cnt_all": int(coords.shape[0]),
                "coverage_fraction": coverage,
                "shift_x": float(dx),
                "shift_y": float(dy),
            }
        )
    return pd.DataFrame(rows).set_index("direction")


def kdtree_domain_map(
    adata: AnnData,
    target: str,
    feature: str,
    sample: str | None = None,
    sample_key: str = "sample_name",
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
    layer: str | None = None,
    radius: float = 15.0,
    cutoffs=None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
    cutoff_method: str | None = None,
    cutoff_samples: Sequence[str] | None = None,
    cutoff_n_repeats: int = 100,
    cutoff_balance_round_to: int = 1000,
    cutoff_balance_n: int | None = None,
    cutoff_random_state: int = 666,
    shift: tuple[float, float] = (0.0, 0.0),
    workers: int = 1,
) -> KDTreeDomainResult:
    """Prepare target/feature radius domains for direct spatial visualization.

    By default, KDTree positivity uses the legacy sample-specific mean. Set
    ``cutoff_method`` to a cohort-wide method from :func:`calculate_cutoffs` to
    use shared **cell-level** thresholds. Explicit values in ``cutoffs`` override
    calculated cutoffs feature-by-feature.
    """
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if cutoff_method is not None and quantile_cutoffs is not None:
        raise ValueError("quantile_cutoffs and cutoff_method cannot be used together.")
    if sample is None:
        idx = np.arange(adata.n_obs, dtype=np.int64)
        sample_name = None
    else:
        if sample_key not in adata.obs.columns:
            raise KeyError(f"sample_key {sample_key!r} not found in adata.obs.")
        idx = np.flatnonzero(adata.obs[sample_key].astype(str).to_numpy() == str(sample))
        if idx.size == 0:
            raise ValueError(f"No observations found for {sample_key}={sample!r}.")
        sample_name = str(sample)
    resolved_cutoffs, _ = _merge_calculated_cutoffs(
        adata, [target, feature], cutoffs, cutoff_method, "cell", sample_key,
        cutoff_samples, coord_cols, spatial_key, layer, 20.0, "mean", 1,
        cutoff_n_repeats, cutoff_balance_round_to, cutoff_balance_n, cutoff_random_state,
    )
    coords = _resolve_coords(adata, idx, coord_cols=coord_cols, spatial_key=spatial_key)
    target_values = _resolve_feature_vector(adata, target, idx, layer=layer)
    feature_values = _resolve_feature_vector(adata, feature, idx, layer=layer)
    ct, tpos, tcoords, ttree, tdomain = _kdtree_prepare_feature(
        coords, target_values, target, radius, resolved_cutoffs, quantile_cutoffs, workers
    )
    cf, fpos, fcoords, ftree, _ = _kdtree_prepare_feature(
        coords, feature_values, feature, radius, resolved_cutoffs, quantile_cutoffs, workers
    )
    anchor_tree = cKDTree(coords)
    fdomain = _domain_from_positive_tree(coords, ftree, radius, query_shift=shift, workers=workers)
    coverage = _shift_coverage(anchor_tree, fcoords, shift, radius, workers=workers)
    return KDTreeDomainResult(
        coords=coords,
        target_domain=tdomain,
        feature_domain=fdomain,
        target_positive=tpos,
        feature_positive=fpos,
        target=str(target),
        feature=str(feature),
        sample=sample_name,
        radius=float(radius),
        cutoff_target=float(ct),
        cutoff_feature=float(cf),
        shift=(float(shift[0]), float(shift[1])),
        coverage_fraction=float(coverage) if np.isfinite(coverage) else np.nan,
    )



def _single_sample_vectors_grid(
    adata: AnnData,
    obs_idx: np.ndarray,
    sample_name: str,
    target: str,
    features: Sequence[str],
    steps: Sequence[float],
    coord_cols: Sequence[str],
    spatial_key: str,
    layer: str | None,
    grid_size: float,
    agg: str,
    cutoffs,
    quantile_cutoffs: Mapping[str, float] | float | None,
    min_cells_per_bin: int,
    round_jaccard: int | None,
) -> pd.DataFrame:
    coords = _resolve_coords(adata, obs_idx, coord_cols=coord_cols, spatial_key=spatial_key)
    flat, counts, occupied, shape, _ = _make_grid(coords, grid_size, min_cells_per_bin)
    target_values = _resolve_feature_vector(adata, target, obs_idx, layer=layer)
    target_grid = _aggregate_grid(target_values, flat, counts, shape, agg=agg)
    cutoff_target = _choose_cutoff(target, target_grid, occupied, cutoffs, quantile_cutoffs)
    target_positive = occupied & np.isfinite(target_grid) & (target_grid > cutoff_target)
    n_bins = int(occupied.sum())
    n_positive_target = int(target_positive.sum())
    rows = []
    for feature in features:
        try:
            feature_values = _resolve_feature_vector(adata, feature, obs_idx, layer=layer)
        except KeyError:
            for step in steps:
                rows.append({"min_djaccard": np.nan, "max_djaccard": np.nan, "feature": feature, "step": float(step), "sample": sample_name})
            continue
        feature_grid = _aggregate_grid(feature_values, flat, counts, shape, agg=agg)
        cutoff_feature = _choose_cutoff(feature, feature_grid, occupied, cutoffs, quantile_cutoffs)
        feature_positive = occupied & np.isfinite(feature_grid) & (feature_grid > cutoff_feature)
        n_positive_feature = int(feature_positive.sum())
        for step in steps:
            stat = _cordstat_from_grids(
                target_grid, feature_grid, occupied, cutoff_target, cutoff_feature,
                int(round(step)), round_jaccard=round_jaccard,
            )
            diff = stat["diff_pct"].to_numpy(dtype=float)
            mn = float(np.nanmin(diff)) if np.isfinite(diff).any() else np.nan
            mx = float(np.nanmax(diff)) if np.isfinite(diff).any() else np.nan
            rows.append(
                {
                    "min_djaccard": mn,
                    "max_djaccard": mx,
                    "feature": feature,
                    "step": float(step),
                    "sample": sample_name,
                    "cutoff_target": cutoff_target,
                    "cutoff_feature": cutoff_feature,
                    "n_bins": n_bins,
                    "n_positive_target": n_positive_target,
                    "n_positive_feature": n_positive_feature,
                    "positive_fraction_target": np.nan if n_bins == 0 else n_positive_target / n_bins,
                    "positive_fraction_feature": np.nan if n_bins == 0 else n_positive_feature / n_bins,
                    "backend": "grid",
                }
            )
    return pd.DataFrame(rows)



def _single_sample_vectors_kdtree(
    adata: AnnData,
    obs_idx: np.ndarray,
    sample_name: str,
    target: str,
    features: Sequence[str],
    steps: Sequence[float],
    coord_cols: Sequence[str],
    spatial_key: str,
    layer: str | None,
    radius: float,
    cutoffs,
    quantile_cutoffs: Mapping[str, float] | float | None,
    direction_mode: str,
    min_coverage: float,
    workers: int,
    round_jaccard: int | None,
) -> pd.DataFrame:
    coords = _resolve_coords(adata, obs_idx, coord_cols=coord_cols, spatial_key=spatial_key)
    anchor_tree = cKDTree(coords)
    target_values = _resolve_feature_vector(adata, target, obs_idx, layer=layer)
    cutoff_target, target_positive, _, _, target_domain = _kdtree_prepare_feature(
        coords, target_values, target, radius, cutoffs, quantile_cutoffs, workers
    )
    n_anchors = int(coords.shape[0])
    n_positive_target = int(np.count_nonzero(target_positive))
    rows = []
    for feature in features:
        try:
            feature_values = _resolve_feature_vector(adata, feature, obs_idx, layer=layer)
        except KeyError:
            for step in steps:
                rows.append({"min_djaccard": np.nan, "max_djaccard": np.nan, "feature": feature, "step": float(step), "sample": sample_name})
            continue
        cutoff_feature, positive, positive_coords, feature_tree, feature_domain = _kdtree_prepare_feature(
            coords, feature_values, feature, radius, cutoffs, quantile_cutoffs, workers
        )
        n_positive_feature = int(np.count_nonzero(positive))
        for step in steps:
            stat = _cordstat_from_kdtree(
                coords=coords,
                anchor_tree=anchor_tree,
                target_domain=target_domain,
                feature_domain=feature_domain,
                feature_positive_coords=positive_coords,
                feature_tree=feature_tree,
                radius=radius,
                step=float(step),
                direction_mode=direction_mode,
                min_coverage=min_coverage,
                workers=workers,
                round_jaccard=round_jaccard,
            )
            diff = stat["diff_pct"].to_numpy(dtype=float)
            mn = float(np.nanmin(diff)) if np.isfinite(diff).any() else np.nan
            mx = float(np.nanmax(diff)) if np.isfinite(diff).any() else np.nan
            cov = stat["coverage_fraction"].to_numpy(dtype=float)
            rows.append(
                {
                    "min_djaccard": mn,
                    "max_djaccard": mx,
                    "feature": feature,
                    "step": float(step),
                    "sample": sample_name,
                    "cutoff_target": cutoff_target,
                    "cutoff_feature": cutoff_feature,
                    "n_anchors": n_anchors,
                    "n_positive_target": n_positive_target,
                    "n_positive_feature": n_positive_feature,
                    "positive_fraction_target": np.nan if n_anchors == 0 else n_positive_target / n_anchors,
                    "positive_fraction_feature": np.nan if n_anchors == 0 else n_positive_feature / n_anchors,
                    "min_direction_coverage": float(np.nanmin(cov)) if np.isfinite(cov).any() else np.nan,
                    "mean_direction_coverage": float(np.nanmean(cov)) if np.isfinite(cov).any() else np.nan,
                    "backend": "kdtree",
                }
            )
    return pd.DataFrame(rows)



def _single_sample_vectors(
    adata: AnnData,
    obs_idx: np.ndarray,
    sample_name: str,
    target: str,
    features: Sequence[str],
    steps: Sequence[float],
    coord_cols: Sequence[str],
    spatial_key: str,
    layer: str | None,
    backend: str,
    grid_size: float,
    agg: str,
    cutoffs: Mapping[str, float] | None,
    quantile_cutoffs: Mapping[str, float] | float | None,
    min_cells_per_bin: int,
    radius: float,
    direction_mode: str,
    min_coverage: float,
    workers: int,
    round_jaccard: int | None,
) -> pd.DataFrame:
    if backend == "grid":
        return _single_sample_vectors_grid(
            adata, obs_idx, sample_name, target, features, steps, coord_cols,
            spatial_key, layer, grid_size, agg, cutoffs, quantile_cutoffs,
            min_cells_per_bin, round_jaccard,
        )
    return _single_sample_vectors_kdtree(
        adata, obs_idx, sample_name, target, features, steps, coord_cols,
        spatial_key, layer, radius, cutoffs, quantile_cutoffs, direction_mode,
        min_coverage, workers, round_jaccard,
    )


def spatial_vector(
    adata: AnnData,
    target: str,
    features: Sequence[str],
    sample: str | None = None,
    sample_key: str = "sample_name",
    steps: Sequence[float] | None = None,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
    layer: str | None = None,
    backend: str = "grid",
    grid_size: float = 20.0,
    agg: str = "mean",
    cutoffs=None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
    cutoff_method: str | None = None,
    cutoff_samples: Sequence[str] | None = None,
    cutoff_n_repeats: int = 100,
    cutoff_balance_round_to: int = 1000,
    cutoff_balance_n: int | None = None,
    cutoff_random_state: int = 666,
    min_cells_per_bin: int = 1,
    radius: float = 15.0,
    direction_mode: str = "sphere",
    min_coverage: float = 0.0,
    workers: int = 1,
    round_jaccard: int | None = 4,
) -> SpatialVectorResult:
    """Generate SPHERE vectors for one tissue/sample with grid or KDTree backend.

    ``cutoff_method=None`` preserves the legacy sample-specific mean threshold.
    Cohort-wide methods are calculated at grid level for ``backend='grid'`` and
    at cell level for ``backend='kdtree'``. Explicit ``cutoffs`` override any
    calculated feature cutoff.
    """
    backend = str(backend).lower()
    features = list(map(str, features))
    steps = _normalize_backend_steps(backend, steps)
    if cutoff_method is not None and quantile_cutoffs is not None:
        raise ValueError("quantile_cutoffs and cutoff_method cannot be used together.")
    if sample is None:
        idx = np.arange(adata.n_obs, dtype=np.int64)
        sample_name = "obj"
    else:
        if sample_key not in adata.obs.columns:
            raise KeyError(f"sample_key {sample_key!r} not found in adata.obs.")
        idx = np.flatnonzero(adata.obs[sample_key].astype(str).to_numpy() == str(sample))
        if idx.size == 0:
            raise ValueError(f"No observations found for {sample_key}={sample!r}.")
        sample_name = str(sample)
    cutoff_level = "grid" if backend == "grid" else "cell"
    resolved_cutoffs, cutoff_result = _merge_calculated_cutoffs(
        adata, [target, *features], cutoffs, cutoff_method, cutoff_level,
        sample_key, cutoff_samples, coord_cols, spatial_key, layer, grid_size,
        agg, min_cells_per_bin, cutoff_n_repeats, cutoff_balance_round_to,
        cutoff_balance_n, cutoff_random_state,
    )
    vectors = _single_sample_vectors(
        adata, idx, sample_name, target, features, steps, coord_cols, spatial_key,
        layer, backend, grid_size, agg, resolved_cutoffs, quantile_cutoffs,
        min_cells_per_bin, radius, direction_mode, min_coverage, workers,
        round_jaccard,
    )
    projected = spatial_vec_proj(vectors)
    magnitude = spatial_vec_magnitude(vectors, features)
    final = vectors.loc[vectors["step"] == max(steps), ["sample", "feature", "min_djaccard", "max_djaccard"]].copy()
    final["projected_score"] = np.nansum(final[["min_djaccard", "max_djaccard"]].to_numpy(dtype=float), axis=1) / 2.0
    sample_mag = pd.DataFrame({"sample": sample_name, "feature": magnitude.index, "vector_len": magnitude.values})
    return SpatialVectorResult(
        vectors=vectors,
        target=target,
        features=features,
        projected_score=projected,
        vector_len=magnitude,
        pool_raw=vectors.copy(),
        sample_projected_score=final,
        sample_vector_len=sample_mag,
        settings={
            "backend": backend,
            "steps": steps,
            "grid_size": grid_size if backend == "grid" else None,
            "radius": radius if backend == "kdtree" else None,
            "direction_mode": direction_mode if backend == "kdtree" else None,
            "min_coverage": min_coverage if backend == "kdtree" else None,
            "agg": agg if backend == "grid" else None,
            "sample_key": sample_key,
            "coord_cols": tuple(coord_cols),
            "spatial_key": spatial_key,
            "layer": layer,
            "round_jaccard": round_jaccard,
            "cutoff_level": cutoff_level,
            "cutoff_method": cutoff_method or "sample_mean_legacy",
            "cutoffs": resolved_cutoffs,
            "cutoff_samples": cutoff_result.settings.get("samples") if cutoff_result is not None else cutoff_samples,
            "cutoff_n_repeats": cutoff_n_repeats if cutoff_method and str(cutoff_method).startswith("balanced_global_") else None,
            "cutoff_balance_round_to": cutoff_balance_round_to if cutoff_method and str(cutoff_method).startswith("balanced_global_") else None,
            "cutoff_balance_n": cutoff_balance_n,
            "cutoff_random_state": cutoff_random_state if cutoff_method and str(cutoff_method).startswith("balanced_global_") else None,
        },
    )



def spatial_vector_x(
    adata: AnnData,
    target: str,
    features: Sequence[str],
    sample_key: str = "sample_name",
    samples: Sequence[str] | None = None,
    steps: Sequence[float] | None = None,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
    layer: str | None = None,
    backend: str = "grid",
    grid_size: float = 20.0,
    agg: str = "mean",
    cutoffs=None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
    cutoff_method: str | None = None,
    cutoff_samples: Sequence[str] | None = None,
    cutoff_n_repeats: int = 100,
    cutoff_balance_round_to: int = 1000,
    cutoff_balance_n: int | None = None,
    cutoff_random_state: int = 666,
    min_cells_per_bin: int = 1,
    radius: float = 15.0,
    direction_mode: str = "sphere",
    min_coverage: float = 0.0,
    workers: int = 1,
    round_jaccard: int | None = 4,
    verbose: bool = True,
) -> SpatialVectorResult:
    """Cohort-level SPHERE analysis using regular-grid or KDTree backend.

    Use ``cutoff_method`` for one shared cutoff per feature across the cohort.
    If ``cutoff_samples`` is omitted, the analyzed ``samples`` define the cutoff
    cohort. Grid and KDTree automatically use grid-level and cell-level cutoff
    calculation, respectively.
    """
    backend = str(backend).lower()
    steps = _normalize_backend_steps(backend, steps)
    if cutoff_method is not None and quantile_cutoffs is not None:
        raise ValueError("quantile_cutoffs and cutoff_method cannot be used together.")
    if sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key {sample_key!r} not found in adata.obs.")
    features = list(map(str, features))
    if samples is None:
        samples = list(pd.unique(adata.obs[sample_key].astype(str)))
    else:
        samples = list(map(str, samples))
    cutoff_cohort = list(map(str, cutoff_samples)) if cutoff_samples is not None else list(samples)
    cutoff_level = "grid" if backend == "grid" else "cell"
    resolved_cutoffs, cutoff_result = _merge_calculated_cutoffs(
        adata, [target, *features], cutoffs, cutoff_method, cutoff_level,
        sample_key, cutoff_cohort, coord_cols, spatial_key, layer, grid_size, agg,
        min_cells_per_bin, cutoff_n_repeats, cutoff_balance_round_to,
        cutoff_balance_n, cutoff_random_state,
    )
    sample_values = adata.obs[sample_key].astype(str).to_numpy()
    raw_parts = []
    for i, sample in enumerate(samples, start=1):
        idx = np.flatnonzero(sample_values == sample)
        if idx.size == 0:
            warnings.warn(f"Skipping sample {sample!r}: no observations.")
            continue
        if verbose:
            extra = f"grid={grid_size:g}" if backend == "grid" else f"radius={radius:g}"
            print(f"[sat4sc:{backend}] {i}/{len(samples)} {sample}: {idx.size:,} cells/spots, {extra}")
        raw_parts.append(
            _single_sample_vectors(
                adata, idx, sample, target, features, steps, coord_cols, spatial_key,
                layer, backend, grid_size, agg, resolved_cutoffs, quantile_cutoffs,
                min_cells_per_bin, radius, direction_mode, min_coverage, workers,
                round_jaccard,
            )
        )
    if not raw_parts:
        raise ValueError("No valid samples were analyzed.")
    pool_raw = pd.concat(raw_parts, ignore_index=True)
    vectors = (
        pool_raw.groupby(["feature", "step"], sort=False, observed=True)[["min_djaccard", "max_djaccard"]]
        .mean()
        .reset_index()
    )
    vectors["sample"] = "obj"
    projected = spatial_vec_proj(vectors)
    magnitude = spatial_vec_magnitude(vectors, features)
    max_step = max(steps)
    sample_projected = pool_raw.loc[
        pool_raw["step"] == max_step,
        ["sample", "feature", "min_djaccard", "max_djaccard"],
    ].copy()
    sample_projected["projected_score"] = np.nansum(
        sample_projected[["min_djaccard", "max_djaccard"]].to_numpy(dtype=float), axis=1
    ) / 2.0
    mag_rows = []
    for sample, df in pool_raw.groupby("sample", sort=False):
        mag = spatial_vec_magnitude(df, features)
        mag_rows.extend({"sample": sample, "feature": k, "vector_len": v} for k, v in mag.items())
    sample_magnitude = pd.DataFrame(mag_rows)
    return SpatialVectorResult(
        vectors=vectors,
        target=target,
        features=features,
        projected_score=projected,
        vector_len=magnitude,
        pool_raw=pool_raw,
        sample_projected_score=sample_projected,
        sample_vector_len=sample_magnitude,
        settings={
            "backend": backend,
            "steps": steps,
            "grid_size": grid_size if backend == "grid" else None,
            "radius": radius if backend == "kdtree" else None,
            "direction_mode": direction_mode if backend == "kdtree" else None,
            "min_coverage": min_coverage if backend == "kdtree" else None,
            "workers": workers if backend == "kdtree" else None,
            "agg": agg if backend == "grid" else None,
            "sample_key": sample_key,
            "samples": samples,
            "coord_cols": tuple(coord_cols),
            "spatial_key": spatial_key,
            "layer": layer,
            "round_jaccard": round_jaccard,
            "cutoff_level": cutoff_level,
            "cutoff_method": cutoff_method or "sample_mean_legacy",
            "cutoffs": resolved_cutoffs,
            "cutoff_samples": cutoff_result.settings.get("samples") if cutoff_result is not None else cutoff_cohort,
            "cutoff_n_repeats": cutoff_n_repeats if cutoff_method and str(cutoff_method).startswith("balanced_global_") else None,
            "cutoff_balance_round_to": cutoff_balance_round_to if cutoff_method and str(cutoff_method).startswith("balanced_global_") else None,
            "cutoff_balance_n": cutoff_balance_n,
            "cutoff_random_state": cutoff_random_state if cutoff_method and str(cutoff_method).startswith("balanced_global_") else None,
        },
    )



def pairwise_projected_scores(
    adata: AnnData,
    features: Sequence[str],
    sample_key: str = "sample_name",
    samples: Sequence[str] | None = None,
    final_step: float | None = None,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    spatial_key: str = "spatial",
    layer: str | None = None,
    backend: str = "grid",
    grid_size: float = 20.0,
    agg: str = "mean",
    cutoffs=None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
    cutoff_method: str | None = None,
    cutoff_samples: Sequence[str] | None = None,
    cutoff_n_repeats: int = 100,
    cutoff_balance_round_to: int = 1000,
    cutoff_balance_n: int | None = None,
    cutoff_random_state: int = 666,
    min_cells_per_bin: int = 1,
    radius: float = 15.0,
    direction_mode: str = "sphere",
    min_coverage: float = 0.0,
    workers: int = 1,
    round_jaccard: int | None = 4,
    verbose: bool = True,
) -> PairwiseResult:
    """Figure-3A-like pairwise projected-score matrix for either backend.

    ``cutoff_method`` applies the same cohort-wide cutoff for each feature to
    every sample. Grid backend calculates those cutoffs on grid scores; KDTree
    backend calculates them on cell/spot scores.
    """
    backend = str(backend).lower()
    if backend not in {"grid", "kdtree"}:
        raise ValueError("backend must be 'grid' or 'kdtree'.")
    if cutoff_method is not None and quantile_cutoffs is not None:
        raise ValueError("quantile_cutoffs and cutoff_method cannot be used together.")
    if final_step is None:
        final_step = float(DEFAULT_STEPS[-1] if backend == "grid" else DEFAULT_KDTREE_STEPS[-1])
    if backend == "grid" and not np.isclose(float(final_step), round(float(final_step))):
        raise ValueError("grid backend requires integer final_step.")
    if sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key {sample_key!r} not found in adata.obs.")
    features = list(map(str, features))
    if samples is None:
        samples = list(pd.unique(adata.obs[sample_key].astype(str)))
    else:
        samples = list(map(str, samples))
    cutoff_cohort = list(map(str, cutoff_samples)) if cutoff_samples is not None else list(samples)
    cutoff_level = "grid" if backend == "grid" else "cell"
    resolved_cutoffs, cutoff_result = _merge_calculated_cutoffs(
        adata, features, cutoffs, cutoff_method, cutoff_level, sample_key,
        cutoff_cohort, coord_cols, spatial_key, layer, grid_size, agg,
        min_cells_per_bin, cutoff_n_repeats, cutoff_balance_round_to,
        cutoff_balance_n, cutoff_random_state,
    )
    sample_values = adata.obs[sample_key].astype(str).to_numpy()
    rows = []
    for i, sample in enumerate(samples, start=1):
        idx = np.flatnonzero(sample_values == sample)
        if idx.size == 0:
            continue
        if verbose:
            print(f"[sat4sc:{backend}] pairwise {i}/{len(samples)} {sample}: {idx.size:,} cells/spots")
        coords = _resolve_coords(adata, idx, coord_cols=coord_cols, spatial_key=spatial_key)
        if backend == "grid":
            flat, counts, occupied, shape, _ = _make_grid(coords, grid_size, min_cells_per_bin)
            grids = {}
            thresholds = {}
            for feature in features:
                values = _resolve_feature_vector(adata, feature, idx, layer=layer)
                grid = _aggregate_grid(values, flat, counts, shape, agg=agg)
                grids[feature] = grid
                thresholds[feature] = _choose_cutoff(feature, grid, occupied, resolved_cutoffs, quantile_cutoffs)
            for target in features:
                for feature in features:
                    if target == feature:
                        score = np.nan
                    else:
                        stat = _cordstat_from_grids(
                            grids[target], grids[feature], occupied,
                            thresholds[target], thresholds[feature], int(round(float(final_step))),
                            round_jaccard=round_jaccard,
                        )
                        d = stat["diff_pct"].to_numpy(dtype=float)
                        score = float(np.nansum([np.nanmin(d), np.nanmax(d)]) / 2.0) if np.isfinite(d).any() else np.nan
                    rows.append({"sample": sample, "target": target, "feature": feature, "projected_score": score})
        else:
            anchor_tree = cKDTree(coords)
            prepared = {}
            for feature in features:
                values = _resolve_feature_vector(adata, feature, idx, layer=layer)
                prepared[feature] = _kdtree_prepare_feature(
                    coords, values, feature, radius, resolved_cutoffs, quantile_cutoffs, workers
                )
            shifted_cache = {}
            for feature in features:
                _, _, pos_coords, ftree, fdomain = prepared[feature]
                j_domains = []
                for dx, dy, name in _direction_shifts(float(final_step), direction_mode=direction_mode):
                    shift = (float(dx), float(dy))
                    cov = _shift_coverage(anchor_tree, pos_coords, shift, radius, workers=workers)
                    if np.isnan(cov) or cov >= min_coverage:
                        dom = _domain_from_positive_tree(coords, ftree, radius, query_shift=shift, workers=workers)
                    else:
                        dom = None
                    j_domains.append((name, dom, cov))
                shifted_cache[feature] = (fdomain, j_domains)
            for target in features:
                tdomain = prepared[target][4]
                for feature in features:
                    if target == feature:
                        score = np.nan
                    else:
                        fdomain, shifted = shifted_cache[feature]
                        j0 = _jaccard(tdomain, fdomain)
                        if round_jaccard is not None and np.isfinite(j0):
                            j0 = round(j0, round_jaccard)
                        diffs = []
                        for _, dom, _ in shifted:
                            if dom is None:
                                diffs.append(np.nan)
                                continue
                            jac = _jaccard(tdomain, dom)
                            if round_jaccard is not None and np.isfinite(jac):
                                jac = round(jac, round_jaccard)
                            diffs.append(jac - j0 if np.isfinite(jac) and np.isfinite(j0) else np.nan)
                        d = np.asarray(diffs, dtype=float)
                        score = float(np.nansum([np.nanmin(d), np.nanmax(d)]) / 2.0) if np.isfinite(d).any() else np.nan
                    rows.append({"sample": sample, "target": target, "feature": feature, "projected_score": score})
    sample_scores = pd.DataFrame(rows)
    matrix = sample_scores.pivot_table(
        index="target", columns="feature", values="projected_score", aggfunc="mean", sort=False
    ).reindex(index=features, columns=features)
    return PairwiseResult(
        matrix=matrix,
        sample_scores=sample_scores,
        features=features,
        settings={
            "backend": backend,
            "final_step": float(final_step),
            "grid_size": grid_size if backend == "grid" else None,
            "radius": radius if backend == "kdtree" else None,
            "direction_mode": direction_mode if backend == "kdtree" else None,
            "min_coverage": min_coverage if backend == "kdtree" else None,
            "workers": workers if backend == "kdtree" else None,
            "agg": agg if backend == "grid" else None,
            "sample_key": sample_key,
            "samples": samples,
            "coord_cols": tuple(coord_cols),
            "spatial_key": spatial_key,
            "layer": layer,
            "round_jaccard": round_jaccard,
            "cutoff_level": cutoff_level,
            "cutoff_method": cutoff_method or "sample_mean_legacy",
            "cutoffs": resolved_cutoffs,
            "cutoff_samples": cutoff_result.settings.get("samples") if cutoff_result is not None else cutoff_cohort,
            "cutoff_n_repeats": cutoff_n_repeats if cutoff_method and str(cutoff_method).startswith("balanced_global_") else None,
            "cutoff_balance_round_to": cutoff_balance_round_to if cutoff_method and str(cutoff_method).startswith("balanced_global_") else None,
            "cutoff_balance_n": cutoff_balance_n,
            "cutoff_random_state": cutoff_random_state if cutoff_method and str(cutoff_method).startswith("balanced_global_") else None,
        },
    )



def spatial_vector_kdtree(*args, **kwargs) -> SpatialVectorResult:
    """Convenience wrapper for ``spatial_vector(..., backend='kdtree')``."""
    kwargs["backend"] = "kdtree"
    return spatial_vector(*args, **kwargs)


def spatial_vector_x_kdtree(*args, **kwargs) -> SpatialVectorResult:
    """Convenience wrapper for ``spatial_vector_x(..., backend='kdtree')``."""
    kwargs["backend"] = "kdtree"
    return spatial_vector_x(*args, **kwargs)


def pairwise_projected_scores_kdtree(*args, **kwargs) -> PairwiseResult:
    """Convenience wrapper for ``pairwise_projected_scores(..., backend='kdtree')``."""
    kwargs["backend"] = "kdtree"
    return pairwise_projected_scores(*args, **kwargs)


def _mean_over_columns(X, cols: np.ndarray) -> np.ndarray:
    if cols.size == 0:
        return np.full(X.shape[0], np.nan, dtype=np.float64)
    sub = X[:, cols]
    if sparse.issparse(sub):
        return np.asarray(sub.mean(axis=1)).ravel().astype(np.float64, copy=False)
    return np.asarray(sub, dtype=np.float64).mean(axis=1)


def add_module_scores(
    adata: AnnData,
    gene_sets: Mapping[str, Sequence[str]],
    layer: str | None = None,
    nbin: int = 10,
    ctrl: int = 100,
    seed: int = 666,
    inplace: bool = True,
) -> pd.DataFrame:
    """Seurat-AddModuleScore-like helper for generating signature columns.

    The procedure follows the same principle as Seurat AddModuleScore: genes are
    binned by average expression; for each signature gene, control genes are drawn
    from its expression bin; signature mean minus control mean is returned.

    Because NumPy and R use different RNG implementations and tie-breaking, values
    should be treated as algorithmically equivalent rather than bit-for-bit equal
    to Seurat's AddModuleScore.
    """
    X = _get_matrix(adata, layer=layer)
    var_names = pd.Index(adata.var_names.astype(str))
    means = np.asarray(X.mean(axis=0)).ravel().astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1e-30, size=means.size)
    order = np.argsort(means + noise, kind="mergesort")
    bins = np.empty(means.size, dtype=np.int32)
    bins[order] = np.minimum((np.arange(means.size) * int(nbin)) // means.size, int(nbin) - 1)

    results = {}
    for name, genes in gene_sets.items():
        present = [g for g in genes if g in var_names]
        if not present:
            warnings.warn(f"Gene set {name!r} has no genes present in adata.var_names; returning NaN.")
            results[name] = np.full(adata.n_obs, np.nan)
            continue
        module_idx = np.array([var_names.get_loc(g) for g in present], dtype=np.int64)
        ctrl_idx = []
        module_set = set(module_idx.tolist())
        for j in module_idx:
            pool = np.flatnonzero(bins == bins[j])
            pool = np.array([x for x in pool if x not in module_set], dtype=np.int64)
            if pool.size == 0:
                continue
            take = min(int(ctrl), pool.size)
            ctrl_idx.extend(rng.choice(pool, size=take, replace=False).tolist())
        ctrl_idx = np.unique(np.asarray(ctrl_idx, dtype=np.int64))
        module_score = _mean_over_columns(X, module_idx)
        control_score = _mean_over_columns(X, ctrl_idx)
        results[name] = module_score - control_score

    df = pd.DataFrame(results, index=adata.obs_names)
    if inplace:
        for col in df.columns:
            adata.obs[col] = df[col].to_numpy()
    return df


# R-style aliases for users porting existing workflows.
spatial_vectorX = spatial_vector_x
spatial_vecProj = spatial_vec_proj
spatial_vecMagnitude = spatial_vec_magnitude
spatial_vectorX_kdtree = spatial_vector_x_kdtree


__all__ = [
    "SpatialAdjusted",
    "BinStatResult",
    "CordStatResult",
    "SpatialVectorResult",
    "PairwiseResult",
    "GridFeatureMap",
    "KDTreeDomainResult",
    "CutoffResult",
    "NicheSampleResult",
    "NicheResult",
    "CUTOFF_METHODS",
    "spatial_adjust",
    "spatial_binstat",
    "calculate_cutoffs",
    "calculate_global_cutoffs",
    "positive_proportions",
    "define_niche",
    "define_niches",
    "grid_feature_map",
    "kdtree_domain_map",
    "spatial_cordstat",
    "spatial_vector",
    "spatial_vector_x",
    "spatial_vectorX",
    "spatial_vector_kdtree",
    "spatial_vector_x_kdtree",
    "spatial_vectorX_kdtree",
    "spatial_vec_proj",
    "spatial_vecProj",
    "spatial_vec_magnitude",
    "spatial_vecMagnitude",
    "pairwise_projected_scores",
    "pairwise_projected_scores_kdtree",
    "add_module_scores",
]
