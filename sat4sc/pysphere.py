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

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

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
    min_cutoff: float | None = 0.0,
    min_count: int = 10,
    pct_cutoff: float = 0.5,
    mask: np.ndarray | None = None,
) -> BinStatResult:
    """Bin a spatial feature and summarize the fraction above a cutoff.

    This ports the calculation performed by the original package's non-exported
    ``spatial_binstat`` helper. Matrix row order is top-to-bottom, matching the R
    helper's visual matrix convention.
    """
    if obj.values is None:
        raise ValueError("SpatialAdjusted object must contain feature values.")
    ny, nx = map(int, bins)
    if ny < 1 or nx < 1:
        raise ValueError("bins must contain positive integers.")
    values = np.asarray(obj.values, dtype=float)
    coords = np.asarray(obj.coords, dtype=float)
    if min_cutoff is None:
        min_cutoff = float(np.nanmean(values))
    x_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max(), nx + 1)
    y_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max(), ny + 1)
    # np.digitize gives lower-to-upper y; reverse row index to mirror the R matrix.
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
                stat_pct[r, c] = float(np.mean(values[take] > float(min_cutoff)))
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
        min_cutoff=float(min_cutoff), min_count=int(min_count), pct_cutoff=float(pct_cutoff),
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
) -> GridFeatureMap:
    """Rasterize one gene/signature to the same grid used by ``backend='grid'``.

    The returned object is designed for direct spatial plotting with
    :func:`sat4sc.pysphere_plotting.plot_grid_feature_map`. ``cutoff`` is optional and is
    only used to mark high-feature bins in visualization; when omitted, the mean
    across occupied finite bins is used, matching SPHERE's default threshold.
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
        if not 0 <= float(quantile_cutoff) <= 1:
            raise ValueError("quantile_cutoff must be in [0, 1].")
        chosen = float(np.quantile(vals, float(quantile_cutoff)))
    elif cutoff is not None:
        chosen = float(cutoff)
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
    cutoffs: Mapping[str, float] | None = None,
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
    if cutoffs is not None and feature in cutoffs:
        return float(cutoffs[feature])
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
    min_cutoffs: Sequence[float] | None = None,
    min_pct_cutoffs: Sequence[float] | None = None,
    operator_steps: Sequence[int] = (4, 4, 4, 4),
    grid_size: float = 1.0,
    agg: str = "mean",
    min_cells_per_bin: int = 1,
    round_jaccard: int | None = 4,
) -> CordStatResult:
    """Compute displacement Jaccard statistics for two SpatialAdjusted objects.

    This function mirrors SPHERE::spatial_cordstat conceptually. For continuous
    coordinates, both objects are rasterized onto one regular grid.
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
    cutoffs: Mapping[str, float] | None = None,
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
    if cutoffs is not None and feature in cutoffs:
        return float(cutoffs[feature])
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
    cutoffs: Mapping[str, float] | None = None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
    shift: tuple[float, float] = (0.0, 0.0),
    workers: int = 1,
) -> KDTreeDomainResult:
    """Prepare target/feature radius domains for direct spatial visualization.

    ``shift`` is in the original coordinate units (typically microns for Xenium)
    and is applied virtually to ``feature`` before it is re-projected onto the
    fixed cell anchors.
    """
    if radius <= 0:
        raise ValueError("radius must be > 0.")
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
    target_values = _resolve_feature_vector(adata, target, idx, layer=layer)
    feature_values = _resolve_feature_vector(adata, feature, idx, layer=layer)
    ct, tpos, tcoords, ttree, tdomain = _kdtree_prepare_feature(
        coords, target_values, target, radius, cutoffs, quantile_cutoffs, workers
    )
    cf, fpos, fcoords, ftree, _ = _kdtree_prepare_feature(
        coords, feature_values, feature, radius, cutoffs, quantile_cutoffs, workers
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
    cutoffs: Mapping[str, float] | None,
    quantile_cutoffs: Mapping[str, float] | float | None,
    min_cells_per_bin: int,
    round_jaccard: int | None,
) -> pd.DataFrame:
    coords = _resolve_coords(adata, obs_idx, coord_cols=coord_cols, spatial_key=spatial_key)
    flat, counts, occupied, shape, _ = _make_grid(coords, grid_size, min_cells_per_bin)
    target_values = _resolve_feature_vector(adata, target, obs_idx, layer=layer)
    target_grid = _aggregate_grid(target_values, flat, counts, shape, agg=agg)
    cutoff_target = _choose_cutoff(target, target_grid, occupied, cutoffs, quantile_cutoffs)
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
                    "n_bins": int(occupied.sum()),
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
    cutoffs: Mapping[str, float] | None,
    quantile_cutoffs: Mapping[str, float] | float | None,
    direction_mode: str,
    min_coverage: float,
    workers: int,
    round_jaccard: int | None,
) -> pd.DataFrame:
    coords = _resolve_coords(adata, obs_idx, coord_cols=coord_cols, spatial_key=spatial_key)
    anchor_tree = cKDTree(coords)
    target_values = _resolve_feature_vector(adata, target, obs_idx, layer=layer)
    cutoff_target, _, _, _, target_domain = _kdtree_prepare_feature(
        coords, target_values, target, radius, cutoffs, quantile_cutoffs, workers
    )
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
                    "n_anchors": int(coords.shape[0]),
                    "n_positive_feature": int(np.count_nonzero(positive)),
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
    cutoffs: Mapping[str, float] | None = None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
    min_cells_per_bin: int = 1,
    radius: float = 15.0,
    direction_mode: str = "sphere",
    min_coverage: float = 0.0,
    workers: int = 1,
    round_jaccard: int | None = 4,
) -> SpatialVectorResult:
    """Generate SPHERE vectors for one tissue/sample with grid or KDTree backend.

    Grid steps are measured in grid cells. KDTree steps are measured directly in
    the coordinate units (normally microns for Xenium).
    """
    backend = str(backend).lower()
    features = list(map(str, features))
    steps = _normalize_backend_steps(backend, steps)
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
    vectors = _single_sample_vectors(
        adata, idx, sample_name, target, features, steps, coord_cols, spatial_key,
        layer, backend, grid_size, agg, cutoffs, quantile_cutoffs,
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
    cutoffs: Mapping[str, float] | None = None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
    min_cells_per_bin: int = 1,
    radius: float = 15.0,
    direction_mode: str = "sphere",
    min_coverage: float = 0.0,
    workers: int = 1,
    round_jaccard: int | None = 4,
    verbose: bool = True,
) -> SpatialVectorResult:
    """Cohort-level SPHERE analysis using either regular-grid or KDTree backend."""
    backend = str(backend).lower()
    steps = _normalize_backend_steps(backend, steps)
    if sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key {sample_key!r} not found in adata.obs.")
    features = list(map(str, features))
    if samples is None:
        samples = list(pd.unique(adata.obs[sample_key].astype(str)))
    else:
        samples = list(map(str, samples))
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
                layer, backend, grid_size, agg, cutoffs, quantile_cutoffs,
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
    cutoffs: Mapping[str, float] | None = None,
    quantile_cutoffs: Mapping[str, float] | float | None = None,
    min_cells_per_bin: int = 1,
    radius: float = 15.0,
    direction_mode: str = "sphere",
    min_coverage: float = 0.0,
    workers: int = 1,
    round_jaccard: int | None = 4,
    verbose: bool = True,
) -> PairwiseResult:
    """Figure-3A-like pairwise projected-score matrix for either backend.

    Only the final displacement is calculated. For KDTree, shifted feature
    domains are cached once per feature/sample and reused across all targets.
    """
    backend = str(backend).lower()
    if backend not in {"grid", "kdtree"}:
        raise ValueError("backend must be 'grid' or 'kdtree'.")
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
                thresholds[feature] = _choose_cutoff(feature, grid, occupied, cutoffs, quantile_cutoffs)
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
                    coords, values, feature, radius, cutoffs, quantile_cutoffs, workers
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
    "spatial_adjust",
    "spatial_binstat",
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
