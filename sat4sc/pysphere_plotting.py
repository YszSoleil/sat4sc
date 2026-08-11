"""Plotting utilities for sat4sc.pysphere results.

These functions are designed to reproduce the visual logic of the SPHERE paper,
including Figure-3A-like projected-score heatmaps, Figure-3B-like vector plots,
Figure-3F-like paired projected-score comparisons, and Figure-4B-like projected
score density + co-localization proportion summaries.
"""

from __future__ import annotations

from typing import Mapping, Sequence
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, norm, wilcoxon

from .pysphere import (
    BinStatResult, CutoffResult, GridFeatureMap, KDTreeDomainResult, NicheResult, PairwiseResult,
    SpatialAdjusted, SpatialVectorResult,
)


def _step_cmap():
    return LinearSegmentedColormap.from_list("sphere_steps", ["steelblue", "orange", "red"])


def _heatmap_cmap():
    # Negative projected score = spatially closer. Orange therefore denotes closer.
    return LinearSegmentedColormap.from_list("sphere_projected", ["#E2A12A", "#EEEEEE", "#8AA0B5"])


def plot_vector(
    result: SpatialVectorResult,
    ax=None,
    label_features: Sequence[str] | None = None,
    label_all: bool = True,
    point_size: float = 22,
    line_width: float = 1.1,
    show_projection: bool = True,
    title: str | None = None,
):
    """Figure-3B-like SPHERE min/max delta-Jaccard vector plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 5.4))
    else:
        fig = ax.figure

    df = result.vectors.copy()
    steps = np.sort(df["step"].unique())
    cmap = _step_cmap()
    norm_step = Normalize(vmin=float(steps.min()), vmax=float(steps.max()))

    for feature in result.features:
        tmp = df.loc[df["feature"] == feature].sort_values("step")
        if tmp.empty:
            continue
        xs = tmp["max_djaccard"].to_numpy(dtype=float)
        ys = tmp["min_djaccard"].to_numpy(dtype=float)
        prev_x, prev_y = 0.0, 0.0
        for x, y, step in zip(xs, ys, tmp["step"]):
            if np.isfinite(x) and np.isfinite(y):
                c = cmap(norm_step(float(step)))
                ax.plot([prev_x, x], [prev_y, y], color=c, lw=line_width)
                ax.scatter(x, y, s=point_size, color=c, edgecolor="none", zorder=3)
                prev_x, prev_y = x, y

    ax.axhline(0, ls=":", lw=0.9, color="0.45")
    ax.axvline(0, ls=":", lw=0.9, color="0.45")
    if show_projection:
        lims = np.array([*ax.get_xlim(), *ax.get_ylim()], dtype=float)
        lo, hi = float(np.nanmin(lims)), float(np.nanmax(lims))
        ax.plot([lo, hi], [lo, hi], ls=":", lw=0.9, color="0.45")

    final_step = steps.max()
    final = df.loc[df["step"] == final_step]
    if label_features is not None:
        final = final[final["feature"].isin(label_features)]
    elif not label_all:
        final = final.iloc[0:0]
    for _, row in final.iterrows():
        if np.isfinite(row["max_djaccard"]) and np.isfinite(row["min_djaccard"]):
            ax.annotate(
                str(row["feature"]),
                (row["max_djaccard"], row["min_djaccard"]),
                xytext=(4, 3), textcoords="offset points", fontsize=9,
            )

    if show_projection and result.projected_score is not None:
        for feature, score in result.projected_score.items():
            if np.isfinite(score):
                ax.scatter(score, score, marker="+", s=34, color="black", linewidths=0.9, zorder=4)

    sm = ScalarMappable(norm=norm_step, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.05)
    backend = result.settings.get("backend", "grid") if result.settings else "grid"
    cbar.set_label("step (coordinate units)" if backend == "kdtree" else "step (grid cells)")
    ax.set_xlabel("max ΔJaccard")
    ax.set_ylabel("min ΔJaccard")
    ax.set_title(title or f"relation to {result.target}")
    return fig, ax


def plot_magnitude_projected(
    result: SpatialVectorResult,
    ax=None,
    title: str | None = None,
    annotate: bool = True,
):
    """Python analogue of SPHERE::spatial_magPlot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.8))
    else:
        fig = ax.figure
    x = result.vector_len.reindex(result.features)
    y = result.projected_score.reindex(result.features)
    ax.axhline(0, ls=":", lw=0.9, color="0.45")
    for feature in result.features:
        xv, yv = x.get(feature, np.nan), y.get(feature, np.nan)
        if not np.isfinite(xv) or not np.isfinite(yv):
            continue
        ax.plot([0, xv], [yv, yv], lw=1.1)
        ax.scatter(xv, yv, marker="+", s=45)
        if annotate:
            ax.annotate(feature, (xv, yv), xytext=(4, 2), textcoords="offset points", fontsize=9)
    ax.set_xlabel("magnitude")
    ax.set_ylabel("projected score")
    ax.set_title(title or f"relation to {result.target}")
    return fig, ax


def plot_projected_heatmap(
    result: PairwiseResult | pd.DataFrame,
    ax=None,
    order: Sequence[str] | None = None,
    triangle: str | None = "lower",
    annotate: bool = False,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = 0.0,
):
    """Figure-3A-like projected-score heatmap."""
    mat = result.matrix.copy() if isinstance(result, PairwiseResult) else result.copy()
    if order is not None:
        mat = mat.reindex(index=order, columns=order)
    arr = mat.to_numpy(dtype=float, copy=True)
    if triangle == "lower":
        arr[np.triu_indices_from(arr, k=0)] = np.nan
    elif triangle == "upper":
        arr[np.tril_indices_from(arr, k=0)] = np.nan
    elif triangle not in (None, "none"):
        raise ValueError("triangle must be 'lower', 'upper', or None.")

    finite = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(np.nanmin(finite)) if finite.size else -1.0
    if vmax is None:
        vmax = float(np.nanmax(finite)) if finite.size else 0.0

    if ax is None:
        side = max(5.0, 0.42 * len(mat) + 2.5)
        fig, ax = plt.subplots(figsize=(side, side))
    else:
        fig = ax.figure
    im = ax.imshow(arr, cmap=_heatmap_cmap(), vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(len(mat.columns)), labels=mat.columns, rotation=55, ha="right")
    ax.set_yticks(np.arange(len(mat.index)), labels=mat.index)
    ax.tick_params(axis="both", length=0)
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i,j]:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("projected score")
    ax.set_title(title or "pairwise spatial relationships")
    return fig, ax


def plot_paired_projected_scores(
    result: SpatialVectorResult,
    feature_a: str,
    feature_b: str,
    ax=None,
    labels: tuple[str, str] | None = None,
    title: str | None = None,
    show_p: bool = True,
    colors: tuple[str, str] = ("#D95F76", "#777777"),
):
    """Figure-3F-like paired comparison of sample-level projected scores."""
    if result.sample_projected_score is None:
        raise ValueError("result.sample_projected_score is required; use spatial_vector_x().")
    df = result.sample_projected_score.pivot(index="sample", columns="feature", values="projected_score")
    if feature_a not in df.columns or feature_b not in df.columns:
        raise KeyError(f"Both {feature_a!r} and {feature_b!r} must be present in result features.")
    pair = df[[feature_a, feature_b]].dropna()
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 5.0))
    else:
        fig = ax.figure
    x = np.array([0.0, 1.0])
    for _, row in pair.iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color="0.72", lw=0.8, alpha=0.85)
    ax.scatter(np.zeros(len(pair)), pair[feature_a], s=28, color=colors[0], edgecolor="none", zorder=3)
    ax.scatter(np.ones(len(pair)), pair[feature_b], s=28, color=colors[1], edgecolor="none", zorder=3)

    p = np.nan
    if len(pair) > 0:
        try:
            p = float(wilcoxon(pair[feature_a], pair[feature_b], zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p = np.nan
    lab = labels or (feature_a, feature_b)
    ax.set_xticks(x, labels=lab)
    ax.set_ylabel("projected score")
    ax.set_xlim(-0.4, 1.4)
    ax.set_title(title or f"relation to {result.target}")
    if show_p and np.isfinite(p):
        ax.text(0.5, 0.98, f"paired Wilcoxon p = {p:.3g}", transform=ax.transAxes, ha="center", va="top")
    return fig, ax, {"n_pairs": len(pair), "pvalue": p}


def one_sided_proportion_test(
    successes: int,
    n: int,
    p0: float = 0.5,
    alternative: str = "less",
    continuity: bool = True,
):
    """Normal-approximation one-sample proportion test with continuity correction.

    With successes=12, n=44, p0=0.5, alternative='less', this reproduces the
    approximately 0.00209 value shown in the paper's Figure 4B.
    """
    if n <= 0:
        return np.nan
    p_hat = successes / n
    se = math.sqrt(p0 * (1 - p0) / n)
    correction = 0.5 / n if continuity else 0.0
    if alternative == "less":
        z = (p_hat - p0 + correction) / se
        p = norm.cdf(z)
    elif alternative == "greater":
        z = (p_hat - p0 - correction) / se
        p = norm.sf(z)
    elif alternative == "two-sided":
        delta = abs(p_hat - p0)
        z = max(delta - correction, 0.0) / se
        p = 2 * norm.sf(z)
    else:
        raise ValueError("alternative must be 'less', 'greater', or 'two-sided'.")
    return float(p)


def plot_colocalization_distribution(
    result: SpatialVectorResult,
    feature: str,
    threshold: float = 0.0,
    p0: float = 0.5,
    ax=None,
    title: str | None = None,
    pie_label: str = "% co-localized",
    density_color: str = "black",
    pie_colors: tuple[str, str] = ("#F2D28B", "#8C8C8C"),
):
    """Figure-4B-like density curve + co-localization pie chart.

    Co-localized samples are defined by sample-level projected_score < threshold,
    matching the paper's use of zero as the projected-score sign boundary.
    """
    if result.sample_projected_score is None:
        raise ValueError("result.sample_projected_score is required; use spatial_vector_x().")
    scores = result.sample_projected_score.loc[
        result.sample_projected_score["feature"] == feature, "projected_score"
    ].dropna().to_numpy(dtype=float)
    if scores.size == 0:
        raise ValueError(f"No sample-level projected scores found for feature {feature!r}.")
    coloc = scores < threshold
    k, n = int(coloc.sum()), int(scores.size)
    pct = 100 * k / n
    p = one_sided_proportion_test(k, n, p0=p0, alternative="less", continuity=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
    else:
        fig = ax.figure
    if scores.size >= 2 and np.nanstd(scores) > 0:
        kde = gaussian_kde(scores)
        pad = max(np.ptp(scores) * 0.12, 1e-6)
        xx = np.linspace(scores.min() - pad, scores.max() + pad, 400)
        ax.plot(xx, kde(xx), lw=1.6, color=density_color)
    else:
        ax.axvline(scores[0], lw=1.6, color=density_color)
    ax.axvline(threshold, ls=":", color="0.35", lw=1.0)
    ax.set_xlabel("projected score")
    ax.set_ylabel("Density")
    ax.set_title(title or f"{result.target} - {feature}, N={n}")

    inset = ax.inset_axes([0.07, 0.50, 0.34, 0.40])
    inset.pie([k, n - k], startangle=90, colors=pie_colors, wedgeprops={"linewidth": 0.6, "edgecolor": "white"})
    inset.text(0, 0, f"{pct:.1f}%", ha="center", va="center", fontsize=11)
    inset.set_title(pie_label, fontsize=9)
    ax.text(0.98, 0.96, f"one-tailed proportion p = {p:.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    return fig, ax, {"n": n, "n_colocalized": k, "percent_colocalized": pct, "pvalue": p, "threshold": threshold}


def plot_spatial_adjusted(
    obj: SpatialAdjusted,
    ax=None,
    point_size: float = 4.0,
    cmap: str = "viridis",
    invert_y: bool = False,
    title: str | None = None,
):
    """Plot one SpatialAdjusted feature map, separating plotting from computation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
    else:
        fig = ax.figure
    if obj.values is None:
        ax.scatter(obj.coords[:, 0], obj.coords[:, 1], s=point_size, edgecolor="none")
    else:
        sc = ax.scatter(
            obj.coords[:, 0], obj.coords[:, 1], c=obj.values, s=point_size,
            cmap=cmap, edgecolor="none", rasterized=True,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=obj.feature or "feature")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()
    ax.set_title(title or (obj.feature or "spatial map"))
    return fig, ax



def plot_grid_feature_map(
    result: GridFeatureMap,
    ax=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    show_positive_outline: bool = False,
    outline_color: str = "black",
    outline_width: float = 0.8,
    invert_y: bool = False,
    title: str | None = None,
):
    """Plot a gene/pathway score after the exact rasterization used by grid SPHERE.

    Empty/unoccupied grid cells are masked. ``show_positive_outline=True`` draws
    the binary high-feature domain defined by ``result.cutoff``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
    else:
        fig = ax.figure
    arr = np.asarray(result.grid, dtype=float)
    masked = np.ma.masked_where(~result.occupied | ~np.isfinite(arr), arr)
    mesh = ax.pcolormesh(
        result.x_edges,
        result.y_edges,
        masked,
        cmap=cmap,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{result.feature} ({result.agg} per grid)")
    if show_positive_outline and np.isfinite(result.cutoff):
        positive = result.occupied & np.isfinite(arr) & (arr > result.cutoff)
        if positive.any() and (~positive & result.occupied).any():
            xc = (result.x_edges[:-1] + result.x_edges[1:]) / 2.0
            yc = (result.y_edges[:-1] + result.y_edges[1:]) / 2.0
            ax.contour(
                xc,
                yc,
                positive.astype(float),
                levels=[0.5],
                colors=[outline_color],
                linewidths=outline_width,
            )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()
    sample = f" | {result.sample}" if result.sample is not None else ""
    ax.set_title(title or f"{result.feature} grid map{sample}")
    return fig, ax


def plot_kdtree_domain_map(
    result: KDTreeDomainResult,
    ax=None,
    point_size: float = 4.0,
    background_size: float = 1.2,
    colors: tuple[str, str, str] = ("#D95F76", "#4C78A8", "#8C6BB1"),
    background_color: str = "0.88",
    show_positive_cells: bool = False,
    invert_y: bool = False,
    title: str | None = None,
):
    """Visualize the radius-defined cell-anchor domains used by KDTree SPHERE.

    Colors correspond to target-only, feature-only, and overlap anchors. If a
    virtual feature shift was requested, the displayed feature domain is the
    re-projected shifted domain.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
    else:
        fig = ax.figure
    coords = result.coords
    target = np.asarray(result.target_domain, dtype=bool)
    feature = np.asarray(result.feature_domain, dtype=bool)
    overlap = target & feature
    target_only = target & ~feature
    feature_only = feature & ~target
    neither = ~(target | feature)
    if neither.any():
        ax.scatter(
            coords[neither, 0], coords[neither, 1], s=background_size,
            color=background_color, edgecolor="none", rasterized=True,
        )
    if target_only.any():
        ax.scatter(
            coords[target_only, 0], coords[target_only, 1], s=point_size,
            color=colors[0], edgecolor="none", label=result.target, rasterized=True,
        )
    if feature_only.any():
        ax.scatter(
            coords[feature_only, 0], coords[feature_only, 1], s=point_size,
            color=colors[1], edgecolor="none", label=result.feature, rasterized=True,
        )
    if overlap.any():
        ax.scatter(
            coords[overlap, 0], coords[overlap, 1], s=point_size * 1.15,
            color=colors[2], edgecolor="none", label="overlap", rasterized=True,
        )
    if show_positive_cells:
        tp = coords[result.target_positive]
        fp = coords[result.feature_positive] + np.asarray(result.shift, dtype=float)
        if tp.size:
            ax.scatter(tp[:, 0], tp[:, 1], s=point_size * 2.0, facecolors="none", edgecolors=colors[0], linewidths=0.5, rasterized=True)
        if fp.size:
            ax.scatter(fp[:, 0], fp[:, 1], s=point_size * 2.0, facecolors="none", edgecolors=colors[1], linewidths=0.5, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()
    ax.legend(frameon=False, markerscale=1.8)
    shift_txt = "" if np.allclose(result.shift, (0.0, 0.0)) else f" | shift=({result.shift[0]:g},{result.shift[1]:g})"
    sample_txt = f" | {result.sample}" if result.sample is not None else ""
    cov_txt = "" if not np.isfinite(result.coverage_fraction) else f" | coverage={result.coverage_fraction:.2f}"
    ax.set_title(title or f"KDTree domains r={result.radius:g}{shift_txt}{sample_txt}{cov_txt}")
    return fig, ax

def plot_spatial_overlap(
    obj1: SpatialAdjusted,
    obj2: SpatialAdjusted,
    ax=None,
    cutoffs=None,
    shift: tuple[float, float] = (0.0, 0.0),
    point_size: float = 5.0,
    colors: tuple[str, str] = ("orange", "steelblue"),
    invert_y: bool = False,
):
    """Binary spatial overlay analogous to ``spatial_cordstat(..., plot_bin=TRUE)``.

    ``cutoffs`` may be the legacy two-value tuple, a feature-keyed mapping, a
    pandas Series, or a :class:`sat4sc.pysphere.CutoffResult` returned by
    ``pysphere.calculate_cutoffs``. ``None`` retains local mean thresholds.
    """
    if obj1.values is None or obj2.values is None:
        raise ValueError("Both objects require feature values.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
    else:
        fig = ax.figure
    if cutoffs is None:
        c1 = float(np.nanmean(obj1.values))
        c2 = float(np.nanmean(obj2.values))
    elif isinstance(cutoffs, CutoffResult):
        mapping = cutoffs.cutoffs.to_dict()
        if obj1.feature not in mapping or obj2.feature not in mapping:
            raise KeyError("CutoffResult must contain both object feature names.")
        c1, c2 = float(mapping[obj1.feature]), float(mapping[obj2.feature])
    elif isinstance(cutoffs, Mapping) or isinstance(cutoffs, pd.Series):
        mapping = dict(cutoffs)
        if obj1.feature not in mapping or obj2.feature not in mapping:
            raise KeyError("cutoffs mapping must contain both object feature names.")
        c1, c2 = float(mapping[obj1.feature]), float(mapping[obj2.feature])
    else:
        c1, c2 = map(float, cutoffs)
    a = np.asarray(obj1.values) > c1
    b = np.asarray(obj2.values) > c2
    ax.scatter(obj1.coords[a, 0], obj1.coords[a, 1], s=point_size, color=colors[0], edgecolor="none", label=obj1.feature, rasterized=True)
    shifted = obj2.coords + np.asarray(shift, dtype=float)
    ax.scatter(shifted[b, 0], shifted[b, 1], s=point_size, facecolors="none", edgecolors=colors[1], linewidths=0.6, label=obj2.feature, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()
    ax.legend(frameon=False)
    ax.set_title(f"{obj1.feature}:{obj2.feature}")
    return fig, ax



def plot_binstat(
    result: BinStatResult,
    ax=None,
    cmap: str = "viridis",
    show_mask: bool = True,
    annotate: bool = False,
):
    """Visualize the binned positive fraction from pysphere.spatial_binstat()."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
    else:
        fig = ax.figure
    im = ax.imshow(result.stat_mtx_pct, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="positive fraction")
    if show_mask:
        yy, xx = np.where(result.stat_mtx_msk == 1)
        ax.scatter(xx, yy, marker="s", facecolors="none", edgecolors="red", s=180, linewidths=0.9)
    if annotate:
        for r in range(result.stat_mtx_pct.shape[0]):
            for c in range(result.stat_mtx_pct.shape[1]):
                v = result.stat_mtx_pct[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(result.feature or "spatial bin statistics")
    ax.set_xlabel("x bin"); ax.set_ylabel("y bin")
    return fig, ax


# R-style aliases
spatial_vecPlot = plot_vector
spatial_magPlot = plot_magnitude_projected


__all__ = [
    "plot_vector",
    "plot_magnitude_projected",
    "plot_projected_heatmap",
    "plot_paired_projected_scores",
    "plot_colocalization_distribution",
    "one_sided_proportion_test",
    "plot_spatial_adjusted",
    "plot_grid_feature_map",
    "plot_kdtree_domain_map",
    "plot_spatial_overlap",
    "plot_binstat",
    "spatial_vecPlot",
    "spatial_magPlot",
]


# -----------------------------------------------------------------------------
# v0.2.1 high-level binary overlay
# -----------------------------------------------------------------------------
def _resolve_binary_overlay_cutoff(values, cutoff):
    """Resolve a binary-overlay cutoff without modifying the input values.

    Supported forms
    ---------------
    ``None`` or ``"mean"``
        Mean of finite values (SPHERE-style default).
    ``"median"``
        Median of finite values.
    ``"zero"``
        Zero.
    numeric
        Explicit cutoff.
    ``("quantile", q)``
        Quantile ``q`` of finite values, with ``0 <= q <= 1``.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    if cutoff is None or cutoff == "mean":
        return float(np.mean(finite))
    if isinstance(cutoff, str):
        if cutoff == "median":
            return float(np.median(finite))
        if cutoff == "zero":
            return 0.0
        raise ValueError("cutoff must be None, 'mean', 'median', 'zero', a numeric value, or ('quantile', q).")
    if isinstance(cutoff, (tuple, list)) and len(cutoff) == 2 and str(cutoff[0]).lower() == "quantile":
        q = float(cutoff[1])
        if not 0 <= q <= 1:
            raise ValueError("Quantile cutoff q must be in [0, 1].")
        return float(np.quantile(finite, q))
    if np.isscalar(cutoff):
        return float(cutoff)
    raise TypeError("Unsupported cutoff specification.")


def plot_binary_overlay(
    adata,
    feature1: str,
    feature2: str,
    sample: str | None = None,
    sample_key: str = "sample_name",
    coord_cols: tuple[str, str] = ("x_centroid", "y_centroid"),
    spatial_key: str = "spatial",
    layer: str | None = None,
    space: str = "cell",
    grid_size: float = 20.0,
    agg: str = "mean",
    cutoff1=None,
    cutoff2=None,
    cutoff_method: str | None = None,
    cohort_cutoffs=None,
    cutoff_samples: Sequence[str] | None = None,
    cutoff_n_repeats: int = 100,
    cutoff_balance_round_to: int = 1000,
    cutoff_balance_n: int | None = None,
    cutoff_random_state: int = 666,
    min_cells_per_bin: int = 1,
    ax=None,
    point_size: float = 5.0,
    background_size: float = 1.2,
    colors: tuple[str, str] = ("#F2D28B", "#707070"),
    background_color: str = "0.90",
    show_background: bool = True,
    feature2_linewidth: float = 0.65,
    invert_y: bool = False,
    title: str | None = None,
):
    """High-level two-feature binary spatial overlay directly from AnnData.

    ``space='cell'`` thresholds original cells/spots; ``space='grid'`` thresholds
    rasterized grid scores. Set ``cutoff_method`` to one of
    ``median_of_sample_medians``, ``mean_of_sample_means``,
    ``balanced_global_median`` or ``balanced_global_mean`` to use the same
    cohort-wide cutoff in every sample. ``cohort_cutoffs`` can instead receive a
    precomputed mapping/Series/CutoffResult. Explicit ``cutoff1`` or ``cutoff2``
    overrides the cohort cutoff for that feature.

    Legacy local cutoff specifications are still supported: ``None``/``'mean'``,
    ``'median'``, ``'zero'``, a numeric value, or ``('quantile', q)``.
    """
    from . import pysphere as _pysphere

    space = str(space).lower()
    if space not in {"cell", "grid"}:
        raise ValueError("space must be either 'cell' or 'grid'.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.5))
    else:
        fig = ax.figure

    cohort_map = _pysphere._coerce_cutoff_mapping(cohort_cutoffs)
    need_global = []
    if cutoff1 is None and feature1 not in cohort_map and cutoff_method is not None:
        need_global.append(feature1)
    if cutoff2 is None and feature2 not in cohort_map and cutoff_method is not None:
        need_global.append(feature2)
    cutoff_result = None
    if need_global:
        cutoff_result = _pysphere.calculate_cutoffs(
            adata,
            features=list(dict.fromkeys(need_global)),
            sample_key=sample_key,
            samples=cutoff_samples,
            level=space,
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
        cohort_map.update(cutoff_result.cutoffs.to_dict())

    resolved_spec1 = cutoff1 if cutoff1 is not None else cohort_map.get(feature1, None)
    resolved_spec2 = cutoff2 if cutoff2 is not None else cohort_map.get(feature2, None)

    if space == "cell":
        obj1 = _pysphere.spatial_adjust(
            adata, feature=feature1, sample=sample, sample_key=sample_key,
            coord_cols=coord_cols, spatial_key=spatial_key, layer=layer,
        )
        obj2 = _pysphere.spatial_adjust(
            adata, feature=feature2, sample=sample, sample_key=sample_key,
            coord_cols=coord_cols, spatial_key=spatial_key, layer=layer,
        )
        if obj1.values is None or obj2.values is None:
            raise ValueError("Both features require numeric values.")
        if obj1.coords.shape != obj2.coords.shape or not np.allclose(obj1.coords, obj2.coords, equal_nan=True):
            raise ValueError("feature1 and feature2 did not resolve to the same spatial coordinates.")
        values1 = np.asarray(obj1.values, dtype=float)
        values2 = np.asarray(obj2.values, dtype=float)
        c1 = _resolve_binary_overlay_cutoff(values1, resolved_spec1)
        c2 = _resolve_binary_overlay_cutoff(values2, resolved_spec2)
        pos1 = np.isfinite(values1) & (values1 > c1)
        pos2 = np.isfinite(values2) & (values2 > c2)
        coords = np.asarray(obj1.coords, dtype=float)
        if show_background:
            ax.scatter(coords[:, 0], coords[:, 1], s=background_size, color=background_color, edgecolor="none", rasterized=True, zorder=1)
        if pos1.any():
            ax.scatter(coords[pos1, 0], coords[pos1, 1], s=point_size, color=colors[0], edgecolor="none", label=feature1, rasterized=True, zorder=2)
        if pos2.any():
            ax.scatter(coords[pos2, 0], coords[pos2, 1], s=point_size, facecolors="none", edgecolors=colors[1], linewidths=feature2_linewidth, label=feature2, rasterized=True, zorder=3)
        info = {
            "space": "cell", "sample": sample, "feature1": feature1, "feature2": feature2,
            "cutoff1": c1, "cutoff2": c2, "cutoff_method": cutoff_method,
            "n_units": int(len(coords)), "n_feature1_positive": int(pos1.sum()),
            "n_feature2_positive": int(pos2.sum()), "n_overlap": int((pos1 & pos2).sum()),
            "feature1_positive_fraction": float(pos1.mean()) if len(pos1) else np.nan,
            "feature2_positive_fraction": float(pos2.mean()) if len(pos2) else np.nan,
        }
    else:
        map1 = _pysphere.grid_feature_map(
            adata, feature=feature1, sample=sample, sample_key=sample_key,
            coord_cols=coord_cols, spatial_key=spatial_key, layer=layer,
            grid_size=grid_size, agg=agg, min_cells_per_bin=min_cells_per_bin,
        )
        map2 = _pysphere.grid_feature_map(
            adata, feature=feature2, sample=sample, sample_key=sample_key,
            coord_cols=coord_cols, spatial_key=spatial_key, layer=layer,
            grid_size=grid_size, agg=agg, min_cells_per_bin=min_cells_per_bin,
        )
        occupied = np.asarray(map1.occupied & map2.occupied, dtype=bool)
        values1 = np.asarray(map1.grid, dtype=float)
        values2 = np.asarray(map2.grid, dtype=float)
        c1 = _resolve_binary_overlay_cutoff(values1[occupied], resolved_spec1)
        c2 = _resolve_binary_overlay_cutoff(values2[occupied], resolved_spec2)
        pos1 = occupied & np.isfinite(values1) & (values1 > c1)
        pos2 = occupied & np.isfinite(values2) & (values2 > c2)
        from matplotlib.colors import ListedColormap
        if show_background:
            bg = np.ma.masked_where(~occupied, np.ones_like(values1, dtype=float))
            ax.pcolormesh(map1.x_edges, map1.y_edges, bg, cmap=ListedColormap([background_color]), shading="flat", vmin=0, vmax=1, rasterized=True, zorder=1)
        fill1 = np.ma.masked_where(~pos1, np.ones_like(values1, dtype=float))
        if pos1.any():
            ax.pcolormesh(map1.x_edges, map1.y_edges, fill1, cmap=ListedColormap([colors[0]]), shading="flat", vmin=0, vmax=1, rasterized=True, zorder=2)
        if pos2.any():
            xc = (map2.x_edges[:-1] + map2.x_edges[1:]) / 2.0
            yc = (map2.y_edges[:-1] + map2.y_edges[1:]) / 2.0
            if pos2.shape[0] > 1 and pos2.shape[1] > 1 and (~pos2).any():
                ax.contour(xc, yc, pos2.astype(float), levels=[0.5], colors=[colors[1]], linewidths=feature2_linewidth, zorder=3)
            else:
                yy, xx = np.where(pos2)
                if len(xx):
                    ax.scatter(xc[xx], yc[yy], marker="s", s=max(point_size * 3.0, 12.0), facecolors="none", edgecolors=colors[1], linewidths=feature2_linewidth, rasterized=True, zorder=3)
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=colors[0], markeredgecolor="none", markersize=8, label=feature1),
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor=colors[1], markersize=8, label=feature2),
        ]
        ax.legend(handles=handles, frameon=False)
        n_units = int(occupied.sum())
        info = {
            "space": "grid", "sample": sample, "feature1": feature1, "feature2": feature2,
            "cutoff1": c1, "cutoff2": c2, "cutoff_method": cutoff_method,
            "grid_size": float(grid_size), "agg": str(agg), "n_units": n_units,
            "n_feature1_positive": int(pos1.sum()), "n_feature2_positive": int(pos2.sum()),
            "n_overlap": int((pos1 & pos2).sum()),
            "feature1_positive_fraction": np.nan if n_units == 0 else float(pos1.sum() / n_units),
            "feature2_positive_fraction": np.nan if n_units == 0 else float(pos2.sum() / n_units),
        }

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()
    sample_txt = f" | {sample}" if sample is not None else ""
    ax.set_title(title or f"{feature1} : {feature2} ({space}){sample_txt}")
    if space == "cell":
        ax.legend(frameon=False)
    return fig, ax, info



# -----------------------------------------------------------------------------
# v0.3.0 niche visualizations
# -----------------------------------------------------------------------------
def _get_niche_sample(result: NicheResult, sample: str | None):
    if sample is None:
        if len(result.sample_results) != 1:
            raise ValueError("sample must be provided when NicheResult contains multiple samples.")
        return next(iter(result.sample_results.values()))
    key = str(sample)
    if key not in result.sample_results:
        raise KeyError(f"Sample {key!r} not found in NicheResult for {result.feature!r}.")
    return result.sample_results[key]


def _categorical_spatial_plot(
    coords_or_grid,
    categories: np.ndarray,
    labels: tuple[str, str, str, str],
    colors: tuple[str, str, str, str],
    space: str,
    ax,
    point_size: float,
    x_edges=None,
    y_edges=None,
    occupied=None,
):
    """Shared 4-category renderer: other, object1, object2, overlap."""
    if space == "cell":
        coords = np.asarray(coords_or_grid, dtype=float)
        for code in range(4):
            mask = categories == code
            if mask.any():
                ax.scatter(
                    coords[mask, 0], coords[mask, 1], s=point_size, color=colors[code],
                    edgecolor="none", label=labels[code], rasterized=True, zorder=1 + code,
                )
    else:
        occupied = np.asarray(occupied, dtype=bool)
        cat = np.asarray(categories, dtype=float)
        masked = np.ma.masked_where(~occupied, cat)
        ax.pcolormesh(
            x_edges, y_edges, masked, cmap=ListedColormap(list(colors)),
            shading="flat", vmin=-0.5, vmax=3.5, rasterized=True,
        )
        handles = [Patch(facecolor=colors[i], edgecolor="none", label=labels[i]) for i in range(4)]
        ax.legend(handles=handles, frameon=False)


def _draw_grid_mask_boundary(
    ax,
    mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    color: str = "black",
    linewidth: float = 0.8,
    zorder: float = 5,
):
    """Draw exact outer boundaries of a boolean grid mask."""
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    segments = []
    for y, x in zip(*np.where(mask)):
        x0, x1 = x_edges[x], x_edges[x + 1]
        y0, y1 = y_edges[y], y_edges[y + 1]
        if y == 0 or not mask[y - 1, x]:
            segments.append([(x0, y0), (x1, y0)])
        if y == ny - 1 or not mask[y + 1, x]:
            segments.append([(x0, y1), (x1, y1)])
        if x == 0 or not mask[y, x - 1]:
            segments.append([(x0, y0), (x0, y1)])
        if x == nx - 1 or not mask[y, x + 1]:
            segments.append([(x1, y0), (x1, y1)])
    if segments:
        lc = LineCollection(segments, colors=[color], linewidths=linewidth, zorder=zorder)
        ax.add_collection(lc)
    return segments


def _resolve_feature_positive_for_niche_plot(
    adata,
    niche_sample,
    feature: str,
    space: str,
    sample_key: str,
    coord_cols,
    spatial_key: str,
    layer: str | None,
    agg: str,
    cutoff,
    cutoff_method: str | None,
    cohort_cutoffs,
    cutoff_samples,
    cutoff_n_repeats: int,
    cutoff_balance_round_to: int,
    cutoff_balance_n: int | None,
    cutoff_random_state: int,
    min_cells_per_bin: int,
):
    from . import pysphere as _pysphere
    mapping = _pysphere._coerce_cutoff_mapping(cohort_cutoffs)
    if cutoff is not None:
        resolved = float(cutoff)
    elif feature in mapping:
        resolved = float(mapping[feature])
    elif cutoff_method is not None:
        cr = _pysphere.calculate_cutoffs(
            adata, features=[feature], sample_key=sample_key, samples=cutoff_samples,
            level=space, method=cutoff_method, coord_cols=coord_cols,
            spatial_key=spatial_key, layer=layer,
            grid_size=float(niche_sample.x_edges[1] - niche_sample.x_edges[0]),
            agg=agg,
            min_cells_per_bin=int(min_cells_per_bin),
            n_repeats=cutoff_n_repeats,
            balance_round_to=cutoff_balance_round_to,
            balance_n=cutoff_balance_n,
            random_state=cutoff_random_state,
        )
        resolved = float(cr.cutoffs[feature])
    else:
        resolved = None

    values_cell = _pysphere._resolve_feature_vector(adata, feature, niche_sample.cell_indices, layer=layer)
    if space == "cell":
        values = np.asarray(values_cell, dtype=float)
        c = _resolve_binary_overlay_cutoff(values, resolved)
        positive = np.isfinite(values) & (values > c)
        return positive, values, c

    grid_size = float(niche_sample.x_edges[1] - niche_sample.x_edges[0])
    flat, counts, occupied, shape, xy_min = _pysphere._make_grid(niche_sample.coords, grid_size, int(min_cells_per_bin))
    if shape != niche_sample.grid.shape or not np.allclose(xy_min, [niche_sample.x_edges[0], niche_sample.y_edges[0]]):
        raise ValueError("Feature grid geometry does not match the niche grid geometry.")
    grid = _pysphere._aggregate_grid(values_cell, flat, counts, shape, agg=agg)
    c = _resolve_binary_overlay_cutoff(grid[niche_sample.occupied & np.isfinite(grid)], resolved)
    positive = niche_sample.occupied & np.isfinite(grid) & (grid > c)
    return positive, grid, c


def plot_niche_overlay(
    niche1: NicheResult,
    niche2: NicheResult,
    sample: str | None = None,
    space: str = "grid",
    ax=None,
    point_size: float = 4.0,
    colors: tuple[str, str, str, str] = ("#D9D9D9", "#E76F51", "#4C78A8", "#8E6BBE"),
    labels: tuple[str, str, str, str] | None = None,
    invert_y: bool = False,
    title: str | None = None,
):
    """Visualize two niches simultaneously as other/niche1/niche2/overlap.

    ``space='grid'`` displays the retained niche grids. ``space='cell'`` displays
    cell/spot membership inherited from those niche grids.
    """
    space = str(space).lower()
    if space not in {"grid", "cell"}:
        raise ValueError("space must be 'grid' or 'cell'.")
    s1 = _get_niche_sample(niche1, sample)
    s2 = _get_niche_sample(niche2, sample)
    if not np.array_equal(s1.cell_indices, s2.cell_indices):
        raise ValueError("The two niche results do not contain the same cells/spots for this sample.")
    if s1.niche_grid.shape != s2.niche_grid.shape or not np.allclose(s1.x_edges, s2.x_edges) or not np.allclose(s1.y_edges, s2.y_edges):
        raise ValueError("The two niche results do not share the same grid geometry.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.5))
    else:
        fig = ax.figure
    labels = labels or ("Other", f"{niche1.feature} niche", f"{niche2.feature} niche", "Overlap")
    if space == "grid":
        a, b = s1.niche_grid, s2.niche_grid
        occupied = s1.occupied & s2.occupied
        categories = np.zeros(a.shape, dtype=np.int8)
        categories[a & ~b] = 1
        categories[~a & b] = 2
        categories[a & b] = 3
        _categorical_spatial_plot(None, categories, labels, colors, "grid", ax, point_size, s1.x_edges, s1.y_edges, occupied)
        n_units = int(occupied.sum())
    else:
        a, b = s1.cell_niche, s2.cell_niche
        categories = np.zeros(len(a), dtype=np.int8)
        categories[a & ~b] = 1
        categories[~a & b] = 2
        categories[a & b] = 3
        _categorical_spatial_plot(s1.coords, categories, labels, colors, "cell", ax, point_size)
        n_units = int(len(a))
        ax.legend(frameon=False)
    if space == "grid":
        counts = {labels[i]: int(np.sum((categories == i) & occupied)) for i in range(4)}
    else:
        counts = {labels[i]: int(np.sum(categories == i)) for i in range(4)}
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()
    sample_txt = s1.sample
    ax.set_title(title or f"{niche1.feature} vs {niche2.feature} niches | {sample_txt} ({space})")
    return fig, ax, {"sample": sample_txt, "space": space, "n_units": n_units, "counts": counts}


def plot_niche_positive_overlay(
    niche: NicheResult,
    adata,
    feature: str,
    sample: str | None = None,
    space: str = "grid",
    sample_key: str = "sample_name",
    coord_cols: tuple[str, str] = ("x_centroid", "y_centroid"),
    spatial_key: str = "spatial",
    layer: str | None = None,
    agg: str = "mean",
    cutoff=None,
    cutoff_method: str | None = "balanced_global_mean",
    cohort_cutoffs=None,
    cutoff_samples: Sequence[str] | None = None,
    cutoff_n_repeats: int = 100,
    cutoff_balance_round_to: int = 1000,
    cutoff_balance_n: int | None = None,
    cutoff_random_state: int = 666,
    ax=None,
    point_size: float = 4.0,
    colors: tuple[str, str, str, str] = ("#D9D9D9", "#E76F51", "#4C78A8", "#8E6BBE"),
    labels: tuple[str, str, str, str] | None = None,
    invert_y: bool = False,
    title: str | None = None,
):
    """Overlay one retained niche with positive status of another feature.

    Four categories are shown: other, niche only, feature-positive only, overlap.
    Grid space uses a grid-level feature cutoff; cell space uses a cell-level
    feature cutoff. Cohort-wide cutoff methods from v0.2.2 are supported.
    """
    space = str(space).lower()
    if space not in {"grid", "cell"}:
        raise ValueError("space must be 'grid' or 'cell'.")
    ns = _get_niche_sample(niche, sample)
    if cutoff_samples is None and cutoff_method is not None:
        cutoff_samples = niche.settings.get("cutoff_samples", niche.settings.get("samples"))
    positive, _, resolved_cutoff = _resolve_feature_positive_for_niche_plot(
        adata, ns, feature, space, sample_key, coord_cols, spatial_key, layer, agg,
        cutoff, cutoff_method, cohort_cutoffs, cutoff_samples, cutoff_n_repeats,
        cutoff_balance_round_to, cutoff_balance_n, cutoff_random_state,
        int(niche.settings.get("min_cells_per_bin", 1)),
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.5))
    else:
        fig = ax.figure
    labels = labels or ("Other", f"{niche.feature} niche", f"{feature}+", "Overlap")
    if space == "grid":
        a = ns.niche_grid
        b = positive
        categories = np.zeros(a.shape, dtype=np.int8)
        categories[a & ~b] = 1; categories[~a & b] = 2; categories[a & b] = 3
        _categorical_spatial_plot(None, categories, labels, colors, "grid", ax, point_size, ns.x_edges, ns.y_edges, ns.occupied)
        n_units = int(ns.occupied.sum())
    else:
        a = ns.cell_niche
        b = positive
        categories = np.zeros(len(a), dtype=np.int8)
        categories[a & ~b] = 1; categories[~a & b] = 2; categories[a & b] = 3
        _categorical_spatial_plot(ns.coords, categories, labels, colors, "cell", ax, point_size)
        n_units = int(len(a))
        ax.legend(frameon=False)
    if space == "grid":
        counts = {labels[i]: int(np.sum((categories == i) & ns.occupied)) for i in range(4)}
    else:
        counts = {labels[i]: int(np.sum(categories == i)) for i in range(4)}
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()
    ax.set_title(title or f"{niche.feature} niche + {feature}+ | {ns.sample} ({space})")
    return fig, ax, {
        "sample": ns.sample, "space": space, "feature": feature,
        "feature_cutoff": float(resolved_cutoff), "cutoff_method": cutoff_method,
        "n_units": n_units, "counts": counts,
    }


def plot_niche_continuous_feature(
    niche: NicheResult,
    adata,
    feature: str,
    sample: str | None = None,
    space: str = "grid",
    sample_key: str = "sample_name",
    coord_cols: tuple[str, str] = ("x_centroid", "y_centroid"),
    spatial_key: str = "spatial",
    layer: str | None = None,
    agg: str = "mean",
    ax=None,
    cmap: str = "viridis",
    color_quantiles: tuple[float, float] | None = (0.01, 0.99),
    vmin: float | None = None,
    vmax: float | None = None,
    point_size: float = 4.0,
    niche_edge_color: str = "black",
    niche_linewidth: float = 0.55,
    invert_y: bool = False,
    title: str | None = None,
):
    """Show continuous feature intensity together with niche membership.

    Visualization encoding
    ----------------------
    ``space='cell'``
        All cells/spots are colored by the continuous feature value. Cells that
        reside in the niche are redrawn with a black outline while retaining the
        same face color.
    ``space='grid'``
        The feature is shown as a continuous raster map and the outer boundary
        of retained niche grids is drawn in black.

    By default the color scale uses the 1st-99th percentiles of finite values to
    reduce domination by extreme outliers; this affects only color scaling, not
    the underlying values or niche definition. Set ``color_quantiles=None`` for
    the full range, or provide explicit ``vmin``/``vmax``.
    """
    from . import pysphere as _pysphere
    space = str(space).lower()
    if space not in {"grid", "cell"}:
        raise ValueError("space must be 'grid' or 'cell'.")
    ns = _get_niche_sample(niche, sample)
    values_cell = _pysphere._resolve_feature_vector(adata, feature, ns.cell_indices, layer=layer)
    if space == "cell":
        values = np.asarray(values_cell, dtype=float)
        coords = ns.coords
        finite = np.isfinite(values)
        plot_vals = values[finite]
    else:
        grid_size = float(ns.x_edges[1] - ns.x_edges[0])
        flat, counts, occupied, shape, xy_min = _pysphere._make_grid(
            ns.coords, grid_size, int(niche.settings.get("min_cells_per_bin", 1))
        )
        if shape != ns.grid.shape or not np.allclose(xy_min, [ns.x_edges[0], ns.y_edges[0]]):
            raise ValueError("Feature grid geometry does not match the niche grid geometry.")
        values = _pysphere._aggregate_grid(values_cell, flat, counts, shape, agg=agg)
        finite = ns.occupied & np.isfinite(values)
        plot_vals = values[finite]
    if plot_vals.size == 0:
        raise ValueError(f"Feature {feature!r} has no finite values in sample {ns.sample!r}.")
    if vmin is None or vmax is None:
        if color_quantiles is None:
            lo, hi = float(np.nanmin(plot_vals)), float(np.nanmax(plot_vals))
        else:
            q0, q1 = map(float, color_quantiles)
            if not (0 <= q0 < q1 <= 1):
                raise ValueError("color_quantiles must satisfy 0 <= q0 < q1 <= 1.")
            lo, hi = np.quantile(plot_vals, [q0, q1]).astype(float)
        if vmin is None:
            vmin = float(lo)
        if vmax is None:
            vmax = float(hi)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("vmin and vmax must be finite.")
    if np.isclose(vmin, vmax):
        eps = max(abs(float(vmin)) * 1e-6, 1e-9)
        vmin, vmax = float(vmin) - eps, float(vmax) + eps
    norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.5))
    else:
        fig = ax.figure
    if space == "cell":
        sc = ax.scatter(
            coords[finite, 0], coords[finite, 1], c=values[finite], cmap=cmap, norm=norm,
            s=point_size, edgecolors="none", rasterized=True, zorder=1,
        )
        niche_mask = finite & ns.cell_niche
        if niche_mask.any():
            ax.scatter(
                coords[niche_mask, 0], coords[niche_mask, 1], c=values[niche_mask],
                cmap=cmap, norm=norm, s=point_size, edgecolors=niche_edge_color,
                linewidths=niche_linewidth, rasterized=True, zorder=2,
            )
    else:
        masked = np.ma.masked_where(~finite, values)
        sc = ax.pcolormesh(
            ns.x_edges, ns.y_edges, masked, cmap=cmap, norm=norm,
            shading="flat", rasterized=True, zorder=1,
        )
        _draw_grid_mask_boundary(
            ax, ns.niche_grid, ns.x_edges, ns.y_edges,
            color=niche_edge_color, linewidth=niche_linewidth, zorder=3,
        )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(feature)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()
    ax.set_title(title or f"{feature} with {niche.feature} niche | {ns.sample} ({space})")
    return fig, ax, {
        "sample": ns.sample, "space": space, "feature": feature,
        "niche_feature": niche.feature, "vmin": float(vmin), "vmax": float(vmax),
        "color_quantiles": color_quantiles,
    }


__all__.append("plot_binary_overlay")
__all__.extend([
    "plot_niche_overlay",
    "plot_niche_positive_overlay",
    "plot_niche_continuous_feature",
])
