"""Plotting utilities for sat4sc.pysphere results.

These functions are designed to reproduce the visual logic of the SPHERE paper,
including Figure-3A-like projected-score heatmaps, Figure-3B-like vector plots,
Figure-3F-like paired projected-score comparisons, and Figure-4B-like projected
score density + co-localization proportion summaries.
"""

from __future__ import annotations

from typing import Sequence
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde, norm, wilcoxon

from .pysphere import BinStatResult, PairwiseResult, SpatialAdjusted, SpatialVectorResult


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
    cbar.set_label("step")
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
    arr = mat.to_numpy(dtype=float)
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


def plot_spatial_overlap(
    obj1: SpatialAdjusted,
    obj2: SpatialAdjusted,
    ax=None,
    cutoffs: tuple[float, float] | None = None,
    shift: tuple[float, float] = (0.0, 0.0),
    point_size: float = 5.0,
    colors: tuple[str, str] = ("orange", "steelblue"),
    invert_y: bool = False,
):
    """Binary spatial overlay analogous to spatial_cordstat(..., plot_bin=TRUE)."""
    if obj1.values is None or obj2.values is None:
        raise ValueError("Both objects require feature values.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
    else:
        fig = ax.figure
    c1 = float(np.nanmean(obj1.values)) if cutoffs is None else float(cutoffs[0])
    c2 = float(np.nanmean(obj2.values)) if cutoffs is None else float(cutoffs[1])
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
    "plot_spatial_overlap",
    "plot_binstat",
    "spatial_vecPlot",
    "spatial_magPlot",
]
