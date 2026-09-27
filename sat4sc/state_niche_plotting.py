"""Plotting helpers for :mod:`sat4sc.state_niche`."""

from __future__ import annotations

from math import ceil
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .state_niche import ClusterAutoK, DEFAULT_COORD_COLS, DEFAULT_SPATIAL_KEY


def stability(
    autok: ClusterAutoK,
    *,
    ax=None,
    show_adjacent: bool = False,
    annotate_peaks: bool = True,
    title: str = "State-niche GMM stability",
):
    """Plot repeated-GMM Fowlkes-Mallows stability across candidate K."""

    if autok.stability_ is None:
        raise RuntimeError("autok.fit(...) must be called before plotting stability.")
    df = autok.stability_
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["k"], df["stability"], marker="o", label="within-K stability")
    if show_adjacent:
        ax.plot(df["k"], df["prev_similarity"], marker=".", label="similarity to K-1")
        ax.plot(df["k"], df["next_similarity"], marker=".", label="similarity to K+1")
    if annotate_peaks:
        peaks = df[df["is_peak"]]
        ax.scatter(peaks["k"], peaks["stability"], s=70, zorder=3, label="local peak")
        for row in peaks.itertuples(index=False):
            ax.annotate(
                f"K={int(row.k)}",
                (row.k, row.stability),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set_xlabel("Number of GMM clusters (K)")
    ax.set_ylabel("Fowlkes–Mallows stability")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    return ax


def _coords_from_adata(adata, spatial_key, coord_cols):
    if spatial_key in adata.obsm:
        return np.asarray(adata.obsm[spatial_key])[:, :2]
    missing = [c for c in coord_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"No adata.obsm[{spatial_key!r}] and missing coordinate columns: {missing}."
        )
    return adata.obs[list(coord_cols)].to_numpy()


def spatial(
    adata,
    *,
    cluster_key: str = "state_niche",
    sample_key: str = "sample_name",
    samples: Sequence[str] | None = None,
    spatial_key: str = DEFAULT_SPATIAL_KEY,
    coord_cols: Sequence[str] = DEFAULT_COORD_COLS,
    ncols: int = 4,
    point_size: float = 1.0,
    invert_y: bool = True,
    figsize_per_panel: tuple[float, float] = (4.0, 4.0),
):
    """Plot state-niche labels in spatial coordinates for one or more samples."""

    if cluster_key not in adata.obs.columns:
        raise KeyError(f"{cluster_key!r} not found in adata.obs.")
    if sample_key not in adata.obs.columns:
        raise KeyError(f"{sample_key!r} not found in adata.obs.")
    coords = _coords_from_adata(adata, spatial_key, coord_cols)
    sample_values = adata.obs[sample_key].astype(str).to_numpy()
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    if samples is None:
        samples = list(pd.unique(sample_values))
    else:
        samples = [str(s) for s in samples]

    categories = sorted(pd.unique(labels), key=lambda x: (len(str(x)), str(x)))
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    color_map = {cat: cmap(i) for i, cat in enumerate(categories)}

    ncols = max(1, min(int(ncols), len(samples)))
    nrows = ceil(len(samples) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, sample in zip(axes_flat, samples):
        mask = sample_values == sample
        for cat in categories:
            m = mask & (labels == cat)
            if np.any(m):
                ax.scatter(
                    coords[m, 0],
                    coords[m, 1],
                    s=point_size,
                    color=color_map[cat],
                    linewidths=0,
                    rasterized=True,
                    label=cat,
                )
        ax.set_title(sample)
        ax.set_aspect("equal")
        if invert_y:
            ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(samples):]:
        ax.axis("off")

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", markersize=5,
            color=color_map[cat], label=str(cat)
        )
        for cat in categories
    ]
    if handles:
        fig.legend(handles=handles, title=cluster_key, loc="center right", frameon=False)
        fig.subplots_adjust(right=0.88)
    return fig, axes


def composition(
    adata,
    *,
    cluster_key: str = "state_niche",
    celltype_key: str = "cell_type",
    normalize: bool = True,
    ax=None,
):
    """Stacked bar plot of cell-type composition inside each state niche."""

    for key in (cluster_key, celltype_key):
        if key not in adata.obs.columns:
            raise KeyError(f"{key!r} not found in adata.obs.")
    tab = pd.crosstab(adata.obs[cluster_key], adata.obs[celltype_key])
    if normalize:
        tab = tab.div(tab.sum(axis=1), axis=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(7, 0.55 * len(tab)), 5))
    tab.plot(kind="bar", stacked=True, ax=ax, width=0.9)
    ax.set_xlabel(cluster_key)
    ax.set_ylabel("Fraction" if normalize else "Cell count")
    ax.legend(title=celltype_key, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    return ax, tab


def sample_composition(
    adata,
    *,
    cluster_key: str = "state_niche",
    sample_key: str = "sample_name",
    normalize: bool = True,
    ax=None,
):
    """Plot state-niche proportions for every sample and return the table."""

    for key in (cluster_key, sample_key):
        if key not in adata.obs.columns:
            raise KeyError(f"{key!r} not found in adata.obs.")
    tab = pd.crosstab(adata.obs[sample_key], adata.obs[cluster_key])
    if normalize:
        tab = tab.div(tab.sum(axis=1), axis=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, 0.6 * len(tab)), 5))
    tab.plot(kind="bar", stacked=True, ax=ax, width=0.9)
    ax.set_xlabel(sample_key)
    ax.set_ylabel("Fraction" if normalize else "Cell count")
    ax.legend(title=cluster_key, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    return ax, tab


__all__ = ["stability", "spatial", "composition", "sample_composition"]
