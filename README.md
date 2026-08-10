# sat4sc

`sat4sc` 是面向单细胞/亚细胞分辨率空间转录组的 Python 工具包。本版本首先实现 `pysphere`：将 Wu et al. 的 R 包 **SPHERE** 的核心算法用 Python 重写，并适配 `AnnData`。

目标接口：

```python
from sat4sc import pysphere
from sat4sc import plotting
```

## 1. 目录结构

```text
sat4sc/
├── pyproject.toml
├── README.md
├── NOTICE.md
└── sat4sc/
    ├── __init__.py
    ├── pysphere.py      # 计算：SPHERE、projected score、magnitude、pairwise matrix、module score
    └── plotting.py      # 绘图：Figure 3A/3B/3F/4B 风格图，以及 magnitude plot
```

## 2. 与原 SPHERE R 包的对应关系

| R SPHERE | sat4sc Python |
|---|---|
| `spatial_adjust()` | `pysphere.spatial_adjust()` |
| `spatial_cordstat()` | `pysphere.spatial_cordstat()` |
| internal `spatial_binstat()` | `pysphere.spatial_binstat()` |
| `spatial_vector()` | `pysphere.spatial_vector()` |
| `spatial_vectorX()` | `pysphere.spatial_vector_x()` / `pysphere.spatial_vectorX()` |
| `spatial_vecProj()` | `pysphere.spatial_vec_proj()` |
| `spatial_vecMagnitude()` | `pysphere.spatial_vec_magnitude()` |
| `spatial_vecPlot()` | `plotting.plot_vector()` |
| `spatial_magPlot()` | `plotting.plot_magnitude_projected()` |
| `spatial_adjust(..., plot=TRUE)` | `plotting.plot_spatial_adjusted()` |
| `spatial_cordstat(..., plot_bin=TRUE)` | `plotting.plot_spatial_overlap()` |
| Figure 3A pairwise heatmap | `pysphere.pairwise_projected_scores()` + `plotting.plot_projected_heatmap()` |
| Figure 3F paired score | `plotting.plot_paired_projected_scores()` |
| Figure 4B distribution + pie | `plotting.plot_colocalization_distribution()` |

## 3. 安装

在仓库根目录：

```bash
pip install -e .
```

之后：

```python
from sat4sc import pysphere, plotting
```

依赖只有：`numpy`, `pandas`, `scipy`, `anndata`, `matplotlib`。

---

# 4. 关键算法

原 SPHERE 的核心逻辑是：固定空间对象 A，将空间对象 B 沿 8 个方向平移，在不同 step 下计算 Jaccard overlap 相对于原始状态的变化：

```text
ΔJaccard = Jaccard(displaced B, A) - Jaccard(original B, A)
```

每个 step 取 8 个方向中的：

```text
min ΔJaccard
max ΔJaccard
```

形成二维向量：

```text
(max ΔJaccard, min ΔJaccard)
```

最终 step 的 projected score 为：

```text
projected_score = (min ΔJaccard + max ΔJaccard) / 2
```

因此：

- projected score 越负：两个对象原始状态越倾向空间靠近/共定位；
- `projected_score < 0` 可作为论文 Figure 4B 使用的共定位方向判定；
- Q3（`max ΔJaccard < 0`）是更严格的“对象 B 原本位于对象 A 核心”的几何定义；
- magnitude 是向量轨迹从原点开始，跨所有 step 的累计欧氏路径长度。

本实现默认保留原 R 包的一个细节：每次 Jaccard 先四舍五入到 4 位小数，再计算 ΔJaccard（`round_jaccard=4`）。

---

# 5. Xenium 与原 Visium SPHERE 的关键差异

原 R 包依赖 Visium 的规则 spot lattice：坐标平移后可以通过完全一致的 `(row, col)` 坐标寻找重叠 spot。

你的 AnnData 使用：

```text
x_centroid
y_centroid
```

它们是连续坐标，因此不能直接做“坐标 + 2 后精确匹配”。本实现先把细胞投影到规则二维网格，再执行原 SPHERE 的 8 方向平移。

默认：

```python
grid_size = 20.0
steps = (2, 4, 6, 8, 10, 12)
```

如果 Xenium 坐标单位是 μm，则：

```text
step 2  = 40 μm
step 4  = 80 μm
...
step 12 = 240 μm
```

**建议 Xenium 正式分析至少对 `grid_size=10, 20, 25 μm` 做敏感性分析。**

如果输入本身就是 Visium array row/col 坐标，则使用：

```python
grid_size = 1
steps = (2, 4, 6, 8, 10, 12)
```

即可最大程度贴近原 R 包。

---

# 6. 从你的 AnnData 直接开始

例如：

```python
adata
# AnnData object with n_obs × n_vars = 506604 × 5001
# obs: ..., 'x_centroid', 'y_centroid', 'sample_name', 'sample_group', ...
# obsm: 'spatial', ...
```

`sat4sc` 默认优先使用：

```python
coord_cols=("x_centroid", "y_centroid")
```

若这两列不存在，才会自动尝试：

```python
adata.obsm["spatial"]
```

## 6.1 feature 可以是什么？

`target` 和 `features` 可以直接是：

1. `adata.obs` 中的数值列，例如：

```python
"hypoxia_score"
"inflammation_score"
"MES1_score"
```

2. `adata.var_names` 中的单基因，例如：

```python
"HIF1A"
"EPAS1"
```

如果想分析离散 cell type，请先转成 0/1 数值列，例如：

```python
adata.obs["BMDM"] = (adata.obs["sub_cell_type"] == "BMDM").astype(float)
```

---

# 7. 计算 signature score

论文使用 Seurat `AddModuleScore(seed=666, nbin=10)`。本包提供一个近似等价的 Python 实现：

```python
from sat4sc import pysphere

gene_sets = {
    "hypoxia": ["CA9", "VEGFA", "BNIP3", "NDRG1"],
    "MES1": ["..."],
    "MES2": ["..."],
}

pysphere.add_module_scores(
    adata,
    gene_sets,
    layer=None,   # 默认 adata.X；若你的 normalized expression 在别处请修改
    nbin=10,
    ctrl=100,
    seed=666,
    inplace=True,
)
```

结果直接写入：

```python
adata.obs["hypoxia"]
adata.obs["MES1"]
adata.obs["MES2"]
```

注意：算法逻辑与 Seurat AddModuleScore 一致，但 NumPy 与 R 的随机数实现不同，因此控制基因抽样不会保证逐细胞数值与 Seurat bit-for-bit 完全一致。如果你需要严格复现论文，最稳妥的方法仍然是先使用作者原基因集得到固定 signature score，然后在 Python 中运行 SPHERE。

---

# 8. 单个样本：对应 `spatial_vector()`

```python
from sat4sc import pysphere, plotting

rs = pysphere.spatial_vector(
    adata,
    sample="sample01",
    sample_key="sample_name",
    target="hypoxia",
    features=["MES1", "MES2", "AC", "BMDM", "NPC1", "OPC"],
    grid_size=20,
    steps=(2, 4, 6, 8, 10, 12),
)

rs.vectors
rs.projected_score
rs.vector_len
```

Figure 3B 风格：

```python
fig, ax = plotting.plot_vector(rs)
fig.savefig("Figure3B_like.pdf", bbox_inches="tight")
fig.savefig("Figure3B_like.png", dpi=600, bbox_inches="tight")
```

---

# 9. 多样本：对应 `spatial_vectorX()`

对于你这种所有样本都位于一个 `adata`、由 `sample_name` 区分的形式：

```python
rs = pysphere.spatial_vector_x(
    adata,
    target="hypoxia",
    features=["MES1", "MES2", "AC", "BMDM", "NPC1", "OPC"],
    sample_key="sample_name",
    grid_size=20,
    steps=(2, 4, 6, 8, 10, 12),
)
```

关键结果：

```python
# 跨样本平均后的 vector，用于论文 Figure 3B 一类队列图
rs.vectors

# 跨样本平均 vector 的 projected score
rs.projected_score

# 跨样本平均 vector 的 magnitude
rs.vector_len

# 每个 sample × feature 的原始 vector
rs.pool_raw

# 每个 sample × feature 的 projected score
# Figure 3F / Figure 4B 需要这个表
rs.sample_projected_score
```

---

# 10. Figure 3A：pairwise projected-score heatmap

假设已经在 `adata.obs` 中准备好了：

```python
features = [
    "AC", "hypoxia", "MES1", "MES2", "BMDM",
    "endothelial", "pericyte", "NPC2", "T_NK",
    "microglia", "oligodendrocyte", "OPC", "NPC1"
]
```

计算 pairwise projected score：

```python
pair_rs = pysphere.pairwise_projected_scores(
    adata,
    features=features,
    sample_key="sample_name",
    grid_size=20,
    final_step=12,
)
```

结果：

```python
pair_rs.matrix
pair_rs.sample_scores
```

绘图：

```python
fig, ax = plotting.plot_projected_heatmap(
    pair_rs,
    order=features,
    triangle="lower",
    title="Spatial associations",
)
fig.savefig("Figure3A_like.pdf", bbox_inches="tight")
fig.savefig("Figure3A_like.png", dpi=600, bbox_inches="tight")
```

`pairwise_projected_scores()` 只计算最终 step，因为 projected score 本身只使用最终 step，所以比对每一对 feature 都完整运行 6 个 step 更快。

---

# 11. Figure 3B：以 Hypoxia 为中心的 vector plot

```python
rs_hypoxia = pysphere.spatial_vector_x(
    adata,
    target="hypoxia",
    features=[
        "MES2", "MES1", "AC", "BMDM", "endothelial", "pericyte",
        "T_NK", "microglia", "NPC2", "NPC1", "OPC", "oligodendrocyte"
    ],
    sample_key="sample_name",
    grid_size=20,
)

fig, ax = plotting.plot_vector(
    rs_hypoxia,
    title=f"relation to {rs_hypoxia.target}",
)
```

---

# 12. Figure 3F：HIF1A vs EPAS1 的 paired projected score

计算：

```python
rs_tumor = pysphere.spatial_vector_x(
    adata,
    target="malignant",
    features=["HIF1A", "EPAS1", "HIF3A"],
    sample_key="sample_name",
    grid_size=20,
)
```

绘制 HIF1A vs EPAS1：

```python
fig, ax, stat = plotting.plot_paired_projected_scores(
    rs_tumor,
    feature_a="HIF1A",
    feature_b="EPAS1",
)

print(stat)
# {'n_pairs': ..., 'pvalue': ...}
```

这里使用每个样本的 paired projected score，并用 `scipy.stats.wilcoxon()` 做配对 Wilcoxon 检验。

---

# 13. Figure 4B：Hypoxia–Inflammation 共定位比例

计算：

```python
rs_hi = pysphere.spatial_vector_x(
    adata,
    target="hypoxia",
    features=["inflammation"],
    sample_key="sample_name",
    grid_size=20,
)
```

绘图：

```python
fig, ax, stat = plotting.plot_colocalization_distribution(
    rs_hi,
    feature="inflammation",
    threshold=0,
)

print(stat)
# n
# n_colocalized
# percent_colocalized
# pvalue
# threshold
```

判定规则：

```python
projected_score < 0
```

即定义为 co-localized / spatially close direction。

`plot_colocalization_distribution()` 中的一侧比例检验默认以：

```text
H0: p = 0.5
H1: p < 0.5
```

并使用 continuity correction。对于论文 Figure 4B 的 `12/44`，该计算给出约：

```text
p = 0.00209
```

与论文图中结果一致。

---

# 14. cutoff 如何定义

原 SPHERE 若没有传 `min.cutoffs`，使用两个 feature 的均值作为二值化阈值。

本实现对**每个样本 rasterized spatial bin 的 feature 值**使用均值：

```python
positive_bin = feature_value > mean(feature_value)
```

这是最接近原 R 包在 Visium spot 上的定义。

也可以手动指定：

```python
rs = pysphere.spatial_vector_x(
    adata,
    target="hypoxia",
    features=["inflammation"],
    cutoffs={
        "hypoxia": 0.12,
        "inflammation": 0.08,
    },
)
```

或者按分位数：

```python
rs = pysphere.spatial_vector_x(
    adata,
    target="hypoxia",
    features=["inflammation"],
    quantile_cutoffs=0.75,
)
```

也可不同 feature 使用不同分位数：

```python
quantile_cutoffs={
    "hypoxia": 0.75,
    "inflammation": 0.75,
}
```

---

# 15. 内存和速度优化

与逐 feature 复制 AnnData 或为每次平移建立完整坐标表相比，本实现做了几项优化：

1. **不复制整个 AnnData**：按 observation index 读取需要的坐标和单个 feature；
2. **每个样本只 rasterize 一次坐标**；
3. **使用 `numpy.bincount` 聚合 cell → spatial bin**；
4. **空间平移不创建新的完整矩阵**，而是用 NumPy slice 比较 source/destination；
5. **Jaccard 使用 boolean array + `count_nonzero`**；
6. Figure 3A 的 pairwise heatmap 只计算 final step，避免无用的完整轨迹计算；
7. 稀疏表达矩阵中的单基因提取不会把整个 `adata.X` densify；
8. `add_module_scores()` 对 sparse matrix 直接求列/行均值，不转换整张表达矩阵。

对于 `506,604 × 5,001` 的 Xenium 数据，建议始终以 `sample_name` 分样本运行，而不是把所有样本拼成一个空间平面。

---

# 16. 推荐你对 Xenium 做的敏感性分析

建议至少比较：

```python
grid_sizes = [10, 20, 25]
```

以及：

```python
steps = (2, 4, 6, 8, 10, 12)
```

对应实际距离：

```text
distance = step × grid_size
```

例如 grid_size=20 μm：

```text
40, 80, 120, 160, 200, 240 μm
```

如果不同 grid size 下：

- projected score 的正负方向稳定；
- feature 排序稳定；
- Figure 3A 的主要空间模块稳定；
- Figure 4B 的共定位样本比例稳定；

则结果会比只选择一个 bin size 更有说服力。

---

# 17. 与原 R SPHERE 的“严格一致”与“Xenium 扩展”需要区分

### 可以严格对应的部分

- 8 个平移方向；
- step 递增；
- Jaccard overlap；
- `ΔJaccard = shifted - original`；
- 每个 step 取 min/max ΔJaccard；
- projected score；
- vector magnitude；
- 跨样本先算单样本 vector，再在同 feature/step 上取平均；
- 默认 mean cutoff；
- Jaccard 四舍五入 4 位。

### Xenium 必须做的扩展

Xenium centroid 是连续坐标，不存在 Visium 的固定 spot lattice，因此必须先定义空间离散化尺度。本实现采用规则网格 rasterization。

因此在论文/方法中建议写成：

> We reimplemented the SPHERE displacement-based spatial relationship framework in Python and adapted it to single-cell-resolution spatial transcriptomics by rasterizing cell centroids onto regular spatial grids before eight-direction displacement analysis.

不要直接声称 Xenium 版本与原 Visium R 实现“完全相同”，因为 spatial unit 已从 Visium spot 改成了人工定义的空间 bin。

---

# 18. 最简完整示例

```python
from sat4sc import pysphere, plotting

# 假设 hypoxia / inflammation 已经在 adata.obs
rs = pysphere.spatial_vector_x(
    adata,
    target="hypoxia",
    features=["inflammation"],
    sample_key="sample_name",
    coord_cols=("x_centroid", "y_centroid"),
    grid_size=20,
    steps=(2, 4, 6, 8, 10, 12),
)

print(rs.projected_score)
print(rs.sample_projected_score.head())

fig, ax, stat = plotting.plot_colocalization_distribution(
    rs,
    feature="inflammation",
    threshold=0,
)

fig.savefig("Figure4B_like.png", dpi=600, bbox_inches="tight")
fig.savefig("Figure4B_like.pdf", bbox_inches="tight")
print(stat)
```

## Citation / attribution

`sat4sc.pysphere` is based on the algorithmic design of SPHERE:

Wu L, Wu G, Zhai Y, et al. *Dissection of spatial hypoxic and inflammatory ecosystem in glioblastoma*. Cancer Letters. 2026;657:218718.

Original R repository: `woolingxiang/SPHERE`.

See `NOTICE.md` for attribution and licensing notes.
