# sat4sc

`sat4sc` 是面向单细胞/亚细胞分辨率空间转录组的 Python 工具包。当前版本重点实现 `pysphere`：将 SPHERE 的空间位移思想重写为 Python/AnnData 工作流，并同时提供适用于 Xenium / CosMx / MERFISH 等连续细胞坐标的两种 backend。

```python
from sat4sc import pysphere, pysphere_plotting
```

当前版本：**v0.2.1**

---

## 1. 目录结构

```text
sat4sc/
├── pyproject.toml
├── README.md
├── NOTICE.md
└── sat4sc/
    ├── __init__.py
    ├── pysphere.py      # 计算：grid/KDTree SPHERE、projected score、magnitude、pairwise matrix、module score
    └── pysphere_plotting.py      # 绘图：Figure 3A/3B/3F/4B、grid spatial map、KDTree domain map
```

---

## 2. v0.2.1 更新

本版本在 v0.2.0 基础上更新绘图接口：

1. 原 `plotting.py` **改名为** `pysphere_plotting.py`；原有绘图函数逻辑保持不变。
2. 新增高层接口：

```python
pysphere_plotting.plot_binary_overlay()
```

它直接接受 `AnnData + feature names + sample`，并支持：

```python
space="cell"
space="grid"
```

用于绘制类似论文 Figure 3D / 3G 的两个 binary spatial objects 的空间叠加。
---

## 3. 三个 spatial plotting 函数的区别

| 函数 | 输入层级 | feature 数量 | 连续/二值 | 空间单位 | 主要用途 |
|---|---|---:|---|---|---|
| `plot_grid_feature_map()` | `GridFeatureMap` | 1 | 连续值为主 | grid | 查看 grid-SPHERE 实际看到的某个基因/通路 score；类似 Figure 3C 的逻辑 |
| `plot_spatial_overlap()` | 两个 `SpatialAdjusted` | 2 | 二值 | 原始 cell/spot | 低层接口；按 cutoff 将两个 feature 二值化后叠加；类似 Figure 3D/G |
| `plot_binary_overlay()` | `AnnData` + 两个 feature 名 | 2 | 二值 | `cell` 或 `grid` | 高层一步式接口；可直接选择真实细胞空间或 grid-SPHERE 空间 |

三者可以简单理解为：

```text
plot_grid_feature_map()
    一个 feature + 连续 score + grid

plot_spatial_overlap()
    两个 feature + binary + 原始 coordinates + 低层接口

plot_binary_overlay()
    两个 feature + binary + cell/grid 可选 + AnnData 高层接口
```

### 3.1 `plot_grid_feature_map()`：单个 feature 的连续 grid map

先生成与 grid-SPHERE 完全相同的 rasterized map：

```python
grid_map = pysphere.grid_feature_map(
    adata,
    feature="Hypoxia_score",
    sample="sample01",
    sample_key="sample_name",
    grid_size=20,
    agg="mean",
)
```

再绘图：

```python
fig, ax = pysphere_plotting.plot_grid_feature_map(
    grid_map,
    cmap="viridis",
    show_positive_outline=True,
)
```

这里颜色表示每个 grid 的连续 `Hypoxia_score`；`show_positive_outline=True` 只额外标出超过 `grid_map.cutoff` 的 high-feature domain。

### 3.2 `plot_spatial_overlap()`：低层 cell/spot binary overlay

先分别构建两个 `SpatialAdjusted`：

```python
hypoxia = pysphere.spatial_adjust(
    adata,
    feature="Hypoxia_score",
    sample="sample01",
)

mes1 = pysphere.spatial_adjust(
    adata,
    feature="MES1_score",
    sample="sample01",
)
```

再叠加：

```python
fig, ax = pysphere_plotting.plot_spatial_overlap(
    hypoxia,
    mes1,
    cutoffs=None,
    colors=("#F2D28B", "#707070"),
)
```

`cutoffs=None` 时，两个 feature 都使用各自有限值的均值作为 cutoff。feature 1 为实心点，feature 2 为空心圈。该函数保留为接近原 SPHERE `spatial_cordstat(..., plot_bin=TRUE)` 的低层接口。

### 3.3 `plot_binary_overlay(space="cell")`：一步式 cell-level overlay

```python
fig, ax, info = pysphere_plotting.plot_binary_overlay(
    adata,
    feature1="Hypoxia_score",
    feature2="MES1_score",
    sample="sample01",
    sample_key="sample_name",
    space="cell",
    cutoff1="mean",
    cutoff2="mean",
    colors=("#F2D28B", "#707070"),
)
```

绘图逻辑：

```text
所有细胞         浅灰背景
feature1+        实心
feature2+        空心轮廓
overlap           feature1 实心上叠 feature2 轮廓
```

`info` 会返回实际 cutoff 以及 positive/overlap 数量：

```python
info
# {
#   "space": "cell",
#   "cutoff1": ...,
#   "cutoff2": ...,
#   "n_feature1_positive": ...,
#   "n_feature2_positive": ...,
#   "n_overlap": ...,
# }
```

### 3.4 `plot_binary_overlay(space="grid")`：一步式 grid-level overlay

```python
fig, ax, info = pysphere_plotting.plot_binary_overlay(
    adata,
    feature1="Hypoxia_score",
    feature2="MES1_score",
    sample="sample01",
    sample_key="sample_name",
    space="grid",
    grid_size=20,
    agg="mean",
    cutoff1="mean",
    cutoff2="mean",
)
```

这里先把两个 feature rasterize 到同一套等面积 grid，再做 binary overlay。因此它展示的是 **grid-SPHERE 真正使用的空间单位**，适合检查进入 Jaccard / ΔJaccard 计算的 spatial objects。

### 3.5 `plot_binary_overlay()` 的 cutoff

`cutoff1` / `cutoff2` 支持：

```python
"mean"                 # 默认；SPHERE-style
"median"
"zero"
0.25                   # 显式数值
("quantile", 0.90)     # 90% quantile
```

例如：

```python
fig, ax, info = pysphere_plotting.plot_binary_overlay(
    adata,
    feature1="Hypoxia_score",
    feature2="HIF1A",
    sample="sample01",
    space="cell",
    cutoff1=("quantile", 0.8),
    cutoff2="zero",
)
```

---

## 4. v0.2.0 功能回顾

相较 v0.1.0，本版本新增两部分。

### 4.1 KDTree backend

直接保留连续细胞坐标，不先做规则网格。算法流程为：

```text
cell coordinates
      ↓
feature threshold
      ↓
positive cells
      ↓
cKDTree radius-defined occupancy on fixed cell anchors
      ↓
fix target A
      ↓
virtually shift feature B in 8 directions
      ↓
re-project shifted B onto fixed anchors by KDTree radius matching
      ↓
Jaccard
      ↓
ΔJaccard
      ↓
min/max ΔJaccard → projected score / magnitude
```

该 backend 是 **cell-resolved SPHERE extension**，不是把原 Visium lattice 的“精确坐标匹配”机械搬到 Xenium。

### 4.2 Grid 后的空间可视化

新增：

```python
pysphere.grid_feature_map()
pysphere_plotting.plot_grid_feature_map()
```

可以直接查看某个基因或通路分数在 **SPHERE 实际使用的 rasterized grid** 上是什么空间分布，并可叠加二值化阈值轮廓。

这非常适合在正式计算 SPHERE 前检查：

- `grid_size` 是否过粗或过细；
- 某个 pathway score 是否形成合理空间结构；
- mean / quantile cutoff 后形成的 positive domain 是否符合预期。

---

## 5. 安装

在仓库根目录：

```bash
pip install -e .
```

之后：

```python
from sat4sc import pysphere, pysphere_plotting
```

依赖：

- `numpy`
- `pandas`
- `scipy`
- `anndata`
- `matplotlib`

---

# 6. 输入 AnnData

例如：

```python
adata
# AnnData object with n_obs × n_vars = 506604 × 5001
# obs: ..., 'x_centroid', 'y_centroid', 'sample_name', 'sample_group', ...
# obsm: 'spatial', ...
# layers: 'counts'
```

默认优先读取：

```python
coord_cols=("x_centroid", "y_centroid")
```

如果这两列不存在，则尝试：

```python
adata.obsm["spatial"]
```

`target` / `features` 可以是：

1. `adata.obs` 中的数值列，例如：

```python
"Hypoxia_score"
"Inflammation_score"
"GlycoScore"
```

2. `adata.var_names` 中的单基因，例如：

```python
"HIF1A"
"EPAS1"
"LDHA"
```

3. 离散细胞类型转成的 0/1 列：

```python
adata.obs["BMDM"] = (adata.obs["sub_cell_type"] == "BMDM").astype(float)
```

---

# 7. SPHERE 的共同数学框架

两种 backend 最终都保留同一套 SPHERE 核心统计量。

固定对象 A，将对象 B 沿 8 个方向移动，在每个 step 计算：

```text
ΔJaccard = Jaccard(A, shifted B) - Jaccard(A, original B)
```

每个 step 取：

```text
min ΔJaccard
max ΔJaccard
```

最终 step 的 projected score：

```text
projected_score = (min ΔJaccard + max ΔJaccard) / 2
```

解释：

- projected score 越负：B 越倾向原本位于 A 内部/靠近 A；
- `projected_score < 0`：可作为论文 Figure 4B 一类的共定位方向判定；
- `max ΔJaccard < 0`：更严格的 Q3 / core-like 关系；
- magnitude：从原点经过各 step vector 的累计路径长度。

本包默认保留原实现的一个细节：Jaccard 可先四舍五入到 4 位再计算 ΔJaccard：

```python
round_jaccard=4
```

---

# 8. Backend 1：regular grid

## 8.1 原理

Xenium 坐标是连续值：

```text
x_centroid
 y_centroid
```

`backend="grid"` 首先把细胞投影到规则二维 grid，然后再做 8 方向位移。

例如：

```python
grid_size = 20
steps = (2, 4, 6, 8, 10, 12)
```

如果坐标单位是 μm，则对应真实位移：

```text
40, 80, 120, 160, 200, 240 μm
```

该 backend 的优势是每个空间单元面积相同，因此更接近原 Visium SPHERE 的空间面积逻辑。

---

## 8.2 单个样本

```python
rs_grid = pysphere.spatial_vector(
    adata,
    sample="sample01",
    sample_key="sample_name",
    target="Hypoxia_score",
    features=["Inflammation_score", "GlycoScore", "BMDM"],
    backend="grid",
    grid_size=20,
    steps=(2, 4, 6, 8, 10, 12),
)
```

结果：

```python
rs_grid.vectors
rs_grid.projected_score
rs_grid.vector_len
```

---

## 8.3 多样本

```python
rs_grid = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score", "GlycoScore", "BMDM"],
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
)
```

关键输出：

```python
# cohort-level average vector
rs_grid.vectors

# cohort-level projected score
rs_grid.projected_score

# 每个 sample × feature 的 projected score
rs_grid.sample_projected_score

# 每个 sample × feature 的 magnitude
rs_grid.sample_vector_len
```

---

# 9. Grid 后的基因/通路 spatial map

## 9.1 生成 rasterized feature map

例如查看某个样本的 Hypoxia score 在 `20 μm` grid 上的空间分布：

```python
grid_map = pysphere.grid_feature_map(
    adata,
    feature="Hypoxia_score",
    sample="sample01",
    sample_key="sample_name",
    grid_size=20,
    agg="mean",
)
```

返回：

```python
grid_map.grid       # 每个 grid 的 feature score
grid_map.occupied   # 哪些 grid 有足够细胞
grid_map.counts     # 每个 grid 的细胞数
grid_map.x_edges
grid_map.y_edges
grid_map.cutoff     # 默认 occupied grid 的 mean
```

---

## 9.2 绘图

```python
fig, ax = pysphere_plotting.plot_grid_feature_map(
    grid_map,
    cmap="viridis",
    show_positive_outline=True,
)

fig.savefig("Hypoxia_grid_spatial.png", dpi=600, bbox_inches="tight")
fig.savefig("Hypoxia_grid_spatial.pdf", bbox_inches="tight")
```

`show_positive_outline=True` 会按照当前 cutoff 画出 high-score domain 的轮廓。

---

## 9.3 使用自定义 cutoff

直接指定数值：

```python
grid_map = pysphere.grid_feature_map(
    adata,
    feature="Hypoxia_score",
    sample="sample01",
    grid_size=20,
    cutoff=0.12,
)
```

或使用分位数：

```python
grid_map = pysphere.grid_feature_map(
    adata,
    feature="Hypoxia_score",
    sample="sample01",
    grid_size=20,
    quantile_cutoff=0.75,
)
```

注意：这里的 cutoff 主要用于 spatial map 上的 binary outline。实际 SPHERE 计算中的 cutoff 仍由 `spatial_vector()` / `spatial_vector_x()` 的 `cutoffs` 或 `quantile_cutoffs` 参数控制。

---

# 10. Backend 2：KDTree cell-resolved SPHERE

## 10.1 核心定义

KDTree backend 不构建完整 cell-cell distance matrix。

而是：

```python
from scipy.spatial import cKDTree
```

利用空间索引查询 radius neighborhood。

假设 feature A 的阳性细胞坐标为：

```text
A-positive cells
```

对于每一个真实 cell anchor，若距离任一 A-positive cell 不超过 `radius`：

```text
A_domain = True
```

feature B 同理。

然后固定 A，将 B 的阳性坐标虚拟平移，再重新投影到同一批固定 cell anchors，计算 Jaccard。

因此 KDTree backend 的空间单位是：

> **radius-defined occupancy on real cell anchors**

而不是规则等面积 grid。

---

## 10.2 推荐参数含义

```python
radius = 15
steps = (25, 50, 75, 100, 125, 150)
```

若坐标单位为 μm：

- `radius=15`：一个阳性细胞影响附近 15 μm 内的 anchor；
- `step=50`：将 feature B 虚拟移动 50 μm。

两者不要混淆。

---

## 10.3 单样本 KDTree SPHERE

```python
rs_kd = pysphere.spatial_vector(
    adata,
    sample="sample01",
    sample_key="sample_name",
    target="Hypoxia_score",
    features=["Inflammation_score", "GlycoScore", "BMDM"],
    backend="kdtree",
    radius=15,
    steps=(25, 50, 75, 100, 125, 150),
    direction_mode="sphere",
    min_coverage=0.7,
    workers=-1,
)
```

也可以使用快捷函数：

```python
rs_kd = pysphere.spatial_vector_kdtree(
    adata,
    sample="sample01",
    target="Hypoxia_score",
    features=["Inflammation_score"],
    radius=15,
    steps=(25, 50, 75, 100, 125, 150),
)
```

---

## 10.4 多样本 KDTree SPHERE

```python
rs_kd = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score", "GlycoScore", "BMDM"],
    sample_key="sample_name",
    backend="kdtree",
    radius=15,
    steps=(25, 50, 75, 100, 125, 150),
    workers=-1,
)
```

快捷形式：

```python
rs_kd = pysphere.spatial_vector_x_kdtree(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score"],
    sample_key="sample_name",
    radius=15,
)
```

若 `steps` 省略，KDTree backend 默认：

```python
(25, 50, 75, 100, 125, 150)
```

---

## 10.5 `direction_mode`

### 与原 SPHERE 一致

```python
direction_mode="sphere"
```

对角方向：

```text
(step, step)
```

因此对角真实欧氏位移为：

```text
sqrt(2) × step
```

### 等欧氏距离模式

```python
direction_mode="euclidean"
```

对角方向自动使用：

```text
(step/√2, step/√2)
```

使 8 个方向的欧氏移动距离全部等于 `step`。

若目的是尽可能贴近原 SPHERE，建议：

```python
direction_mode="sphere"
```

---

## 10.6 组织边界 coverage

KDTree backend 会计算每个方向虚拟平移后，feature-positive cells 仍能匹配到真实组织 anchor 的比例：

```text
coverage_fraction
```

可以设置：

```python
min_coverage=0.7
```

低于该值的方向不参与该 step 的 min/max ΔJaccard。

每个 sample × feature × step 的结果中还会保存：

```python
min_direction_coverage
mean_direction_coverage
```

如果希望完全不过滤：

```python
min_coverage=0
```

---

# 11. KDTree domain spatial visualization

可以直接查看 KDTree backend 实际使用的 target / feature occupancy domain。

## 11.1 原始状态

```python
kd_map = pysphere.kdtree_domain_map(
    adata,
    target="Hypoxia_score",
    feature="Inflammation_score",
    sample="sample01",
    radius=15,
)

fig, ax = pysphere_plotting.plot_kdtree_domain_map(
    kd_map,
    point_size=3,
)
```

图中区分：

- target-only anchors；
- feature-only anchors；
- overlap anchors；
- background anchors。

---

## 11.2 查看虚拟平移后的 B domain

例如把 inflammation 向左移动 50 μm：

```python
kd_shift = pysphere.kdtree_domain_map(
    adata,
    target="Hypoxia_score",
    feature="Inflammation_score",
    sample="sample01",
    radius=15,
    shift=(-50, 0),
)

fig, ax = pysphere_plotting.plot_kdtree_domain_map(
    kd_shift,
    show_positive_cells=True,
)
```

标题会同时显示：

```text
radius
shift
sample
coverage_fraction
```

这可以非常直观地检查 SPHERE “移动 B 后与 A 更重叠还是更分离”的几何逻辑。

---

# 12. Figure 3B 风格 vector plot：两种 backend 通用

Grid：

```python
fig, ax = pysphere_plotting.plot_vector(rs_grid)
```

KDTree：

```python
fig, ax = pysphere_plotting.plot_vector(rs_kd)
```

KDTree 图中的 step colorbar 会标记为：

```text
step (coordinate units)
```

Grid 则为：

```text
step (grid cells)
```

---

# 13. Figure 3A：pairwise projected-score heatmap

## 13.1 Grid

```python
features = [
    "AC", "Hypoxia_score", "MES1", "MES2", "BMDM",
    "Endothelial", "Pericyte", "NPC2", "T_NK",
    "Microglia", "Oligodendrocyte", "OPC", "NPC1"
]

pair_grid = pysphere.pairwise_projected_scores(
    adata,
    features=features,
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
    final_step=12,
)

fig, ax = pysphere_plotting.plot_projected_heatmap(
    pair_grid,
    order=features,
    triangle="lower",
)
```

## 13.2 KDTree

```python
pair_kd = pysphere.pairwise_projected_scores(
    adata,
    features=features,
    sample_key="sample_name",
    backend="kdtree",
    radius=15,
    final_step=150,
    workers=-1,
)

fig, ax = pysphere_plotting.plot_projected_heatmap(
    pair_kd,
    order=features,
    triangle="lower",
)
```

KDTree pairwise 模式会缓存每个 feature 在最终 step 的 8 个 shifted domain，再对不同 target 重用，以减少重复 KDTree 查询。

---

# 14. Figure 3F：paired projected score

无论 grid 还是 KDTree，只要使用 cohort-level `spatial_vector_x()`，都可直接画：

```python
rs_hif = pysphere.spatial_vector_x(
    adata,
    target="malignant",
    features=["HIF1A", "EPAS1", "HIF3A"],
    sample_key="sample_name",
    backend="kdtree",
    radius=15,
    steps=(25, 50, 75, 100, 125, 150),
)

fig, ax, stat = pysphere_plotting.plot_paired_projected_scores(
    rs_hif,
    feature_a="HIF1A",
    feature_b="EPAS1",
)
```

统计使用：

```text
paired Wilcoxon
```

---

# 15. Figure 4B：Hypoxia–Inflammation 共定位

## Grid

```python
rs_hi_grid = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score"],
    backend="grid",
    grid_size=20,
)
```

## KDTree

```python
rs_hi_kd = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score"],
    backend="kdtree",
    radius=15,
    steps=(25, 50, 75, 100, 125, 150),
)
```

两者都使用相同绘图函数：

```python
fig, ax, stat = pysphere_plotting.plot_colocalization_distribution(
    rs_hi_kd,
    feature="Inflammation_score",
    threshold=0,
)
```

判定：

```python
projected_score < 0
```

输出：

```python
stat
# n
# n_colocalized
# percent_colocalized
# pvalue
# threshold
```

---

# 16. cutoff 如何定义

## Grid backend

默认在每个样本中，对 rasterized occupied grid 的 feature 值取均值：

```text
positive grid = grid score > mean(grid score)
```

## KDTree backend

默认在每个样本中，对原始 cell-level feature 值取均值：

```text
positive cell = cell score > mean(cell score)
```

之后再通过 `radius` 扩展为 occupancy domain。

---

## 16.1 手动 cutoff

```python
rs = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score"],
    backend="kdtree",
    cutoffs={
        "Hypoxia_score": 0.12,
        "Inflammation_score": 0.08,
    },
)
```

## 16.2 quantile cutoff

统一分位数：

```python
quantile_cutoffs=0.75
```

或不同 feature：

```python
quantile_cutoffs={
    "Hypoxia_score": 0.75,
    "Inflammation_score": 0.80,
}
```

---

# 17. 计算 pathway/module score

提供 Seurat `AddModuleScore` 风格实现：

```python
gene_sets = {
    "Hypoxia_score": ["CA9", "VEGFA", "BNIP3", "NDRG1"],
    "GlycoScore": ["HK1", "HK2", "PFKP", "ALDOA", "PGK1", "ENO2", "PKM", "LDHA"],
}

pysphere.add_module_scores(
    adata,
    gene_sets,
    layer=None,
    nbin=10,
    ctrl=100,
    seed=666,
    inplace=True,
)
```

由于 NumPy 和 R 的随机数实现不同，不保证与 Seurat bit-for-bit 完全一致。

---

# 18. 性能与内存

## Grid backend

主要优化：

1. 不复制完整 AnnData；
2. 每个样本只构建一次 grid；
3. 使用 `numpy.bincount` 做 cell → grid 聚合；
4. 位移通过 array slice 实现；
5. 不生成 cell-cell distance matrix。

## KDTree backend

主要优化：

1. 使用 `scipy.spatial.cKDTree`，不创建 N×N distance matrix；
2. 每个 feature 只建立一个 positive-cell KDTree；
3. shifted B 通过查询 `anchor_coords - shift` 实现，不需要每次重建平移后的 KDTree；
4. pairwise 模式缓存最终 step 的 shifted domains；
5. `workers=-1` 可使用 SciPy KDTree 的多线程查询，但在 Slurm 环境建议与申请 CPU 数匹配，避免过度占用。

对于 50 万级细胞，始终建议按 `sample_name` 分样本计算。

---

# 19. Grid vs KDTree：推荐怎么用

### 推荐主分析：Grid

适合：

- Hypoxia niche vs Inflammation niche；
- GlycoHigh niche；
- 大尺度肿瘤生态位；
- 希望不同空间位置按相同面积权重统计。

### 推荐敏感性分析/单细胞扩展：KDTree

适合：

- 某细胞类型 vs niche；
- Myeloid vs Tumor state；
- Endothelial vs Pericyte；
- gene-positive cells vs pathway niche；
- 希望尽量保留原始 cell centroid。

最稳妥的发表策略是：

```text
Grid SPHERE 作为主结果
        +
KDTree SPHERE 作为 cell-resolved sensitivity analysis
```

比较：

- projected score 相关性；
- Q3/Q4 分类一致率；
- feature 排序一致性；
- Figure 4B 共定位样本比例稳定性。

---

# 20. 推荐 Xenium 敏感性分析

Grid：

```python
grid_sizes = [10, 20, 25]
```

KDTree：

```python
radii = [10, 15, 20]
```

位移距离：

```python
steps = (25, 50, 75, 100, 125, 150)
```

建议检查不同参数下：

- projected score 正负方向；
- magnitude；
- pairwise heatmap 模块；
- co-localization 分类；
- 组织边界 coverage。

---

# 21. 最简完整示例

```python
from sat4sc import pysphere, pysphere_plotting

# 1. 先看 grid 后的 Hypoxia 空间分布
gm = pysphere.grid_feature_map(
    adata,
    feature="Hypoxia_score",
    sample="sample01",
    grid_size=20,
)

fig, ax = pysphere_plotting.plot_grid_feature_map(
    gm,
    show_positive_outline=True,
)

# 2. Grid SPHERE
rs_grid = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score"],
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
)

# 3. KDTree SPHERE
rs_kd = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score"],
    sample_key="sample_name",
    backend="kdtree",
    radius=15,
    steps=(25, 50, 75, 100, 125, 150),
    min_coverage=0.7,
    workers=-1,
)

# 4. KDTree occupancy 可视化
km = pysphere.kdtree_domain_map(
    adata,
    target="Hypoxia_score",
    feature="Inflammation_score",
    sample="sample01",
    radius=15,
)

fig, ax = pysphere_plotting.plot_kdtree_domain_map(km)

# 5. Figure 3B-like
fig, ax = pysphere_plotting.plot_vector(rs_kd)

# 6. Figure 4B-like
fig, ax, stat = pysphere_plotting.plot_colocalization_distribution(
    rs_kd,
    feature="Inflammation_score",
    threshold=0,
)

print(stat)
```

---

## Citation / attribution

`sat4sc.pysphere` is based on the algorithmic design of SPHERE:

Wu L, Wu G, Zhai Y, et al. *Dissection of spatial hypoxic and inflammatory ecosystem in glioblastoma*. Cancer Letters. 2026;657:218718.

Original R repository: `woolingxiang/SPHERE`.

The KDTree backend is a sat4sc extension for continuous single-cell spatial coordinates and should not be described as mathematically identical to the original Visium-lattice implementation.

See `NOTICE.md` for attribution and licensing notes.
