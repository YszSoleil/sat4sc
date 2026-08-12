# sat4sc

`sat4sc` 是面向单细胞/亚细胞分辨率空间转录组的 Python 工具包。当前版本重点实现 `pysphere`：将 SPHERE 的空间位移思想重写为 Python/AnnData 工作流，并提供 regular-grid SPHERE、spatial niche、cohort-wide cutoff、pairwise projected score 以及相关可视化。

```python
from sat4sc import pysphere, pysphere_plotting
```

当前版本：**v0.4.0**

> 推荐使用 `backend="grid"`。`backend="kdtree"` 在 v0.4.0 中保留为**实验性功能**，适合需要 cell-resolved radius-domain 行为的探索性分析，但不建议作为常规默认 backend。

---

## 1. v0.4.0 主要更新：SPHERE 可以直接使用 niche 身份

v0.3.x 中，SPHERE 的 binary spatial object 主要来自：

```text
continuous feature / gene expression
        ↓
cutoff
        ↓
feature-positive cells / grids
        ↓
Jaccard / shifted Jaccard / ΔJaccard
```

v0.4.0 新增 `niche_features=` 参数。对于你指定的一个或多个 feature，可以不再使用 `xxx_positive cell/grid`，而是直接把已有 spatial niche 的 `in_niche cell/grid` 当作该 feature 的 binary spatial object：

```text
                       ┌─ ordinary feature ─ cutoff ─ positive cell/grid ─┐
feature / gene set ────┤                                                   ├─ Jaccard / ΔJaccard / vector
                       └─ mapped niche ─────────────── in_niche cell/grid ─┘
```

也就是说，在后续 SPHERE 计算中：

```text
in_niche grid  ≡ 原来的 feature-positive grid
in_niche cell  ≡ 原来的 feature-positive cell
```

这里的“等价”仅指它们在后续 binary spatial calculation 中扮演相同角色；niche 本身仍然是由 `define_niche()` / `define_niches()` 根据 positive grids + spatial continuity 定义得到的。

### 1.1 新参数：`niche_features`

高层函数现在支持：

```python
niche_features={
    "feature_name_1": niche_result_1,
    "feature_name_2": niche_result_2,
}
```

其中 value 必须是 `pysphere.define_niche()` 或 `pysphere.define_niches()` 返回的 `NicheResult`。

支持该参数的主要接口：

```python
pysphere.spatial_vector()
pysphere.spatial_vector_x()
pysphere.pairwise_projected_scores()
pysphere.kdtree_domain_map()   # experimental KDTree utility
```

如果 `niche_features=None` 或完全不传，行为与 v0.3.x 一致。

---

## 2. 最常用的新用法

### 2.1 先定义多个 niche

例如已有两个 pathway score：

```python
niches = pysphere.define_niches(
    adata,
    features=["Hypoxia_score", "Inflammation_score"],
    sample_key="sample_name",
    grid_size=20,
    agg="mean",
    cutoff_method="balanced_global_mean",
    cutoff_n_repeats=100,
    cutoff_balance_round_to=1000,
    cutoff_random_state=666,
    min_connected_grids=3,
    connectivity=8,
)

hypoxia_niche = niches["Hypoxia_score"]
inflammation_niche = niches["Inflammation_score"]
```

### 2.2 target 使用 niche，feature 仍使用原来的 positive grid

例如：Hypoxia 使用已经定义好的 niche，而 LDHA 仍然根据自己的表达量和 cutoff 生成 positive grid：

```python
rs = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["LDHA"],
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
    cutoff_method="balanced_global_mean",
    niche_features={
        "Hypoxia_score": hypoxia_niche,
    },
)
```

此时：

```text
Hypoxia_score → hypoxia_niche.niche_grid → binary target
LDHA          → cutoff → LDHA-positive grid → binary feature
```

Hypoxia 不会在 SPHERE 内再次根据 score/cutoff 二值化。

### 2.3 target 和部分 features 都使用 niche

```python
rs = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score", "LDHA", "VEGFA"],
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
    cutoff_method="balanced_global_mean",
    niche_features={
        "Hypoxia_score": hypoxia_niche,
        "Inflammation_score": inflammation_niche,
    },
)
```

此时：

```text
Hypoxia_score       → in_niche grid
Inflammation_score  → in_niche grid
LDHA                → positive grid
VEGFA               → positive grid
```

它们随后进入同一套 Jaccard / shifted Jaccard / ΔJaccard 计算。

### 2.4 只有某一个 feature 使用 niche

也可以只替换 features 中的一个，而 target 继续按 positive grid：

```python
rs = pysphere.spatial_vector_x(
    adata,
    target="GlycoScore",
    features=["Hypoxia_score", "Inflammation_score"],
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
    cutoff_method="balanced_global_mean",
    niche_features={
        "Hypoxia_score": hypoxia_niche,
    },
)
```

---

## 3. pairwise projected-score matrix 中使用 niche

`pairwise_projected_scores()` 同样支持一个或多个 feature 使用 niche 身份。

```python
pair = pysphere.pairwise_projected_scores(
    adata,
    features=[
        "Hypoxia_score",
        "Inflammation_score",
        "GlycoScore",
        "LDHA",
    ],
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
    final_step=12,
    cutoff_method="balanced_global_mean",
    niche_features={
        "Hypoxia_score": hypoxia_niche,
        "Inflammation_score": inflammation_niche,
    },
)

pair.matrix
pair.sample_scores
```

对于所有涉及 `Hypoxia_score` 或 `Inflammation_score` 的 pair，都会直接使用对应 `niche_grid`；`GlycoScore` 和 `LDHA` 仍然使用 cutoff-derived positive grid。

因此同一个 pairwise matrix 中可以混合：

```text
niche vs niche
niche vs positive
positive vs niche
positive vs positive
```

---

## 4. 输出如何区分 positive 与 niche source

`spatial_vector()` / `spatial_vector_x()` 的 raw vector 表中新增：

```text
binary_source_target
binary_source_feature
n_binary_target
n_binary_feature
binary_fraction_target
binary_fraction_feature
```

`binary_source_*` 为：

```text
"positive"  # 普通 cutoff-derived positive cell/grid
"niche"     # 直接使用 in_niche cell/grid
```

为了兼容旧代码，原来的：

```text
n_positive_target
n_positive_feature
positive_fraction_target
positive_fraction_feature
```

仍然保留。当某个对象来源为 niche 时，这些旧列记录的是**实际参与 SPHERE 的 active binary units**，也就是 in-niche cell/grid 数量；新代码建议优先查看 `n_binary_*` 和 `binary_source_*`。

对于 niche source：

```text
cutoff_target / cutoff_feature = NaN
```

因为该对象已经由 niche identity 二值化，不会在 SPHERE 阶段再次计算 cutoff。

结果对象的 `settings` 中还会记录：

```python
rs.settings["binary_sources"]
rs.settings["niche_features"]
rs.settings["backend_status"]
```

---

## 5. niche 与 SPHERE grid 必须使用相同空间几何

当 `backend="grid"` 且使用 `niche_features` 时，sat4sc 会进行严格检查。

推荐：定义 niche 与后续 SPHERE 使用完全相同的：

```python
grid_size=20
min_cells_per_bin=1
```

并且使用同一个 AnnData / 同一批 cell coordinates。

例如：

```python
hypoxia_niche = pysphere.define_niche(
    adata,
    feature="Hypoxia_score",
    sample_key="sample_name",
    grid_size=20,
    min_cells_per_bin=1,
    cutoff_method="balanced_global_mean",
    min_connected_grids=3,
)

rs = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score"],
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
    min_cells_per_bin=1,
    niche_features={"Hypoxia_score": hypoxia_niche},
)
```

若 grid size、occupied-grid mask、坐标或样本不一致，会直接报错，而不是静默地把不同空间网格混在一起。

---

## 6. spatial niche 定义

niche 的基本流程仍然是：

```text
continuous feature score
        ↓
shared grid-level cutoff
        ↓
positive grids
        ↓
4/8-neighbor connected components
        ↓
component size >= min_connected_grids
        ↓
spatial niche
```

因此：

```text
positive grid ≠ niche
```

只有属于足够大、空间连续的 positive-grid component 的 grid 才成为 niche。

### 6.1 定义一个 niche

```python
hypoxia_niche = pysphere.define_niche(
    adata,
    feature="Hypoxia_score",
    sample_key="sample_name",
    grid_size=20,
    agg="mean",
    cutoff_method="balanced_global_mean",
    cutoff_n_repeats=100,
    cutoff_balance_round_to=1000,
    cutoff_random_state=666,
    min_connected_grids=3,
    connectivity=8,
    annotate_obs=True,
    obs_prefix="hypoxia",
)
```

当 `annotate_obs=True` 时，可以写入类似：

```python
adata.obs["hypoxia_positive_grid"]
adata.obs["hypoxia_niche"]
adata.obs["hypoxia_niche_id"]
```

cell-level niche membership 的定义为：该 cell 所在 grid 是否属于 retained niche component。

### 6.2 4-neighbor 与 8-neighbor

默认：

```python
connectivity=8
```

即上、下、左、右以及四个对角方向均可以连接。

若希望更严格：

```python
connectivity=4
```

只允许共享边界的 grid 相连。

### 6.3 niche 结果

```python
hypoxia_niche.summary
hypoxia_niche.sample_results["sample01"].positive_grid
hypoxia_niche.sample_results["sample01"].niche_grid
hypoxia_niche.sample_results["sample01"].cell_niche
```

`niche_grid` 和 `cell_niche` 正是 v0.4.0 可以直接送入后续 SPHERE binary calculation 的身份。

---

## 7. cohort-wide cutoff

普通 positive cell/grid 仍可使用 cohort-wide cutoff：

```python
cutoff_result = pysphere.calculate_cutoffs(
    adata,
    features=["Hypoxia_score", "Inflammation_score"],
    sample_key="sample_name",
    level="grid",
    method="balanced_global_mean",
)
```

常用方法：

```text
median_of_sample_medians
mean_of_sample_means
balanced_global_median
balanced_global_mean
```

在 v0.4.0 中，如果某个 feature 已写入 `niche_features`，SPHERE 阶段会跳过该 feature 的 cutoff 计算；只有未映射的普通 feature 才需要 SPHERE cutoff。

---

## 8. regular-grid SPHERE（推荐）

Xenium / CosMx / MERFISH 等数据通常使用连续 cell centroid 坐标。`backend="grid"` 会先把细胞投影到规则二维 grid，再执行 8 方向位移。

例如：

```python
grid_size = 20
steps = (2, 4, 6, 8, 10, 12)
```

若坐标单位是 μm，对应位移距离为：

```text
40, 80, 120, 160, 200, 240 μm
```

### 8.1 单个样本

```python
rs_grid = pysphere.spatial_vector(
    adata,
    sample="sample01",
    sample_key="sample_name",
    target="Hypoxia_score",
    features=["Inflammation_score", "GlycoScore", "LDHA"],
    backend="grid",
    grid_size=20,
    steps=(2, 4, 6, 8, 10, 12),
)
```

### 8.2 多样本

```python
rs_grid = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score", "GlycoScore", "LDHA"],
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
)
```

关键输出：

```python
rs_grid.vectors
rs_grid.projected_score
rs_grid.vector_len
rs_grid.pool_raw
rs_grid.sample_projected_score
rs_grid.sample_vector_len
```

---

## 9. SPHERE 的共同统计量

固定对象 A，将对象 B 沿 8 个方向移动，在每个 step 计算：

```text
ΔJaccard = Jaccard(A, shifted B) - Jaccard(A, original B)
```

每个 step 保留：

```text
min ΔJaccard
max ΔJaccard
```

最终 step：

```text
projected_score = (min ΔJaccard + max ΔJaccard) / 2
```

同时计算 vector-path magnitude。

无论 binary object 来源是：

```text
positive grid/cell
```

还是：

```text
in_niche grid/cell
```

进入 binary object 以后，后面的 Jaccard / ΔJaccard / projected score / magnitude 计算完全共用同一套逻辑。

---

## 10. KDTree backend：实验性功能

`backend="kdtree"` 保留连续 cell coordinates，并把 active cells（普通 positive cells 或 v0.4.0 的 in-niche cells）扩展为 radius-defined occupancy domain，再做空间位移和 Jaccard 计算。

简要示例：

```python
rs_kdtree = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score"],
    backend="kdtree",
    radius=15,
    niche_features={
        "Hypoxia_score": hypoxia_niche,
    },
)
```

在此模式下：

```text
普通 feature → positive cells → KDTree radius domain
niche feature → in_niche cells → KDTree radius domain
```

**注意：KDTree backend 在 v0.4.0 中属于实验性功能。对于正式、常规和需要稳定比较的分析，建议优先使用 `backend="grid"`。**

---

## 11. 输入 AnnData

典型输入：

```python
adata
# obs: ..., 'x_centroid', 'y_centroid', 'sample_name', ...
# layers: 'counts', ...
```

默认优先读取：

```python
coord_cols=("x_centroid", "y_centroid")
```

如果不存在，则可使用：

```python
adata.obsm["spatial"]
```

`target` / `features` 可以是：

```text
1. adata.obs 中的连续数值列，例如 pathway score
2. adata.var_names 中的单基因
3. adata.obs 中人为创建的 0/1 数值列
4. v0.4.0 中映射到 NicheResult 的 feature 名
```

当一个 feature 被 `niche_features` 映射后，即使后续 SPHERE 不需要它的原始连续数值，也仍建议 mapping key 与生成该 niche 时的 feature 名保持一致，便于结果追踪。

---

## 12. 安装

在仓库根目录：

```bash
pip install -e .
```

然后：

```python
from sat4sc import pysphere, pysphere_plotting
print(__import__("sat4sc").__version__)
# 0.4.0
```

依赖：

```text
numpy >= 1.24
pandas >= 2.0
scipy >= 1.10
anndata >= 0.10
matplotlib >= 3.7
```

---

## 13. 主要函数概览

### SPHERE / spatial calculation

```python
pysphere.spatial_adjust()
pysphere.spatial_binstat()
pysphere.spatial_cordstat()
pysphere.spatial_vector()
pysphere.spatial_vector_x()
pysphere.pairwise_projected_scores()
pysphere.spatial_vec_proj()
pysphere.spatial_vec_magnitude()
```

### niche / cutoff

```python
pysphere.calculate_cutoffs()
pysphere.positive_proportions()
pysphere.define_niche()
pysphere.define_niches()
```

### spatial maps / utilities

```python
pysphere.grid_feature_map()
pysphere.kdtree_domain_map()   # experimental
pysphere.add_module_scores()
```

### plotting

```python
from sat4sc import pysphere_plotting
```

包括 grid feature map、binary overlay、niche overlay、niche + positive feature、niche + continuous feature，以及 SPHERE vector / projected-score 等绘图接口。

---

## 14. v0.4.0 向后兼容性

旧代码：

```python
rs = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score"],
    backend="grid",
    grid_size=20,
)
```

无需修改，仍然使用：

```text
Hypoxia-positive grid vs Inflammation-positive grid
```

只有显式加入：

```python
niche_features={...}
```

时，相应 feature 才会切换到 `in_niche` 身份。

因此 v0.4.0 可以在同一分析中自由混合 legacy positive object 与 niche object，而不会改变未指定 feature 的旧行为。

---

## 15. 推荐 workflow

```python
from sat4sc import pysphere, pysphere_plotting

# 1. 为需要“空间连续 niche”身份的 pathway 定义 niche
niches = pysphere.define_niches(
    adata,
    features=["Hypoxia_score", "Inflammation_score"],
    sample_key="sample_name",
    grid_size=20,
    cutoff_method="balanced_global_mean",
    min_connected_grids=3,
    connectivity=8,
)

# 2. SPHERE：指定哪些对象使用 niche；其他对象继续按 positive grid
rs = pysphere.spatial_vector_x(
    adata,
    target="Hypoxia_score",
    features=["Inflammation_score", "GlycoScore", "LDHA"],
    sample_key="sample_name",
    backend="grid",
    grid_size=20,
    cutoff_method="balanced_global_mean",
    niche_features={
        "Hypoxia_score": niches["Hypoxia_score"],
        "Inflammation_score": niches["Inflammation_score"],
    },
)

# 3. 检查实际 binary source
print(rs.settings["binary_sources"])
print(rs.pool_raw[[
    "sample", "feature",
    "binary_source_target", "binary_source_feature",
    "n_binary_target", "n_binary_feature",
    "min_djaccard", "max_djaccard",
]].head())

# 4. downstream
print(rs.projected_score)
print(rs.sample_projected_score)
```

这也是 v0.4.0 最推荐的 niche-aware SPHERE 使用方式。

---

## License / attribution

See `LICENSE` and `NOTICE.md`.
