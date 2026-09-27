# sat4sc

`sat4sc` 是面向单细胞/亚细胞分辨率空间转录组的 Python 工具包。当前包含两个核心模块：`pysphere` 用于 SPHERE 空间位移/共定位分析；`state_niche` 用于基于 latent representation（推荐 `X_scVI`）构建 transcriptional-state spatial niche。

```python
from sat4sc import pysphere, pysphere_plotting
from sat4sc import state_niche, state_niche_plotting
```

当前版本：**v0.5.0**

> 推荐使用 `backend="grid"`。`backend="kdtree"` 在 v0.4.0 中保留为**实验性功能**，适合需要 cell-resolved radius-domain 行为的探索性分析，但不建议作为常规默认 backend。

---

## 1. v0.5.0 主要更新：Transcriptional-state niche

v0.5.0 新增：

```text
sat4sc/state_niche.py
sat4sc/state_niche_plotting.py
```

用于从单细胞空间转录组的 latent representation 中定义**局部转录状态空间生态位**。

核心思想：

```text
每个中心细胞
    ↓
在同一 sample 内寻找最近 K 个空间邻居
    ↓
提取邻居 latent representation（推荐 X_scVI）
    ↓
对每个 latent dimension 计算 mean + std
    ↓
得到邻域 transcriptional-state representation
    ↓
StandardScaler
    ↓
Gaussian Mixture Model (GMM)
    ↓
State Niche
```

默认：

```text
K = 25
include_center = False
aggregations = ("mean", "std")
```

因此默认定义强调的是**中心细胞周围的 extrinsic transcriptional microenvironment**，而不是把中心细胞自身的 `X_scVI` 再次用于聚类。

### 1.1 输入要求

典型 AnnData：

```python
adata
# obs:
#   sample_name
#   x_centroid
#   y_centroid
#   cell_type
#   ...
#
# obsm:
#   spatial
#   X_scVI
#   ...
```

`state_niche` 本身**不负责训练 scVI**；只要求指定的 latent representation 已经存在于 `adata.obsm`。因此也可以使用：

```text
X_scVI
X_scANVI
X_pca
其他用户自定义低维 representation
```

### 1.2 构建 25-NN state-niche features

```python
from sat4sc import state_niche as sn

feat = sn.build_features(
    adata,
    use_rep="X_scVI",
    sample_key="sample_name",
    spatial_key="spatial",
    n_neighbors=25,
    aggregations=("mean", "std"),
    include_center=False,
    scale=True,
)
```

空间近邻会**按 `sample_name` 分别构建**，不会跨切片寻找邻居。

如果 `adata.obsm["spatial"]` 不存在，会自动尝试：

```python
adata.obs[["x_centroid", "y_centroid"]]
```

默认写入：

```python
adata.obsm["X_state_niche_mean"]
adata.obsm["X_state_niche_std"]
adata.obsm["X_state_niche_raw"]
adata.obsm["X_state_niche"]

adata.obs["state_niche_knn25_rmax"]
adata.obs["state_niche_knn25_rmean"]
```

其中：

```text
X_state_niche_raw = [neighbor_mean, neighbor_std]
X_state_niche     = globally standardized X_state_niche_raw
```

如果 `X_scVI` 有 D 个 latent dimensions，则默认 state-niche representation 有 `2D` 维。

### 1.3 为什么是 mean + std

`mean` 描述局部邻域的平均 transcriptional state；`std` 描述邻域内部的 transcriptional heterogeneity。

因此两个 cell-type composition 相似的区域，例如：

```text
A: LDHA+ Tumor + glycolytic/hypoxic Myeloid
B: CCN3+ Tumor + GPR34+ relatively quiescent Myeloid
```

可能在传统 composition niche 中接近，但在 state-niche representation 中被分开。

### 1.4 ClusterAutoK：重复 GMM + Fowlkes–Mallows stability

```python
autok = sn.ClusterAutoK(
    n_clusters=(2, 20),
    max_runs=10,
    covariance_type="full",
    random_state=123,
)

autok.fit(
    adata,
    use_rep="X_state_niche",
    sample_key="sample_name",
    max_cells_per_sample=10000,
)
```

查看：

```python
autok.stability_
autok.peaks_
autok.best_k_
```

`ClusterAutoK` 是 **CellCharter-inspired stability scan**，但不是对 CellCharter 源码的直接复制。对每个候选 K 重复拟合 GMM，并使用 Fowlkes–Mallows index 比较不同随机初始化下的 clustering：

```text
within_stability
    = 同一个 K 多次 GMM run 之间的平均 pairwise FMI
```

同时报告：

```text
prev_similarity  # K 与 K-1 的聚类结构相似度
next_similarity  # K 与 K+1 的聚类结构相似度
```

`is_peak` 标记 within-K stability 的局部峰值；`best_k_` 是其中 stability 最高的候选值。如果没有严格局部峰值，则返回全局 stability 最大的 K。

**不建议只凭 `best_k_` 机械选择最终 K。** 最终 K 还应同时检查：

```text
spatial coherence
cross-sample recurrence
biological interpretability
over-fragmentation
```

### 1.5 约 60 万细胞时推荐 balanced GMM fitting

对于 Xenium 等大数据，建议先对每个样本最多抽取固定数量细胞用于 AutoK/GMM fitting，再将模型预测到全部细胞：

```python
autok.fit(
    adata,
    use_rep="X_state_niche",
    sample_key="sample_name",
    max_cells_per_sample=10000,
)
```

这样可以同时：

```text
1. 降低重复 GMM 的计算量
2. 避免细胞数特别多的样本主导 GMM
```

fit set 会记录到：

```python
adata.obs["state_niche_fit_set"]
```

### 1.6 将候选 K 预测到所有细胞

例如选 K=12：

```python
autok.predict(
    adata,
    k=12,
    use_rep="X_state_niche",
    key_added="state_niche",
)
```

得到：

```python
adata.obs["state_niche"]
adata.obs["state_niche_max_prob"]
adata.obs["state_niche_entropy"]
adata.obsm["X_state_niche_prob"]
```

其中 posterior probability / entropy 可用于识别 niche boundary 或 transition cells。

### 1.7 已经确定 K 时一步运行

如果已经决定使用固定 K：

```python
res = sn.state_niche(
    adata,
    use_rep="X_scVI",
    sample_key="sample_name",
    spatial_key="spatial",
    n_neighbors=25,
    aggregations=("mean", "std"),
    include_center=False,
    n_clusters=12,
    max_cells_per_sample=10000,
    key_added="state_niche",
)
```

或者已经构建好 `X_state_niche` 后：

```python
res = sn.fit(
    adata,
    n_clusters=12,
    use_rep="X_state_niche",
    key_added="state_niche",
    sample_key="sample_name",
    max_cells_per_sample=10000,
)
```

### 1.8 可视化

```python
from sat4sc import state_niche_plotting as snpl
```

稳定性曲线：

```python
ax = snpl.stability(autok)
```

空间分布：

```python
fig, axes = snpl.spatial(
    adata,
    cluster_key="state_niche",
    sample_key="sample_name",
)
```

每个 niche 的 cell-type composition：

```python
ax, table = snpl.composition(
    adata,
    cluster_key="state_niche",
    celltype_key="cell_type",
)
```

每个 sample 的 niche composition：

```python
ax, table = snpl.sample_composition(
    adata,
    cluster_key="state_niche",
    sample_key="sample_name",
)
```

### 1.9 推荐的完整工作流

```python
from sat4sc import state_niche as sn
from sat4sc import state_niche_plotting as snpl

# 1. 构建 neighborhood transcriptional-state representation
sn.build_features(
    adata,
    use_rep="X_scVI",
    sample_key="sample_name",
    spatial_key="spatial",
    n_neighbors=25,
    aggregations=("mean", "std"),
)

# 2. AutoK
model = sn.ClusterAutoK(
    n_clusters=(2, 20),
    max_runs=10,
    covariance_type="full",
    random_state=123,
)

model.fit(
    adata,
    use_rep="X_state_niche",
    sample_key="sample_name",
    max_cells_per_sample=10000,
)

print(model.stability_)
print("stability peaks:", model.peaks_)
print("suggested K:", model.best_k_)

# 3. 人工结合 stability + spatial pattern + biology 选择最终 K
selected_k = model.best_k_

# 4. 预测全数据
model.predict(
    adata,
    k=selected_k,
    key_added="state_niche",
)

# 5. 可视化
snpl.stability(model)
snpl.spatial(adata, cluster_key="state_niche")
snpl.composition(
    adata,
    cluster_key="state_niche",
    celltype_key="cell_type",
)
```

### 1.10 推荐 sensitivity analyses

正式分析建议至少比较：

```text
KNN = 10 / 25 / 50
mean-only vs mean+std
neighbor-only vs center-inclusive
balanced-fit vs all-cell fit
不同 stability peak 对应的 cluster solution
```

特别是 `25-NN` 在不同细胞密度区域对应的物理尺度可能不同，因此建议查看：

```python
adata.obs["state_niche_knn25_rmax"]
adata.obs["state_niche_knn25_rmean"]
```

确认不同样本和组织区域的空间尺度是否合理。

---

## 2. v0.4.0：SPHERE 可以直接使用 niche 身份

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

## 3. 最常用的新用法

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

## 4. pairwise projected-score matrix 中使用 niche

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

## 5. 输出如何区分 positive 与 niche source

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

## 6. niche 与 SPHERE grid 必须使用相同空间几何

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

## 7. spatial niche 定义

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

## 8. cohort-wide cutoff

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

## 9. regular-grid SPHERE（推荐）

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

## 10. SPHERE 的共同统计量

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

## 11. KDTree backend：实验性功能

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

## 12. 输入 AnnData

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

## 13. 安装

在仓库根目录：

```bash
pip install -e .
```

然后：

```python
from sat4sc import (
    pysphere, pysphere_plotting,
    state_niche, state_niche_plotting,
)
print(__import__("sat4sc").__version__)
# 0.5.0
```

依赖：

```text
numpy >= 1.24
pandas >= 2.0
scipy >= 1.10
anndata >= 0.10
matplotlib >= 3.7
scikit-learn >= 1.3
```

---

## 14. 主要函数概览

### transcriptional-state niche

```python
from sat4sc import state_niche

state_niche.build_spatial_neighbors()
state_niche.build_features()
state_niche.balanced_fit_indices()
state_niche.ClusterAutoK()
state_niche.fit()
state_niche.state_niche()
```

### state-niche plotting

```python
from sat4sc import state_niche_plotting

state_niche_plotting.stability()
state_niche_plotting.spatial()
state_niche_plotting.composition()
state_niche_plotting.sample_composition()
```

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

## 15. v0.4.0 向后兼容性

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

## 16. 推荐 workflow

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
