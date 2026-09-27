# 基于邻域 scVI 状态聚合的 Transcriptional-state niche 技术方案

## 1. 项目目标

本方案拟在单细胞分辨率空间转录组数据中定义一种 **Transcriptional-state niche（转录状态空间生态位）**。

与传统基于邻域细胞类型组成的 cellular neighborhood 不同，本方法不直接使用邻域内各细胞类型的比例作为特征，而是利用已经训练好的 **scVI latent representation** 表征每个细胞的转录状态，再将中心细胞周围最近 25 个细胞的 scVI latent 信息进行聚合，得到该中心细胞所处局部微环境的 transcriptional-state representation。

核心定义：

\[
N_i^{state}=[\mu_i,\sigma_i]
\]

其中，对中心细胞 \(i\) 的 25 个空间近邻 \(N_{25}(i)\)：

\[
\mu_{i,d}=\frac{1}{25}\sum_{j\in N_{25}(i)} Z_{j,d}
\]

\[
\sigma_{i,d}=\sqrt{\frac{1}{25}\sum_{j\in N_{25}(i)}(Z_{j,d}-\mu_{i,d})^2}
\]

其中 \(Z=X_{scVI}\)，\(d\) 表示 scVI latent dimension。

若 `X_scVI` 的维度为 \(D\)，则每个细胞最终获得 \(2D\) 维 niche feature：

```text
[mean_scVI_dim1 ... mean_scVI_dimD,
 std_scVI_dim1  ... std_scVI_dimD]
```

随后：

```text
邻域 transcriptional-state features
        ↓
全局标准化
        ↓
GMM clustering
        ↓
不同 cluster number 的 CellCharter-like stability analysis
        ↓
选择稳定的 K
        ↓
得到 State Niche 0 ... State Niche K-1
```

本方案的重点不是给中心细胞重新进行 transcriptomic clustering，而是回答：

> **这个细胞所处的局部空间环境整体处于什么 transcriptional state？**

因此，主分析中**不将中心细胞自身的 `X_scVI` 加入 niche feature**，只使用其周围细胞的信息。

---

## 2. 输入数据

预期输入为 AnnData，例如：

```python
AnnData object with n_obs × n_vars = 621806 × 5001
    obs:
        'x_centroid',
        'y_centroid',
        'sample_name',
        'sample_group',
        ...
        'cell_type',
        'sub_cell_type',
        'sub_celltype'

    obsm:
        'spatial',
        'X_scVI',
        'X_umap',
        'X_scANVI'

    layers:
        'counts'
```

### 2.1 必需字段

至少要求：

```text
obs:
    sample_name
    x_centroid
    y_centroid

obsm:
    X_scVI
    spatial
```

建议同时保留：

```text
sample_group
patient_id（如有）
cell_type
sub_cell_type
sub_celltype
```

如果 `obsm['spatial']` 需要重新构建，可统一：

```python
adata.obsm["spatial"] = (
    adata.obs[["x_centroid", "y_centroid"]]
    .to_numpy()
    .astype("float32")
)
```

---

## 3. 总体分析流程

```text
                         AnnData
                            │
                            ▼
                ┌─────────────────────┐
                │  Input QC / sanity │
                └─────────────────────┘
                            │
                            ▼
                     X_scVI latent
                            │
                            ▼
             按 sample_name 分别建立空间 kNN
                            │
                            ▼
                 每个细胞取最近 25 个邻居
                      （不包含自身）
                            │
                            ▼
               邻居 X_scVI 聚合：mean + std
                            │
                            ▼
                  X_state_niche_raw
                            │
                            ▼
                    Global scaling
                            │
                            ▼
                    X_state_niche
                            │
                            ▼
        ┌─────────────────────────────────────┐
        │ CellCharter-like AutoK stability  │
        │       GMM + repeated fitting       │
        └─────────────────────────────────────┘
                            │
                            ▼
                 Stability vs cluster number
                            │
                            ▼
                 选择候选 K / 最终 K
                            │
                            ▼
                     Final GMM
                            │
                            ▼
                State Niche 0 ... K-1
                            │
          ┌─────────────────┼───────────────────┐
          ▼                 ▼                   ▼
   cell composition   pathway/state       spatial pattern
          │                 │                   │
          └─────────────────┼───────────────────┘
                            ▼
                  biological annotation
                            │
                            ▼
                Grade1_A vs Grade1_NA
                  niche enrichment
```

---

## 4. Step 0：输入数据 QC

在任何空间邻域计算之前，先进行数据结构检查。

记录：

```python
adata.shape
adata.obs["sample_name"].value_counts()
adata.obs["sample_group"].value_counts()
adata.obsm["X_scVI"].shape
adata.obsm["spatial"].shape
```

检查：

- `X_scVI.shape[0] == adata.n_obs`
- `spatial.shape[0] == adata.n_obs`
- `X_scVI` 不包含 NaN / Inf
- `spatial` 不包含 NaN / Inf
- 每个 sample 至少有 >25 个细胞
- `sample_name` 不缺失

### 4.1 scVI latent QC

计算每一个 latent dimension 的：

```text
mean
std
min
max
NaN count
Inf count
```

输出建议：

```text
00_QC/scVI_latent_summary.csv
00_QC/scVI_latent_distribution.pdf
```

注意：

> 不建议直接根据 scVI latent dimension 解释具体生物学意义。

scVI latent 的主要作用是提供低维、去噪、尽可能 batch-corrected 的 transcriptional representation。真正的 niche 生物学解释应回到原始基因表达、pathway / gene-set score、cell type / subtype composition 和 spatial localization。

---

## 5. Step 1：空间邻域构建

### 5.1 主分析定义

对每个细胞：

> 在**同一个 `sample_name` 内部**寻找最近的 25 个其他细胞。

不能跨 sample 建图。

定义：

\[
N_{25}(i)=25\text{ nearest spatial neighbors of cell }i
\]

距离：

\[
d_{ij}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}
\]

主分析使用：

```text
K_spatial = 25
exclude_self = True
```

### 5.2 为什么必须按 sample 分开计算

不同组织切片之间的 x/y 坐标系彼此独立，因此绝不能对全部样本统一寻找空间近邻。

正确做法：

```text
for sample in sample_name:
    subset sample
    construct KDTree
    query 25 neighbors
```

推荐使用：

```python
scipy.spatial.cKDTree
```

而不是构建全局 dense distance matrix。

### 5.3 推荐实现

```python
from scipy.spatial import cKDTree

coords = adata.obsm["spatial"][sample_mask]

tree = cKDTree(coords)
dist, idx = tree.query(coords, k=26)

neighbor_idx = idx[:, 1:26]
neighbor_dist = dist[:, 1:26]
```

理论上第一个 neighbor 为自身，因此查询 26 个，删除自身后保留 25 个。

建议保存：

```text
01_neighbors/
    knn25_indices/
    knn25_distances/
```

或者按 sample 输出 `.npz`。

---

## 6. 邻域尺度 QC

虽然主分析固定为 25 nearest neighbors，但必须检查不同组织区域中“25 个邻居”对应的物理半径是否差异过大。

对每个中心细胞计算：

```text
distance_to_25th_neighbor
median_neighbor_distance
mean_neighbor_distance
```

保存：

```python
adata.obs["state_niche_knn25_rmax"]
adata.obs["state_niche_knn25_rmean"]
```

建议绘制：

```text
01_neighbors/
    knn25_radius_by_sample_violin.pdf
    knn25_radius_histogram_by_sample.pdf
```

如果少数区域的第 25 个邻居距离异常大，则在 sensitivity analysis 中增加：

```text
25-NN + maximum radius
```

例如最多取 25 个邻居，同时要求距离 ≤50 μm。但主分析仍保持纯 25-NN，以保证定义简单且所有细胞 feature dimension 一致。

---

## 7. Step 2：构建 Transcriptional-state niche feature

设：

```python
Z = adata.obsm["X_scVI"]
```

假设：

```text
Z.shape = (621806, D)
```

对每个中心细胞的 25 个邻居：

```python
Z_neighbors.shape = (25, D)

neighbor_mean = Z_neighbors.mean(axis=0)
neighbor_std  = Z_neighbors.std(axis=0, ddof=0)

state_feature = np.concatenate(
    [neighbor_mean, neighbor_std]
)
```

最终 feature dimension：

```text
2 × D
```

### 7.1 为什么同时使用 mean + std

`mean` 描述邻域总体 transcriptional state；`std` 描述邻域内部 transcriptional heterogeneity。

例如两个区域都可能具有相似的平均状态，但一个区域内部由两种明显不同的细胞状态混合而成，另一个区域内部较一致。仅使用 mean 可能将两者合并，而 std 可以保留这一差异。

### 7.2 不包含中心细胞自身

主分析定义为：

```text
center cell
    ↓
25 neighboring cells
    ↓
mean + std
```

而不是：

```text
center X_scVI
+
neighbor mean
+
neighbor std
```

也就是：

\[
N_i^{state}=[\mu_{neighbors},\sigma_{neighbors}]
\]

而不是：

\[
[z_i,\mu_{neighbors},\sigma_{neighbors}]
\]

这样可以减少 clustering 被中心细胞自身 identity 主导的风险，使结果更接近 **extrinsic transcriptional microenvironment**。

---

## 8. 推荐的 AnnData 存储字段

```python
adata.obsm["X_state_niche_mean"]
adata.obsm["X_state_niche_std"]
adata.obsm["X_state_niche_raw"]
adata.obsm["X_state_niche"]
```

其中：

```text
X_state_niche_mean : n_cells × D
X_state_niche_std  : n_cells × D
X_state_niche_raw  : n_cells × 2D
X_state_niche      : n_cells × 2D，标准化后
```

---

## 9. Step 3：feature scaling

不能直接把 neighbor mean 和 neighbor std 拼起来后直接 GMM，因为两类特征的尺度可能不同。

推荐：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_state_niche = scaler.fit_transform(X_state_niche_raw)
```

即：

\[
x'=\frac{x-\mu}{\sigma}
\]

### 9.1 使用 global scaling，而不是 sample-wise scaling

推荐将所有 sample 合并后做一个全局 `StandardScaler`。

不建议每个 sample 独立 z-score，因为这样会强制不同 sample 的 feature distribution 更相似，可能人为消除真实存在的组间差异。

前提是：

> `X_scVI` 本身已经使用合理的 batch correction / `sample_name` batch 建模。

---

## 10. Step 4：GMM clustering

对：

```python
adata.obsm["X_state_niche"]
```

使用 Gaussian Mixture Model（GMM）进行 clustering。

相比 K-means，GMM 可以允许不同 cluster 具有不同 covariance、不同 variance、非球形结构并存在一定程度重叠，更适合连续 transcriptional-state manifold。

---

## 11. 直接借用 CellCharter 的 GMM / AutoK 框架

推荐不自行重新实现 stability framework。

我们只自定义：

```text
X_state_niche
```

之后把它直接作为 CellCharter 的 `use_rep` 输入。

概念代码：

```python
import cellcharter as cc

autok = cc.tl.ClusterAutoK(
    n_clusters=(2, 20),
    max_runs=10,
    convergence_tol=0.001
)

autok.fit(
    adata_fit,
    use_rep="X_state_niche"
)
```

这样做的好处是：

> 邻域 feature 的构建是定制的，但 GMM clustering 和 cluster-number stability selection 尽可能沿用 CellCharter 原有实现。

---

## 12. CellCharter-like cluster-number stability

CellCharter 的 `ClusterAutoK` 会针对一系列 cluster number：

```text
K = 2, 3, 4, ... 20
```

重复进行 GMM clustering。

推荐参数：

```python
n_clusters = (2, 20)
max_runs = 10
convergence_tol = 0.001
```

CellCharter 使用 **Fowlkes–Mallows similarity** 来衡量聚类解之间的相似性，并根据相邻 K 的稳定性结构识别 stability peaks。

因此不要把它简单理解成：

```text
同一个 K 重复 10 次
→ 算 ARI
→ 最大者就是最佳 K
```

更准确的理解是：

> 对多个 K 反复拟合 GMM，并考察聚类结构在相邻 K 下的持久性；稳定性曲线中的局部峰值是值得进一步检查的候选 K。

### 12.1 Fowlkes–Mallows index

对两个 clustering partition：

\[
U, V
\]

Fowlkes–Mallows index：

\[
FM=\frac{TP}{\sqrt{(TP+FP)(TP+FN)}}
\]

其中：

- TP：两个 clustering 中均被分到同一 cluster 的 cell pair
- FP：在一个 clustering 中同 cluster、另一个中不同 cluster
- FN：反之

范围：

\[
0\le FM\le 1
\]

越接近 1，两个 cluster solution 越相似。

---

## 13. AutoK 输出

至少保存：

```text
03_autok/
    state_niche_stability.csv
    state_niche_stability.pdf
    autok_object/
```

建议 `stability.csv` 包含：

```text
K
stability
is_peak
rank
```

并记录：

```python
autok.best_k
autok.peaks
```

---

## 14. K 的选择原则

不建议机械地直接取 stability 最高的 K。

优先检查：

```text
global maximum
+
其他明显 local peaks
```

例如 K=7、10、13 都是 stability peak，则进一步比较：

1. **Stability**：是否为明显峰值。
2. **Spatial coherence**：是否形成连续/重复出现的空间结构，而不是 salt-and-pepper。
3. **Cross-sample reproducibility**：是否出现在多个独立患者中。
4. **Biological interpretability**：是否可通过 cell composition、gene expression、pathway activity 解释。
5. **Over-fragmentation**：较高 K 是否只是把一个 coherent niche 拆成多个高度相似 cluster。

推荐搜索范围：

```text
主分析 K = 2–20
```

如果 K=18–20 仍持续出现高稳定性，可再扩展至 2–30。

---

## 15. 62 万细胞下的推荐 fitting 策略

AutoK 重复 GMM 是本流程最耗时的步骤。

推荐使用 **balanced fitting subset**：

```text
每个 sample 随机最多抽取 10,000 cells
```

用于：

```text
AutoK
+
GMM fitting
```

随后使用拟合好的模型对全部 621,806 cells 进行预测。

流程：

```text
all cells
   │
   ├── 对全部细胞构建 niche features
   │
   ▼
balanced cells per sample
   │
   ▼
AutoK / GMM fitting
   │
   ▼
trained model
   │
   ▼
predict all 621,806 cells
```

这样不仅加速计算，也可避免细胞数量特别多的 sample 主导 GMM niche definition。

推荐：

```python
N_FIT_PER_SAMPLE = 10000
RANDOM_SEED = 123
```

并增加：

```python
adata.obs["state_niche_fit_set"]
```

作为 Boolean 标记。

### 15.1 主分析与 sensitivity

主分析：

```text
balanced fit → predict all cells
```

敏感性分析：

```text
full-data GMM
```

比较：

```text
ARI
NMI
cluster proportion
spatial pattern
cell-type composition
```

---

## 16. GPU

如果：

```python
torch.cuda.is_available() == True
```

推荐使用 CellCharter / TorchGMM GPU backend。

概念参数：

```python
autok = cc.tl.ClusterAutoK(
    n_clusters=(2, 20),
    max_runs=10,
    convergence_tol=0.001,
    model_params={
        "covariance_type": "full",
        "trainer_params": {
            "accelerator": "gpu",
            "devices": 1
        }
    }
)
```

具体字段应以当前安装的 CellCharter 版本 API 为准。

---

## 17. Final GMM

假设最终选择 K=12，仅作为示例：

```python
adata.obs["state_niche_k12"] = autok.predict(
    adata,
    use_rep="X_state_niche",
    k=12
)
```

注意：

> 不能因为此前其他 CellCharter 分析用过 k=12，就预设本方法也必须为 12。

当前方法的 neighborhood representation 已经不同，必须重新进行 stability selection。

---

## 18. GMM assignment uncertainty

GMM 的一个优势是可获得 posterior probability：

\[
P(cluster=k|x_i)
\]

如果当前 CellCharter / TorchGMM 接口允许，建议保存：

```python
adata.obsm["state_niche_gmm_prob"]
adata.obs["state_niche_max_prob"]
adata.obs["state_niche_entropy"]
```

解释：

```text
max probability 高
→ cluster assignment 较明确

max probability 低
→ 可能处于 niche boundary / transition state
```

这对于空间生态位尤其有意义，因为实际 niche 很可能存在连续过渡。

---

## 19. Step 5：空间可视化

每个 sample 绘制：

```text
x_centroid
vs
y_centroid
```

颜色：

```text
state_niche_final
```

建议输出：

```text
04_spatial/
    all_samples_state_niche_grid.pdf
    per_sample/
        Case01_state_niche.pdf
        Case02_state_niche.pdf
        ...
```

### 19.1 检查 niche 是否具有真实空间结构

如果一个 niche 呈现明显的随机 salt-and-pepper pattern，则需要怀疑它主要反映 latent noise，而不是 spatial microenvironment。

建议计算：

1. **Same-niche neighbor fraction**

\[
f_i=\frac{\#\{neighbors\ with\ same\ niche\}}{25}
\]

2. **Moran's I**：对每个 niche 的 0/1 indicator 计算空间自相关。
3. **Connected component size**：检查空间 patch size。
4. **Sample recurrence**：检查 niche 是否在多个独立 sample 中重复出现。

---

## 20. Step 6：Niche biological characterization

State Niche 的 cluster label 本身没有直接生物学意义，必须进行二次 annotation。

建议从四个层次解释。

### 20.1 Layer 1：Cell-type composition

对每个 niche 计算：

```text
cell_type
sub_cell_type
sub_celltype
```

的：

```text
fraction
observed / expected
odds ratio
```

输出：

```text
05_characterization/
    niche_celltype_fraction.xlsx
    niche_subcelltype_fraction.xlsx
    niche_celltype_heatmap.pdf
```

### 20.2 Layer 2：Gene-expression programs

不要解释 scVI latent dimension 本身，而应回到原始/标准化表达。

重点候选：

#### Tumor

```text
LDHA
CA9
VEGFA
NDRG1
BNIP3
SSTR2
CCN3
```

#### Myeloid

```text
MRC2
SGK1
GPR34
SPP1
C1QC
MRC1
CD163
```

#### Endothelial / Pericyte

```text
ESM1
KDR
PECAM1
PLVAP
RGS5
PDGFRB
CSPG4
```

### 20.3 Layer 3：Pathway scores

重点比较：

```text
Glycolysis
Hypoxia
Inflammatory response
TNF/NF-κB
IFN response
Angiogenesis
EMT
ECM remodeling
Oxidative phosphorylation
```

可使用已有 UCell / AUCell / GSVA-like scoring。

### 20.4 Layer 4：Spatial context

例如：

```text
distance to vessel
distance to Endothelial
distance to Pericyte
Tumor–Myeloid proximity
Tumor–Endothelial proximity
```

为每个 niche 量化 vascular、immune、tumor-core/interface 等空间偏好。

---

## 21. Hypoxic/glycolytic Tumor–Myeloid niche 的识别逻辑

真正值得寻找的不是“哪个 cluster 编号等于 hypoxia niche”，而是某个 State Niche 是否同时满足：

```text
LDHA / glycolysis ↑
hypoxia score ↑
Tumor enriched
MRC2+/SGK1+ Myeloid enriched
Tumor–Myeloid proximity ↑
vascular / hypoxia spatial relationship altered
Grade1_A enrichment
```

只有多层证据一致时，才将其注释为类似：

> **Hypoxic/Glycolytic Tumor–Myeloid Niche**

---

## 22. Step 7：Grade1_A vs Grade1_NA

不能直接把 621,806 cells 当成独立统计学重复做组间检验。

主要统计单位应为：

> **sample**

先计算每个 sample 内：

\[
P_{s,k}=\frac{N_{cells\ in\ niche\ k}}{N_{all\ cells\ in\ sample}}
\]

得到：

```text
sample × niche proportion matrix
```

然后比较 Grade1_A vs Grade1_NA。

### 22.1 基础统计

样本量较小时，可先使用：

```text
Wilcoxon rank-sum test
Benjamini-Hochberg FDR
```

并报告 effect size。

输出：

```text
06_group_comparison/
    niche_abundance_by_sample.xlsx
    niche_A_vs_NA_statistics.xlsx
```

### 22.2 更严格的 compositional analysis

如果 niche abundance 成为论文核心结果，建议进一步考虑：

```text
scCODA
crumblr
Dirichlet / logistic-normal model
```

因为：

\[
\sum_k P_{s,k}=1
\]

niche proportion 属于 compositional data。

---

## 23. Step 8：Sensitivity analyses

### A. Neighborhood size

主分析：

```text
25-NN
```

辅助：

```text
10-NN
50-NN
```

比较 niche stability、spatial pattern、biological annotation。

### B. Mean-only vs Mean+Std

比较：

```text
Model A: mean(X_scVI neighbors)
Model B: mean + std
```

如果 mean+std 能稳定分离 biologically meaningful niche，而 mean-only 不能，则说明 neighborhood heterogeneity 提供额外信息。

### C. Center-inclusive vs Neighbor-only

比较：

```text
neighbor-only:
[mean, std]

center-inclusive:
[z_center, mean, std]
```

如果 center-inclusive clustering 几乎退化为 Tumor / Myeloid / Endothelial 等中心细胞身份，而 neighbor-only 能识别跨 cell identity 的共同微环境，则直接支持主方案设计。

### D. Balanced fit vs all-cell fit

比较两套结果之间：

```text
ARI
NMI
cluster composition correlation
```

### E. Stability peaks / clustree

对 stability curve 中多个 peak 分别可视化，并建议制作类似 clustree 的结果，观察 K 增加时 cluster 如何 split。

---

## 24. 推荐保存的 AnnData 字段

最终 AnnData 可增加：

```python
adata.obsm["X_state_niche_mean"]
adata.obsm["X_state_niche_std"]
adata.obsm["X_state_niche_raw"]
adata.obsm["X_state_niche"]

adata.obs["state_niche_knn25_rmax"]
adata.obs["state_niche_knn25_rmean"]
adata.obs["state_niche_fit_set"]

adata.obs["state_niche_k8"]
adata.obs["state_niche_k10"]
adata.obs["state_niche_k12"]
adata.obs["state_niche_final"]

# 如果可获得 posterior
adata.obs["state_niche_max_prob"]
adata.obs["state_niche_entropy"]
```

---

## 25. 推荐输出目录

```text
state_niche_analysis/
│
├── 00_QC/
│   ├── scVI_latent_summary.csv
│   └── sample_cell_counts.csv
│
├── 01_neighbors/
│   ├── knn25_indices/
│   ├── knn25_distances/
│   ├── knn25_radius_by_sample_violin.pdf
│   └── knn25_radius_summary.csv
│
├── 02_features/
│   ├── X_state_niche_mean.npy
│   ├── X_state_niche_std.npy
│   ├── X_state_niche_scaled.npy
│   └── scaler.pkl
│
├── 03_autok/
│   ├── stability.csv
│   ├── stability.pdf
│   ├── candidate_K/
│   └── autok_model/
│
├── 04_spatial/
│   ├── all_samples_state_niche.pdf
│   └── per_sample/
│
├── 05_characterization/
│   ├── celltype/
│   ├── subtype/
│   ├── genes/
│   └── pathways/
│
├── 06_group_comparison/
│   ├── niche_abundance_by_sample.xlsx
│   ├── Grade1_A_vs_Grade1_NA.xlsx
│   └── figures/
│
├── 07_sensitivity/
│   ├── knn10/
│   ├── knn50/
│   ├── mean_only/
│   ├── center_inclusive/
│   └── full_fit/
│
└── final/
    ├── adata_state_niche.pkl
    ├── niche_annotation.xlsx
    └── figures/
```

---

## 26. 推荐脚本结构

```text
00_state_niche_config.py
01_state_niche_validate_input.py
02_state_niche_build_knn25.py
03_state_niche_build_features.py
04_state_niche_autok_gmm.py
05_state_niche_fit_final_gmm.py
06_state_niche_spatial_plots.py
07_state_niche_characterization.py
08_state_niche_group_comparison.py
09_state_niche_sensitivity.py
run_state_niche.sh
```

### 26.1 配置参数建议

```python
RANDOM_SEED = 123

SAMPLE_KEY = "sample_name"
GROUP_KEY = "sample_group"

SPATIAL_KEY = "spatial"
SCVI_KEY = "X_scVI"

N_NEIGHBORS = 25

STATE_NICHE_RAW_KEY = "X_state_niche_raw"
STATE_NICHE_KEY = "X_state_niche"

AUTOK_MIN = 2
AUTOK_MAX = 20
AUTOK_MAX_RUNS = 10
AUTOK_CONVERGENCE_TOL = 0.001

N_FIT_PER_SAMPLE = 10000
```

---

## 27. 核心算法伪代码

```python
# INPUT
Z = adata.obsm["X_scVI"].astype("float32")
coords = adata.obsm["spatial"].astype("float32")

# SPATIAL KNN
for sample in adata.obs["sample_name"].unique():
    idx_global = np.where(
        adata.obs["sample_name"].values == sample
    )[0]

    coords_s = coords[idx_global]
    tree = cKDTree(coords_s)

    dist_s, idx_s = tree.query(
        coords_s,
        k=26
    )

    idx_s = idx_s[:, 1:26]
    dist_s = dist_s[:, 1:26]

    idx_global_neighbors = idx_global[idx_s]

# NEIGHBOR STATE AGGREGATION
for chunk in chunks:
    z_neighbor = Z[neighbor_idx_chunk]
    # cells × 25 × latent_dim

    mu = z_neighbor.mean(axis=1)
    sd = z_neighbor.std(axis=1)

    X_chunk = np.concatenate(
        [mu, sd],
        axis=1
    )

# SCALE
scaler = StandardScaler()
adata.obsm["X_state_niche"] = (
    scaler
    .fit_transform(adata.obsm["X_state_niche_raw"])
    .astype("float32")
)

# BALANCED FIT SET
for sample:
    random sample <= 10000 cells

# AUTOK

autok = cc.tl.ClusterAutoK(
    n_clusters=(2, 20),
    max_runs=10,
    convergence_tol=0.001
)

autok.fit(
    adata_fit,
    use_rep="X_state_niche"
)

# FINAL PREDICTION
adata.obs["state_niche_final"] = autok.predict(
    adata,
    use_rep="X_state_niche",
    k=selected_k
)
```

---

## 28. 内存优化

对于 621,806 cells，不建议一次性长期保留：

```text
621806 × 25 × D
```

三维数组。

推荐 chunk：

```python
CHUNK_SIZE = 20000
```

预分配：

```python
X_mean = np.empty(
    (adata.n_obs, D),
    dtype=np.float32
)

X_std = np.empty(
    (adata.n_obs, D),
    dtype=np.float32
)
```

逐 chunk 读取 neighbor index，计算 mean/std 后立即写入对应行。

---

## 29. 为什么本方法可能比 composition niche 更适合当前问题

传统 composition niche：

\[
N_i^{composition}=[p_1,p_2,...,p_K]
\]

回答：

> 这里有哪些细胞？

但可能无法区分：

### Region A

```text
Tumor + Myeloid
LDHA+ Tumor
MRC2+/SGK1+ glycolytic/hypoxic Myeloid
```

### Region B

```text
Tumor + Myeloid
CCN3+ Tumor
GPR34+ relatively quiescent Myeloid
```

如果两者 cell composition 相似，composition niche 可能将其归为同一类。

而邻域 scVI latent state 理论上能够进一步区分这两种 transcriptional-state ecosystem。

---

## 30. 与标准 CellCharter 的关系

本方案不是标准 CellCharter 的直接复刻。

标准 CellCharter 的核心思路是：

```text
cell latent representation
+
spatial-neighbor aggregated latent representation
→
GMM spatial clustering
```

本方案做两点针对性调整：

### Modification 1

固定使用：

```text
25 nearest cells
```

而不是 graph hop。

### Modification 2

使用：

```text
neighbor mean + neighbor std
```

并且：

```text
exclude center-cell X_scVI
```

因此本方案更强调：

> **extrinsic transcriptional-state microenvironment**

而不是：

> **cell state + spatial context**

但 GMM clustering 与 cluster-number stability 仍借用 CellCharter 框架。

---

## 31. 方法学表述建议

论文 Methods 中不建议写：

> “We developed a completely novel spatial niche method.”

更稳妥的表述为：

> We constructed a neighborhood-level transcriptional-state representation inspired by latent-space neighborhood aggregation frameworks. For each cell, its 25 nearest spatial neighbors within the same tissue section were identified. The scVI latent representations of these neighboring cells were summarized by the dimension-wise mean and standard deviation, without including the latent representation of the index cell itself. The resulting neighborhood-state features were standardized and clustered using a Gaussian mixture model. Candidate cluster numbers were evaluated using repeated GMM fitting and CellCharter-style clustering stability analysis.

后续若用于论文，应补充：

```text
neighbor number sensitivity
stability analysis
sample-level reproducibility
spatial coherence
biological annotation
```

---

## 32. 关键风险

### 风险 1：X_scVI 主要编码 cell identity

如果 scVI latent 的主要 variance 来自 Tumor vs Myeloid vs Endothelial，那么 neighbor mean 仍可能主要反映 cell-type composition。

解决：

```text
state niche
vs
composition niche
```

做系统比较。如果两者几乎一一对应，则说明新增信息有限。

### 风险 2：sample-specific latent structure

即使 scVI 做了 batch correction，也可能残留 sample-specific effect。

因此必须检查：

```text
niche × sample
```

组成。如果某 niche 超过 80–90% 来自单个 sample，应高度谨慎。

### 风险 3：GMM 是对连续状态的离散化

实际组织微环境可能是 continuous gradient，因此建议保存 posterior probability，并识别 low-confidence boundary cells。

### 风险 4：cluster stability ≠ biological validity

即使某个 K stability 很高，也不意味着该 K 下所有 cluster 都有独立生物学意义。

最终 K 应同时考虑：

```text
stability
spatial coherence
cross-sample reproducibility
biological interpretability
```

---

## 33. 推荐的第一版主参数

| 参数 | 推荐 |
|---|---|
| Spatial neighborhood | 25 nearest cells |
| Cross-sample neighbors | 禁止 |
| Center cell included | 否 |
| Representation | `X_scVI` |
| Aggregation | mean + std |
| Feature scaling | global StandardScaler |
| Clustering | GMM |
| Covariance | full，若显存/速度允许 |
| K search | 2–20 |
| AutoK runs | 10 |
| convergence tolerance | 0.001 |
| Stability framework | CellCharter `ClusterAutoK` |
| Fit dataset | 每 sample 最多随机 10,000 cells |
| Prediction | 全部 621,806 cells |
| Group comparison statistical unit | sample |
| Main sensitivity | KNN=10/25/50；mean-only；center-inclusive；balanced/full fit |

---

## 34. 最终证据链

本项目最终不是简单得到：

```text
Cluster 0
Cluster 1
Cluster 2
...
```

而是建立：

```text
scVI latent state
        │
        ▼
25-cell spatial neighborhood
        │
        ▼
mean + std state representation
        │
        ▼
stable GMM niches
        │
        ▼
spatially coherent recurrent niches
        │
        ▼
cell-type/subtype composition
        │
        ▼
gene + pathway annotation
        │
        ▼
identify hypoxic/glycolytic Tumor–Myeloid niche
        │
        ▼
Grade1_A enrichment
        │
        ▼
Tumor–Myeloid / vascular / immune relationships
        │
        ▼
biological and validation hypotheses
```

---

## 35. 最值得重点验证的结果

如果本方法最终识别出一个 niche 同时满足：

```text
1. Grade1_A 中显著富集
2. Tumor + Myeloid 均明显参与
3. LDHA / glycolysis score 高
4. hypoxia score 高
5. MRC2+ / SGK1+ myeloid enriched
6. CCN3+ / relatively quiescent state 相对减少
7. 空间上形成连续 patch，而非随机散点
8. 在多个 Grade1_A 病例中重复出现
9. 对 KNN=10/25/50 具有一定鲁棒性
10. 对不同 stability peak 的 K 仍能找到对应生态位
```

则可以较有说服力地定义为：

> **Grade1_A-associated hypoxic/glycolytic Tumor–Myeloid transcriptional-state niche**

这将比单纯基于 Tumor/Myeloid composition 定义的 niche 更接近当前项目真正希望捕获的生物学状态。

---

## 36. 推荐参考

1. **CellCharter documentation — CosMx NSCLC tutorial**  
   https://cellcharter.readthedocs.io/en/latest/notebooks/cosmx_human_nsclc.html

2. **CellCharter API — ClusterAutoK**  
   https://cellcharter.readthedocs.io/en/latest/generated/cellcharter.tl.ClusterAutoK.html

3. **CellCharter API — Cluster**  
   https://cellcharter.readthedocs.io/en/stable/generated/cellcharter.tl.Cluster.html

CellCharter 的标准流程同样使用低维 latent representation、空间邻域聚合、GMM clustering，并通过重复聚类与 Fowlkes–Mallows-based stability 辅助选择 cluster number。本方案保留 GMM/AutoK 框架，但将 neighborhood representation 改为针对当前科学问题的 **25-nearest-neighbor、neighbor-only、mean+std scVI state representation**。

---

## 一句话总结

本方案定义的不是：

> “这里有哪些细胞？”

而是：

> **“围绕每一个细胞的局部空间环境，在整体转录状态及其异质性上属于哪一种可重复的微环境？”**

核心技术路线：

\[
\boxed{
25NN
\rightarrow
mean(X_{scVI})+std(X_{scVI})
\rightarrow
standardization
\rightarrow
GMM
\rightarrow
CellCharter-like\ stability
\rightarrow
Transcriptional-state\ niches
}
\]
