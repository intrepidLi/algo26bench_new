# algo26bench v2 —— 框架设计文档

> 目标：为 `refer/` 中 9 篇工业排序模型论文建立一个**可复现、可扩展、且不损害各模型原生特性**的 benchmark 框架。
>
> 状态：设计稿。代码尚未落地。
>
> 覆盖论文：HyFormer、InterFormer、MixFormer、OneTrans、ULTRA-HSTU、EST、Kunlun、LONGER、TokenFormer

---

## 目录

1. [问题陈述](#1-问题陈述)
2. [诊断：现有两套框架的失效点](#2-诊断现有两套框架的失效点)
3. [设计输入：9 篇论文的分歧矩阵](#3-设计输入9-篇论文的分歧矩阵)
4. [核心设计决策](#4-核心设计决策)
5. [分层架构](#5-分层架构)
6. [L0 core：契约层](#6-l0-core契约层)
7. [L1 data：数据层](#7-l1-data数据层)
8. [L2 nn：算子库](#8-l2-nn算子库)
9. [L3 recipes：论文实现层](#9-l3-recipes论文实现层)
10. [L4 engine：训练/评测引擎](#10-l4-engine训练评测引擎)
11. [L5 bench：实验编排与报告](#11-l5-bench实验编排与报告)
12. [保真度治理机制](#12-保真度治理机制)
13. [配置系统](#13-配置系统)
14. [评测赛道设计](#14-评测赛道设计)
15. [SOLID / KISS 自检](#15-solid--kiss-自检)
16. [目录结构](#16-目录结构)
17. [迁移与实施路径](#17-迁移与实施路径)
18. [未决问题](#18-未决问题)

---

## 1. 问题陈述

我们要同时满足两个通常互相拉扯的目标：

- **可比性**：9 个模型跑在同一份数据、同一套指标、同一个训练预算下，结论才有意义。
- **保真度**：每个模型的 distinguishing mechanism 必须真实存在。如果 EST 没有 CSA、HyFormer 的三条序列被拍成一条、LONGER 的因果方向被写反，那么 benchmark 测的不是这些论文，是它们的赝品。

现有的 `UniRank` 用牺牲后者换取了前者。这不可接受，因为**被牺牲掉的恰好就是论文的贡献本身**——benchmark 的全部意义就在于测量这些贡献。

本设计的立场：**可比性应当由"实验编排"保证，而不是由"强制统一模型内部表示"保证。**

---

## 2. 诊断：现有两套框架的失效点

### 2.1 UniRank：统一点放错了位置

UniRank 的根本错误是**把统一点放在 `forward()` 的入口**。dataloader 直接产出 `(batch_dict, item_dict, mask)`，其中 `item_dict` 已经完成 side-info join 并 pad 成 `[B, T+1, item_info_dim]`，且约定 **target 在最后一个位置**（`unirank_dataloader.py:546-598`）。

这是一连串模型决策被写死在数据层：

- 只能有**一条**序列，且只有**一个** `item_info_dim`
- target 与 history 共享同一套 embedding 宽度
- 只能是 padded dense，不能 jagged
- 只有一个候选（`K=1`）
- 序列必须已经按 `max_len` 截断/补齐

后果在 15 个模型里逐条可见：

| 位置 | 症状 |
|---|---|
| `UniMixer.py:90-91,178-183` | 整条序列 `(max_len+1)*item_info_dim` 拍平进 ChunkTokenizer，`mask` 在 forward 里**完全不用** |
| `HyFormer.py:248-250,308-309` | 注释写明"为 FlashAttention 走 mask-free SDPA"——padding 参与了注意力计算 |
| `EST.py:161` | 一行注释删掉 Content Sparse Attention（EST 两大贡献之一，论文消融值 +0.14% GAUC） |
| `HeMix.py:43,148-149` | 硬编码 `real_seq_len=20` |
| `LONGER.py:63-67` | `max_len % num_groups == 0` 的约束检查写在模型里，而不是 pipeline 里 |
| `TokenFormer.py:308-311` | 无效 key 没有在 attention score 上屏蔽，只在输出后乘 mask |
| `Zenith.py:69-75` | 硬编码特征名 `user_id` / `item_id` |
| `INFNet.py:53` | 把论文的 behavior type 数量等同于数据集的 label 数量 |
| 全部 15 个 | 每个模型自己重算一遍 `item_info_dim` / `non_item_dim`（约 55-75 行） |

还有两处结构性问题：

- **God class**：`rank_model.py` 的 `BaseModel` 一个类同时负责 fit / evaluate / checkpoint / early-stop / 双优化器构建 / `torch.compile` / DDP gather / AMP。这直接导致 `UniMixer.py:158-173` 为了做 tau 退火而重写 `train_step`——LSP 违反的典型信号。
- **配置膨胀**：`model_config.yaml` 2293 行 / 136 个 experiment 条目，模型超参和训练超参混在一起；`feature_embedding.py:119-128` 用 `eval()` 解析配置。

### 2.2 algo26bench_framework：契约对但太窄

`RankBatch` / `ModelOutput` / `DataSpec` 的类型化契约、`True = valid` 的统一掩码语义、`tests/test_layers.py` 把 RankMixer rewire 钉成精确置换——这些都做对了，应当继承。

但它的开放性没做完：

- `ExperimentConfig.model` 的静态类型就是 `HyFormerConfig`（`core/config.py:144-160`），不是判别联合
- `train.py:95` 直接调 `build_hyformer`，绕过了自己的 `MODEL_REGISTRY`——加第二个模型必须改入口，OCP 违反
- `RankBatch` 仍然假设 "padded dense + named sequences + 单候选"。ULTRA-HSTU 的全 jagged、TokenFormer 的 item/action 交错双 token 且一次打 K 个候选、LONGER 的 `d → K·d` token merge，都塞不进去
- 只有 `BinaryClassificationTask`，没有 task registry；`grouped_auc()` 写了但 trainer 从不调用；`group_ids` 在 batch 里但没人用
- Trainer 是 smoke 级别：无 early stopping、无 LR schedule、无 checkpoint 选择、无 epoch 概念

结论：**contract 层照抄，registry / config / batch 泛型化重做，engine 补齐。**

---

## 3. 设计输入：9 篇论文的分歧矩阵

这是整个设计最重要的输入。这 9 篇不是一个设计的 9 个变体，而是**互相矛盾的 9 个答案**，其中数篇是对彼此的直接反驳。

### 3.1 三组正面冲突（必须保留，不能调和）

**冲突 1：NS 特征的 tokenize 方式**
- OneTrans 实测 Auto-Split **优于** semantic grouping（group-wise 为 −0.10% CTR AUC / −0.30% UAUC）
- HyFormer 基于归纳偏置**选择** semantic grouping
- MixFormer 用纯 auto-split，且**特征在拼接向量中的顺序是语义承载的**（UI-decoupling 要求 user 侧特征连续且在前）
- EST 每个原始特征一个 token，且**每个 token 有自己的 FFN 和自己的 W_Q/W_O**

**冲突 2：多序列策略**
- HyFormer：每条序列独立 query token 集合，**明确反对合并**（合并 −0.06% AUC），并以此解释 MTGR/OneTrans 为何落后
- OneTrans：合并成一条按时间戳交错的流（时间戳无关的版本 −0.09% CTR AUC，去掉 [SEP] 再 −0.13%）
- InterFormer / Kunlun：沿 embedding 维拼接 `R^{T×kd}` 再用 MaskNet/`MLP_lce` 压回 `R^{T×d}`——**要求所有序列等长且时间对齐**
- ULTRA-HSTU：Mixture of Transducers，每条序列独立 transducer，**各自的深度和容量**
- Kunlun 的 event-level personalization：每种事件类型有自己的 `d_model` / `heads` / `n_tokens` / `L` / `w`（click `(256,8,32,3,100)` vs impression `(128,4,16,2,50)`）

**冲突 3：序列侧的注意力原语**
- ULTRA-HSTU 的头条研究结论：cross-attention（STCA、LONGER 一系）在 ~9 层饱和，self-attention 持续改善（9 层 −0.14% vs −0.38% C-NE，12 层 −0.14% vs −0.46%）
- HyFormer 和 MixFormer 都以 cross-attention 为序列侧核心原语
- MixFormer 实测在 **NS 特征上**用真 self-attention 相比 parameter-free HeadMixing 增益恰好 **+0.00%**
- OneTrans 实测 causal 与 full attention 是平手（+0.00%），选 causal 纯粹是为了 KV cache

**一个 benchmark 如果在设计上偏袒任何一方，就没有资格测量这场争论。**

### 3.2 完整分歧矩阵

| 轴 | HyFormer | InterFormer | MixFormer | OneTrans | ULTRA-HSTU | EST | Kunlun | LONGER | TokenFormer |
|---|---|---|---|---|---|---|---|---|---|
| token 流 | NS/Seq 严格分离 | 双 arch 并行 | NS=Q, Seq=KV | **单条 causal 流** | Seq 单流，NS 折进序列位置 | **双流，仅 B→N** | 双塔交织 | 序列塔 + 顶层晚融合 | **单条 monolithic 流** |
| "token" 是 | 语义特征组 / 行为 / global query | 单字段 / 行为 | NS 拼接向量的连续切片 | auto-split 切片 / 行为 / [SEP] | item+action **相加**后的事件 | 单原始特征 / 单行为 | 单特征 / 单行为 | **K 个行为 merge 后的组** | 字段 / item / **action** / target / sep |
| NS tokenizer | semantic group | 每稀疏字段一 token + dense 合一 token | auto-split（顺序敏感） | **auto-split** | 无（折进序列） | 每特征一 token + per-token FFN | 每字段一 token | 部分 global token，其余绕过 | 前缀 token，`l_f` 层后**硬丢弃** |
| 序列注意力 | 可插拔：full / cross / SwiGLU | full MHA + RoPE，保形 | cross-attn | causal self-attn | **SiLU（无 softmax）+ semi-local mask** | 无（CSA 用冻结 Gram 矩阵） | **GDPA（无 softmax）+ 双向滑窗** | cross-causal 后 self-causal | full causal → **收缩窗口** |
| 因果方向 | — | 非因果 | — | 标准 causal | causal | — | **双向**滑窗 | **反向**（序列 latest→earliest，mask 为 `j≥i`） | 标准 causal |
| 层是否同构 | ✗ layer 0 生成 query | ✗ CLS 仅 layer 1 前置 | ✓ | ✗ pyramid 每层长度变 | ✗ AT 两段长度 + MoT 异构 | ✓ | ✗ **CompSkip 奇偶层跳不同模块，且奇层复用前层 HSP** | ✗ layer 1 是 cross | ✗ 每层 mask 不同 |
| 序列长度 | 3000 + 50 + 50 | 50–1000（10 条） | 512–10000 | ~1190–1500 | **3072–16384** | 300 + 50（lifelong 5000） | >1000 | **2000–10000** | KuaiRand-27K |
| 任务 | 单 CTR | 单 CTR | 多（Finish/Skip） | 多（CTR/CVR） | 多（consumption/engagement） | 单 CTR | 多 | 单 CVR | **多动作 softmax CE** |
| 主指标 | query-AUC | AUC/gAUC/NE | AUC/UAUC | AUC/UAUC | **NE** | AUC/**GAUC** | **NE + MFU** | AUC/LogLoss | **Macro AUC** |
| 公开数据集 | 无 | **Amazon/TaobaoAd/KuaiVideo (BARS)** | 无 | 无 | **KuaiRand** | 无 | 无 | 无 | **KuaiRand-27K** |

### 3.3 会杀死"天真共享抽象"的机制清单

按框架的哪个表面被压迫来归类。这份清单是后面每一条设计决策的直接依据。

#### A. 数据/collate 表面

| 机制 | 论文 | 要求 |
|---|---|---|
| 多序列不对齐宽度 | HyFormer | 每条序列独立 sparse dim（64 或 224）、独立 side-info schema、独立长度；**禁止强制对齐** |
| 全 jagged 张量 | ULTRA-HSTU、Kunlun | 端到端不 pad；per-sample 长度元数据要一路传到 kernel tile 调度 |
| 按时间戳跨序列交错 | OneTrans | 排序发生在 **collate 内部**，时间戳必须活到那时 |
| 每行为两个 token | TokenFormer | `SL = M + 2T + K + N_sep`，item 与 action 交错占位 |
| 一次打 K 个候选 | TokenFormer | 一个 sample = (user context, K candidates)，label 是 K 维向量 |
| request 级分组 | MixFormer (RLB)、OneTrans (KV cache)、EST (user-candidate 解耦)、LONGER (KV cache serving) | 同一 request 的候选在 batch 中连续，且带 request 边界张量 |
| 特征布局顺序约束 | MixFormer | user 侧特征必须连续且在前，才能让 head index `< N_U` ⟺ user 侧 |
| 冻结的第二张 embedding 表 | EST | content embedding 在 `R^{d_M}`（≠ `d`），不可训练，只用于产生相似度图 |
| 预计算稀疏注意力图 | EST | `Ω_γ ∈ Z^{L×K}` 和 `E_γ ∈ R^{L×K}` 作为额外输入张量，层间共享且不可微 |
| GSU 检索作为预处理 | EST | `B_c` 是从 5000 长 lifelong 序列按 (user, candidate) top-50 检索出来的 |
| 每行为原始时间戳 | Kunlun (ROTE)、LONGER | 时间差要进 attention kernel；LONGER 还要把它作为 **side info 拼接**（改变输入宽度），同时另有一套**相加**的可学习位置 |
| 跨 rank 协调的 sampler | ULTRA-HSTU (LBSL) | sampler 内部做 all-reduce，读取模型的代价指数 γ，周期性重标定 |
| 训练/推理长度不同 | ULTRA-HSTU | Stochastic Length 后训练平均 1110，推理 3072 |
| 两套互不兼容的数据组织 | TokenFormer | User-Centric（一 sample = 整条用户轨迹，全程 NTP）vs New-Impression-Only（一 sample = 一次曝光） |
| per-sample loss index 集合 | TokenFormer | `I_loss ⊂ {1..SL}` 作为数据依赖的输入 |

#### B. 模型/forward 表面

| 机制 | 论文 | 要求 |
|---|---|---|
| 双流不同残差路径 | EST | block 签名是 `(N, B) → (N, B)`，两流不同长度、不同 FFN 类型、单向可见 |
| 层间携带状态 | Kunlun (CompSkip) | 签名是 `(X, S, H_prev, layer_cfg) → (X, S, H_curr)`，奇层消费前层 HSP 输出 |
| 每 token 独立权重 | EST、OneTrans、MixFormer、HyFormer | `W_Q, W_O ∈ R^{L_N × d × d}`；参数量是 token 数的函数；需要 batched einsum |
| 层内 Q 与 KV 长度不同 | OneTrans (pyramid)、LONGER | `(B,L,D) → (B,L,D)` 的 block 签名失效；OneTrans 1190→12，按 32 取整 |
| 模型中途改 hidden dim | LONGER (token merge) | `L → L/K`，`d → K·d`，下游每个 block 参数量按 `K²d²` 涨 |
| 嵌套 transformer | LONGER (InnerTrans) | 张量 reshape 成 `[B, L/K, K, d]`，在 `K` 轴上跑完整 transformer block |
| per-sample 动态生成权重 | InterFormer (PFFN)、Kunlun (GDPA) | `W_PFFN = X_sum W ∈ R^{d×d}` 逐样本生成，是 batched bmm 不是 `nn.Linear` |
| token 轴（非 channel 轴）压缩 | InterFormer (LCE)、Kunlun | `R^{d×n} → R^{d×n_sum}`，要求 NS token 数**静态、有序、固定** |
| 逐层重算 K/V | HyFormer、MixFormer | 原始序列 embedding 必须对**每一个** block 保持存活，不能编码一次就丢 |
| 非 softmax 注意力 | ULTRA-HSTU (SiLU)、Kunlun (per-head 任意激活)、EST (无归一化) | SDPA/FlashAttention 不能直接用；Kunlun 温度是 `maxlen(seq)`（数据依赖标量） |
| 逐层不同 mask | TokenFormer | mask 是 `(layer, token_types, positions) → mask` 的函数；收缩窗口 `w=[32,16]`；`l_f` 层后按 **token 类型**列屏蔽 |
| 非 arange 的 position id | TokenFormer (静态字段全 `p=0`，target `p=SL+1`)、Kunlun (ROTE) | position id 必须是 tokenizer 产出的 per-sample 张量 |
| block 自己拥有残差 | TokenFormer (NLIR) | `I = X + σ(XW_g) ⊙ A`，gate 来自 **层输入**而非 attention 输出 |
| 结构性 mask（非 token mask） | MixFormer (UI-decoupling) | `M[i,j]` 是特征 head 组之间的单向流控，token-sequence 抽象里没有它的位置 |
| 候选在序列内部 | ULTRA-HSTU | candidate 占据序列位置，action 项被零屏蔽；**没有独立的 target 输入** |
| 异构 global token | LONGER | target / 可学习 CLS / UID embedding / 预计算高阶交叉特征，四个不同来源 |
| 特征分区 MoE | Kunlun | M 个 Wukong expert，分区是在**特征**上而非 token 路由 |

#### C. 训练/运行时表面

| 机制 | 论文 | 要求 |
|---|---|---|
| 稀疏/稠密双优化器 | 全部 | sparse Adagrad + dense RMSProp/AdamW，不同 LR |
| 分组梯度裁剪 | OneTrans | dense 90 / sparse 120 |
| 异步稀疏更新 | HyFormer、MixFormer | dense 用 step k−1 的梯度，sparse 用 step k，差一步 staleness |
| 每 epoch 稀疏参数重置 | EST | 每个 epoch 开始把 sparse 重置回 `θ_s^(0)`，dense 继承——+0.44% GAUC |
| 流式 next-batch 评测 | OneTrans | 按时间流式，每个 batch 先 eval 后 train，按天算 AUC 再宏平均 |
| 自定义 autograd | HyFormer（序列去重 pooling op）、ULTRA-HSTU（选择性重物化） | 手写 forward/backward |
| 模块并行调度 | InterFormer | Interaction arch（通信受限）与 Sequence arch（计算受限）重叠执行——+20% QPS |
| 混合精度分层配置 | ULTRA-HSTU（BF16/FP8/INT4）、LONGER（按组件） | 精度是逐模块配置 |
| serving 图 ≠ 训练图 | EST、LONGER、OneTrans、TokenFormer、MixFormer | 头条效率数字全在另一张图上测的 |

---

## 4. 核心设计决策

只有一个真正的架构选择要做，其余的都是它的推论。

### 决策：engine 对 batch 类型泛型化，batch 类型对 engine 不透明

```python
TBatch = TypeVar("TBatch", bound="DeviceMovable")

class Recipe(Protocol[TBatch]):
    """一篇论文 = 一个 Recipe。这是框架里唯一的窄腰。"""
    name: str
    def required_features(self) -> FeatureRequest: ...
    def build_collator(self, ctx: DataContext) -> Collator[TBatch]: ...
    def build_module(self, ctx: DataContext) -> nn.Module: ...   # forward(TBatch) -> ModelOutput
    def build_objective(self) -> Objective[TBatch]: ...
    def fidelity_card(self) -> FidelityCard: ...
    def capability(self) -> Capability: ...
```

engine 对 `TBatch` 的全部要求只有两条：支持 `.to(device)`，能报告 `.batch_size`。别的一概不知道。

**为什么这一条同时满足 KISS 和 SOLID：**

- **KISS**：框架不再需要设计一个"能容纳 9 篇论文的通用 batch 结构"——那个东西不存在，UniRank 试过了，结果是 `(batch_dict, item_dict, mask)` 加 15 处 hack。取消这个需求，问题就消失了。
- **OCP**：加一篇论文 = 加一个目录 + 一行 `@register_recipe`。engine / data / core 零改动。
- **DIP**：engine 依赖抽象 `Recipe[TBatch]`，不依赖任何具体张量布局。

§3.3 表 B 里的所有机制——LONGER 的 `d→K·d`、ULTRA-HSTU 的 jagged、TokenFormer 的 per-sample position id、EST 预计算的 top-K 图、Kunlun 的层间携带状态——**全部落在窄腰以下**，engine 碰不到，也不需要碰。

### 推论 1：数据层只吐原始 ragged 记录

pad 是 collator 的职责，不是 loader 的职责。dataset 不做 side-info join，不约定 target 位置，不 pad，不截断。

### 推论 2：不设模型基类

UniRank 的保真度是死在 `MultiTaskModel` 这个基类里的。**没有基类就没有 LSP 可违反。** 模型满足结构化 Protocol 即可，共享通过 `nn/` 算子库（组合）而非继承。

### 推论 3：Trainer 是永不被继承的具体类

变化点全部外置为 policy 和 hook。`UniMixer` 重写 `train_step` 是坏味道；`TauAnnealHook` 是它的正解。

---

## 5. 分层架构

```
                    依赖方向（严格单向，向下）
┌─────────────────────────────────────────────────────────────┐
│ L5  bench/     实验编排、报告、排行榜、CLI                    │
│                依赖：全部（唯一允许 import recipes 的层）      │
├─────────────────────────────────────────────────────────────┤
│ L4  engine/    Trainer / Evaluator / Profiler / Hooks        │
│                依赖：仅 core                                  │
├─────────────────────────────────────────────────────────────┤
│ L3  recipes/   一篇论文一个自包含包                           │
│                依赖：core, data, nn                          │
├─────────────────────────────────────────────────────────────┤
│ L2  nn/        可独立 import 的算子库（不认识本框架）          │
│                依赖：仅 torch                                 │
├─────────────────────────────────────────────────────────────┤
│ L1  data/      RawDataset / Sampler / view 原语              │
│                依赖：仅 core                                  │
├─────────────────────────────────────────────────────────────┤
│ L0  core/      纯契约：Protocol + dataclass，无 torch 逻辑     │
│                依赖：无                                       │
└─────────────────────────────────────────────────────────────┘
```

**关键约束（用 import-linter 在 CI 强制）：**

- `engine` **不得** import `recipes` 或 `data`
- `nn` **不得** import `core` / `data` / `engine`——它是一个可以被别的项目直接抄走的普通库
- `recipes/<A>` **不得** import `recipes/<B>`——论文之间没有依赖，共享的东西必须先沉进 `nn/`
- 只有 `bench` 可以 import `recipes`

最后一条是 UniRank 那种 "EST 和 OneTrans 的 `MixedFFN` 互为近似复制"（`EST.py:285-331` vs `OneTrans.py:346-394`）的结构性解药：想复用就必须先提炼到 `nn/`，提炼的动作本身会强迫你想清楚它到底是不是同一个东西。

---

## 6. L0 core：契约层

**规模上限：不超过 400 行，且不含任何 torch 计算逻辑。** 只有 Protocol、dataclass、枚举。

### 6.1 特征声明

模型**声明式地**向数据层要数据，而不是被动接受数据层给什么。

```python
class FieldKind(Enum):
    CATEGORICAL = auto()
    NUMERIC     = auto()
    PRETRAINED  = auto()   # EST 的冻结 content embedding
    TIMESTAMP   = auto()

@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: FieldKind
    vocab_size: int | None = None
    width: int = 1              # PRETRAINED 时是 d_M
    trainable: bool = True      # EST 的 content 表为 False

@dataclass(frozen=True)
class SequenceRequest:
    """一条命名序列。宽度和长度由 recipe 自己定，框架不做跨序列对齐。"""
    name: str
    fields: tuple[FieldSpec, ...]
    max_len: int | None          # None = 不截断，交给 collator
    layout: Layout               # RAGGED | PADDED
    need_timestamps: bool = False

@dataclass(frozen=True)
class FeatureRequest:
    scalars: tuple[FieldSpec, ...]
    sequences: tuple[SequenceRequest, ...]
    candidates: tuple[FieldSpec, ...]
    num_candidates: int = 1          # TokenFormer 一次打 K 个
    scalar_order: ScalarOrder | None = None   # MixFormer 的 user/item 连续布局约束
    group_by_request: bool = False            # RLB / KV cache 需要
```

`FeatureRequest` 是三个 §3.3-A 机制的落点：HyFormer 声明三条序列且**不**要求对齐宽度；MixFormer 通过 `scalar_order` 声明 user 侧必须连续在前；EST 通过 `trainable=False` + `width=d_M` 声明冻结的第二张表。

注意 `FeatureRequest` **不含 labels**——见下一节。

### 6.1.1 任务由数据集定义，不由 recipe 定义

**已决（2026-09-04）。** 这是一个降低复杂度的决定，值得单独说明。

任务集合是**数据集的属性**，recipe 适配数据集给出的任务集合，而不是反过来声明"我要 CTR 和 CVR"。理由：TAAC2026（`algo26bench` 在用的 2000w 数据集）只有 pCVR 单标签，而 KuaiRand 有多标签但**也可以按单标签跑**。如果让 recipe 声明任务，那么 OneTrans 在 TAAC2026 上就无法运行——而这显然是错的，OneTrans 完全可以只训一个头。

```python
@dataclass(frozen=True)
class TaskSpec:
    name: str                   # "pcvr" / "click" / "like" / ...
    kind: TaskKind              # BINARY | MULTI_ACTION_SOFTMAX
    label_column: str
    num_classes: int = 2        # MULTI_ACTION_SOFTMAX 时 > 2

@dataclass(frozen=True)
class TaskSet:
    tasks: tuple[TaskSpec, ...]     # 由 DataConfig 提供，进 DataContext
```

于是：

- `Recipe.build_module(ctx)` 从 `ctx.task_set` 读出要建几个 head——所以同一份 recipe 代码在 TAAC2026 上是单头、在 KuaiRand 上是多头，**无需任何 if 分支**
- `Recipe.build_objective()` 同理，从 `ctx.task_set` 决定损失的组合方式
- 数据集配置里可以显式收窄任务集合（KuaiRand 跑单标签就是 `tasks: [click]`），这是配置层的一行，不是代码层的分支

代价是 TokenFormer 有一处需要特殊处理：它的原生形态是**多动作 softmax CE**（一个 head 打 A 维动作空间），不是 A 个独立的 binary head。这通过 `TaskKind.MULTI_ACTION_SOFTMAX` 表达——数据集若提供了这种任务类型，TokenFormer 用原生形态；若数据集只给 `BINARY`（如 TAAC2026），TokenFormer 退化成单 binary head，并在它的 `FidelityCard` 里标 `APPROXIMATED`。这是**声明出来的**降级，不是藏在代码里的。

### 6.2 原始批次

```python
@dataclass
class RawBatch:
    """数据层产出。model-agnostic。未 pad、未 join、未约定 target 位置。"""
    scalars:    dict[str, Tensor]            # [B] 或 [B, w]
    sequences:  dict[str, RaggedTensor]      # 命名多序列，各自宽度/长度
    candidates: dict[str, Tensor]            # [B, K]
    timestamps: dict[str, RaggedTensor]
    labels:     dict[str, Tensor]            # 键集合 == ctx.task_set 的 label_column
    request_id: Tensor | None                # [B]
    row_meta:   dict[str, Tensor]            # user_id (分组指标)、event_time (按天评测)
```

`RaggedTensor` = `(values: Tensor, offsets: Tensor)`，CSR 风格。**不 pad** 是刻意的：ULTRA-HSTU 和 Kunlun 全程 jagged，LONGER 要 padded，TokenFormer 要交错——把 pad 提前做掉就等于替它们做了决定。

**已决（2026-09-04）：`RawBatch` 一律用 torch tensor，不用 numpy。** 统一一种张量类型消掉了"边界在哪"这个持续的认知负担，也让 `data/view.py` 的原语只有一套实现。仅当阶段 1 用 ULTRA-HSTU（全 jagged，最容易暴露拷贝开销）实测出 numpy 在 collate 路径上有**显著**优势时，才在该路径局部退回 numpy，并且要在代码里注明实测数据。默认不这么做。

### 6.3 模型输出与目标

```python
@dataclass
class ModelOutput:
    logits: Mapping[str, Tensor]        # task name -> [B, ...]
    aux_losses: Mapping[str, Tensor] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)   # TokenFormer 的 effective rank

class Objective(Protocol[TBatch]):
    task_set: TaskSet                       # 来自 ctx，不是 recipe 自己编的
    def loss(self, out: ModelOutput, batch: TBatch) -> Tensor: ...
    def scores(self, out: ModelOutput) -> Mapping[str, Tensor]: ...
```

`Objective` 是独立协议而非 Trainer 的一部分。绝大多数 recipe 用共享的 `BinaryTaskObjective`（按 `task_set` 自动展开成 N 个 BCE 头）；TokenFormer 在数据集提供 `MULTI_ACTION_SOFTMAX` 时用自己的实现。损失函数与**指标**是两件事——损失由 `Objective` 定，指标由 Evaluator 定，见 §10.4。

### 6.4 可选能力协议（ISP）

不实现就不承担。这是 UniRank 那个"`multi_masks` 每 batch 都算但只有 INFNet 用"的解药。

```python
@runtime_checkable
class CachedScoring(Protocol):
    """EST / LONGER / OneTrans / TokenFormer / MixFormer 的 serving 路径"""
    def prefill(self, request: Any) -> Any: ...
    def score_candidates(self, cache: Any, candidates: Any) -> Tensor: ...

@runtime_checkable
class HasParamGroups(Protocol):
    def param_groups(self) -> list[ParamGroup]: ...   # sparse/dense 标注 + 各自 clip 阈值

@runtime_checkable
class NeedsSampler(Protocol):
    def build_sampler(self, ctx: DataContext) -> Sampler: ...   # ULTRA-HSTU 的 LBSL

@runtime_checkable
class HasFusedKernel(Protocol):
    """Kunlun / ULTRA-HSTU：同一数学的 reference 与 fused 两套实现"""
    def use_fused(self, enabled: bool) -> None: ...
```

`HasFusedKernel` 值得单独说明：Kunlun 整篇论文的论点是 MFU 而非 AUC，纯 PyTorch 实现能复现它的数学但**复现不了它的主张**。把 reference / fused 做成同一个 recipe 的两个开关，才能既保证数值正确性可测（两者对拍），又保证效率赛道有意义。

**已决（2026-09-04）：Kunlun 的 Triton kernel 暂不实现**，因此该协议首轮**没有实现者**。它仍然留在 core 里，理由有二：一是 Kunlun 的 MFU 主张要在 `FidelityCard` 里标 `OMITTED`，而"存在这个协议但该 recipe 未实现"正是这个标注的结构性对应物；二是 ULTRA-HSTU 的自定义 attention（SiLU + semi-local mask，无法直接用 SDPA/FlashAttention）后续若要补 kernel，落点就是这里。协议只有约 3 行，留着的成本可以忽略。

---

## 7. L1 data：数据层

### 7.1 职责边界

**做**：读取存储、schema 管理、按 `FeatureRequest` 投影出需要的列、切分、分布式分片、按 request 分组、提供 view 原语。

**不做**：pad、side-info join 后的形状约定、target 位置约定、tokenize、embedding。

### 7.2 数据源

```
data/source/
    parquet_blocked.py     # 内部 academic_2000w PCVR：~19.8M 训练行，4 条序列域
    kuairand.py            # ULTRA-HSTU / TokenFormer 的公开对照
    taobao_ad.py           # InterFormer 的 BARS 对照
    amazon_electronics.py  # InterFormer 的 BARS 对照
    kuai_video.py          # InterFormer 的 BARS 对照
    taac2025.py            # 内部
    synthetic.py           # 长度扫描：合成到各模型原生长度（2k–16k）
```

内部数据沿用 `algo26bench_framework/algobench/data/legacy_pcvr.py` 的 adapter 思路——**不重写 legacy 的 parquet 读取，只做适配**。这条已被验证是对的，保留。

### 7.3 view 原语

collator 归模型所有，但不该让每个模型从零手写 pad 逻辑（那就变回 UniRank 的 15 份 copy-paste）。给一组**可组合的纯函数**：

```python
data/view.py
    pad_to(ragged, max_len, side="right") -> (values, valid_mask, lengths)
    to_jagged(ragged) -> NestedTensor                      # ULTRA-HSTU / Kunlun
    truncate(ragged, n, keep="recent") -> RaggedTensor
    interleave_by_time(seqs, timestamps, sep_token) -> ...  # OneTrans
    group_by_request(batch) -> RequestGroupedBatch          # MixFormer RLB / EST 解耦
    join_side_info(ids, table) -> ...
    merge_tokens(padded, k, mode="concat") -> ...           # LONGER token merge
    bucketize_time_delta(ts, row_ts, boundaries) -> ...     # 沿用 legacy 的 64 边界，可选
```

再加一个 `StandardTabularCollator`，覆盖"单序列 + padded + target 独立"这个最常见形态。**关键：它是可选的默认值，不是基类，谁都可以不用。**

掩码语义在 core 里单点定义为 **`True = valid`**（沿用 `algo26bench_framework`）。legacy 的 `True = padding` 由 adapter 显式转换，并由测试钉住。

### 7.4 Sampler

```python
data/sampler/
    sharded.py              # 标准 DDP 分片
    request_grouped.py      # 保证同 request 的候选落在同一 rank 且连续
    stochastic_length.py    # ULTRA-HSTU 的 SL
    load_balanced.py        # LBSL：sampler 内部 all-reduce，按 γ 平衡各 rank 负载
```

`LoadBalancedStochasticLength` 是一个 sampler 做集合通信、读模型代价指数的例子。之所以它能存在，正是因为 `NeedsSampler` 是**可选协议**——别的 8 个 recipe 完全不受影响。

---

## 8. L2 nn：算子库

**这一层是一个普通的 PyTorch 库，不认识 `Recipe`、不认识 `RawBatch`、不认识 `Trainer`。** 它应该能被复制到任何别的项目里直接用。

```
nn/
  attention/
    sdpa.py            # 标准，带正确的 mask（不是 mask-free）
    silu_attn.py       # ULTRA-HSTU：φ(QKᵀ)⊙M·V，无 softmax
    gdpa.py            # Kunlun：seq=Q，非序列生成 K/V，per-head 激活，温度 maxlen
    cross_causal.py    # LONGER：Q 集合 ≠ KV 集合
    lightweight_cross.py  # EST 的 LCA：token-specific W_Q/W_O
    mask/
      causal.py, reverse_causal.py     # LONGER 的反向因果
      semi_local.py                    # ULTRA-HSTU：局部带 + 全局前缀窗
      shrinking_window.py              # TokenFormer：w=[32,16] + 类型列屏蔽
      sliding_bidirectional.py         # Kunlun：[t-w, t+w]
      head_decouple.py                 # MixFormer：特征 head 组之间的单向流控
  mixing/
    head_mixing.py     # MixFormer：parameter-free，reshape+transpose
    rank_mixer.py      # HyFormer/RankMixer：精确置换（已有测试可迁移）
    per_token_ffn.py   # EST/OneTrans/MixFormer/HyFormer 共用
    token_mix.py
  tokenize/
    auto_split.py, group_wise.py, field_wise.py, chunk.py
  pos/
    rope.py, rote.py, learned_absolute.py, time_delta_sideinfo.py
  pool/
    pma.py             # InterFormer
    hsp.py             # Kunlun：分层 seed pooling
    sum_kron_linear.py # Kunlun：Kronecker 分解
  hyper/
    pffn.py            # InterFormer：per-sample 生成 d×d 权重
  compress/
    lce.py             # InterFormer/Kunlun：token 轴压缩
    token_merge.py     # LONGER：L→L/K, d→K·d
    inner_trans.py     # LONGER：组内嵌套 transformer
```

**这里是"共享"的正确位置。** UniRank 的 `MixedFFN` 在 EST 和 OneTrans 里各写一份、`MultiHeadTokenMixing` 在共享层和 HeMix 里两套不同实现——这类重复的根因是没有一个"论文无关的算子库"作为沉淀点。

同时注意：**不是所有相似的东西都该合并。** `sdpa` 和 `silu_attn` 长得像但数学不同，合并成一个带 `activation=` 参数的类是过度抽象。判据：如果两个实现的**论文消融值不同**，它们就是两个东西。

---

## 9. L3 recipes：论文实现层

### 9.1 一篇论文一个自包含目录

```
recipes/est/
    __init__.py      # @register_recipe("est")
    config.py        # ESTConfig dataclass
    spec.py          # required_features()：双序列 + 冻结 content 表 + 预计算 top-K 图
    collate.py       # ESTBatch + ESTCollator（request 分组，B_u 广播）
    model.py         # 双流 block：(N, B) -> (N, B)
    objective.py
    card.py          # FidelityCard
    tests/
        test_lca.py       # 钉住 W_Q/W_O 是 [L_N, d, d] 而非 [d, d]
        test_csa.py       # 钉住 CSA 无参数、不回传梯度、层间复用同一 top-K
        test_decouple.py  # 钉住 B_u 的 K/V 每 request 只算一次
```

### 9.2 两个极端案例的可行性验证

设计的有效性取决于**最不兼容的两个能不能都装下**。

**HyFormer**（三条不对齐的序列 + 逐层重算 K/V）：

```python
def required_features(self) -> FeatureRequest:
    return FeatureRequest(
        scalars=self.ns_fields,
        sequences=(
            SequenceRequest("longterm", self.lt_fields, max_len=3000, layout=PADDED),
            SequenceRequest("search",   self.se_fields, max_len=50,   layout=PADDED),
            SequenceRequest("feed",     self.fe_fields, max_len=50,   layout=PADDED),
        ),
        candidates=self.target_fields,
    )
```

三条序列的 `fields` 各不相同、宽度各不相同（64 / 224 sparse dim），框架不做任何对齐——这正是 HyFormer 论文里 "eliminating the need for forced alignment" 的直接兑现，也正是 UniRank 做不到的。逐层重算 K/V 由 `HyFormerBatch` 携带**原始序列 embedding**、每个 block 自己持有 `W_K^l / W_V^l` 实现，engine 完全不参与。

**ULTRA-HSTU**（全 jagged + candidate 在序列内部 + LBSL sampler）：

```python
@dataclass
class UltraHSTUBatch:
    tokens: NestedTensor          # jagged [B, L_i, d]，item + action 已相加
    action_mask: NestedTensor     # candidate 位置的 action 项置零
    candidate_pos: NestedTensor   # candidate 在序列内的位置索引
    labels: dict[str, NestedTensor]
    def to(self, device): ...
    @property
    def batch_size(self): ...
```

**没有 `target` 字段**——因为这个模型里 candidate 就是序列位置。engine 对此一无所知也不需要知道。LBSL 通过 `NeedsSampler` 挂进去。

如果这两个都装得下，剩下 7 个就都装得下。

### 9.3 Recipe 之间禁止互相 import

想复用就沉进 `nn/`。这条由 CI 的 import-linter 强制。

---

## 10. L4 engine：训练/评测引擎

### 10.1 Trainer：一个具体类，永不继承

```python
class Trainer:
    def __init__(self, loop: LoopPolicy, optim: OptimPolicy,
                 precision: PrecisionPolicy, hooks: Sequence[Hook]): ...
    def run(self, module, objective, train_loader, evaluator) -> RunResult: ...
```

目标规模 ~150 行。对比 UniRank 的 `BaseModel`（fit / train_epoch / train_step / eval_step / evaluate / checkpoint_and_earlystop / 双优化器构建 / torch.compile / DDP gather / AMP，全在一个类里，约 600 行）。

### 10.2 变化点外置为 policy

```python
engine/loops.py
    EpochLoop                 # EST（配合每 epoch 稀疏重置）、InterFormer
    StepLoop                  # 冒烟/扫描
    StreamingNextBatchLoop    # OneTrans：先 eval 后 train 同一 batch，按天宏平均

engine/optim.py
    ParamGroupPolicy          # sparse Adagrad + dense RMSProp/AdamW，分组 clip（90/120）
    AsyncSparsePolicy         # HyFormer/MixFormer：dense 用 g_{k-1}，sparse 用 g_k

engine/precision.py
    UniformPrecision          # bf16 / fp16 / fp32
    PerModulePrecision        # ULTRA-HSTU（BF16/FP8/INT4）、LONGER（按组件）
```

**`AsyncSparsePolicy` 是必须实现项（已决 2026-09-04），不是可选优化。** HyFormer 和 MixFormer 都规定 dense 参数使用 step *k*−1 的梯度而 sparse 参数立即更新（`W_k = W_{k-1} + g_{k-1}` vs `W_k = W_{k-1} + g_k`），即 sparse 比 dense 领先一步。TAAC2026 是 2000w 样本量级，这个 staleness 会真实地作用在收敛轨迹上，不能当作纯吞吐优化而略过——略过就意味着这两个 recipe 的训练动力学与论文不同。

实现上它是 `OptimPolicy` 的一个实现而非 Trainer 的分支：持有上一步的 dense 梯度快照，在 `step()` 里对两个参数组施加不同的更新时序。Trainer 本身不知道有这回事。配套一个数值测试：连续两步中 dense 参数的更新量必须等于第一步的梯度，而 sparse 的等于当步梯度。

### 10.3 Hook

```python
class Hook(Protocol):
    def on_run_start(self, s: TrainState): ...
    def on_epoch_start(self, s: TrainState): ...    # EST 的稀疏参数重置
    def on_step_start(self, s: TrainState): ...
    def on_step_end(self, s: TrainState): ...       # UniMixer 的 tau 退火
    def on_epoch_end(self, s: TrainState): ...
```

内置：`EarlyStopping`、`Checkpoint`、`TensorBoard`、`ThroughputProbe`、`GradNormProbe`。
Recipe 提供：`SparseResetHook`（EST）、`TauAnnealHook`、`LBSLRecalibrateHook`（ULTRA-HSTU）。

**这是 OCP 的检验点**：`UniMixer.py:158-173` 重写 `train_step` 只为做 tau 退火同步——在新架构里这是一个 15 行的 hook，不触碰训练循环。

### 10.4 Evaluator

与 Trainer 正交，独立协议。

```python
engine/evaluate.py
    HoldoutEvaluator          # 标准
    StreamingDailyEvaluator   # OneTrans：按天算指标再宏平均
    GroupedEvaluator          # 分组 AUC，按 user_id 或 query_id
```

指标是**无状态纯函数**（`metrics.py`），不依赖 torch 之外的任何东西。`algo26bench_framework/algobench/metrics.py` 已经是对的，直接迁移。

**已决（2026-09-04）：主指标是 AUC 和 LogLoss。** 排行榜按这两个排序，其余全部降为辅助指标，只在明细里出现。

| 指标 | 地位 | 说明 |
|---|---|---|
| **AUC** | 主 | 全部 recipe、全部数据集 |
| **LogLoss** | 主 | 全部 recipe、全部数据集 |
| GAUC / UAUC | 辅助 | 按 `row_meta.user_id` 分组；数据集有分组键时才算 |
| NE | 辅助 | Kunlun / ULTRA-HSTU 的论文主指标，保留以便对照原文 |
| Macro-AUC | 辅助 | 多任务时的跨任务平均 |

这条决定顺带解掉了原来的一个隐患：NE 依赖"训练集平均正例率"作为基线，跨数据集比较 NE 需要统一基线口径，而 LogLoss 没有这个问题。把 NE 降为辅助指标后，基线口径只需在同一数据集内部保持一致即可，`metrics.normalized_entropy` 显式接收基线参数而非内部推断。

多任务时（KuaiRand 多标签），AUC/LogLoss **逐任务报，不做隐式平均**——平均值放在 Macro-AUC 列里，但排序看具体任务。TAAC2026 只有 pCVR 单任务，不涉及。

### 10.5 Profiler

Kunlun 的核心主张是 MFU，只报 AUC 的 harness 表达不了它。

```python
engine/profile.py
    count_params(module)       -> (dense, sparse)
    count_flops(module, batch) -> float          # 记录测量时的 batch size
    measure_mfu(...)           -> float
    measure_latency(...)       -> LatencyStats   # p50/p99
    measure_peak_memory(...)   -> int
```

FLOPs 必须**连同 batch size 一起记录**——各论文报的 batch size 不同（HyFormer/InterFormer/OneTrans 2048，MixFormer 1500，EST 1000/PPU），脱离 batch size 的 FLOPs 数字不可比。

---

## 11. L5 bench：实验编排与报告

```
bench/
    run.py         # CLI：单次实验
    sweep.py       # 扫描（替代 UniRank 的 autotuner，去掉硬编码 python 路径）
    report.py      # 排行榜 + 保真度卡片渲染
    tracks.py      # 赛道定义
    configs/
```

报告的每一行至少包含：

| 列 | 来源 |
|---|---|
| **AUC**（主，逐任务） | Evaluator |
| **LogLoss**（主，逐任务） | Evaluator |
| 参数：**dense（对齐轴）** / sparse 分列 | Profiler（EST 的 "Params(M)" 是只算 dense 的） |
| 辅助质量：GAUC / NE / Macro-AUC | Evaluator |
| FLOPs @ batch_size | Profiler |
| MFU | Profiler（Kunlun 的核心变量） |
| 训练 step 时延 / 峰值显存 | Profiler |
| 推理 p50/p99（若实现 `CachedScoring` 则含 cached 路径） | Profiler |
| **保真度**：faithful / approximated / omitted 计数 | FidelityCard |
| **能力标注**：序列长度是否达到该模型的有意义区间 | Capability |

排序按主指标（AUC / LogLoss），对齐轴是 dense 参数量。

最后两列是这个 benchmark 与 UniRank 的本质区别。

---

## 12. 保真度治理机制

这是直接回应"牺牲原本模型特性不可接受"的部分。

**光靠架构挡不住。** `EST.py:161` 那行注释是人写的，任何架构都拦不住人写。所以要把保真度**变成可测的、会出现在排行榜上的东西**。

### 12.1 FidelityCard

每个 recipe 必须声明论文的关键机制及其状态。

```python
class Status(Enum):
    FAITHFUL     = "faithful"
    APPROXIMATED = "approximated"
    OMITTED      = "omitted"

@dataclass(frozen=True)
class Mechanism:
    description: str
    status: Status
    paper_ref: str                # 论文里的公式号/表号
    pinned_by: str | None = None  # FAITHFUL 必填：钉住它的测试
    reason: str | None = None     # 非 FAITHFUL 必填
    measured_impact: str | None = None   # 论文消融值

@dataclass(frozen=True)
class FidelityCard:
    paper: str
    arxiv: str
    mechanisms: tuple[Mechanism, ...]
```

示例：

```python
FidelityCard(paper="EST", arxiv="2602.10811v1", mechanisms=(
    Mechanism(
        "LCA: token-specific W_Q/W_O of shape [L_N, d, d]，W_K/W_V 跨行为 token 共享",
        Status.FAITHFUL, paper_ref="Eq.4",
        pinned_by="recipes/est/tests/test_lca.py::test_per_token_weight_shape"),
    Mechanism(
        "CSA: 无参数 top-K 注意力，权重来自冻结 content embedding 的 Gram 矩阵",
        Status.OMITTED, paper_ref="Alg.1",
        reason="已决 2026-09-04：首轮不引入多模态 content embedding。"
               "TAAC2026 与 KuaiRand 均无此字段；Taobao-mm 可能有，待评估后再议。",
        measured_impact="论文消融：+0.14% GAUC @ +0.12% FLOPs"),
    Mechanism(
        "每 epoch 稀疏参数重置回 θ_s^(0)，dense 继承",
        Status.FAITHFUL, paper_ref="§4.1",
        pinned_by="recipes/est/tests/test_sparse_reset.py"),
))
```

**CI 规则：**

1. `FAITHFUL` 但 `pinned_by` 为空 → 拒绝
2. `pinned_by` 指向的测试不存在或不通过 → 拒绝
3. `OMITTED` / `APPROXIMATED` 但 `reason` 为空 → 拒绝
4. 排行榜自动渲染这张表；`OMITTED` 的模型在报告里带显式标记

这样，"一行注释删掉核心贡献"这件事仍然可以做——有时确实必须做（公开数据集就是没有 content embedding）——但**它会出现在报告里**，读者知道自己在看什么。

### 12.2 机制单测

`algo26bench_framework/tests/test_layers.py` 已经做对了一半：它把 RankMixer 的 rewire 钉成精确的置换、验证 PerTokenFFN 各 token 的 bias 相互独立、验证全 mask 的 key 输出为零。这个模式推广到每个 distinguishing mechanism。

优先级最高的几个（都是"静默出错"风险，不是"报错"风险）：

| 测试 | 防的错 |
|---|---|
| LONGER 反向因果 | 序列是 latest→earliest，mask 是 `j ≥ i`；用 `torch.tril` 会**静默**变成完全相反的语义，且 KV cache 复用随之非法 |
| 掩码语义 | legacy `True=padding` vs 新 `True=valid`；转错不报错，只是 AUC 低一点 |
| padding 不变性 | 给 batch 加 padding，logits 不变。一条测试同时抓住 UniRank 那 5 处 mask-free SDPA |
| TokenFormer position id | 静态字段必须全为 `p=0`（这样 `R_{0-0}=I`，字段间退化为纯语义点积），target 为 `p=SL+1` |
| MixFormer UI mask | `M[i,j]` user→item 单向；验证 user 侧 head 的输出与 candidate 无关 |
| EST per-token 权重 | `W_Q.shape == (L_N, d, d)`，不是 `(d, d)` |
| ULTRA-HSTU action 屏蔽 | candidate 位置的 action embedding 必须为零（否则标签泄漏） |
| Kunlun CompSkip | 奇层确实消费了前层的 HSP 输出，而不是重算 |
| 异步稀疏更新时序 | dense 用 g_{k-1} / sparse 用 g_k；连续两步的更新量必须能区分出这一步差 |
| 任务集合适配 | 同一 recipe 在单任务与多任务 `TaskSet` 下都能建出正确的 head 数，且无代码分支 |
| reference vs fused 对拍 | `HasFusedKernel` 的两条路径数值一致（首轮无实现者，测试待启用） |

### 12.3 Capability 声明

不是所有模型在所有序列长度下都有意义。

```python
@dataclass(frozen=True)
class Capability:
    min_meaningful_seq_len: int = 0
    needs_multi_sequence: bool = False       # HyFormer / InterFormer / ULTRA-HSTU(MoT)
    needs_raw_timestamp: bool = False        # OneTrans 交错、Kunlun ROTE、LONGER 时间差
    needs_semantic_groups: bool = False      # HyFormer 的 semantic grouping
    needs_action_axis: bool = False          # ULTRA-HSTU item+action、TokenFormer 双 token
    needs_content_embedding: bool = False    # EST 的 CSA
    needs_request_grouping: bool = False     # RLB / KV cache
    supports_multitask: bool = True
```

LONGER 声明 `min_meaningful_seq_len=2000`，ULTRA-HSTU 声明 `3072`，MixFormer 声明 `512`。在 KuaiRand（长度 256）上跑时，harness **在报告里标注**"该配置低于模型的有意义区间"，而不是假装可比。

这一点有论文依据：ULTRA-HSTU 自己就点名 STCA 在 KuaiRand 短序列上因为昂贵的 pre-attention 投影而吃亏。

`Capability` 与 `DataConfig` 的交叉校验在 `ExperimentConfig.validate()` 里做，产出的不是异常而是**报告注记**——因为"能力不匹配但仍然跑一下"往往是有意义的（比如想看 LONGER 在短序列下退化到什么程度）。审计后新增的四个字段（`needs_raw_timestamp` / `needs_semantic_groups` / `needs_action_axis` / `needs_content_embedding`）让 §14.3.1 的结论自动传导到报告里：例如 OneTrans 跑在 QK_Video 上会被自动标注"缺原始时间戳，timestamp-aware fusion 已降级"，无需人工维护一张兼容性表格。

---

## 13. 配置系统

### 13.1 问题

UniRank 的 `model_config.yaml` 是 2293 行 / 136 个 experiment 条目，模型架构超参和训练超参混在一起，同一组超参按 模型×数据集 的笛卡尔积重复，`feature_embedding.py:119` 用 `eval()` 解析。

`algo26bench_framework` 的方向对（dataclass + 严格拒绝未知 key），但 `ExperimentConfig.model` 静态绑死 `HyFormerConfig`。

### 13.2 方案

四段式，各自独立、正交组合：

```
configs/
  data/       taac2026.yaml, kuairand.yaml, kuairand_click_only.yaml,
              taobao_ad.yaml, synthetic_len8k.yaml       # 每份含 task_set 定义
  recipe/     est.yaml, hyformer.yaml, longer.yaml, ...  # 只有架构超参
  training/   epoch_standard.yaml, streaming_daily.yaml, smoke.yaml
  experiment/ est@taac2026.yaml                          # 组合三者 + 少量覆盖
```

注意 `data/` 下的每份配置都携带 `task_set`——`taac2026.yaml` 是单 pCVR，`kuairand.yaml` 是多标签，`kuairand_click_only.yaml` 是同一份数据收窄到单任务。**"KuaiRand 跑单标签"因此是一份数据配置，不是代码里的开关。**

模型配置用**按 `name` 判别的联合**，每个 recipe 自带 dataclass：

```python
@dataclass
class ExperimentConfig:
    data: DataConfig
    recipe: RecipeConfig     # 判别联合：由 recipe.name 决定具体类型
    training: TrainingConfig
    def validate(self) -> None: ...   # 跨段校验，如 capability vs data.max_seq_len
```

三条硬规则：

1. 未知 key **拒绝**（沿用 `algo26bench_framework`，抓 typo 极有效）
2. **禁止 `eval()`**
3. recipe 配置**只放架构超参**；`batch_size` / `epochs` / `lr` 属于 training 段

---

## 14. 评测赛道设计

### 14.1 为什么必须分赛道

公开数据集的序列长度是 50–256，而 LONGER（2000–10000）、MixFormer（512–10000）、ULTRA-HSTU（3072–16384）、OneTrans（~1500）的核心机制在长度 100 下基本是 no-op 甚至负优化。在 KuaiRand 上跑一个 LONGER 然后说"它不如 X"，是没有意义的结论。

同时，9 篇里有 3 篇（EST、LONGER、TokenFormer）**头条效率数字是在另一张计算图上测的**——把 serving 路径和训练路径混为一谈也测不出它们在讲什么。

### 14.2 三条赛道

**Track A — 质量（可比性优先）**

- 数据：**TAAC2026（2000w，单 pCVR）为主**，KuaiRand（多标签，也可配成单标签）为公开对照，其余公开集（KuaiVideo / TaobaoAd / Amazon-Electronics）按需
- 全部 9 个 recipe
- **主指标：AUC、LogLoss**；GAUC / NE / Macro-AUC 为辅助列
- 训练预算按**等参数量**对齐（见 §14.4）
- 每行带 Capability 标注

**Track B — 效率（保真度优先）**

- 数据：合成 + 内部长序列，长度扫到各模型的**原生区间**（512 / 2k / 8k / 16k）
- 指标：FLOPs @ batch_size、MFU、step 时延、峰值显存、推理 p50/p99
- 实现 `CachedScoring` 的额外跑 cached serving 路径（这是 EST / LONGER / OneTrans / TokenFormer / MixFormer 头条数字的所在）
- `HasFusedKernel` 的分 reference / fused 两行（首轮无实现者，Kunlun 的 MFU 列带 `OMITTED` 标记）

**Track C — Scaling**

- 拟合 `Δmetric = α·C^β`，报告 α（截距）和 β（指数）
- 至少 4 个 compute 点
- 这是 HyFormer / EST / LONGER / Kunlun / ULTRA-HSTU 共同的论证形式，单点结果表达不了

### 14.3 数据集覆盖现状

| 数据集 | 序列结构 | 任务集合（由数据集定义） | 原始时间戳 | 语义特征名 | 覆盖论文的原生设定 |
|---|---|---|---|---|---|
| **TAAC2026 / academic_2000w（主）** | **4 序列域，宽度不齐（9/14/12/10 字段），长度 256/256/512/512** | 单任务 pCVR | ✅ 逐域 `ts_fid` | ❌ 匿名 fid | **HyFormer 多序列（唯一）**、MixFormer UI-decoupling |
| KuaiRand | 单序列 + action(65) | 6 任务，可配置收窄 | ✅ `full_timestamp_seq` | ✅ | ULTRA-HSTU、TokenFormer、HyFormer 语义分组 |
| **TencentGR_10M** | 单序列 + action(4) | **2 任务：is_click / is_conversion** | 待查 | 部分（fid 编号） | **OneTrans 的原生 CTR/CVR** |
| Taobao | 单序列 + action(17) | 4 任务 | ✅ | ✅ | InterFormer |
| MerRec | 单序列 + action(33) | 5 任务 | ✅ | ✅ | — |
| TAAC2025 | 单序列 | 待查 | ✅ | 待查 | — |
| QK_Video | 单序列 + action(17) | 4 任务 | ❌ **唯一没有** | ✅ | — |
| Taobao-mm | 待评估 | 待评估 | 待评估 | 待评估 | **可能含多模态 content embedding** —— EST 的 CSA 唯一潜在载体 |
| 合成长序列 | 可配 | 可配 | 可配 | — | Track B/C 唯一可行的载体（长度 2k–16k） |

审计依据与逐条含义见 §14.3.1。三条要点：**TAAC2026 是唯一能验证多序列分歧的数据集**（公开集全是单序列）；**公开数据集是唯一能验证 item/action 双 token 机制的**（TAAC2026 没有独立 action 轴）；**HyFormer 的语义分组只能在有语义特征名的公开集上测**。三者互补，缺一不可——这是不局限于单一数据集的具体理由。

**已决（2026-09-04）：任务集合由数据集给定，recipe 适配之**（机制见 §6.1.1）。所以 MixFormer（原生 Finish/Skip）、OneTrans（原生 CTR/CVR）、ULTRA-HSTU（原生 consumption/engagement）在 TAAC2026 上都退化成单 pCVR 头，在 KuaiRand 上按配置展开成多头。这是一行配置，不是代码分支。

唯一需要在 `FidelityCard` 里显式标注的是 TokenFormer：它的原生形态是多动作 softmax CE，在只有 BINARY 任务的数据集上会降级为单 binary head，标 `APPROXIMATED`。

### 14.3.1 Schema 审计结果（2026-09-04 实查）

对 TAAC2026 与 UniRank 的 6 个数据集做了 schema 实查，结论与设计初稿的假设**相反**，需要记录。

#### 结论 1：逐行为原始时间戳普遍存在，但被两个框架的 dataloader 各自丢掉了

| 数据集 | 磁盘上有原始逐行为时间戳 | 当前 loader 暴露的形态 |
|---|---|---|
| **TAAC2026** | ✅ 4 个序列域各有 `ts_fid`（seq_a=39, seq_b=67, seq_c=27, seq_d=26）+ 行级 `timestamp` 列 | ❌ 仅 `{domain}_time_bucket`，65 桶 |
| KuaiRand | ✅ `full_timestamp_seq`（raw `time_ms`，按时间排序） | ❌ 完全不读 |
| Taobao | ✅ 同上 | ❌ 完全不读 |
| MerRec | ✅ 同上 | ❌ 完全不读 |
| TAAC2025 | ✅ 同上 | ❌ 完全不读 |
| QK_Video | ❌ 预处理只写 `full_item_seq` / `full_action_seq` | — |

两处丢失点都是纯粹的 dataloader 层信息销毁，与数据本身无关：

- `algo26bench/data/dataset.py:774-792` 先算 `time_diff = row_ts - seq_ts`，再用 64 个边界桶化成 65 个 id，**只把 `{domain}_time_bucket` 放进 batch**，原始 `ts` 就地丢弃
- `UniRank/unirank/pytorch/dataloaders/unirank_dataloader.py:417` 的 `need_user_cols` 硬编码为 `["user_index", "full_item_seq", "full_action_seq"]`——`full_timestamp_seq` 明明就在同一个 parquet 里，从来没被读过

这两处正是本设计反复批评的"数据层替模型做了决定"的教科书案例，而且是本次审计里**唯一一个两套框架都犯的同一个错**。

**设计含义**（已并入 §6.2 / §7.3）：`RawBatch.timestamps` 必须携带**原始时间戳**，桶化下沉为 `data/view.py` 的一个原语（`bucketize_time_delta`，沿用现有的 64 个边界以保持与 legacy 可对照），由 recipe 自行选用。于是：

- OneTrans 的 timestamp-aware 跨序列交错**可以原生实现**（论文实测比 timestamp-agnostic 好 0.09% CTR AUC / 0.22% UAUC），不必降级
- Kunlun 的 ROTE 需要 `log(1 + Δt/τ_scale)` 的连续值，桶化后无法还原——现在可以实现
- LONGER 的"绝对时间差作为 side info 拼接"（改变输入宽度）与"可学习绝对位置相加"这两套并存的位置机制，也都可以实现
- 只有 QK_Video 上这些机制必须降级，`Capability` 里加一条 `needs_raw_timestamp` 即可自动标注

#### 结论 2：TAAC2026 的匿名化影响范围比预期小得多

特征确实只有 fid 编号（1–132），没有语义名。但结构分区**是保留的**：`user_int`（54 个）/ `item_int`（17 个）/ `user_dense`（17 个）/ `item_dense`（4 个）四分区清晰。所以：

| 机制 | 在 TAAC2026 上是否可行 | 说明 |
|---|---|---|
| MixFormer 的 UI-decoupling | ✅ **可行** | 它需要的正是 user/item 二分并保证 user 侧连续在前，四分区足够 |
| OneTrans / MixFormer 的 Auto-Split | ✅ 可行 | 本来就不依赖语义 |
| EST 的"每原始特征一个 token" | ✅ 可行 | 依赖的是特征边界，不是特征名 |
| TokenFormer 的字段前缀 token | ✅ 可行 | 同上 |
| **HyFormer 的 semantic grouping** | ❌ **不可行** | 这是匿名化的**唯一**真实受害者 |
| LONGER 的 global token 含"高阶交叉特征" | ⚠️ 部分 | 无法按语义挑选，只能取 target + CLS + UID |

HyFormer 在 TAAC2026 上只能退化为 auto-split，标 `APPROXIMATED`。值得注意的是这个降级的代价有上界：**OneTrans 实测 auto-split 比 group-wise 好 0.10% CTR AUC**，方向与 HyFormer 的选择相反。所以这不是"把 HyFormer 弄残了"，而是"§3.1 冲突 1 在这个数据集上只能取一侧"——而 KuaiRand 等公开数据集有语义特征名，可以两侧都测。**这正是不局限于单一数据集的价值所在。**

#### 结论 3：TAAC2026 的 4 个序列域天然宽度不齐，恰好是 HyFormer 的原生设定

| 序列域 | 字段数 | ts_fid |
|---|---|---|
| seq_a | 9 | 39 |
| seq_b | 14 | 67 |
| seq_c | 12 | 27 |
| seq_d | 10 | 26 |

四条序列的 side-info schema 和宽度各不相同，**这正是 HyFormer 论文强调的 "eliminating the need for forced alignment of side information or sparse dimensions across sequences"**，也正是 UniRank 的单一 `item_info_dim` 装不下的东西。TAAC2026 因此是 9 篇论文里最适合验证多序列分歧（§3.1 冲突 2）的数据集，而公开数据集全是单序列。

反过来，公开数据集清一色是"单序列 + `action` 字段"（action vocab 分别为 KuaiRand 65 / Taobao 17 / MerRec 33 / QK_Video 17 / TencentGR 4）。ULTRA-HSTU 的 item+action 相加、TokenFormer 的 item/action 交错双 token，在这些数据集上都能原生实现——而在 TAAC2026 上没有独立的 action 轴。**两类数据集覆盖的机制是互补的，不能只留一个。**

#### 结论 4：任务集合实查

| 数据集 | 任务数 | 标签 |
|---|---|---|
| TAAC2026 | 1 | pCVR |
| KuaiRand | 6 | is_click / is_follow / is_like / is_comment / is_forward / long_view |
| MerRec | 5 | Like / Cart / Offer / Checkout / Purchase |
| Taobao | 4 | is_click / cart / fav / buy |
| QK_Video | 4 | click / follow / like / share |
| **TencentGR_10M** | 2 | **is_click / is_conversion** |

`TencentGR_10M_Action` 的 (is_click, is_conversion) **正好是 OneTrans 论文的原生任务组合（CTR / CVR）**，此前没注意到。它应当作为 OneTrans 的首选对照数据集加入 Track A。

### 14.4 公平口径：等参数量

**已决（2026-09-04）：Track A 的主表按等参数量对齐。**

各论文自己用的口径并不一致（HyFormer 等参数量、EST 等 dense 参数量、Kunlun 等 GFLOPs），选参数量的理由是它最容易控制且与实现质量无关——等 wall-clock 会奖励"kernel 写得好"，等 FLOPs 会因各家 FLOPs 统计口径不同而扯皮，而参数量是确定性的。

两条配套规则：

1. **dense / sparse 分开计数并分开对齐。** embedding 表的规模由数据集词表决定，各 recipe 之间差异不大且不是模型贡献；真正要对齐的是 dense 参数量。这也与 EST 论文的 "Params (M)" 口径一致（它只算 dense）。
2. **FLOPs、MFU、wall-clock 仍然照常测量并出现在报告里**，只是不作为对齐轴。等参数量下 FLOPs 的差异本身就是结论的一部分——这正是 HyFormer（3.9 TFLOPs）对比 MTGR/OneTrans（21.9 TFLOPs）那张表在讲的事。

超参搜索阶段可以再引入等 FLOPs 的对照表，作为主表的附录。

---

## 15. SOLID / KISS 自检

### SOLID

| 原则 | 具体做法 | 修掉的具体病灶 |
|---|---|---|
| **SRP** | Trainer 只管步进；Evaluator 只管打分；Collator 只管建 batch；metrics 是无状态纯函数；Profiler 只管量 | `rank_model.py` 的 `BaseModel` 一个类扛 8 项职责 |
| **OCP** | 加论文 = 加目录 + 一行 register，engine/data/core 零改动；变化点是 policy 和 hook | `UniMixer.py:158` 重写 `train_step` 做 tau 退火；`train.py:95` 硬编码 `build_hyformer` |
| **LSP** | **不设模型基类**，用结构化 Protocol——没有基类就没有 LSP 可违反 | 15 个模型被塞进 `MultiTaskModel`，保真度死在这里 |
| **ISP** | `CachedScoring` / `NeedsSampler` / `HasParamGroups` / `HasFusedKernel` 是独立可选协议 | `multi_masks` 每 batch 都算但只有 INFNet 用（`unirank_dataloader.py:539`） |
| **DIP** | engine 依赖 `Recipe[TBatch]`，TBatch 不透明；data 依赖 `FeatureRequest`，不认识任何模型 | dataloader 把 target 放最后一位——数据层替模型做了决定 |

### KISS

| 做法 | 对照 |
|---|---|
| core 层无逻辑，只有类型，≤400 行 | — |
| Trainer 是一个具体类，永不继承 | `BaseModel` ~600 行 + 各模型重写 |
| 一个 registry，不是四个 | — |
| 配置：per-recipe dataclass + 判别联合 | 2293 行单体 YAML / 136 条目 |
| 禁止 `eval()` | `feature_embedding.py:119` |
| **取消"设计一个通用 batch 结构"这个需求本身** | `(batch_dict, item_dict, mask)` + 15 处 hack |
| 相似但消融值不同的算子**不合并** | 防止过度抽象——这是 UniRank 的反方向陷阱 |

最后一条值得强调：KISS 的失败模式有两个方向。UniRank 是"抽象过度"（强行统一），但矫枉过正会变成"抽象不足"（9 份完全独立的代码，无法比较）。判据是：**共享发生在 `nn/` 算子层（论文无关的数学），不发生在 `recipes/` 结构层（论文特有的组织方式）。**

---

## 16. 目录结构

```
algo26bench_v2/
├── DESIGN.md
├── pyproject.toml
├── .importlinter                    # 强制层间依赖方向
│
├── core/                            # L0，≤400 行，无 torch 逻辑
│   ├── types.py                     # FieldSpec / SequenceRequest / FeatureRequest / RawBatch / TaskSpec / TaskSet
│   ├── protocols.py                 # Recipe / Collator / Objective / Hook / LoopPolicy
│   ├── capabilities.py              # CachedScoring / NeedsSampler / HasParamGroups / HasFusedKernel
│   ├── fidelity.py                  # FidelityCard / Mechanism / Status / Capability
│   └── registry.py
│
├── data/                            # L1
│   ├── schema.py
│   ├── source/                      # parquet_blocked, kuairand, taobao_ad, amazon, synthetic, taac2025
│   ├── sampler/                     # sharded, request_grouped, stochastic_length, load_balanced
│   ├── view.py                      # pad_to / to_jagged / interleave_by_time / merge_tokens / ...
│   └── standard_collator.py         # 可选默认值，非基类
│
├── nn/                              # L2，可独立 import 的普通库
│   ├── attention/  {sdpa, silu_attn, gdpa, cross_causal, lightweight_cross, mask/}
│   ├── mixing/     {head_mixing, rank_mixer, per_token_ffn, token_mix}
│   ├── tokenize/   {auto_split, group_wise, field_wise, chunk}
│   ├── pos/        {rope, rote, learned_absolute, time_delta_sideinfo}
│   ├── pool/       {pma, hsp, sum_kron_linear}
│   ├── hyper/      {pffn}
│   └── compress/   {lce, token_merge, inner_trans}
│
├── recipes/                         # L3，一篇论文一个自包含包
│   ├── hyformer/   {config, spec, collate, model, objective, card, tests/}
│   ├── interformer/
│   ├── mixformer/
│   ├── onetrans/
│   ├── ultrahstu/
│   ├── est/
│   ├── kunlun/
│   ├── longer/
│   └── tokenformer/
│
├── engine/                          # L4，仅依赖 core
│   ├── trainer.py                   # ~150 行，永不继承
│   ├── loops.py                     # Epoch / Step / StreamingNextBatch
│   ├── optim.py                     # ParamGroupPolicy / AsyncSparsePolicy
│   ├── precision.py
│   ├── distributed.py
│   ├── hooks/                       # early_stop, checkpoint, tensorboard, probes
│   ├── evaluate.py                  # Holdout / StreamingDaily / Grouped
│   ├── metrics.py                   # 纯函数：auc, gauc, normalized_entropy, logloss, macro_auc
│   └── profile.py                   # params / flops / mfu / latency / memory
│
├── bench/                           # L5，唯一可 import recipes 的层
│   ├── run.py, sweep.py, report.py, tracks.py
│   └── configs/  {data/, recipe/, training/, experiment/}
│
└── tests/
    ├── contract/                    # 契约测试：任何 recipe 都必须通过
    │   ├── test_padding_invariance.py    # 一条测试抓住所有 mask-free 问题
    │   ├── test_mask_semantics.py
    │   ├── test_determinism.py
    │   └── test_fidelity_card.py         # CI 规则 1-3
    └── layers/                      # nn/ 的算子级测试
```

---

## 17. 迁移与实施路径

### 阶段 0：地基（无模型）

- `core/` 全部契约
- `engine/trainer.py` + `EpochLoop` + `ParamGroupPolicy` + `metrics.py`
- `data/` 的 PCVR adapter（复用 `algo26bench_framework/algobench/data/legacy_pcvr.py`）+ KuaiRand + synthetic
- `tests/contract/` 全部
- 从 `algo26bench_framework` 迁移：`metrics.py`、`core/schema.py` 的掩码语义、`tests/test_layers.py` 的 RankMixer 置换测试

**退出条件**：synthetic 数据上跑通一个 trivial recipe（LR + pooling），契约测试全绿。

### 阶段 1：可行性验证（最不兼容的两个）

- `recipes/hyformer/`（三条不对齐序列 + 逐层重算 K/V）
- `recipes/ultrahstu/`（全 jagged + candidate 在序列内 + LBSL sampler）

**退出条件**：两者都跑通，且 `engine/` 和 `core/` 在实现过程中**零改动**。如果需要改，说明窄腰设计有问题，回到阶段 0 重新设计——**这是整个计划里最重要的一个检查点**。

### 阶段 2：单流派系

- `recipes/onetrans/`（pyramid + KV cache + StreamingNextBatchLoop）
- `recipes/tokenformer/`（per-layer mask + 双数据组织 + 多动作 CE）
- `recipes/longer/`（token merge + 反向因果 + KV cache serving）

这批集中验证 `LoopPolicy` 和 `CachedScoring` 的设计。

### 阶段 3：双流派系

- `recipes/est/`、`recipes/mixformer/`、`recipes/interformer/`、`recipes/kunlun/`

这批集中验证 `nn/` 算子库的沉淀是否到位——EST 与 OneTrans 的 `MixedFFN`、EST 与 InterFormer 的 cross-attention 变体，应该在这个阶段合并进 `nn/`，且**合并前先确认它们的消融值是同一件事**。

两处已决的范围裁剪：

- **Kunlun 的 Triton fused kernel 不实现**（已决 2026-09-04）。GDPA / HSP / SWA 的**数学**全部按论文实现，只是走 PyTorch reference 路径。后果是 Kunlun 的核心主张（MFU 17% → 37%、PFFN block 6× MFU）无法复现，必须在 `FidelityCard` 里标 `OMITTED` 并写明原因，同时该 recipe 在 Track B 的 MFU 列带显式标记。相应地 `HasFusedKernel` 协议先只保留定义，暂无实现者——它仍然值得留在 core 里，因为 ULTRA-HSTU 的自定义 attention kernel 后续可能用到。
- **EST 的 CSA 不实现**（已决 2026-09-04）。首轮不引入多模态 content embedding，EST 只实现 LCA + per-token FFN + 每 epoch 稀疏重置三项。标 `OMITTED`，理由写在卡片里。若后续确认 Taobao-mm 带 content embedding，再作为增量补上。

### 阶段 4：赛道与报告

- `bench/tracks.py` + `report.py`
- Track B 的 Profiler 全套（MFU / cached serving 路径）
- Track C 的 scaling 拟合

### 关于 UniRank 的 15 个 model_zoo

不做机械迁移。其中 RankMixer、HiFormer、Zenith、SSR、HeMix、TokenMixer、UniMixer、INFNet 不在 `refer/` 的 9 篇范围内，属于另一批工作。若要接入，走同样的 recipe 流程，**且必须补 FidelityCard**——迁移的价值恰恰在于这个过程会逼出"当初到底删了什么"。

---

## 18. 决策记录与剩余未决

### 18.1 已决（2026-09-04）

| # | 议题 | 决定 | 落点 |
|---|---|---|---|
| 1 | 多任务标签口径 | **任务集合由数据集定义，recipe 适配之**。TAAC2026 单 pCVR，KuaiRand 多标签且可配置收窄为单标签 | §6.1.1、§14.3 |
| 2 | 主指标 | **AUC + LogLoss**。GAUC / NE / Macro-AUC 降为辅助列 | §10.4、§11、§14.2 |
| 3 | 异步稀疏更新 | **必须实现**。TAAC2026 是 2000w 样本量级，staleness 会真实作用在收敛轨迹上，不是纯吞吐优化 | §10.2 |
| 4 | Kunlun Triton kernel | **不实现**。数学按论文实现走 PyTorch reference，MFU 主张标 `OMITTED` 并在报告里显式标记；其余机制正常 | §17 阶段 3 |
| 5 | EST content embedding | **首轮不引入多模态**。CSA 标 `OMITTED`；Taobao-mm 可能带此字段，待评估 | §12.1、§17 阶段 3 |
| 6 | RawBatch 张量类型 | **一律 torch**。仅当阶段 1 用 ULTRA-HSTU 实测出 numpy 在 collate 路径有显著优势时，才局部退回并注明实测数据 | §6.2 |
| 7 | 训练预算公平口径 | **等参数量（dense 为对齐轴，sparse 分列）**。FLOPs / MFU / wall-clock 照常测量但不作为对齐轴；超参搜索阶段再补等 FLOPs 附表 | §14.4 |

三条决定的连带效果值得记一笔：

- 决策 1 让 `FeatureRequest` 里的 `labels` 字段消失，`Recipe` 的接口反而更窄了——这是往 KISS 的方向走。
- 决策 2 顺带解掉了原第 2 号未决（NE 的 background entropy 基线跨数据集不可比）：NE 降为辅助指标后，基线只需在同一数据集内一致，`metrics.normalized_entropy` 显式接收基线参数而非内部推断。
- 决策 4 和 5 都是**范围裁剪而非质量妥协**——两者都通过 `FidelityCard` 的 `OMITTED` 状态显式暴露在报告里，读者知道 Kunlun 那一行的 MFU 不代表论文主张。这正是 §12 那套机制存在的意义。

### 18.2 已由 schema 审计解决（2026-09-04，详见 §14.3.1）

原第 2 号未决（TAAC2026 是否保留逐行为时间戳）**已确认为"有"**，且发现范围远大于预期：TAAC2026 的 4 个序列域各有 `ts_fid`，KuaiRand / Taobao / MerRec / TAAC2025 也都在 parquet 里存了 `full_timestamp_seq`，只有 QK_Video 没有。两套现有框架的 dataloader 各自把它丢掉了（`algo26bench/data/dataset.py:774-792` 桶化后丢弃原值；`UniRank/.../unirank_dataloader.py:417` 的 `need_user_cols` 从来没读过这一列）。

因此 OneTrans 的 timestamp-aware 交错、Kunlun 的 ROTE、LONGER 的时间差 side-info **都不需要降级**，只要 `RawBatch.timestamps` 携带原值、桶化下沉为可选原语即可。这条从"阶段 2 前置风险"变成了"阶段 0 的一个明确需求"。

同时确认匿名化的唯一真实受害者是 **HyFormer 的 semantic grouping**（TAAC2026 上退化为 auto-split，标 `APPROXIMATED`），而 MixFormer 的 UI-decoupling 因为 `user_int` / `item_int` 四分区存在而**不受影响**。

### 18.3 剩余未决

1. **Taobao-mm 是否真的带 content embedding，以及字段形态**。这决定 EST 的 CSA 能否作为增量补回。需要先做一次数据探查（字段名、维度 `d_M`、覆盖率）再判断，不急于阶段 3 之前。

2. **TencentGR_10M 与 TAAC2025 的补充探查**。TencentGR_10M 的 (is_click, is_conversion) 正好是 OneTrans 的原生 CTR/CVR 组合，应加入 Track A，但它的原始时间戳情况尚未确认（其余 4 个公开集都有）。TAAC2025 的任务集合与语义特征名也未查。两者都是十几分钟的探查量，建议与阶段 0 的 data adapter 一并做掉。

3. **TAAC2026 最长 512 的序列域对长序列 recipe 的约束**。LONGER（原生 2000–10000）、ULTRA-HSTU（3072–16384）在 512 长度下的 Capability 会一直标"低于有意义区间"。Track B/C 靠合成数据补，但 Track A 主表上这两行的解释力有限——是否在主表额外加注，还是接受"公开可比性优先"？审计后这个问题更尖锐了：**没有任何一个现有数据集的序列长度达到 2000**，所以这两个 recipe 的原生优势在 Track A 上完全无法体现。

4. **等参数量对齐的具体操作方式**。dense 参数量对齐到哪个档位（比如 100M，与 EST 论文一致），以及各 recipe 通过调哪个超参去凑（层数 vs 宽度）。各论文结论不一致——EST 实测**深度比宽度更 scale**（depth `0.46·P^0.14` vs width `0.63·P^0.08`），OneTrans 也说深度优于宽度但深度有串行延迟代价。倾向统一按调宽度对齐、深度固定在各论文默认值，但需要在阶段 4 用实验确认这不会系统性偏袒某一方。
