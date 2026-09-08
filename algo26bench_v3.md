# algo26bench v3 —— 更简的学术复现框架

> **v3 相对 v2 的立场**：v2 的窄腰（`Recipe[TBatch]`）思路是对的，但把 DESIGN.md 里的每一条抽象都当成了必须落地的需求，结果长出了一批**为工业场景准备但学术复现用不到**的接口（`Capability` 的 7 个布尔位、`FidelityStatus` 的 4 个状态、`LoopPolicy`/`OptimPolicy`/`Hook` 的三条独立协议、`HasFusedKernel` / `CachedScoring` / `NeedsSampler` 的可选协议、`RawBatch` 这个中间层……）。v3 的目标是：**只保留能落到 9 篇论文代码里的抽象，删掉其余**，并且把 v2 里为了 ULTRA-HSTU / EST / Kunlun 挖的但目前还没填的坑一次性填掉。
>
> **范围**：`refer/` 里的 9 篇论文（HyFormer / OneTrans / MixFormer / InterFormer / LONGER / TokenFormer / ULTRA-HSTU / EST / Kunlun），加公开 pCVR 数据集（TAAC2026 academic_2000w）。不涉及 serving、MFU 主张、Triton kernel、跨节点分布式。
>
> **写作时间**：2026-09-06。基于对 v2 全部代码的实际审计（不是照抄 DESIGN.md）。

---

## 目录

1. [v2 现状盘点：什么是对的，什么该删](#1-v2-现状盘点)
2. [v3 的四条硬约束](#2-v3-的四条硬约束)
3. [分层：三层足够，不需要五层](#3-分层三层足够不需要五层)
4. [core：契约层（≤200 行）](#4-core契约层200-行)
5. [data：数据层](#5-data数据层)
6. [nn：算子库](#6-nn算子库)
7. [recipes：论文实现层](#7-recipes论文实现层)
8. [engine：训练/评测](#8-engine训练评测)
9. [bench：CLI 与配置](#9-bench-cli-与配置)
10. [9 篇论文的落点](#10-9-篇论文的落点)
11. [保真度治理：一张卡片就够](#11-保真度治理一张卡片就够)
12. [迁移路径](#12-迁移路径)
13. [被砍掉的抽象及理由](#13-被砍掉的抽象及理由)

---

## 1. v2 现状盘点

审计当前 `src/algo26bench` 得到的判断。要留的、要砍的、要补的，全部有代码依据。

### 1.1 v2 做对的（v3 全部继承）

| 项 | 位置 | 为什么留 |
|---|---|---|
| `Recipe.prepare(context) → PreparedRecipe[TBatch]` 窄腰 | `core/protocols.py:36-39` | HyFormer/OneTrans 两份 batch 布局差异不小（一个 per-domain PaddedSequence，一个多一份 merge_source/merge_position），engine 完全不认识 —— 这条已经跑通 |
| `RawExample` 承载原始 ragged 序列 + **原始时间戳** | `data/pcvr.py:559-596` | v2 修好了 legacy 把 ts 桶化后丢掉的问题，OneTrans 的 timestamp-aware merge 才能真正工作 |
| `@register_recipe("name")` + `strict_dataclass` | `core/registry.py` | OCP 落地的最小成品，加论文对 engine 零改动 |
| `FidelityCard.validate()` 门禁 | `core/fidelity.py:22-26` | FAITHFUL 必须挂 test_id，能真挡"一行注释删掉核心贡献"的场景 |
| `MaskSafeMultiheadAttention` 的 padding-safe 语义 | `nn/attention.py:9-25` | 一行测试就能钉住"padding 不变性" |

### 1.2 v2 做错的（v3 要砍或改）

| 症状 | 代码位置 | v3 处置 |
|---|---|---|
| **`RawBatch` 是个只被 collator 内部用一次的中间层** | `data/view.py:39-114` 里 `collate_raw_examples` 返回 `RawBatch`，`HyFormerCollator`/`OneTransCollator` 拿到后立刻拆散 | **删掉 `RawBatch`**。collator 直接从 `Sequence[RawExample]` 生产自己的 batch |
| **`HyFormerObjective` 和 `OneTransObjective` 90% 重复，`BinaryObjective` 已经写了但没人用** | `recipes/hyformer/objective.py:13-45` vs `recipes/onetrans/objective.py:*` vs `engine/objectives.py:20-56` | recipe 直接用共享 `BinaryObjective`，删掉两个 recipe 各自的 objective 类 |
| **`Trainer.fit` 一个函数塞了 loop / optimizer / clip / max_steps / checkpoint** | `engine/trainer.py:114-172` | 拆一次即可，不做 LoopPolicy/OptimPolicy 三条独立协议。见 §8 |
| **`Capability` 是 4 个布尔位 + notes tuple，从未在决策中被读过** | `core/fidelity.py:48-53`，唯一用法是 `PreparedRecipe.capability` 字段，写了不读 | **删掉**。相关信息进 `FidelityCard.mechanisms` 的 reason 字段 |
| **`FidelityStatus` 有 4 个状态**（FAITHFUL / ADAPTED / OMITTED / OUT_OF_SCOPE） | `core/fidelity.py:7-11` | 缩到 2 个：`FAITHFUL` / `ADAPTED`。理由见 §11 |
| **HyFormer 的 semantic grouping 在 pcvr 上实际是"user/item 粗分组"，卡片里标 FAITHFUL** | `recipes/hyformer/recipe.py:87-92` | 应标 ADAPTED（说清"匿名 fid 无法真语义分组"）。这是**诚实度问题**，不是架构问题，但 v3 的 2-状态 FidelityCard 会强迫更精确的表达 |
| **`MixedCausalAttention._mixed_projection` 用 for-loop 逐 token 分派** | `nn/attention.py:150-162`；`MixedFFN` 同病（`nn/mixing.py:126-140`） | 用 `scatter` 或按 kind 分组的 batched einsum 重写，一次搞定所有 MixedXxx |
| **`data/view.py` 只有 `pad_ragged_sequence` 一个原语，没有 `to_jagged` / `interleave_by_time` / `merge_tokens`** | `data/view.py` 全部 | 补三个原语（见 §5） |
| **`TaskSpec` 只有 name + label_key，无 kind** | `core/types.py:38-42` | 加 `kind: TaskKind`（BINARY / MULTI_ACTION_SOFTMAX） |
| **`Trainer` 里两个 hard-coded 优化器（Adagrad + AdamW）** | `engine/trainer.py:97-112` | 保留，但把"哪些是 sparse"改为 `param_groups()` 由 recipe 声明。见 §8 |

### 1.3 v2 没做但必须补（不然剩下 7 篇装不下）

- **jagged 张量原语**：ULTRA-HSTU 3072–16384 长度，padded 会 OOM
- **多动作 softmax objective**：TokenFormer 的原生形态
- **epoch-level hook**：EST 的每 epoch 稀疏参数重置，论文实测 +0.44% GAUC，是**核心贡献之一**
- **cross-attention 变体的一个抽象点**：InterFormer / EST / LONGER 的 cross-attention 各自不同，但都用得上一个"query set ≠ kv set"的模块

---

## 2. v3 的四条硬约束

学术复现框架的复杂度上限。任何抽象都要过这四条：

1. **本次学术使用不需要就不写。** 具体判据：如果加抽象是为了"未来某天 recipe 可能要 XX"，砍掉。发生了再加。
2. **抽象必须至少被 2 个 recipe 用到才提炼。** `nn/` 里的算子亦然。1 个 recipe 用的东西留在 recipe 目录，不进公共层。
3. **契约必须能用 <10 行 dataclass/Protocol 表达完。** 超过就是抽象错了。
4. **每层 <150 行是硬指标。** core 全部合计 <200 行。超了说明某段代码放错层。

这四条与 DESIGN.md §15 的 KISS 立场一致，但 v3 是"用四条硬约束去检查每一个抽象"，而 v2 是"先设计一遍，再挑几处套 KISS"。区别在于删除动作是否真的发生。

---

## 3. 分层：三层足够

```
              依赖方向（向下）
┌────────────────────────────────────────────┐
│ bench/     CLI + config 加载                │
│            依赖：全部                        │
├────────────────────────────────────────────┤
│ recipes/   一篇论文一个目录（含 batch/model/ │
│            collator/card/tests）             │
│            依赖：core, data, nn             │
├────────────────────────────────────────────┤
│ nn/        论文无关的 PyTorch 算子           │
│            依赖：仅 torch                    │
├────────────────────────────────────────────┤
│ data/      RawExample loader + view 原语     │
│            依赖：仅 core                     │
├────────────────────────────────────────────┤
│ core/      Protocol + dataclass，≤200 行      │
│            依赖：无                          │
├────────────────────────────────────────────┤
│ engine/    Trainer + Evaluator + metrics    │
│            依赖：仅 core                     │
└────────────────────────────────────────────┘
```

**为什么砍掉 v2 DESIGN.md 里的"L5 bench"作为独立层**：`bench/run.py` 只有 96 行，本质是 argparse + config 加载 + 调 `Trainer.fit`。单独一层过度设计。

**为什么把 engine 放在最下面而不是 recipes 上面**：因为 recipes 不需要认识 engine（recipes 只暴露 `PreparedRecipe`，engine 消费它）。engine 只依赖 core，不依赖 recipes —— 这是 DESIGN.md §5 里就想要的方向。v2 目前也是这么做的，v3 保持。

---

## 4. core：契约层（≤200 行）

**唯一目标：定义 recipe 与 engine 之间的窄腰。**

### 4.1 types.py（<80 行）

```python
# 特征/序列声明（数据集告诉 recipe 有什么）
@dataclass(frozen=True)
class CategoricalField:
    name: str
    vocab_size: int
    width: int = 1

@dataclass(frozen=True)
class SequenceDomainSpec:
    name: str
    fields: tuple[CategoricalField, ...]
    max_len: int

@dataclass(frozen=True)
class DataSpec:
    scalar_fields: tuple[CategoricalField, ...]
    candidate_fields: tuple[CategoricalField, ...]
    dense_dim: int
    sequence_domains: tuple[SequenceDomainSpec, ...]

# 任务定义（数据集给定，recipe 适配）
class TaskKind(Enum):
    BINARY = auto()
    MULTI_ACTION_SOFTMAX = auto()   # TokenFormer 的原生形态

@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_key: str
    kind: TaskKind = TaskKind.BINARY
    num_classes: int = 2            # SOFTMAX 时 > 2

@dataclass(frozen=True)
class TaskSet:
    tasks: tuple[TaskSpec, ...]

@dataclass(frozen=True)
class DataContext:
    data_spec: DataSpec
    task_set: TaskSet

# 数据层的输出（recipe collator 的输入）
@dataclass
class RawSequence:
    fields: Mapping[str, Tensor]      # [L] per field
    timestamps: Tensor                # [L]，原始时间戳，不桶化

@dataclass
class RawExample:
    scalars: Mapping[str, Tensor]
    dense: Tensor
    candidates: Mapping[str, Tensor]
    sequences: Mapping[str, RawSequence]
    labels: Mapping[str, Tensor]
    sample_id: Tensor
    group_id: Tensor | None = None

# 模型输出（objective 和 evaluator 的输入）
@dataclass
class ModelOutput:
    logits: Mapping[str, Tensor]
    aux_losses: Mapping[str, Tensor] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

@dataclass
class LossPacket:
    loss_sum: Tensor
    weight: Tensor
    @property
    def mean(self): return self.loss_sum / self.weight.clamp_min(1.0)

@dataclass
class MetricPacket:
    task: str
    scores: Tensor
    targets: Tensor
    sample_ids: Tensor
    group_ids: Tensor | None = None
```

**相对 v2 的删减**：
- 删掉 `RaggedSequence` 和 `RawBatch`。原因：这两个在 v2 里只作为 `collate_raw_examples` 的中间产物存在，recipe 拿到后立刻拆散。让 collator 直接从 `Sequence[RawExample]` 生产 batch，中间层不需要。
- 删掉 `MetricPacket.weights` 字段。v2 里唯一的用法是 `torch.ones_like(...)`（`recipes/hyformer/objective.py:37`），从未真正加权。

### 4.2 protocols.py（<40 行）

```python
class DeviceMovable(Protocol):
    @property
    def batch_size(self) -> int: ...
    def to(self, device: Any) -> "DeviceMovable": ...

TBatch = TypeVar("TBatch", bound=DeviceMovable)

class Objective(Protocol[TBatch]):
    def loss(self, output: ModelOutput, batch: TBatch) -> LossPacket: ...
    def metrics(self, output: ModelOutput, batch: TBatch) -> Sequence[MetricPacket]: ...

class Callback(Protocol):
    """训练循环钩子。EST 的 sparse reset、LR warmup 都是它。"""
    def on_epoch_start(self, state: "TrainState") -> None: ...
    def on_step_end(self, state: "TrainState") -> None: ...

@dataclass
class PreparedRecipe(Generic[TBatch]):
    name: str
    module: nn.Module                # forward(TBatch) -> ModelOutput
    collator: Callable[[Sequence[RawExample]], TBatch]
    objective: Objective[TBatch]
    fidelity: FidelityCard
    callbacks: tuple[Callback, ...] = ()          # 新增：recipe 声明自己的钩子
    param_groups: Callable[[nn.Module], list[dict]] | None = None  # 新增：sparse/dense 分组
```

**相对 v2 的变化**：
- 加 `callbacks` 字段：EST 每 epoch 稀疏重置、任何后续的 tau 退火都通过它挂载
- 加 `param_groups` 字段：让 recipe 自己声明"哪些参数是 embedding"，不再在 `Trainer` 里用 `".tables." in name` 字符串匹配（`engine/trainer.py:93`）—— 这个字符串匹配对新加的 recipe 会静默失效
- 删掉 v2 里的 `capability: Capability` 字段

### 4.3 fidelity.py（<50 行）

```python
class Status(Enum):
    FAITHFUL = "faithful"    # 数学与论文一致，且有测试钉住
    ADAPTED  = "adapted"     # 因公开数据集/学术范围裁剪而降级，必须写清 why 和影响

@dataclass(frozen=True)
class Mechanism:
    description: str
    status: Status
    paper_ref: str
    test_id: str | None = None       # FAITHFUL 必填
    reason: str | None = None        # ADAPTED 必填，说明降级方式和量化影响

    def __post_init__(self) -> None:
        if self.status == Status.FAITHFUL and not self.test_id:
            raise ValueError(f"FAITHFUL mechanism needs a test_id: {self.description}")
        if self.status == Status.ADAPTED and not self.reason:
            raise ValueError(f"ADAPTED mechanism needs a reason: {self.description}")

@dataclass(frozen=True)
class FidelityCard:
    recipe: str
    paper: str                       # arxiv id
    mechanisms: tuple[Mechanism, ...]
    def counts(self) -> dict[str, int]: ...
```

**相对 v2 的变化**：`FidelityStatus` 从 4 状态砍到 2 状态。理由：
- `OUT_OF_SCOPE`（v2）= "工业系统机制，学术范围外" → 但 v3 只做学术复现，"out of scope" 就是**默认状态**，不需要显式标记。真要标就并入 `ADAPTED` 的 reason，一句话说清"serving-only，学术复现不实现"。
- `OMITTED`（v2）= "决定不做" → 与 `ADAPTED` 的区别是"完全没做" vs "做了但简化"。这个区分在报告里没有实际影响力：读者关心的是"这个数字能不能与论文比"，两种情况都是"不能直接比"。合并成 `ADAPTED` + 详细 reason 更清楚。

结果：2 状态足以表达全部 9 篇论文的机制状态，且 `Mechanism.__post_init__` 用 2 行代码就能强制"FAITHFUL 挂测试，ADAPTED 写理由"。

### 4.4 registry.py（<30 行）

保留 v2 的 `register_recipe` / `available_recipes` / `build_recipe` / `strict_dataclass`。这一块已经是最小实现，不动。

---

## 5. data：数据层

### 5.1 view.py（<200 行）

**目标：一组可组合的纯函数**，覆盖 9 篇论文的 collate 需要，不设 `StandardCollator` 基类。

```python
# 已有，保留
def pad_ragged(examples, name, max_len, *, left_pad=False, keep_recent=True) -> PaddedSequence

# 新增，为 ULTRA-HSTU / Kunlun
def to_jagged(examples, name) -> NestedTensor
# torch.nested.nested_tensor(...)；对全 jagged 场景

# 新增，为 OneTrans（v2 是在 recipe 内部 for-loop 做的，抽出来）
def interleave_by_time(examples, domain_order, *, sep_token: bool) -> InterleavePlan
# 返回 (source_index, position_index, is_sep) 三个张量，长度对齐

# 新增，为 LONGER
def merge_tokens(padded, k) -> PaddedSequence
# reshape [B, L, d] → [B, L/K, K*d]
```

**没有 `RawBatch`**。collator 直接从 `Sequence[RawExample]` 生产 recipe-specific batch。v2 的 `collate_raw_examples` 那 80 行验证代码转成 `validate_examples(examples, spec)` 单独一个函数，collator 需要就调，不需要就跳。

### 5.2 pcvr.py

**完全保留 v2 现状**。`data/pcvr.py` 是本次审计里发现的最扎实的一块代码：schema 加载正确、ts_fid 保留、`keep_recent=True` 与业界实践一致、`_USER_DENSE_DIM_OVERRIDE={130: 259}` 沿用 legacy。这条不需要重写。

### 5.3 synthetic.py

保留 v2 现状，只用于 CI 冒烟。**不追加 KuaiRand / Amazon / Taobao 等公开数据集** —— v3 明确聚焦 pCVR，其它数据集是不同 track 的事，别在这个框架里揽。

---

## 6. nn：算子库

### 6.1 保留（v2 已有）

```
nn/
  attention.py         MaskSafeMultiheadAttention, MixedCausalAttention
  sequence.py          TransformerSequenceEncoder, LongerSequenceEncoder, PointwiseSequenceEncoder
  mixing.py            SwiGLU, PerTokenFFN, RankMixerRewire, QueryBoosting, MixedFFN
  tokenize.py          EmbeddingBank, SemanticGroupTokenizer, AutoSplitTokenizer, SequenceTokenizer, DenseTokenizer
  norms.py             RMSNorm
```

### 6.2 修

- `MixedCausalAttention` / `MixedFFN` 用 batched einsum 代替 for-loop。数学等价，代码更短，性能可用。约 30 行。

### 6.3 补（一次到位，剩 7 篇一次全覆盖）

```
nn/
  cross_attention.py       # 通用 cross-attn：query set ≠ kv set，配可插拔 mask
                           # InterFormer / EST-LCA / LONGER 全用它
  mask.py                  # causal / reverse_causal / semi_local / shrinking_window / ui_decoupled
                           # 每个 mask 是一个纯函数：(query_pos, key_pos, extras) → bool tensor
  pos.py                   # RoPE / ROTE / learned_absolute / time_delta_side_info
  token_merge.py           # LONGER 的 L→L/K, d→K·d，配套 inner-transformer
  hyper.py                 # per-sample 动态生成 W：InterFormer PFFN / Kunlun GDPA 的 mini 版
  compress.py              # LCE：token 轴的可学习压缩
```

**判据**：每个新增文件都被至少 2 篇论文用到（见 §10 落点表格）。仅 1 篇用到的算子留在该 recipe 目录，不进 `nn/`。

**举例反面**：EST 的冻结第二张 embedding 表 —— **只有 EST 一家用**，且本身是"预计算相似度图作为额外输入"的一次性机制，**不进 `nn/`**，放 `recipes/est/inputs.py`。

---

## 7. recipes：论文实现层

### 7.1 目录结构

```
recipes/
  hyformer/
    __init__.py    @register_recipe("hyformer")
    config.py      HyFormerConfig（dataclass，from_dict 走 strict_dataclass）
    batch.py       HyFormerBatch + HyFormerCollator（不继承任何东西）
    model.py       HyFormerModel（不继承任何东西）
    card.py        FidelityCard 声明
    tests/
      test_multi_sequence.py
      test_padding_invariance.py
      test_layerwise_kv.py
  onetrans/
  mixformer/
  interformer/
  longer/
  tokenformer/
  ultrahstu/
  est/
  kunlun/
```

### 7.2 一个 recipe 的模板

```python
# recipes/xxx/__init__.py
from .config import XxxConfig
from .model import XxxModel
from .batch import XxxCollator, XxxBatch
from .card import CARD
from algo26bench.core.registry import register_recipe
from algo26bench.core.protocols import PreparedRecipe
from algo26bench.engine.objectives import BinaryObjective, SoftmaxObjective

@register_recipe("xxx")
class XxxRecipe:
    name = "xxx"

    def __init__(self, config: XxxConfig) -> None:
        self.config = config

    @classmethod
    def from_dict(cls, raw: dict) -> "XxxRecipe":
        return cls(XxxConfig.from_dict(raw))

    def prepare(self, ctx: DataContext) -> PreparedRecipe[XxxBatch]:
        module = XxxModel(ctx, self.config)
        collator = XxxCollator(ctx, self.config)
        objective = _choose_objective(ctx.task_set)   # BinaryObjective 或 SoftmaxObjective
        CARD.validate()
        return PreparedRecipe(
            name=self.name,
            module=module,
            collator=collator,
            objective=objective,
            fidelity=CARD,
            callbacks=self._callbacks(),                # 大多数 recipe 是 ()
            param_groups=lambda m: _split_sparse_dense(m),
        )
```

**每个 recipe 目录预计规模 ≤400 行**（config + batch + model + card + tests），远小于 v2 现在 hyformer ~500 行 + 5 处遍布 nn/ 的支撑代码的复杂度。

### 7.3 与 v2 的两个具体差异

1. **objective 不再 recipe-owned**。删掉 `recipes/hyformer/objective.py` 和 `recipes/onetrans/objective.py`，全部走 `engine/objectives.py` 的共享实现。理由：这两个类 90% 重复，仅 metrics 里 weights 那行有微小差别，且这一行本身就是死代码（都是 `torch.ones_like`）。TokenFormer 加进来时用 `SoftmaxObjective`，也是共享。
2. **recipe 声明 param_groups**。`Trainer` 不再猜"哪些是 embedding"（v2 用 `".tables." in name` 字符串匹配，`engine/trainer.py:93`）—— 这个匹配对 EST 的冻结第二张表、Kunlun 的 MoE expert 参数都会静默失效。让 recipe 显式说，是 3 行代码但责任清楚。

---

## 8. engine：训练/评测

### 8.1 Trainer（≤120 行）

```python
@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 1
    max_steps: int | None = None
    dense_lr: float = 1e-3
    sparse_lr: float = 1e-2                # embedding table 用
    weight_decay: float = 0.0
    gradient_clip: float = 5.0
    seed: int = 2026
    device: str = "auto"
    num_workers: int = 0
    output_dir: str | None = None

@dataclass
class TrainState:
    step: int
    epoch: int
    module: nn.Module
    optimizers: list[torch.optim.Optimizer]

class Trainer:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)

    def fit(self, prepared, train_ds, valid_ds) -> dict:
        _set_seed(self.cfg.seed)
        module = prepared.module.to(self.device)
        optimizers = self._build_optimizers(module, prepared.param_groups)
        state = TrainState(step=0, epoch=0, module=module, optimizers=optimizers)

        loader = self._loader(train_ds, prepared.collator, shuffle=True)
        for epoch in range(self.cfg.epochs):
            state.epoch = epoch
            for cb in prepared.callbacks: cb.on_epoch_start(state)   # EST 的 sparse reset
            for batch in loader:
                if self.cfg.max_steps and state.step >= self.cfg.max_steps: break
                self._step(state, batch, prepared.objective)
                for cb in prepared.callbacks: cb.on_step_end(state)   # tau 退火之类
            if self.cfg.max_steps and state.step >= self.cfg.max_steps: break

        metrics = self.evaluate(prepared, valid_ds)
        return _summary(prepared, state, metrics, self.cfg)

    def _step(self, state, batch, objective):
        batch = batch.to(self.device)
        for opt in state.optimizers: opt.zero_grad(set_to_none=True)
        output = state.module(batch)
        loss = objective.loss(output, batch).mean
        if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.module.parameters(), self.cfg.gradient_clip)
        for opt in state.optimizers: opt.step()
        state.step += 1

    @torch.no_grad()
    def evaluate(self, prepared, ds) -> dict: ...
```

**相对 v2 的变化**：

| 变化 | 理由 |
|---|---|
| `fit` 拆出 `_step` 方法 | 从 60 行的 fit 里抠出 20 行的 _step，可读性和可测性都上来了。不做 LoopPolicy —— OneTrans 的 streaming eval 是 evaluator 的事，不是 loop 的事 |
| **加 `callbacks` 钩子**（on_epoch_start / on_step_end） | 支撑 EST 的每 epoch 稀疏参数重置（论文实测 +0.44% GAUC）。就 2 个钩子点，覆盖 9 篇里全部真实需求 |
| **删掉 `LoopPolicy` / `OptimPolicy` / `PrecisionPolicy`** | v2 DESIGN.md 说要外置的三条独立协议，实际上 9 篇论文里只有 EST 的 sparse reset 需要 loop 定制，而它可以用 callback 表达。OneTrans 的 streaming eval 是 evaluator 层的事。学术复现里 mixed precision 用 `torch.amp.autocast` 一行搞定，不需要 policy |
| **`param_groups` 从 recipe 传入而非 Trainer 猜** | 见 §7.3 |
| **`AsyncSparsePolicy` 不做** | v2 DESIGN.md §10.2 说它"必须实现"，但**论文原文 Sec 3.8.2 明确说这个 staleness 对模型质量无影响**（"empirical results indicate that this staleness does not degrade convergence quality"）。这是纯吞吐优化，学术复现不需要。在 hyformer 卡片里标 ADAPTED + 引用原文这句话即可 |

### 8.2 objectives.py

```python
class BinaryObjective:
    def __init__(self, task_set: TaskSet) -> None:
        self.task_set = task_set
        for t in task_set.tasks:
            assert t.kind == TaskKind.BINARY
    def loss(self, output, batch) -> LossPacket: ...          # 逐 task BCE 加和
    def metrics(self, output, batch) -> list[MetricPacket]: ... # 逐 task sigmoid → MetricPacket

class SoftmaxObjective:
    """TokenFormer 的原生形态，A 维 softmax CE。"""
    def __init__(self, task_set: TaskSet) -> None:
        assert len(task_set.tasks) == 1
        assert task_set.tasks[0].kind == TaskKind.MULTI_ACTION_SOFTMAX
    def loss(self, output, batch) -> LossPacket: ...
    def metrics(self, output, batch) -> list[MetricPacket]: ...   # per-class one-vs-rest AUC

def choose_objective(task_set: TaskSet) -> Objective:
    kinds = {t.kind for t in task_set.tasks}
    if kinds == {TaskKind.BINARY}: return BinaryObjective(task_set)
    if kinds == {TaskKind.MULTI_ACTION_SOFTMAX}: return SoftmaxObjective(task_set)
    raise ValueError(f"unsupported task mix: {kinds}")
```

**判据**：这两个 objective 覆盖 9 篇论文的全部原生任务形态。加第三种只有在数据集里出现 `TaskKind.PAIRWISE_RANKING` 之类的新种类时才需要 —— 现在没有。

### 8.3 metrics.py

保留 v2 的 `binary_auc` / `binary_logloss` / `aggregate_metric_packets`。这三个函数已经是最小实现。**加一个 `grouped_auc`** —— v2 里 `aggregate_metric_packets` 已经算了 group_auc（`engine/metrics.py:63-77`），单独提出来做纯函数，方便测试。

### 8.4 evaluator

不设独立类。`Trainer.evaluate` 就是全部 —— 9 篇论文里只有 OneTrans 有 streaming next-batch 评测的需求，而它可以在 `Trainer.evaluate` 里加一个 `mode="streaming"` 分支或在 bench/run.py 里跑不同的 loop。**先不做，等真正加 OneTrans-streaming 时再决定。**

这里做一个"欠着的抽象"是刻意的：v3 的立场是"发生了再加"。

---

## 9. bench：CLI 与配置

```python
# bench/run.py（<100 行）
def run(config_path: Path) -> dict:
    raw = json.loads(config_path.read_text())
    recipe_name = raw["recipe"]
    ctx, train_ds, valid_ds = _build_data(raw["data"])
    recipe = build_recipe(recipe_name, raw.get("model", {}))
    prepared = recipe.prepare(ctx)
    return Trainer(TrainingConfig.from_dict(raw.get("training", {}))).fit(prepared, train_ds, valid_ds)
```

**保留 v2 现状**。config 分四段（`recipe` / `model` / `data` / `training`），每段独立 dataclass 拒绝未知 key。

**不做** DESIGN.md §13.2 的 `configs/{data,recipe,training,experiment}` 四段目录 —— 9 篇论文的运行入口是 9 个 json 文件，直接放 `configs/` 平铺就够了。加了目录反而要写 include/merge 逻辑。

---

## 10. 9 篇论文的落点

**判据是"新增了什么"和"复用了什么"。每一行的"新增" ≤5 项，"复用"能列出的都算成本平摊。**

| 论文 | 新增 nn/ 算子 | 新增 recipes/xxx/ 特有代码 | 复用 |
|---|---|---|---|
| **HyFormer** | 无（v2 已实现） | v2 已有 | — |
| **OneTrans** | 无（v2 已实现） | v2 已有 | — |
| **MixFormer** | `mask.py::ui_decoupled_mask` | UI-decoupling 结构 mask 的构造函数 | `AutoSplitTokenizer`、`SwiGLU`、`MaskSafeMultiheadAttention` + 结构 mask |
| **InterFormer** | `hyper.py::PFFN`、`pos.py::rope`、`compress.py::lce` | 双 arch 交替调度的 forward 结构 | `MaskSafeMultiheadAttention`、`SequenceTokenizer` |
| **LONGER** | `mask.py::reverse_causal_mask`、`token_merge.py`、`pos.py::time_delta_side_info` | LONGER 特有的 target/CLS/UID 四来源 global token | `TransformerSequenceEncoder`（自 self-attn 部分）、cross-attention |
| **TokenFormer** | `mask.py::shrinking_window_mask` + 逐层不同 mask 的支持 | per-sample position id 生成、双数据组织的 collate 分支 | `MixedCausalAttention`（可以稍加改造复用）、`PerTokenFFN` |
| **ULTRA-HSTU** | `attention.py::SiLUAttention`（φ(QKᵀ)·M·V，无 softmax）、`mask.py::semi_local_mask` | item+action 相加、candidate-in-sequence 的 collate、`view.py::to_jagged` 的应用 | — |
| **EST** | `cross_attention.py`（token-specific W_Q/W_O）+ `MixedFFN`（可复用） | 冻结第二张 embedding 表、预计算 top-K 图作为额外输入、`SparseResetCallback` | `PerTokenFFN`、共享 dense 优化路径 |
| **Kunlun** | `attention.py::GDPA`（seq=Q, non-seq 生成 K/V，per-head 激活）、`mask.py::sliding_bidirectional_mask`、`pool.py::hsp` | 层间携带状态的 forward（block 签名多一个 `hidden_state`） | `MixedFFN`、`RMSNorm` |

**关键观察**：新增算子的总量约是 6-8 个，且**每一个至少 2 篇论文用**（复用列显式列出）。这符合 §2 的第 2 条约束。

**明确降级项（在 FidelityCard 里标 ADAPTED）**：
- Kunlun 的 Triton fused kernel → 不做，走 PyTorch reference。理由：与学术 AUC 无关，是 MFU 主张的载体。
- EST 的 CSA（Content Sparse Attention）→ 不做（TAAC2026 没有多模态 content embedding）。EST 只实现 LCA + per-token FFN + sparse reset callback 三项。
- 所有 recipe 的 KV cache / FlashAttention / async allreduce → 全部走 PyTorch 参考实现。
- HyFormer 在 pcvr 上的 semantic grouping → 因匿名 fid 只能做 user/item 粗分组，ADAPTED。

这些降级项的**测量影响**在 reason 字段里量化写清楚（引用论文消融表的数字）—— 读者一眼能看出这条被裁掉大概花多少 AUC。

---

## 11. 保真度治理：一张卡片就够

### 11.1 CI 检查规则

只需要 3 条，都是 `Mechanism.__post_init__` 直接抛错，不需要额外 CI 脚本：

1. `status == FAITHFUL` 且 `test_id` 为空 → 抛错
2. `status == ADAPTED` 且 `reason` 为空 → 抛错
3. `pytest` 收集时找不到 `test_id` 对应的测试 → 一个 conftest.py 里的 `pytest_collection_modifyitems` 30 行搞定

### 11.2 关键机制的必备测试（每 recipe 3-5 条）

只列容易"静默出错"的：

| 测试 | 挡的错 |
|---|---|
| `test_padding_invariance` | 给 batch 加 padding，logits 不变。一条测试同时抓住"mask-free SDPA"、"padded 位置参与 softmax"、"pool 时把 padded 位置也算进去"三种 bug |
| `test_mask_semantics` | `True = valid` 语义一致，legacy 转换点显式标记 |
| `test_reverse_causal_mask`（LONGER） | 序列 latest→earliest，mask 是 `j ≥ i` 而不是 `torch.tril` |
| `test_kv_recomputed_per_layer`（HyFormer） | 每层的 `sequence_encoder` 都消费**原始** tokenized 序列（Eq. 8） |
| `test_layerwise_shrinking_window`（TokenFormer） | 逐层 mask 由 `(layer, positions) → mask` 决定，且末层按 token 类型列屏蔽 |
| `test_sparse_reset_on_epoch_start`（EST） | callback 在 `on_epoch_start` 把稀疏参数写回 `θ_s^(0)`，dense 不动 |

这 6 条测试加上现有的 5 条测试文件（`tests/test_hyformer.py` 等），全 9 篇覆盖率约 30-40 条测试。不算多，也不算少。

### 11.3 报告

`Trainer.fit` 返回的 summary 字典里加两项：

```json
{
  "recipe": "hyformer",
  "auc": 0.6234,
  "logloss": 0.4189,
  "params_dense": 12_345_678,
  "params_sparse": 98_765_432,
  "fidelity": {"faithful": 6, "adapted": 2},
  "fidelity_adapted": [
    {"description": "semantic_grouping", "reason": "TAAC2026 匿名 fid，退化为 user/item 粗分组"},
    ...
  ]
}
```

**不做**独立的排行榜渲染代码 —— jq / pandas 一行就能出表，不值得写一个 report.py。

---

## 12. 迁移路径

### 阶段 0（1 天）：core 收敛

- 删 `RawBatch` / `RaggedSequence` / `Capability` / 4 状态 `FidelityStatus`
- `TaskSpec` 加 `kind`
- `PreparedRecipe` 加 `callbacks` / `param_groups`
- `Mechanism.__post_init__` 拉起 CI 门禁

**退出条件**：hyformer / onetrans 的现有测试全绿。核心是"抽象只是变了名字或缩了范围，行为没变"。

### 阶段 1（0.5 天）：objective 去重

- `engine/objectives.py` 的 `BinaryObjective` 已有，加 `SoftmaxObjective`
- 删 `recipes/hyformer/objective.py` 和 `recipes/onetrans/objective.py`
- `recipe.prepare` 走 `choose_objective(ctx.task_set)`

### 阶段 2（2 天）：view 原语 + nn 补齐

- `data/view.py` 补 `to_jagged` / `interleave_by_time` / `merge_tokens`
- `nn/` 补 `cross_attention.py` / `mask.py`（5 种 mask）/ `pos.py` / `hyper.py` / `compress.py` / `token_merge.py`
- 每加一个算子跟一条测试

### 阶段 3（10-14 天）：剩下 7 篇

按依赖顺序：LONGER → InterFormer → MixFormer → TokenFormer → EST → Kunlun → ULTRA-HSTU。前四个用现有基建，一篇 1-2 天；后三个每篇 2-3 天（EST 靠 callback，Kunlun 靠层间状态 forward 签名，ULTRA-HSTU 靠 jagged）。

**总工期**：约 3 周，与 v2 DESIGN.md §17 的 4 个阶段接近，但**没有 UniRank 15 个模型的机械迁移**、**没有 4 条赛道**、**没有 scaling law 拟合** —— 这些都是学术复现范围外，v3 不揽。

---

## 13. 被砍掉的抽象及理由

汇总一下 v3 相对 v2 DESIGN.md（1180 行）明确不做的事，每条一句话说清为什么：

| 砍掉的 | 理由 |
|---|---|
| `RawBatch` 中间层 | 只是 collator 内部的临时结构，让 collator 直接生产 recipe batch |
| `Capability` dataclass | 4 个布尔位从未在决策中被读过，进 reason 字段更好 |
| `FidelityStatus.OUT_OF_SCOPE` / `OMITTED` | 学术复现里"没做"就一种：ADAPTED + 说清原因 |
| `LoopPolicy` / `OptimPolicy` / `PrecisionPolicy` 独立协议 | 9 篇里只有 EST 需要 loop 定制，一个 callback 就够 |
| `AsyncSparsePolicy` | 论文原文说 staleness 对质量无影响，纯吞吐优化 |
| `HasFusedKernel` / `CachedScoring` / `NeedsSampler` 可选协议 | 3 个协议**没有实现者**，是 v2 里"预留 hook 但从未挂载"的教科书案例 |
| `RankBatch` 通用批次 | v2 已经删过一次（好），DESIGN.md 又想加回来（错） |
| 4 条赛道（Quality / Efficiency / Scaling / Track A/B/C）| MFU / FLOPs / p99 latency 都是 serving 主张，学术复现不做 |
| `bench/sweep.py` 超参搜索 | shell for-loop 加 config 覆盖，不需要独立模块 |
| `bench/report.py` 排行榜渲染 | jq 或 pandas 一行 |
| `configs/{data,recipe,training,experiment}` 四段目录 | 9 个 recipe = 9 个平铺 json，够了 |
| `import-linter` 层间强制 | 已经用目录约定表达了依赖方向，加工具是过度工程 |
| `UniRank 15 model_zoo` 机械迁移 | `refer/` 只 9 篇，其它不在范围 |

---

## 与 v2 DESIGN.md 的立场差异

DESIGN.md 是**为工业级 benchmark 而写的设计**，覆盖了 MFU / serving cache / async allreduce / 分组梯度裁剪 / stochastic length sampler / 混合精度分层配置 / scaling law 拟合等等。这些是好东西，但**学术复现用不到**。v3 的立场：

- **v2 DESIGN.md**：为 9 篇论文的**全部主张**（AUC + MFU + serving latency + scaling）复现设计。
- **v3**：为 9 篇论文的**算法主张**（AUC / logloss / gAUC）复现设计。效率与 serving 的主张全部走 ADAPTED，在 FidelityCard 里说清"这一项走参考实现，与论文的效率数字不可比"。

这个立场差异带来的具体化简：core 从 v2 的 ~400 行目标（含 Capability / 4 状态 fidelity / 3 条 policy protocol / 4 个能力 protocol）缩到 v3 的 ≤200 行；engine 从 v2 DESIGN.md 描述的 loop/optim/precision/distributed/hooks 五个文件缩到 v3 的 trainer + objectives + metrics 三个文件。

**如果未来想做 Track B/C**（效率、scaling），v3 是可以扩展过去的 —— `nn/` 算子层可以直接沿用，`Trainer` 加 `profile` 方法即可。但那时框架有明确的用户需求驱动，不是提前铺开一堆用不着的接口。

---

## 一句话总结

v3 = v2 的窄腰 + v2 已跑通的 nn/data 层 + `PreparedRecipe.callbacks` + `TaskKind` + `SoftmaxObjective` + 6 个新算子文件 − 3 个中间层 dataclass − 4 状态 fidelity − 3 条未实现的 policy 协议 − 4 条 track − DESIGN.md 里所有"未来可能会用到"的 hook 点。

工期与 v2 落地 9 篇的估计相当（3 周），但**框架代码量少约 40%**，且**每一处抽象都能指出至少 2 个使用者**。


初赛数据：/apdcephfs_qy3/share_470749/joefzhou/data/alg_2026_sample/20260323_150w_anonymized/split_v2/train/native_parquet_v2