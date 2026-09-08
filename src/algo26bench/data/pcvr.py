"""TAAC2026 / academic_2000w PCVR parquet adapter.

Reads the same ``schema.json`` + native parquet layout as ``algo26bench``, but
stops at ``RawExample``: no padding, no time-bucket replacement, no embedding.
Sequence truncation keeps the most recent ``max_len`` events; recipes pad.
"""

from __future__ import annotations

import glob
import json
import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from algo26bench.core.registry import strict_dataclass
from algo26bench.core.types import (
    CategoricalField,
    DataContext,
    DataSpec,
    RawExample,
    RawSequence,
    SequenceDomainSpec,
    TaskSet,
    TaskSpec,
)

# Same production override as algo26bench/data/dataset.py: fid 130's tail is
# unrelated stats, so the effective dense width is 259 rather than the schema
# max_len.
_USER_DENSE_DIM_OVERRIDE = {130: 259}
_TASK_SET = TaskSet((TaskSpec("pcvr", "pcvr"),))


@dataclass(frozen=True)
class _IntColumn:
    name: str
    parquet_column: str
    vocab_size: int
    width: int
    unused: bool = False


@dataclass(frozen=True)
class _DenseColumn:
    parquet_column: str
    width: int
    offset: int


@dataclass(frozen=True)
class _SequenceLayout:
    name: str
    prefix: str
    max_len: int
    ts_column: str | None
    fields: tuple[_IntColumn, ...]


@dataclass(frozen=True)
class PCVRSchema:
    context: DataContext
    user_int: tuple[_IntColumn, ...]
    item_int: tuple[_IntColumn, ...]
    user_dense: tuple[_DenseColumn, ...]
    item_dense: tuple[_DenseColumn, ...]
    sequences: tuple[_SequenceLayout, ...]


@dataclass(frozen=True)
class PCVRDataConfig:
    data_dir: str
    schema_path: str | None = None
    valid_ratio: float = 0.01
    train_ratio: float = 1.0
    train_row_groups: int = 0
    seq_max_lens: Mapping[str, int] = field(default_factory=dict)
    clip_vocab: bool = True
    keep_recent: bool = True
    shuffle: bool = True
    buffer_rows: int = 4096
    read_batch_size: int = 1024
    seed: int = 2026
    # When eval_data_dir is set, the "validation" set becomes the predict
    # parquet directory with labels sourced from answer_path (a JSON dict
    # mapping user_id string -> {0,1}). In that mode the training dataset
    # uses the full data_dir (no valid_ratio slice removed).
    eval_data_dir: str | None = None
    eval_schema_path: str | None = None
    answer_path: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PCVRDataConfig":
        normalized = dict(raw)
        if "seq_max_lens" in normalized:
            seq_max_lens = normalized["seq_max_lens"]
            if not isinstance(seq_max_lens, Mapping):
                raise ValueError("seq_max_lens must be a mapping of domain -> max_len")
            normalized["seq_max_lens"] = {
                str(name): int(length) for name, length in seq_max_lens.items()
            }
        config = strict_dataclass(cls, normalized)
        if not config.data_dir:
            raise ValueError("pcvr data_dir is required")
        if not 0.0 < config.valid_ratio < 1.0:
            raise ValueError("valid_ratio must be in (0, 1)")
        if not 0.0 < config.train_ratio <= 1.0:
            raise ValueError("train_ratio must be in (0, 1]")
        if config.train_row_groups < 0:
            raise ValueError("train_row_groups cannot be negative")
        if config.read_batch_size <= 0 or config.buffer_rows < 0:
            raise ValueError("read_batch_size must be positive")
        if (config.eval_data_dir is None) != (config.answer_path is None):
            raise ValueError(
                "eval_data_dir and answer_path must be set together"
            )
        return config


def _field_name(family: str, feature_id: int) -> str:
    return f"{family}_{feature_id}"


def _embedding_vocab(vocab_size: int) -> int:
    return max(int(vocab_size), 1)


def load_pcvr_schema(
    schema_path: str | Path,
    seq_max_lens: Mapping[str, int] | None = None,
) -> PCVRSchema:
    raw = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    overrides = seq_max_lens or {}

    user_int = tuple(
        _IntColumn(
            _field_name("user", int(fid)),
            f"user_int_feats_{int(fid)}",
            _embedding_vocab(int(vs)),
            int(dim),
            unused=int(vs) <= 0,
        )
        for fid, vs, dim in raw["user_int"]
    )
    item_int = tuple(
        _IntColumn(
            _field_name("item", int(fid)),
            f"item_int_feats_{int(fid)}",
            _embedding_vocab(int(vs)),
            int(dim),
            unused=int(vs) <= 0,
        )
        for fid, vs, dim in raw["item_int"]
    )

    user_dense_cols: list[_DenseColumn] = []
    offset = 0
    for fid, dim in raw.get("user_dense", []):
        width = int(_USER_DENSE_DIM_OVERRIDE.get(int(fid), dim))
        user_dense_cols.append(
            _DenseColumn(f"user_dense_feats_{int(fid)}", width, offset)
        )
        offset += width
    user_dense_total = offset
    item_dense_cols: list[_DenseColumn] = []
    for fid, dim in raw.get("item_dense", []):
        width = int(dim)
        item_dense_cols.append(
            _DenseColumn(f"item_dense_feats_{int(fid)}", width, offset)
        )
        offset += width
    item_dense_total = offset - user_dense_total

    sequences: list[_SequenceLayout] = []
    sequence_specs: list[SequenceDomainSpec] = []
    for domain in sorted(raw["seq"]):
        cfg = raw["seq"][domain]
        ts_fid = cfg.get("ts_fid")
        prefix = str(cfg["prefix"])
        fields: list[_IntColumn] = []
        for fid, vs in cfg["features"]:
            if ts_fid is not None and int(fid) == int(ts_fid):
                continue
            fields.append(
                _IntColumn(
                    _field_name(domain, int(fid)),
                    f"{prefix}_{int(fid)}",
                    _embedding_vocab(int(vs)),
                    1,
                    unused=int(vs) <= 0,
                )
            )
        if not fields:
            raise ValueError(f"{domain}: schema has no non-timestamp sequence fields")
        max_len = int(overrides.get(domain, 256))
        sequences.append(
            _SequenceLayout(
                name=str(domain),
                prefix=prefix,
                max_len=max_len,
                ts_column=None if ts_fid is None else f"{prefix}_{int(ts_fid)}",
                fields=tuple(fields),
            )
        )
        sequence_specs.append(
            SequenceDomainSpec(
                name=str(domain),
                fields=tuple(
                    CategoricalField(column.name, column.vocab_size, column.width)
                    for column in fields
                ),
                max_len=max_len,
            )
        )

    context = DataContext(
        data_spec=DataSpec(
            scalar_fields=tuple(
                CategoricalField(column.name, column.vocab_size, column.width)
                for column in user_int
            ),
            candidate_fields=tuple(
                CategoricalField(column.name, column.vocab_size, column.width)
                for column in item_int
            ),
            user_dense_dim=user_dense_total,
            item_dense_dim=item_dense_total,
            sequence_domains=tuple(sequence_specs),
        ),
        task_set=_TASK_SET,
    )
    return PCVRSchema(
        context=context,
        user_int=user_int,
        item_int=item_int,
        user_dense=tuple(user_dense_cols),
        item_dense=tuple(item_dense_cols),
        sequences=tuple(sequences),
    )


def _sanitize_ids(
    values: np.ndarray,
    vocab_size: int,
    *,
    clip: bool,
    column: str,
    unused: bool = False,
) -> np.ndarray:
    cleaned = np.asarray(values, dtype=np.int64).copy()
    cleaned[cleaned <= 0] = 0
    if unused:
        cleaned[:] = 0
        return cleaned
    oob = cleaned >= vocab_size
    if not np.any(oob):
        return cleaned
    if not clip:
        raise ValueError(
            f"{column}: ids outside [0, {vocab_size}), "
            f"range=[{int(cleaned.min())}, {int(cleaned.max())}]"
        )
    cleaned[oob] = 0
    return cleaned


def _window_copy(values: np.ndarray, start: int, end: int) -> np.ndarray:
    """Copy ``values[start:end]`` into a length-(end-start) array, clipping to the source."""

    width = end - start
    out = np.zeros((width,), dtype=np.asarray(values).dtype)
    src_start = max(0, start)
    src_end = min(int(values.size), end)
    if src_end > src_start:
        dst = src_start - start
        out[dst : dst + (src_end - src_start)] = values[src_start:src_end]
    return out


def _list_row(
    offsets: np.ndarray,
    values: np.ndarray,
    row: int,
) -> np.ndarray:
    start = int(offsets[row])
    end = int(offsets[row + 1])
    if end <= start:
        return values[:0]
    return np.asarray(values[start:end])


def _pad_or_trim(values: np.ndarray, width: int, dtype: np.dtype) -> np.ndarray:
    out = np.zeros((width,), dtype=dtype)
    use = min(int(values.size), width)
    if use:
        out[:use] = values[:use]
    return out


def _group_id(user_id: object) -> torch.Tensor:
    text = str(user_id)
    try:
        value = int(text)
    except ValueError:
        value = zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF
    return torch.tensor(value, dtype=torch.long)


def _answer_key(user_id: object) -> str:
    """Match answer.json's string-int keys ("861152", not "861152.0")."""
    text = str(user_id)
    try:
        return str(int(text))
    except ValueError:
        return text


def load_answer_labels(path: str | Path) -> dict[str, int]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"answer file {path} is not a JSON object")
    return {str(k): int(v) for k, v in raw.items()}


def list_row_groups(data_dir: str) -> list[tuple[str, int, int]]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files in {data_dir}")
    groups: list[tuple[str, int, int]] = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for index in range(parquet_file.metadata.num_row_groups):
            groups.append(
                (path, index, int(parquet_file.metadata.row_group(index).num_rows))
            )
    return groups


def split_row_groups(
    groups: Sequence[tuple[str, int, int]],
    *,
    valid_ratio: float,
    train_ratio: float,
    train_row_groups: int,
) -> tuple[list[int], tuple[int, int]]:
    total = len(groups)
    n_valid = max(1, int(total * valid_ratio)) if total > 1 else 0
    n_train = total - n_valid
    if train_row_groups > 0:
        selected = min(int(train_row_groups), n_train)
    elif train_ratio < 1.0:
        selected = max(1, int(n_train * train_ratio))
    else:
        selected = n_train
    if selected < n_train:
        train_indices = [
            ((2 * index + 1) * n_train) // (2 * selected) for index in range(selected)
        ]
    else:
        train_indices = list(range(n_train))
    return train_indices, (n_train, total)


class PCVRParquetDataset(IterableDataset[RawExample]):
    """Row-level PCVR reader. DataLoader + recipe collator own batching."""

    def __init__(
        self,
        schema: PCVRSchema,
        row_groups: Sequence[tuple[str, int, int]],
        *,
        clip_vocab: bool = True,
        keep_recent: bool = True,
        shuffle: bool = False,
        buffer_rows: int = 0,
        read_batch_size: int = 1024,
        seed: int = 2026,
        is_training: bool = True,
        answer_labels: Mapping[int, int] | None = None,
    ) -> None:
        super().__init__()
        if not row_groups:
            raise ValueError("PCVR dataset was given no row groups")
        self.schema = schema
        self.row_groups = tuple(row_groups)
        self.clip_vocab = clip_vocab
        self.keep_recent = keep_recent
        self.shuffle = shuffle
        self.buffer_rows = buffer_rows
        self.read_batch_size = read_batch_size
        self.seed = seed
        self.is_training = is_training
        self.answer_labels = answer_labels
        self.num_rows = sum(rows for _, _, rows in self.row_groups)

    def __iter__(self) -> Iterator[RawExample]:
        groups = self.row_groups
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            groups = tuple(
                group
                for index, group in enumerate(groups)
                if index % worker.num_workers == worker.id
            )
        buffer: list[RawExample] = []
        flush_count = 0
        sample_index = 0
        needed = set(self._required_columns())
        for path, row_group, _ in groups:
            parquet_file = pq.ParquetFile(path)
            names = parquet_file.schema_arrow.names
            missing = sorted(needed - set(names))
            if missing:
                raise KeyError(f"{path}: missing columns {missing}")
            columns = [name for name in names if name in needed]
            for record_batch in parquet_file.iter_batches(
                batch_size=self.read_batch_size,
                row_groups=[row_group],
                columns=columns,
            ):
                examples = self._examples_from_batch(record_batch, sample_index)
                sample_index += len(examples)
                if self.shuffle and self.buffer_rows > 1:
                    buffer.extend(examples)
                    while len(buffer) >= self.buffer_rows:
                        yield from self._flush(buffer, flush_count)
                        flush_count += 1
                else:
                    yield from examples
        if buffer:
            yield from self._flush(buffer, flush_count)

    def _required_columns(self) -> tuple[str, ...]:
        columns = ["timestamp", "label_type", "user_id"]
        columns.extend(column.parquet_column for column in self.schema.user_int)
        columns.extend(column.parquet_column for column in self.schema.item_int)
        columns.extend(column.parquet_column for column in self.schema.user_dense)
        columns.extend(column.parquet_column for column in self.schema.item_dense)
        for sequence in self.schema.sequences:
            columns.extend(column.parquet_column for column in sequence.fields)
            if sequence.ts_column is not None:
                columns.append(sequence.ts_column)
        return tuple(columns)

    def _flush(
        self, buffer: list[RawExample], flush_count: int
    ) -> Iterator[RawExample]:
        generator = torch.Generator().manual_seed(self.seed + flush_count)
        order = torch.randperm(len(buffer), generator=generator).tolist()
        for index in order:
            yield buffer[index]
        buffer.clear()

    def _examples_from_batch(
        self, batch: pa.RecordBatch, start_index: int
    ) -> list[RawExample]:
        by_name = {name: batch.column(name) for name in batch.schema.names}
        rows = batch.num_rows
        user_ids = by_name["user_id"].to_pylist()
        if self.answer_labels is not None:
            labels = np.array(
                [
                    int(self.answer_labels.get(_answer_key(uid), 0))
                    for uid in user_ids
                ],
                dtype=np.bool_,
            )
        elif self.is_training:
            labels = (
                by_name["label_type"]
                .fill_null(0)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
                == 2
            )
        else:
            labels = np.zeros(rows, dtype=np.bool_)

        user_values = {
            column.name: self._int_column(by_name[column.parquet_column], column)
            for column in self.schema.user_int
        }
        item_values = {
            column.name: self._int_column(by_name[column.parquet_column], column)
            for column in self.schema.item_int
        }
        dense = np.zeros((rows, self.schema.context.data_spec.dense_dim), dtype=np.float32)
        for column in (*self.schema.user_dense, *self.schema.item_dense):
            dense[:, column.offset : column.offset + column.width] = self._dense_column(
                by_name[column.parquet_column], column.width, rows
            )

        sequences = {
            sequence.name: self._sequence_columns(by_name, sequence, rows)
            for sequence in self.schema.sequences
        }

        examples: list[RawExample] = []
        for row in range(rows):
            raw_sequences = {}
            for sequence in self.schema.sequences:
                fields, ts_values = sequences[sequence.name]
                raw_sequences[sequence.name] = RawSequence(
                    fields={
                        name: torch.from_numpy(values[row].copy())
                        for name, values in fields.items()
                    },
                    timestamps=torch.from_numpy(ts_values[row].copy()),
                )
            examples.append(
                RawExample(
                    scalars={
                        name: torch.from_numpy(np.asarray(values[row]).copy())
                        for name, values in user_values.items()
                    },
                    dense=torch.from_numpy(dense[row].copy()),
                    candidates={
                        name: torch.from_numpy(np.asarray(values[row]).copy())
                        for name, values in item_values.items()
                    },
                    sequences=raw_sequences,
                    labels={
                        "pcvr": torch.tensor(float(labels[row]), dtype=torch.float32)
                    },
                    sample_id=torch.tensor(start_index + row, dtype=torch.long),
                    group_id=_group_id(user_ids[row]),
                )
            )
        return examples

    def _int_column(self, arrow_col: pa.Array, column: _IntColumn) -> list[np.ndarray]:
        rows = len(arrow_col)
        if column.width == 1 and not (
        pa.types.is_list(arrow_col.type) or pa.types.is_large_list(arrow_col.type)
    ):
            values = _sanitize_ids(
                arrow_col.fill_null(0).to_numpy(zero_copy_only=False),
                column.vocab_size,
                clip=self.clip_vocab,
                column=column.parquet_column,
                unused=column.unused,
            )
            return [values[row : row + 1].reshape(()) for row in range(rows)]
        offsets = arrow_col.offsets.to_numpy()
        raw_values = arrow_col.values.to_numpy()
        rows_out: list[np.ndarray] = []
        for row in range(rows):
            sliced = _pad_or_trim(
                _list_row(offsets, raw_values, row), column.width, np.int64
            )
            rows_out.append(
                _sanitize_ids(
                    sliced,
                    column.vocab_size,
                    clip=self.clip_vocab,
                    column=column.parquet_column,
                    unused=column.unused,
                )
            )
        return rows_out

    def _dense_column(
        self, arrow_col: pa.Array, width: int, rows: int
    ) -> np.ndarray:
        offsets = arrow_col.offsets.to_numpy()
        raw_values = arrow_col.values.to_numpy()
        out = np.zeros((rows, width), dtype=np.float32)
        for row in range(rows):
            out[row] = _pad_or_trim(
                _list_row(offsets, raw_values, row).astype(np.float32, copy=False),
                width,
                np.float32,
            )
        return out

    def _sequence_columns(
        self,
        by_name: Mapping[str, pa.Array],
        sequence: _SequenceLayout,
        rows: int,
    ) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
        prepared: dict[str, tuple[np.ndarray, np.ndarray, _IntColumn]] = {}
        for column in sequence.fields:
            arrow_col = by_name[column.parquet_column]
            prepared[column.name] = (
                arrow_col.offsets.to_numpy(),
                arrow_col.values.to_numpy(),
                column,
            )
        ts_offsets = ts_values = None
        if sequence.ts_column is not None:
            ts_col = by_name[sequence.ts_column]
            ts_offsets = ts_col.offsets.to_numpy()
            ts_values = ts_col.values.to_numpy()

        fields: dict[str, list[np.ndarray]] = {name: [] for name in prepared}
        timestamps: list[np.ndarray] = []
        for row in range(rows):
            lengths = [
                int(offsets[row + 1] - offsets[row])
                for offsets, _, _ in prepared.values()
            ]
            if ts_offsets is not None:
                lengths.append(int(ts_offsets[row + 1] - ts_offsets[row]))
            raw_len = max(lengths, default=0)
            keep = min(raw_len, sequence.max_len)
            start = raw_len - keep if self.keep_recent else 0
            end = start + keep
            for name, (offsets, values, column) in prepared.items():
                fields[name].append(
                    _sanitize_ids(
                        _window_copy(_list_row(offsets, values, row), start, end),
                        column.vocab_size,
                        clip=self.clip_vocab,
                        column=column.parquet_column,
                        unused=column.unused,
                    )
                )
            if ts_offsets is None:
                timestamps.append(np.zeros((keep,), dtype=np.int64))
            else:
                timestamps.append(
                    _window_copy(
                        _list_row(ts_offsets, ts_values, row).astype(np.int64, copy=False),
                        start,
                        end,
                    )
                )
        return fields, timestamps


def build_pcvr_datasets(
    config: PCVRDataConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[PCVRSchema, PCVRParquetDataset, PCVRParquetDataset]:
    """Build train/valid datasets, optionally sharded by DDP rank.

    Under DDP, each rank sees a disjoint slice of the training row groups
    (round-robin), and every rank sees the full validation set (only rank 0
    evaluates, other ranks skip the eval pass).

    When ``config.eval_data_dir`` and ``config.answer_path`` are set, the
    "valid" dataset is built from that predict parquet directory with labels
    looked up in the answer JSON (keyed by ``user_id``); in that mode the
    training set uses every row group in ``config.data_dir`` (no valid slice
    removed) since a real held-out eval set exists.
    """

    schema_path = config.schema_path or os.path.join(config.data_dir, "schema.json")
    schema = load_pcvr_schema(schema_path, config.seq_max_lens)
    groups = list_row_groups(config.data_dir)

    external_eval = config.eval_data_dir is not None
    if external_eval:
        # Use every training row group; sub-select via train_row_groups if set.
        n_train = len(groups)
        if config.train_row_groups > 0:
            selected = min(int(config.train_row_groups), n_train)
        elif config.train_ratio < 1.0:
            selected = max(1, int(n_train * config.train_ratio))
        else:
            selected = n_train
        if selected < n_train:
            train_indices = [
                ((2 * index + 1) * n_train) // (2 * selected)
                for index in range(selected)
            ]
        else:
            train_indices = list(range(n_train))
        valid_range = (n_train, n_train)  # unused
    else:
        train_indices, valid_range = split_row_groups(
            groups,
            valid_ratio=config.valid_ratio,
            train_ratio=config.train_ratio,
            train_row_groups=config.train_row_groups,
        )
    if world_size > 1:
        # Round-robin sharding: rank r takes indices [r, r+ws, r+2ws, ...].
        # This preserves the even-spacing property of split_row_groups.
        train_indices = train_indices[rank::world_size]
        if not train_indices:
            raise ValueError(
                f"rank {rank} received no training row groups; "
                f"reduce world_size or increase train_row_groups"
            )
    train = PCVRParquetDataset(
        schema,
        [groups[index] for index in train_indices],
        clip_vocab=config.clip_vocab,
        keep_recent=config.keep_recent,
        shuffle=config.shuffle,
        buffer_rows=config.buffer_rows,
        read_batch_size=config.read_batch_size,
        seed=config.seed + rank,
        is_training=True,
    )
    if external_eval:
        eval_schema_path = config.eval_schema_path or os.path.join(
            config.eval_data_dir, "schema.json"
        )
        # Reuse the train schema so vocab/dense layouts match the model; the
        # predict schema is only referenced to sanity-check column presence.
        _ = load_pcvr_schema(eval_schema_path, config.seq_max_lens)
        answer_labels = load_answer_labels(config.answer_path)
        eval_groups = list_row_groups(config.eval_data_dir)
        valid = PCVRParquetDataset(
            schema,
            eval_groups,
            clip_vocab=config.clip_vocab,
            keep_recent=config.keep_recent,
            shuffle=False,
            buffer_rows=0,
            read_batch_size=config.read_batch_size,
            seed=config.seed,
            is_training=False,
            answer_labels=answer_labels,
        )
    else:
        valid_groups = groups[valid_range[0] : valid_range[1]]
        if not valid_groups:
            valid_groups = [groups[-1]]
        valid = PCVRParquetDataset(
            schema,
            valid_groups,
            clip_vocab=config.clip_vocab,
            keep_recent=config.keep_recent,
            shuffle=False,
            buffer_rows=0,
            read_batch_size=config.read_batch_size,
            seed=config.seed,
            is_training=True,
        )
    return schema, train, valid
