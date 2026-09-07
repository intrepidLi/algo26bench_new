"""DDP setup helpers.

`torchrun` populates ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` env vars; we
initialise NCCL from those.  Single-process runs (``WORLD_SIZE`` unset or
``1``) fall through as no-ops so the same Trainer code path serves both.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from algo26bench.core.protocols import DistributedInfo


def read_env() -> DistributedInfo:
    """Read ``torchrun``-style env vars into a DistributedInfo."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedInfo()
    return DistributedInfo(
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ.get("LOCAL_RANK", os.environ["RANK"])),
        world_size=world_size,
    )


def setup(info: DistributedInfo, *, backend: str = "nccl") -> None:
    """Initialise the process group and pin the CUDA device for this rank."""

    if not info.enabled:
        return
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)


def shutdown(info: DistributedInfo) -> None:
    if info.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(info: DistributedInfo) -> None:
    if info.enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


__all__ = ["barrier", "read_env", "setup", "shutdown"]
