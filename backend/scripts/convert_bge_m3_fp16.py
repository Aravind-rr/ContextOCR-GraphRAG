"""Create memory-efficient FP16 safetensor shards from the official BGE-M3 checkpoint."""

from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


SOURCE = Path("models/bge-m3")
TARGET = Path("models/bge-m3-fp16")
MAX_SHARD_BYTES = 300 * 1024 * 1024


def converted_size(tensor: torch.Tensor) -> int:
    element_size = 2 if tensor.is_floating_point() else tensor.element_size()
    return tensor.numel() * element_size


def main() -> None:
    checkpoint = SOURCE / "pytorch_model.bin"
    if not checkpoint.is_file():
        raise SystemExit(f"Missing source checkpoint: {checkpoint}")
    TARGET.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key, tensor in state.items():
        size = converted_size(tensor)
        if current and current_size + size > MAX_SHARD_BYTES:
            groups.append(current)
            current, current_size = [], 0
        current.append(key)
        current_size += size
    if current:
        groups.append(current)

    weight_map: dict[str, str] = {}
    total_size = 0
    for index, keys in enumerate(groups, 1):
        name = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
        shard: dict[str, torch.Tensor] = {}
        for key in keys:
            tensor = state[key]
            converted = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
            shard[key] = converted.contiguous().clone()
            weight_map[key] = name
            total_size += shard[key].numel() * shard[key].element_size()
        save_file(shard, TARGET / name, metadata={"format": "pt"})
        print(f"Wrote {name}", flush=True)
        del shard
        gc.collect()

    (TARGET / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
    for source in SOURCE.iterdir():
        if source.is_file() and source.name not in {"pytorch_model.bin"} and source.suffix != ".pt":
            shutil.copy2(source, TARGET / source.name)
        elif source.is_dir() and source.name == "1_Pooling":
            shutil.copytree(source, TARGET / source.name, dirs_exist_ok=True)
    print(f"FP16 BGE-M3 checkpoint ready at {TARGET.resolve()}", flush=True)


if __name__ == "__main__":
    main()
