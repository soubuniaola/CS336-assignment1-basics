from __future__ import annotations



import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_idx = torch.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,),
    )

    x = torch.stack(
        [
            torch.from_numpy(dataset[i:i+context_length].astype(np.int64))
            for i in starting_idx
        ],
        dim=0
    )

    y = torch.stack(
        [
            torch.from_numpy(dataset[i+1:i+1+context_length].astype(np.int64))
            for i in starting_idx
        ],
        dim=0
    )

    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x.to(device)
        y.to(device)

    return x, y

