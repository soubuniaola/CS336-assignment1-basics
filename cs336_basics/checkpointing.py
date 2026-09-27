import os
import torch
import typing
from torch import Tensor, nn, optim

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    obj_dict = dict(
        model_state_dict=model_state_dict,
        optimizer_state_dict=optimizer_state_dict,
        iteration=iteration,
    )
    torch.save(obj_dict, out)

def load_checkpoint(src, model, optimizer):
    obj_dict = torch.load(src)
    model.load_state_dict(obj_dict["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(obj_dict["optimizer_state_dict"])
    iteration = obj_dict["iteration"]

    return iteration