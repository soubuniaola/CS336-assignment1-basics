"""Train the Assignment 1 Transformer language model.

Run this module from the repository root, for example:

    python -m cs336_basics.training_together \
        --train-data cs336_basics/tokenized_dataset/train_tokens.bin \
        --val-data cs336_basics/tokenized_dataset/valid_tokens.bin \
        --checkpoint-path checkpoints/tinystories.pt

The input files must contain a flat, headerless sequence of token IDs. Their
dtype must match ``--data-dtype`` (the tokenizer script originally wrote
``int32``). Both files are memory-mapped, so they need not fit in RAM.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.basic_transformer_lm import BasicTransformerLM
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.function_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW, cos_lr


@dataclass(frozen=True)
class TrainConfig:
    """Command-line configuration for a training run."""

    train_data: Path
    val_data: Path
    checkpoint_path: Path
    resume_from: Path | None
    data_dtype: str
    device: str
    seed: int

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    d_ff: int
    num_heads: int
    rope_theta: float

    batch_size: int
    train_steps: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    cosine_decay_steps: int
    beta1: float
    beta2: float
    weight_decay: float
    eps: float
    max_grad_norm: float | None

    log_every: int
    eval_every: int
    eval_batches: int
    checkpoint_every: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)

    data = parser.add_argument_group("data and output")
    data.add_argument("--train-data", type=Path, required=True, help="Flat binary training-token file.")
    data.add_argument("--val-data", type=Path, required=True, help="Flat binary validation-token file.")
    data.add_argument("--checkpoint-path", type=Path, required=True, help="Checkpoint file to overwrite periodically.")
    data.add_argument("--resume-from", type=Path, default=None, help="Checkpoint from which to resume.")
    data.add_argument(
        "--data-dtype",
        choices=("uint16", "uint32", "int32", "int64"),
        default="int32",
        help="On-disk token dtype. It must match the preprocessing output.",
    )
    data.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or a device such as cuda:0.")
    data.add_argument("--seed", type=int, default=42)

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=_positive_int, default=10_000)
    model.add_argument("--context-length", type=_positive_int, default=256)
    model.add_argument("--d-model", type=_positive_int, default=512)
    model.add_argument("--num-layers", type=_positive_int, default=4)
    model.add_argument("--d-ff", type=_positive_int, default=1_344)
    model.add_argument("--num-heads", type=_positive_int, default=16)
    model.add_argument("--rope-theta", type=float, default=10_000.0)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--batch-size", type=_positive_int, default=64)
    optimization.add_argument("--train-steps", type=_positive_int, default=20_000)
    optimization.add_argument("--max-lr", type=float, default=3e-4)
    optimization.add_argument("--min-lr", type=float, default=3e-5)
    optimization.add_argument("--warmup-steps", type=int, default=500)
    optimization.add_argument(
        "--cosine-decay-steps",
        type=int,
        default=None,
        help="End of cosine decay; defaults to --train-steps.",
    )
    optimization.add_argument("--beta1", type=float, default=0.9)
    optimization.add_argument("--beta2", type=float, default=0.95)
    optimization.add_argument("--weight-decay", type=float, default=0.1)
    optimization.add_argument("--eps", type=float, default=1e-8)
    optimization.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="L2 gradient-clipping threshold; use 0 to disable clipping.",
    )

    reporting = parser.add_argument_group("reporting")
    reporting.add_argument("--log-every", type=_positive_int, default=10)
    reporting.add_argument("--eval-every", type=_positive_int, default=250)
    reporting.add_argument("--eval-batches", type=_positive_int, default=20)
    reporting.add_argument("--checkpoint-every", type=_positive_int, default=1_000)

    args = parser.parse_args()
    cosine_decay_steps = args.cosine_decay_steps if args.cosine_decay_steps is not None else args.train_steps
    max_grad_norm = None if args.max_grad_norm == 0 else args.max_grad_norm

    config = TrainConfig(
        train_data=args.train_data,
        val_data=args.val_data,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume_from,
        data_dtype=args.data_dtype,
        device=args.device,
        seed=args.seed,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        cosine_decay_steps=cosine_decay_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        eps=args.eps,
        max_grad_norm=max_grad_norm,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_every=args.checkpoint_every,
    )
    validate_config(config)
    return config


def validate_config(config: TrainConfig) -> None:
    if config.d_model % config.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads")
    if config.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if config.cosine_decay_steps <= config.warmup_steps:
        raise ValueError("--cosine-decay-steps must be greater than --warmup-steps")
    if not 0.0 <= config.min_lr <= config.max_lr:
        raise ValueError("learning rates must satisfy 0 <= min_lr <= max_lr")
    if not 0.0 <= config.beta1 < 1.0 or not 0.0 <= config.beta2 < 1.0:
        raise ValueError("AdamW betas must be in [0, 1)")
    if config.weight_decay < 0.0 or config.eps <= 0.0:
        raise ValueError("--weight-decay must be non-negative and --eps must be positive")
    if config.max_grad_norm is not None and config.max_grad_norm < 0.0:
        raise ValueError("--max-grad-norm must be non-negative")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but MPS is unavailable")
    return device


def open_token_data(path: Path, dtype: str, context_length: int) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(f"Token file does not exist: {path}")

    # mode="r" prevents an accidental training-time mutation of the corpus.
    dataset = np.memmap(path, dtype=np.dtype(dtype), mode="r")
    if len(dataset) <= context_length:
        raise ValueError(
            f"{path} contains {len(dataset)} tokens, but more than {context_length} are required"
        )
    return dataset


def sample_batch(
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample next-token examples and guarantee placement on every device type."""
    inputs, targets = get_batch(dataset, batch_size, context_length, str(device))
    # get_batch already uses pinned asynchronous copies for CUDA. These calls
    # are no-ops there, but also cover CPU and MPS consistently.
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_validation_loss(
    model: BasicTransformerLM,
    dataset: np.memmap,
    config: TrainConfig,
    device: torch.device,
) -> float:
    """Average validation loss over fresh randomly sampled batches."""
    model.eval()
    total_loss = 0.0
    for _ in range(config.eval_batches):
        inputs, targets = sample_batch(dataset, config.batch_size, config.context_length, device)
        total_loss += cross_entropy(model(inputs), targets).item()
    model.train()
    return total_loss / config.eval_batches


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)

    train_data = open_token_data(config.train_data, config.data_dtype, config.context_length)
    val_data = open_token_data(config.val_data, config.data_dtype, config.context_length)

    model = BasicTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        num_heads=config.num_heads,
        rope_theta=config.rope_theta,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.eps,
    )

    # Checkpoints store the number of completed updates, so the range starts
    # at this value without repeating the last optimizer step.
    start_step = 0
    if config.resume_from is not None:
        if not config.resume_from.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {config.resume_from}")
        start_step = load_checkpoint(config.resume_from, model, optimizer)
        print(f"resumed_from={config.resume_from} completed_steps={start_step}", flush=True)
    if start_step >= config.train_steps:
        raise ValueError(
            f"checkpoint is already at step {start_step}, not below --train-steps={config.train_steps}"
        )

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    steps_since_log = 0
    training_time_since_log = 0.0

    for step in range(start_step, config.train_steps):
        step_started = time.perf_counter()
        learning_rate = cos_lr(
            step,
            min_lr=config.min_lr,
            max_lr=config.max_lr,
            warmup_iters=config.warmup_steps,
            cosine_annealing_iters=config.cosine_decay_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = sample_batch(train_data, config.batch_size, config.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}: {loss.item()}")
        loss.backward()
        if config.max_grad_norm is not None:
            clip_gradient(model.parameters(), config.max_grad_norm)
        optimizer.step()
        training_time_since_log += time.perf_counter() - step_started
        steps_since_log += 1

        completed_steps = step + 1
        if completed_steps % config.log_every == 0 or completed_steps == 1:
            tokens_per_second = (
                steps_since_log * config.batch_size * config.context_length / training_time_since_log
            )
            print(
                f"step={completed_steps}/{config.train_steps} train_loss={loss.item():.4f} "
                f"lr={learning_rate:.3e} tokens_per_second={tokens_per_second:,.0f}",
                flush=True,
            )
            steps_since_log = 0
            training_time_since_log = 0.0

        if completed_steps % config.eval_every == 0 or completed_steps == config.train_steps:
            val_loss = estimate_validation_loss(model, val_data, config, device)
            perplexity = math.exp(val_loss) if val_loss < 100 else math.inf
            print(
                f"step={completed_steps}/{config.train_steps} val_loss={val_loss:.4f} "
                f"val_perplexity={perplexity:.2f}",
                flush=True,
            )

        if completed_steps % config.checkpoint_every == 0:
            save_checkpoint(model, optimizer, completed_steps, config.checkpoint_path)
            print(f"checkpoint={config.checkpoint_path} completed_steps={completed_steps}", flush=True)

    # Always persist the final state, even when train_steps is not a multiple
    # of checkpoint_every.
    save_checkpoint(model, optimizer, config.train_steps, config.checkpoint_path)
    print(f"training_complete checkpoint={config.checkpoint_path}", flush=True)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
