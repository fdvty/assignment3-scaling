#!/usr/bin/env python3
"""Local 4090 reproduction of the Assignment 3 scaling-law problem.

This is not the official Stanford API path. It runs a small byte-level language
model experiment locally, using the same high-level IsoFLOPs idea:

1. Pick several compute budgets C.
2. For each C, train models with different non-embedding parameter counts N for
   D ~= C / (6N) tokens.
3. Select the lowest validation loss for each C.
4. Fit power laws for N_opt(C) and D_opt(C), then extrapolate.

The script is intentionally self-contained so it can run on machines that only
have Python + PyTorch installed.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TARGET_COMPUTE = 1.0e11


@dataclass(frozen=True)
class ModelSpec:
    num_layers: int
    d_model: int
    num_heads: int

    @property
    def non_embedding_params(self) -> int:
        return 12 * self.num_layers * self.d_model * self.d_model

    @property
    def name(self) -> str:
        return f"L{self.num_layers}_D{self.d_model}_H{self.num_heads}"


@dataclass(frozen=True)
class PowerLaw:
    intercept_log10: float
    exponent: float
    r2_log10: float

    def predict(self, compute_budget: float) -> float:
        return 10 ** self.intercept_log10 * compute_budget**self.exponent


def compute_budget(non_embedding_params: int, train_tokens: int) -> float:
    return 6.0 * non_embedding_params * train_tokens


def round_train_tokens(tokens: float, *, tokens_per_step: int) -> int:
    rounded = max(tokens_per_step, int(tokens // tokens_per_step) * tokens_per_step)
    return rounded


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(num_layers=1, d_model=16, num_heads=2),
        ModelSpec(num_layers=1, d_model=24, num_heads=4),
        ModelSpec(num_layers=1, d_model=32, num_heads=4),
        ModelSpec(num_layers=1, d_model=48, num_heads=4),
        ModelSpec(num_layers=1, d_model=64, num_heads=4),
        ModelSpec(num_layers=2, d_model=64, num_heads=4),
        ModelSpec(num_layers=2, d_model=96, num_heads=6),
        ModelSpec(num_layers=2, d_model=128, num_heads=8),
    ]


def default_compute_budgets() -> list[float]:
    return [3.0e9, 1.0e10, 3.0e10, 1.0e11, 3.0e11, 1.0e12]


def load_byte_data(path: Path, *, max_bytes: int, val_tokens: int):
    import torch

    with path.open("rb") as f:
        raw = f.read(max_bytes if max_bytes > 0 else -1)
    min_size = val_tokens + 100_000
    if len(raw) < min_size:
        raise ValueError(f"data file is too small for this experiment: {path}")
    values = torch.tensor(list(raw), dtype=torch.long)
    return values[:-val_tokens], values[-val_tokens:]


class ByteTransformerLM:
    @staticmethod
    def build(
        *,
        vocab_size: int,
        seq_len: int,
        spec: ModelSpec,
        dropout: float,
    ):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, spec.d_model)
                self.position_embedding = nn.Embedding(seq_len, spec.d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=spec.d_model,
                    nhead=spec.num_heads,
                    dim_feedforward=4 * spec.d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.layers = nn.TransformerEncoder(layer, num_layers=spec.num_layers)
                self.norm = nn.LayerNorm(spec.d_model)
                self.lm_head = nn.Linear(spec.d_model, vocab_size, bias=False)
                mask = torch.full((seq_len, seq_len), float("-inf"))
                self.register_buffer("causal_mask", torch.triu(mask, diagonal=1))

            def forward(self, input_ids):
                batch_size, sequence_length = input_ids.shape
                positions = torch.arange(
                    sequence_length, device=input_ids.device
                ).expand(batch_size, sequence_length)
                hidden = self.token_embedding(input_ids) + self.position_embedding(
                    positions
                )
                hidden = self.layers(
                    hidden,
                    mask=self.causal_mask[:sequence_length, :sequence_length],
                    is_causal=True,
                )
                hidden = self.norm(hidden)
                return self.lm_head(hidden)

        return Model()


def cosine_lr(
    *,
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_frac: float,
    final_lr_frac: float,
) -> float:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / denom)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (final_lr_frac + (1.0 - final_lr_frac) * cosine)


def validation_loss(model, val_tokens, *, seq_len: int, batch_size: int, device: str):
    import torch
    import torch.nn.functional as F

    model.eval()
    n_sequences = max(1, (len(val_tokens) - 1) // seq_len)
    n_batches = math.ceil(n_sequences / batch_size)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_sequence = batch_idx * batch_size
            end_sequence = min(n_sequences, start_sequence + batch_size)
            starts = torch.arange(start_sequence, end_sequence) * seq_len
            xs = torch.stack([val_tokens[s : s + seq_len] for s in starts]).to(device)
            ys = torch.stack(
                [val_tokens[s + 1 : s + seq_len + 1] for s in starts]
            ).to(device)
            logits = model(xs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                ys.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += int(ys.numel())
    model.train()
    return total_loss / total_tokens


def run_one(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    device = args.device
    train_tokens_all, val_tokens = load_byte_data(
        args.data, max_bytes=args.max_bytes, val_tokens=args.val_tokens
    )
    spec = ModelSpec(args.num_layers, args.d_model, args.num_heads)
    tokens_per_step = args.batch_size * args.seq_len
    total_train_tokens = round_train_tokens(
        args.total_train_tokens, tokens_per_step=tokens_per_step
    )
    max_tokens = len(train_tokens_all) - args.seq_len - 1
    if total_train_tokens > max_tokens:
        raise ValueError(
            f"requested {total_train_tokens} train tokens, but data only supports {max_tokens}"
        )
    total_steps = total_train_tokens // tokens_per_step

    model = ByteTransformerLM.build(
        vocab_size=256,
        seq_len=args.seq_len,
        spec=spec,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    start_time = time.perf_counter()
    train_losses: list[float] = []
    for step in range(total_steps):
        lr = cosine_lr(
            step=step,
            total_steps=total_steps,
            peak_lr=args.peak_lr,
            warmup_frac=args.warmup_frac,
            final_lr_frac=args.final_lr_frac,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        offset = step * tokens_per_step
        chunk = train_tokens_all[offset : offset + tokens_per_step + 1]
        xs = chunk[:-1].reshape(args.batch_size, args.seq_len).to(device)
        ys = chunk[1:].reshape(args.batch_size, args.seq_len).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            logits = model(xs)
            loss = F.cross_entropy(logits.reshape(-1, 256), ys.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        train_losses.append(float(loss.item()))

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start_time
    val_loss = validation_loss(
        model,
        val_tokens,
        seq_len=args.seq_len,
        batch_size=args.val_batch_size,
        device=device,
    )
    actual_compute_budget = compute_budget(
        spec.non_embedding_params, total_train_tokens
    )
    result = {
        "status": "completed",
        "model_name": spec.name,
        "num_layers": spec.num_layers,
        "d_model": spec.d_model,
        "num_heads": spec.num_heads,
        "non_embedding_params": spec.non_embedding_params,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "total_train_tokens": total_train_tokens,
        "target_compute_budget": args.target_compute_budget,
        "actual_compute_budget": actual_compute_budget,
        "final_val_loss": val_loss,
        "mean_train_loss": sum(train_losses) / len(train_losses),
        "train_seconds": train_seconds,
        "tokens_per_second": total_train_tokens / train_seconds,
        "peak_lr": args.peak_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": device,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")
    return result


def fit_power_law(points: Iterable[tuple[float, float]]) -> PowerLaw:
    xs = [math.log10(x) for x, _ in points]
    ys = [math.log10(y) for _, y in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("cannot fit power law with identical x values")
    exponent = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - exponent * x_mean
    rss = sum((y - (intercept + exponent * x)) ** 2 for x, y in zip(xs, ys))
    tss = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - rss / tss if tss else 1.0
    return PowerLaw(intercept, exponent, r2)


def fit_loss_line(points: Iterable[tuple[float, float]]) -> tuple[float, float, float]:
    xs = [math.log10(x) for x, _ in points]
    ys = [y for _, y in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    tss = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - rss / tss if tss else 1.0
    return intercept, slope, r2


def load_results(path: Path) -> list[dict[str, object]]:
    results = []
    for line in path.read_text().splitlines():
        if line.strip():
            result = json.loads(line)
            if result.get("status") == "completed":
                results.append(result)
    return results


def best_by_budget(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, dict[str, object]] = {}
    for result in results:
        budget = float(result["target_compute_budget"])
        current = grouped.get(budget)
        if current is None or float(result["final_val_loss"]) < float(
            current["final_val_loss"]
        ):
            grouped[budget] = result
    return [grouped[budget] for budget in sorted(grouped)]


def format_scientific(value: float) -> str:
    return f"{value:.4e}"


def write_svg_plot(
    *,
    output: Path,
    title: str,
    y_label: str,
    points: list[tuple[float, float]],
    law: PowerLaw,
    target_compute: float,
) -> None:
    width = 860
    height = 560
    left = 92
    right = 36
    top = 58
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = math.floor(min(math.log10(x) for x, _ in points))
    x_max = math.ceil(max(math.log10(target_compute), max(math.log10(x) for x, _ in points)))
    line_points = []
    for idx in range(120):
        lx = x_min + (x_max - x_min) * idx / 119
        x = 10**lx
        line_points.append((x, law.predict(x)))
    y_values = [y for _, y in points] + [y for _, y in line_points]
    y_min = math.floor(min(math.log10(y) for y in y_values))
    y_max = math.ceil(max(math.log10(y) for y in y_values))

    def sx(x: float) -> float:
        return left + (math.log10(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - math.log10(y)) / (y_max - y_min) * plot_h

    path = " ".join(
        f"{'M' if idx == 0 else 'L'} {sx(x):.2f} {sy(y):.2f}"
        for idx, (x, y) in enumerate(line_points)
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#111827}.tick{font-size:12px;fill:#4b5563}.label{font-size:14px;font-weight:600}.title{font-size:21px;font-weight:700}.caption{font-size:12px;fill:#4b5563}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{width / 2}" y="32" text-anchor="middle">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]
    for power in range(x_min, x_max + 1):
        x = sx(10**power)
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        elements.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle">1e{power}</text>')
    for power in range(y_min, y_max + 1):
        y = sy(10**power)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        elements.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">1e{power}</text>')
    elements.append(f'<path d="{path}" fill="none" stroke="#2563eb" stroke-width="3"/>')
    for x, y in points:
        elements.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5.5" fill="#dc2626"/>')
    tx = sx(target_compute)
    elements.append(f'<line x1="{tx:.2f}" y1="{top}" x2="{tx:.2f}" y2="{top + plot_h}" stroke="#64748b" stroke-dasharray="6 5"/>')
    elements.append(f'<text class="label" x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle">Compute C = 6ND, log scale</text>')
    elements.append(f'<text class="label" x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})">{html.escape(y_label)}</text>')
    elements.append(f'<text class="caption" x="{left}" y="{height - 48}">log10(y) = {law.intercept_log10:.4f} + {law.exponent:.4f} log10(C), R2 = {law.r2_log10:.4f}</text>')
    elements.append("</svg>")
    output.write_text("\n".join(elements) + "\n")


def fit_and_report(args: argparse.Namespace) -> None:
    results = load_results(args.results)
    if not results:
        raise ValueError(f"no completed results in {args.results}")
    best = best_by_budget(results)
    if len(best) < 2:
        raise ValueError("need completed runs for at least two compute budgets")

    model_points = [
        (float(row["target_compute_budget"]), float(row["non_embedding_params"]))
        for row in best
    ]
    data_points = [
        (float(row["target_compute_budget"]), float(row["total_train_tokens"]))
        for row in best
    ]
    loss_points = [
        (float(row["target_compute_budget"]), float(row["final_val_loss"]))
        for row in best
    ]
    model_law = fit_power_law(model_points)
    data_law = fit_power_law(data_points)
    loss_intercept, loss_slope, loss_r2 = fit_loss_line(loss_points)
    predicted_loss = loss_intercept + loss_slope * math.log10(args.target_compute)
    predicted_n = model_law.predict(args.target_compute)
    predicted_d = data_law.predict(args.target_compute)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_svg_plot(
        output=out_dir / "local_model_size_scaling.svg",
        title="Local 4090 IsoFLOPs Model-Size Scaling",
        y_label="N_opt non-embedding parameters",
        points=model_points,
        law=model_law,
        target_compute=args.target_compute,
    )
    write_svg_plot(
        output=out_dir / "local_dataset_size_scaling.svg",
        title="Local 4090 IsoFLOPs Dataset-Size Scaling",
        y_label="D_opt train tokens",
        points=data_points,
        law=data_law,
        target_compute=args.target_compute,
    )

    lines = [
        "# Local 4090 Scaling-Law Results",
        "",
        "This is a local reproduction of Assignment 3 Problem 3.3, not an official Stanford API submission.",
        f"Fitted result file: `{args.results}`.",
        "",
        "## Best IsoFLOPs Runs",
        "",
        "| target C | model | N non-emb params | train tokens | val loss | seconds |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in best:
        lines.append(
            f"| {format_scientific(float(row['target_compute_budget']))} | {row['model_name']} | {format_scientific(float(row['non_embedding_params']))} | {format_scientific(float(row['total_train_tokens']))} | {float(row['final_val_loss']):.6f} | {float(row['train_seconds']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Fits",
            "",
            f"- N_opt(C) = 10^{model_law.intercept_log10:.6f} * C^{model_law.exponent:.6f}; R2(log10) = {model_law.r2_log10:.6f}.",
            f"- D_opt(C) = 10^{data_law.intercept_log10:.6f} * C^{data_law.exponent:.6f}; R2(log10) = {data_law.r2_log10:.6f}.",
            f"- Loss fit: L(C) = {loss_intercept:.6f} + {loss_slope:.6f} log10(C); R2 = {loss_r2:.6f}.",
            "",
            "## Prediction",
            "",
            f"For target C = {format_scientific(args.target_compute)}, predicted N_opt = {format_scientific(predicted_n)}, predicted D_opt = {format_scientific(predicted_d)}, predicted validation loss = {predicted_loss:.6f}.",
        ]
    )
    (out_dir / "local_scaling_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run_grid(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        args.output.unlink()

    train_tokens_all, _ = load_byte_data(
        args.data, max_bytes=args.max_bytes, val_tokens=args.val_tokens
    )
    max_train_tokens = len(train_tokens_all) - args.seq_len - 1
    for budget in default_compute_budgets():
        for spec in default_model_specs():
            tokens_per_step = args.batch_size * args.seq_len
            train_tokens = round_train_tokens(
                budget / (6.0 * spec.non_embedding_params),
                tokens_per_step=tokens_per_step,
            )
            if train_tokens < args.min_train_tokens:
                continue
            if train_tokens > max_train_tokens:
                print(
                    f"skipping C={format_scientific(budget)} {spec.name}: D={train_tokens} exceeds available train tokens {max_train_tokens}",
                    flush=True,
                )
                continue
            one_args = argparse.Namespace(**vars(args))
            one_args.num_layers = spec.num_layers
            one_args.d_model = spec.d_model
            one_args.num_heads = spec.num_heads
            one_args.total_train_tokens = train_tokens
            one_args.target_compute_budget = budget
            print(
                f"running C={format_scientific(budget)} {spec.name} N={spec.non_embedding_params} D={train_tokens}",
                flush=True,
            )
            result = run_one(one_args)
            print(
                f"completed val_loss={result['final_val_loss']:.6f} seconds={result['train_seconds']:.2f}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_data = (
        repo_root / "assignment1-basics" / "tests" / "fixtures" / "tinystories_sample_5M.txt"
    )
    default_results = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "local_4090_scaling"
        / "results.jsonl"
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_one_parser = subparsers.add_parser("run-one")
    add_train_args(run_one_parser, default_data, default_results)
    run_one_parser.add_argument("--num-layers", type=int, required=True)
    run_one_parser.add_argument("--d-model", type=int, required=True)
    run_one_parser.add_argument("--num-heads", type=int, required=True)
    run_one_parser.add_argument("--total-train-tokens", type=int, required=True)
    run_one_parser.add_argument("--target-compute-budget", type=float, default=0.0)

    grid_parser = subparsers.add_parser("run-grid")
    add_train_args(grid_parser, default_data, default_results)
    grid_parser.add_argument("--min-train-tokens", type=int, default=4096)
    grid_parser.add_argument("--resume", action="store_true")

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--results", type=Path, default=default_results)
    fit_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "local_4090_scaling",
    )
    fit_parser.add_argument("--target-compute", type=float, default=DEFAULT_TARGET_COMPUTE)
    return parser.parse_args()


def add_train_args(parser: argparse.ArgumentParser, default_data: Path, default_results: Path) -> None:
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_results)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-bytes", type=int, default=64_000_000)
    parser.add_argument("--val-tokens", type=int, default=2**18)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--final-lr-frac", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)


def main() -> None:
    args = parse_args()
    if args.command == "run-one":
        result = run_one(args)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "run-grid":
        run_grid(args)
    elif args.command == "fit":
        fit_and_report(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
