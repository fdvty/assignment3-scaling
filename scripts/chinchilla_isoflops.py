#!/usr/bin/env python3
"""Solve the Assignment 3 IsoFLOPs scaling-law exercise.

The handout recommends using, for each fixed compute budget, the run with the
lowest final loss as the IsoFLOPs optimum. We then fit log-log power laws:

    N_opt(C) = alpha_N * C ** beta_N
    D_opt(C) = alpha_D * C ** beta_D

where D_opt is computed from the Transformer training FLOPs approximation
C = 6 N D.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TARGET_COMPUTE_BUDGETS = (1e23, 1e24)


@dataclass(frozen=True)
class Run:
    parameters: int
    compute_budget: float
    final_loss: float

    @property
    def dataset_tokens(self) -> float:
        return self.compute_budget / (6.0 * self.parameters)


@dataclass(frozen=True)
class PowerLaw:
    intercept_log10: float
    exponent: float
    r2_log10: float

    def predict(self, compute_budget: float) -> float:
        return 10 ** self.intercept_log10 * compute_budget**self.exponent


def load_runs(path: Path) -> list[Run]:
    raw_runs = json.loads(path.read_text())
    return [
        Run(
            parameters=int(run["parameters"]),
            compute_budget=float(run["compute_budget"]),
            final_loss=float(run["final_loss"]),
        )
        for run in raw_runs
    ]


def best_runs_by_compute(runs: Iterable[Run]) -> list[Run]:
    best: dict[float, Run] = {}
    for run in runs:
        current_best = best.get(run.compute_budget)
        if current_best is None or run.final_loss < current_best.final_loss:
            best[run.compute_budget] = run
    return [best[compute] for compute in sorted(best)]


def fit_power_law(points: Iterable[tuple[float, float]]) -> PowerLaw:
    xs = [math.log10(x) for x, _ in points]
    ys = [math.log10(y) for _, y in points]
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need at least two points to fit a power law")

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("all x values are identical")

    exponent = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - exponent * x_mean
    residual_sum_squares = sum(
        (y - (intercept + exponent * x)) ** 2 for x, y in zip(xs, ys)
    )
    total_sum_squares = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - residual_sum_squares / total_sum_squares
    return PowerLaw(intercept_log10=intercept, exponent=exponent, r2_log10=r2)


def format_scientific(value: float) -> str:
    return f"{value:.4e}"


def svg_log_plot(
    *,
    points: list[tuple[float, float]],
    law: PowerLaw,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    width = 900
    height = 620
    margin_left = 96
    margin_right = 36
    margin_top = 62
    margin_bottom = 82
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_min_log = math.floor(min(math.log10(x) for x, _ in points))
    x_max_log = math.ceil(max(24.0, max(math.log10(x) for x, _ in points)))

    line_samples: list[tuple[float, float]] = []
    for idx in range(160):
        x_log = x_min_log + (x_max_log - x_min_log) * idx / 159
        x = 10**x_log
        line_samples.append((x, law.predict(x)))

    y_values = [y for _, y in points] + [y for _, y in line_samples]
    y_min_log = math.floor(min(math.log10(y) for y in y_values))
    y_max_log = math.ceil(max(math.log10(y) for y in y_values))
    if y_min_log == y_max_log:
        y_max_log += 1

    def sx(x: float) -> float:
        return margin_left + (math.log10(x) - x_min_log) / (
            x_max_log - x_min_log
        ) * plot_width

    def sy(y: float) -> float:
        return margin_top + (y_max_log - math.log10(y)) / (
            y_max_log - y_min_log
        ) * plot_height

    def tick_values(lo: int, hi: int) -> list[float]:
        return [10.0**power for power in range(lo, hi + 1)]

    x_ticks = tick_values(x_min_log, x_max_log)
    y_ticks = tick_values(y_min_log, y_max_log)
    line_path = " ".join(
        f"{'M' if idx == 0 else 'L'} {sx(x):.2f} {sy(y):.2f}"
        for idx, (x, y) in enumerate(line_samples)
    )

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #202124; }",
        ".tick { font-size: 12px; fill: #4b5563; }",
        ".axis-label { font-size: 15px; font-weight: 600; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".caption { font-size: 13px; fill: #4b5563; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.1f}" y="32" text-anchor="middle">{html.escape(title)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]

    for tick in x_ticks:
        x_coord = sx(tick)
        elements.extend(
            [
                f'<line x1="{x_coord:.2f}" y1="{margin_top}" x2="{x_coord:.2f}" y2="{margin_top + plot_height}" stroke="#e2e8f0"/>',
                f'<text class="tick" x="{x_coord:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle">1e{int(math.log10(tick))}</text>',
            ]
        )

    for tick in y_ticks:
        y_coord = sy(tick)
        elements.extend(
            [
                f'<line x1="{margin_left}" y1="{y_coord:.2f}" x2="{margin_left + plot_width}" y2="{y_coord:.2f}" stroke="#e2e8f0"/>',
                f'<text class="tick" x="{margin_left - 12}" y="{y_coord + 4:.2f}" text-anchor="end">1e{int(math.log10(tick))}</text>',
            ]
        )

    elements.extend(
        [
            f'<path d="{line_path}" fill="none" stroke="#2563eb" stroke-width="3"/>',
        ]
    )

    for x, y in points:
        elements.extend(
            [
                f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5.5" fill="#dc2626" stroke="#7f1d1d" stroke-width="1"/>',
                f'<title>C={format_scientific(x)}, value={format_scientific(y)}</title>',
            ]
        )

    for target in TARGET_COMPUTE_BUDGETS:
        x_coord = sx(target)
        elements.append(
            f'<line x1="{x_coord:.2f}" y1="{margin_top}" x2="{x_coord:.2f}" y2="{margin_top + plot_height}" stroke="#64748b" stroke-dasharray="6 5"/>'
        )

    elements.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#0f172a" stroke-width="1.5"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#0f172a" stroke-width="1.5"/>',
            f'<text class="axis-label" x="{margin_left + plot_width / 2:.1f}" y="{height - 24}" text-anchor="middle">Compute budget C (FLOPs, log scale)</text>',
            f'<text class="axis-label" x="22" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {margin_top + plot_height / 2:.1f})">{html.escape(y_label)}</text>',
            f'<text class="caption" x="{margin_left}" y="{height - 54}">Red points: lowest-loss run per IsoFLOPs profile. Blue line: log-log least-squares power law. Dashed lines: 1e23 and 1e24 FLOPs.</text>',
            f'<text class="caption" x="{margin_left}" y="{height - 38}">Fit: log10(y) = {law.intercept_log10:.4f} + {law.exponent:.4f} log10(C), R2(log10) = {law.r2_log10:.4f}</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(elements) + "\n")


def write_summary(
    *,
    output_path: Path,
    best_runs: list[Run],
    model_law: PowerLaw,
    dataset_law: PowerLaw,
) -> None:
    lines = [
        "# Chinchilla IsoFLOPs Results",
        "",
        "For each compute budget, the optimum is the run with the lowest final loss.",
        "",
        "## IsoFLOPs Optima",
        "",
        "| C FLOPs | N_opt parameters | D_opt tokens | final loss |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for run in best_runs:
        lines.append(
            f"| {format_scientific(run.compute_budget)} | {format_scientific(run.parameters)} | {format_scientific(run.dataset_tokens)} | {run.final_loss:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Fits",
            "",
            f"- Model size: N_opt(C) = 10^{model_law.intercept_log10:.6f} * C^{model_law.exponent:.6f}; R2(log10) = {model_law.r2_log10:.6f}.",
            f"- Dataset size: D_opt(C) = 10^{dataset_law.intercept_log10:.6f} * C^{dataset_law.exponent:.6f}; R2(log10) = {dataset_law.r2_log10:.6f}.",
            "",
            "## Predictions",
            "",
            "| C FLOPs | predicted N_opt parameters | predicted D_opt tokens |",
            "| ---: | ---: | ---: |",
        ]
    )
    for compute_budget in TARGET_COMPUTE_BUDGETS:
        lines.append(
            f"| {format_scientific(compute_budget)} | {format_scientific(model_law.predict(compute_budget))} | {format_scientific(dataset_law.predict(compute_budget))} |"
        )

    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root / "data" / "isoflops_curves.json",
        help="Path to isoflops_curves.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "artifacts" / "chinchilla_isoflops",
        help="Directory for generated plots and summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.data)
    best_runs = best_runs_by_compute(runs)
    model_points = [(run.compute_budget, float(run.parameters)) for run in best_runs]
    dataset_points = [(run.compute_budget, run.dataset_tokens) for run in best_runs]

    model_law = fit_power_law(model_points)
    dataset_law = fit_power_law(dataset_points)

    svg_log_plot(
        points=model_points,
        law=model_law,
        output_path=args.out_dir / "model_size_scaling.svg",
        title="IsoFLOPs Scaling Law for Model Size",
        y_label="N_opt parameters (log scale)",
    )
    svg_log_plot(
        points=dataset_points,
        law=dataset_law,
        output_path=args.out_dir / "dataset_size_scaling.svg",
        title="IsoFLOPs Scaling Law for Dataset Size",
        y_label="D_opt tokens (log scale)",
    )
    write_summary(
        output_path=args.out_dir / "summary.md",
        best_runs=best_runs,
        model_law=model_law,
        dataset_law=dataset_law,
    )

    print("IsoFLOPs optima")
    print("C FLOPs,N_opt parameters,D_opt tokens,final loss")
    for run in best_runs:
        print(
            f"{format_scientific(run.compute_budget)},{format_scientific(run.parameters)},{format_scientific(run.dataset_tokens)},{run.final_loss:.6f}"
        )

    print()
    print(
        f"Model fit: log10(N) = {model_law.intercept_log10:.6f} + {model_law.exponent:.6f} log10(C), R2 = {model_law.r2_log10:.6f}"
    )
    print(
        f"Dataset fit: log10(D) = {dataset_law.intercept_log10:.6f} + {dataset_law.exponent:.6f} log10(C), R2 = {dataset_law.r2_log10:.6f}"
    )

    print()
    print("Predictions")
    for compute_budget in TARGET_COMPUTE_BUDGETS:
        print(
            f"C={format_scientific(compute_budget)}: N_opt={format_scientific(model_law.predict(compute_budget))}, D_opt={format_scientific(dataset_law.predict(compute_budget))}"
        )
    print()
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
