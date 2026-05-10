#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INTERESTING_PREFIXES = (
    "turboquant_encode",
    "turboquant_decode",
    "turboquant_fused",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_ms(value: float | int | None) -> str:
    return f"{float(value or 0.0):10.3f}"


def _worker_generate_ms(run_stats: dict[str, Any], variant: str) -> float:
    timing = run_stats.get("workers", {}).get(variant, {}).get("timing", {})
    return float(timing.get("generate_seconds", 0.0)) * 1000.0


def _print_worker_summary(run_stats: dict[str, Any]) -> None:
    workers = run_stats.get("workers", {})
    children = run_stats.get("child_processes", {})
    print("=== Worker Summary ===")
    for name in ("plain", "baseline", "fused"):
        stats = workers.get(name)
        if not stats:
            continue
        timing = stats.get("timing", {})
        throughput = stats.get("throughput", {})
        kv_cache = stats.get("kv_cache", {})
        log_stats = children.get(name, {}).get("log_stats", {})
        kv_tokens = (
            log_stats.get("kv_cache_capacity_tokens")
            or kv_cache.get("capacity_tokens")
            or kv_cache.get("capacity_tokens_estimate")
        )
        print(
            f"{name:8s} "
            f"init_s={float(timing.get('init_seconds', 0.0)):8.3f} "
            f"generate_s={float(timing.get('generate_seconds', 0.0)):8.3f} "
            f"out_tok/s={float(throughput.get('output_tokens_per_second', 0.0)):8.3f} "
            f"total_tok/s={float(throughput.get('total_tokens_per_second', 0.0)):8.3f} "
            f"kv_tokens={kv_tokens}"
        )

    plain_tps = workers.get("plain", {}).get("throughput", {}).get(
        "output_tokens_per_second"
    )
    fused_tps = workers.get("fused", {}).get("throughput", {}).get(
        "output_tokens_per_second"
    )
    if plain_tps and fused_tps:
        print(f"fused/plain out_tok/s ratio: {float(fused_tps) / float(plain_tps):.4f}x")
    print()


def _iter_profile_rows(stats: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = [
        (name, stat)
        for name, stat in stats.items()
        if name.startswith(INTERESTING_PREFIXES)
    ]
    rows.sort(key=lambda item: float(item[1].get("total_ms", 0.0)), reverse=True)
    return rows


def _print_profile_table(
    run_stats: dict[str, Any],
    variant: str,
    *,
    top: int,
) -> None:
    child = run_stats.get("child_processes", {}).get(variant, {})
    profile = child.get("turboquant_profile", {})
    stats = profile.get("stats", {})
    if not stats:
        return

    generate_ms = _worker_generate_ms(run_stats, variant)
    print(f"=== Profile Breakdown: {variant} ===")
    print(f"profile_files={profile.get('num_files', 0)} generate_ms={generate_ms:.3f}")
    print(
        f"{'name':78s} {'calls':>7s} {'total_ms':>10s} {'avg_ms':>9s} "
        f"{'pct_gen':>8s} {'min_ms':>10s} {'max_ms':>10s} {'vectors':>9s}"
    )
    for name, stat in _iter_profile_rows(stats)[:top]:
        calls = int(stat.get("calls", 0))
        total_ms = float(stat.get("total_ms", 0.0))
        avg_ms = total_ms / calls if calls else 0.0
        pct_gen = (total_ms / generate_ms * 100.0) if generate_ms > 0 else 0.0
        print(
            f"{name[:78]:78s} "
            f"{calls:7d} {_fmt_ms(total_ms)} {avg_ms:9.3f} "
            f"{pct_gen:8.2f} {_fmt_ms(stat.get('min_ms'))} "
            f"{_fmt_ms(stat.get('max_ms'))} {int(stat.get('vectors', 0)):9d}"
        )
    print()


def _print_m4_stage_table(run_stats: dict[str, Any], variant: str) -> None:
    stats = (
        run_stats.get("child_processes", {})
        .get(variant, {})
        .get("turboquant_profile", {})
        .get("stats", {})
    )
    prefix = "turboquant_fused_kv_update_attention.decode.m4_attention"
    total = stats.get(f"{prefix}.total")
    if not total:
        return
    total_ms = float(total.get("total_ms", 0.0))
    print(f"=== M4 Stage Breakdown: {variant} ===")
    for suffix in ("prepare_inputs", "custom_op", "output_cast", "total"):
        key = f"{prefix}.{suffix}"
        stat = stats.get(key)
        if not stat:
            continue
        calls = int(stat.get("calls", 0))
        stage_total = float(stat.get("total_ms", 0.0))
        avg = stage_total / calls if calls else 0.0
        pct = stage_total / total_ms * 100.0 if total_ms > 0 else 0.0
        print(
            f"{suffix:15s} calls={calls:5d} total_ms={stage_total:10.3f} "
            f"avg_ms={avg:8.3f} pct_m4={pct:7.2f}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    run_stats = _load_json(args.output_dir / "run_stats.json")
    print(f"Results: {args.output_dir}")
    _print_worker_summary(run_stats)
    for variant in ("baseline", "fused"):
        _print_profile_table(run_stats, variant, top=args.top)
        _print_m4_stage_table(run_stats, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
