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
    "turboquant_chunked_prefill",
    "turboquant_prefill",
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
    for suffix in (
        "prepare_inputs",
        "pretransform_query",
        "custom_op",
        "output_cast",
        "total",
    ):
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


def _avg_ms(stats: dict[str, Any], key: str) -> float:
    stat = stats.get(key, {})
    calls = int(stat.get("calls", 0))
    if calls <= 0:
        return 0.0
    return float(stat.get("total_ms", 0.0)) / calls


def _total_ms(stats: dict[str, Any], key: str) -> float:
    return float(stats.get(key, {}).get("total_ms", 0.0))


def _calls(stats: dict[str, Any], key: str) -> int:
    return int(stats.get(key, {}).get("calls", 0))


def _mode_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 1_000_000


def _print_m4_debug_sweep_summary(output_dir: Path) -> bool:
    mode_dirs = sorted(
        (path for path in output_dir.glob("mode_*") if (path / "run_stats.json").exists()),
        key=_mode_number,
    )
    if not mode_dirs:
        return False

    prefix = "turboquant_fused_kv_update_attention.decode.m4_attention"
    rows: dict[int, dict[str, float]] = {}
    print(f"Results: {output_dir}")
    print("=== M4 Debug Sweep Summary ===")
    print(
        f"{'mode':>4s} {'gen_s':>8s} {'out_tok/s':>9s} "
        f"{'forward':>9s} {'encode':>9s} {'enc_op':>9s} "
        f"{'m4_total':>9s} "
        f"{'custom':>9s} {'prepare':>9s} {'pre_xform':>9s} "
        f"{'out_cast':>9s} {'ref':>9s}"
    )
    mode_stats: dict[int, dict[str, Any]] = {}
    for mode_dir in mode_dirs:
        mode = _mode_number(mode_dir)
        run_stats = _load_json(mode_dir / "run_stats.json")
        mode_stats[mode] = run_stats
        fused = run_stats.get("workers", {}).get("fused", {})
        timing = fused.get("timing", {})
        throughput = fused.get("throughput", {})
        stats = (
            run_stats.get("child_processes", {})
            .get("fused", {})
            .get("turboquant_profile", {})
            .get("stats", {})
        )
        row = {
            "generate_s": float(timing.get("generate_seconds", 0.0)),
            "out_tps": float(throughput.get("output_tokens_per_second", 0.0)),
            "forward": _avg_ms(
                stats, "turboquant_fused_kv_update_attention.forward"
            ),
            "encode": _avg_ms(stats, "turboquant_encode_cache_update.total"),
            "encode_op": _avg_ms(
                stats, "turboquant_encode_cache_update.custom.op"
            ),
            "m4_total": _avg_ms(stats, f"{prefix}.total"),
            "custom": _avg_ms(stats, f"{prefix}.custom_op"),
            "prepare": _avg_ms(stats, f"{prefix}.prepare_inputs"),
            "pretransform": _avg_ms(stats, f"{prefix}.pretransform_query"),
            "out_cast": _avg_ms(stats, f"{prefix}.output_cast"),
            "ref": _avg_ms(
                stats, "turboquant_fused_kv_update_attention.reference.total"
            ),
        }
        rows[mode] = row
        print(
            f"{mode:4d} {row['generate_s']:8.3f} {row['out_tps']:9.3f} "
            f"{row['forward']:9.3f} {row['encode']:9.3f} "
            f"{row['encode_op']:9.3f} "
            f"{row['m4_total']:9.3f} {row['custom']:9.3f} "
            f"{row['prepare']:9.3f} {row['pretransform']:9.3f} "
            f"{row['out_cast']:9.3f} {row['ref']:9.3f}"
        )

    def delta(label: str, lhs: int, rhs: int) -> None:
        if lhs in rows and rhs in rows:
            value = rows[lhs]["custom"] - rows[rhs]["custom"]
            print(f"{label:42s} {value:9.3f} ms/call")

    print()
    print("=== M4 Debug Derived Custom-Op Deltas ===")
    delta("full - full_no_store (mode0 - mode5)", 0, 5)
    delta("query_transform (mode7 - mode6)", 7, 6)
    delta("zero_store_output (mode8 - mode6)", 8, 6)
    delta("current_score+online (mode9 - mode8)", 9, 8)
    delta("current path query delta (mode1 - mode9)", 1, 9)
    delta("history_score only (mode2 - mode7)", 2, 7)
    delta("history+softmax+v no_store (mode5 - mode7)", 5, 7)
    delta("query+history score (mode2 - mode6)", 2, 6)
    print()

    _print_m4_debug_cost_model(rows, mode_stats, prefix)
    return True


def _print_m4_debug_cost_model(
    rows: dict[int, dict[str, float]],
    mode_stats: dict[int, dict[str, Any]],
    prefix: str,
) -> None:
    if 0 not in rows or 0 not in mode_stats:
        return

    stats0 = (
        mode_stats[0]
        .get("child_processes", {})
        .get("fused", {})
        .get("turboquant_profile", {})
        .get("stats", {})
    )
    m4_calls = _calls(stats0, f"{prefix}.custom_op")
    generate_ms = _worker_generate_ms(mode_stats[0], "fused")
    full_custom = rows[0]["custom"]
    if m4_calls <= 0 or full_custom <= 0.0:
        return

    def row_delta(lhs: int, rhs: int) -> float | None:
        if lhs not in rows or rhs not in rows:
            return None
        return rows[lhs]["custom"] - rows[rhs]["custom"]

    components: list[tuple[str, float, str]] = []
    if 6 in rows:
        components.append(("m4_kernel_floor", rows[6]["custom"], "mode6 custom"))
    for name, lhs, rhs, note in (
        ("query_transform", 7, 6, "mode7 - mode6"),
        ("history_score", 2, 7, "mode2 - mode7"),
        ("history_online_v", 5, 7, "mode5 - mode7"),
        ("output_store", 0, 5, "mode0 - mode5"),
        ("zero_store_floor", 8, 6, "mode8 - mode6"),
        ("current_score_online", 9, 8, "mode9 - mode8"),
    ):
        value = row_delta(lhs, rhs)
        if value is not None:
            components.append((name, value, note))

    direct_components = (
        ("m4_prepare_inputs", rows[0]["prepare"], "mode0 stage"),
        ("m4_pretransform_query", rows[0]["pretransform"], "mode0 stage"),
        ("m4_output_cast", rows[0]["out_cast"], "mode0 stage"),
        ("encode_custom_op", rows[0]["encode_op"], "mode0 avg"),
    )

    print("=== M4 Debug Cost Model (mode0 call-count estimate) ===")
    print(
        f"{'component':24s} {'avg_ms':>9s} {'est_total_ms':>12s} "
        f"{'pct_m4_custom':>14s} {'pct_gen':>8s}  source"
    )
    for name, avg, note in components:
        total = avg * m4_calls
        pct_custom = avg / full_custom * 100.0 if full_custom else 0.0
        pct_gen = total / generate_ms * 100.0 if generate_ms > 0 else 0.0
        print(
            f"{name:24s} {avg:9.3f} {total:12.3f} "
            f"{pct_custom:14.2f} {pct_gen:8.2f}  {note}"
        )

    print()
    print("=== Adjacent Module Averages (mode0) ===")
    print(f"{'module':24s} {'avg_ms':>9s} {'total_ms':>12s} {'pct_gen':>8s}")
    for name, avg, note in direct_components:
        if name == "encode_custom_op":
            total = _total_ms(stats0, "turboquant_encode_cache_update.custom.op")
        elif name == "m4_prepare_inputs":
            total = _total_ms(stats0, f"{prefix}.prepare_inputs")
        elif name == "m4_pretransform_query":
            total = _total_ms(stats0, f"{prefix}.pretransform_query")
        else:
            total = _total_ms(stats0, f"{prefix}.output_cast")
        pct_gen = total / generate_ms * 100.0 if generate_ms > 0 else 0.0
        print(f"{name:24s} {avg:9.3f} {total:12.3f} {pct_gen:8.2f}  {note}")
    print()

    candidates: list[tuple[float, str]] = []
    for name, avg, _ in components:
        if avg > 0:
            candidates.append((avg * m4_calls, name))
    encode_total = _total_ms(stats0, "turboquant_encode_cache_update.custom.op")
    if encode_total > 0:
        candidates.append((encode_total, "encode_custom_op"))
    pretransform_total = _total_ms(stats0, f"{prefix}.pretransform_query")
    if pretransform_total > 0:
        candidates.append((pretransform_total, "m4_pretransform_query"))
    prepare_total = _total_ms(stats0, f"{prefix}.prepare_inputs")
    if prepare_total > 0:
        candidates.append((prepare_total, "m4_prepare_inputs"))
    out_cast_total = _total_ms(stats0, f"{prefix}.output_cast")
    if out_cast_total > 0:
        candidates.append((out_cast_total, "m4_output_cast"))
    candidates.sort(reverse=True)
    if candidates:
        print("=== Suggested Optimization Order ===")
        for total, name in candidates[:5]:
            pct_gen = total / generate_ms * 100.0 if generate_ms > 0 else 0.0
            print(f"{name:24s} est_total_ms={total:10.3f} pct_gen={pct_gen:7.2f}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    if not (args.output_dir / "run_stats.json").exists():
        if _print_m4_debug_sweep_summary(args.output_dir):
            return 0
        raise SystemExit(f"run_stats.json not found under {args.output_dir}")

    run_stats = _load_json(args.output_dir / "run_stats.json")
    print(f"Results: {args.output_dir}")
    _print_worker_summary(run_stats)
    for variant in ("baseline", "fused"):
        _print_profile_table(run_stats, variant, top=args.top)
        _print_m4_stage_table(run_stats, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
