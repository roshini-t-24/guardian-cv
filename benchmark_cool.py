"""Module E — Hardware profiler for AWS Graviton (aarch64) vs x86_64.

    python benchmark_cool.py                       # 150 timed iterations, 4 kernels + full pipeline
    python benchmark_cool.py --parallel            # also measure real multi-process scaling
    python benchmark_cool.py --price-hourly 0.07   # add cost-per-stream from YOUR instance price
    python benchmark_cool.py --compare other.json  # compare two measured reports

Everything reported is measured on the machine running this script, on real
video frames from the reference clips. Nothing is looked up or assumed:

  * No vendor latency table. Cross-architecture numbers come only from
    running this script on both machines and passing --compare.
  * Streams per node = measured aggregate pipeline fps / target fps.
    With --parallel the aggregate comes from actually running one worker
    per vCPU; without it, a clearly labelled single-core extrapolation.
  * Cost is computed only if you supply --price-hourly. The script ships no
    price list.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.telemetry import compute_visual_telemetry  # noqa: E402
from utils import video_generator as _vg  # noqa: E402
from utils.vision_pipeline import (  # noqa: E402
    ClinicalVisionPipeline,
    compute_dense_flow,
    flow_sway_magnitude,
    knee_flexion,
)

REPORT_PATH = ROOT / "cool_benchmark_report.json"
DEFAULT_ITERATIONS = 150
WARMUP = 10
TARGET_FPS = 15.0
STREAM_TIERS = (1, 10, 24, 50, 100)
CLIP_FOR_BENCH = "clip2_symmetric_baseline.avi"


# -- environment ---------------------------------------------------------------
def detect_architecture() -> str:
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    return machine or "unknown"


def environment() -> dict:
    return {
        "architecture": detect_architecture(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "cpu_count": os.cpu_count() or 1,
        "opencv_threads": cv2.getNumThreads(),
        "cpu_features": cv2.getCPUFeaturesLine(),
    }


def _clip_path() -> Path:
    """Locate a benchmark clip, generating the reference clips if possible."""
    video_dir = ROOT / "videos"
    generator = getattr(_vg, "ensure_clips", None)
    if generator is not None and not (video_dir / CLIP_FOR_BENCH).exists():
        generator(str(video_dir))
    preferred = video_dir / CLIP_FOR_BENCH
    if preferred.exists():
        return preferred
    existing = sorted([*video_dir.glob("*.avi"), *video_dir.glob("*.mp4")]) if video_dir.exists() else []
    if existing:
        return existing[0]
    raise SystemExit(
        "No .avi/.mp4 found in videos/ and utils/video_generator.py has no ensure_clips(). "
        "Put any .avi or .mp4 in videos/ or update utils/video_generator.py."
    )


def load_frames(count: int = 60) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(_clip_path()))
    frames: List[np.ndarray] = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 2:
        raise RuntimeError("could not read benchmark frames from the reference clip")
    return frames


# -- timing ---------------------------------------------------------------------
def _time_loop(step: Callable[[int], None], iterations: int) -> dict:
    for i in range(WARMUP):
        step(i)
    samples: List[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        step(i + WARMUP)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    mean = statistics.fmean(samples)
    return {
        "iterations": iterations,
        "mean_ms": round(mean, 3),
        "p50_ms": round(samples[len(samples) // 2], 3),
        "p95_ms": round(samples[min(int(len(samples) * 0.95), len(samples) - 1)], 3),
        "min_ms": round(samples[0], 3),
        "fps": round(1000.0 / mean, 2) if mean > 0 else 0.0,
    }


def run_kernel_suite(
    iterations: int = DEFAULT_ITERATIONS,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """Time the four core kernels on real frames. `progress(name, done, total)`."""
    frames = load_frames()
    n = len(frames)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    mog = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=16, detectShadows=False)
    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))

    kernels: Dict[str, Callable[[int], None]] = {
        "MOG2": lambda i: mog.apply(frames[i % n]),
        "CLAHE": lambda i: clahe.apply(grays[i % n]),
        "Canny": lambda i: cv2.Canny(grays[i % n], 35, 115),
        "Farneback": lambda i: cv2.calcOpticalFlowFarneback(
            grays[i % n], grays[(i + 1) % n], None, 0.5, 3, 15, 3, 5, 1.2, 0
        ),
    }
    results: Dict[str, dict] = {}
    for idx, (name, step) in enumerate(kernels.items()):
        if progress:
            progress(name, idx, len(kernels))
        results[name] = _time_loop(step, iterations)
    if progress:
        progress("done", len(kernels), len(kernels))
    return results


def run_pipeline_bench(iterations: int = DEFAULT_ITERATIONS) -> dict:
    """Time the FULL per-frame pipeline: segmentation, joints, telemetry, flow."""
    frames = load_frames()
    n = len(frames)
    pipe = ClinicalVisionPipeline()
    state = {"prev": None, "mask": None}

    def step(i: int) -> None:
        frame = frames[i % n]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = pipe.segment_foreground(frame)
        joints, bbox = pipe.extract_joints(mask)
        knee_flexion(joints)
        compute_visual_telemetry(gray, state["mask"])
        if state["prev"] is not None:
            mag, _ = compute_dense_flow(state["prev"], gray)
            flow_sway_magnitude(mag, bbox, mask)
        state["prev"], state["mask"] = gray, mask

    return _time_loop(step, iterations)


# -- real multi-process scaling ---------------------------------------------------
def _worker(frames_per_worker: int, ready, go, out) -> None:
    cv2.setNumThreads(1)
    frames = load_frames()
    n = len(frames)
    pipe = ClinicalVisionPipeline()
    prev = mask = None
    ready.put(1)
    go.wait()
    t0 = time.perf_counter()
    for i in range(frames_per_worker):
        frame = frames[i % n]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = pipe.segment_foreground(frame)
        joints, bbox = pipe.extract_joints(mask)
        knee_flexion(joints)
        compute_visual_telemetry(gray, mask)
        if prev is not None:
            mag, _ = compute_dense_flow(prev, gray)
            flow_sway_magnitude(mag, bbox, mask)
        prev = gray
    out.put(frames_per_worker / (time.perf_counter() - t0))


def run_parallel_scaling(frames_per_worker: int = 90) -> dict:
    """Run one pipeline worker per vCPU simultaneously; sum their throughput."""
    workers = os.cpu_count() or 1
    ctx = mp.get_context("spawn")
    ready, out, go = ctx.Queue(), ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=_worker, args=(frames_per_worker, ready, go, out))
             for _ in range(workers)]
    for p in procs:
        p.start()
    for _ in procs:
        ready.get()
    go.set()
    per_worker = [out.get() for _ in procs]
    for p in procs:
        p.join()
    return {
        "workers": workers,
        "per_worker_fps": [round(v, 2) for v in per_worker],
        "aggregate_fps": round(sum(per_worker), 2),
    }


# -- FinOps (derived from measurements only) ---------------------------------------
def finops(
    pipeline: dict,
    parallel: Optional[dict],
    cpu_count: int,
    target_fps: float = TARGET_FPS,
    price_hourly: Optional[float] = None,
) -> dict:
    if parallel:
        aggregate_fps, basis = parallel["aggregate_fps"], "measured: one worker per vCPU"
    else:
        aggregate_fps = pipeline["fps"] * cpu_count
        basis = ("extrapolated: single-core fps x vCPU count "
                 "(optimistic, run with --parallel to measure)")
    streams_per_node = int(aggregate_fps // target_fps)
    table = []
    for streams in STREAM_TIERS:
        nodes = -(-streams // streams_per_node) if streams_per_node else None
        row = {"streams": streams, "nodes_required": nodes}
        if price_hourly is not None and nodes is not None:
            row["usd_per_hour"] = round(nodes * price_hourly, 4)
            row["usd_per_stream_hour"] = round(nodes * price_hourly / streams, 4)
        table.append(row)
    return {
        "target_fps_per_stream": target_fps,
        "aggregate_fps": round(aggregate_fps, 2),
        "basis": basis,
        "streams_per_node": streams_per_node,
        "price_hourly_usd": price_hourly,
        "price_note": ("supplied by caller" if price_hourly is not None
                       else "not computed: pass --price-hourly with your instance price"),
        "table": table,
    }


# -- report ------------------------------------------------------------------------
def build_report(
    iterations: int = DEFAULT_ITERATIONS,
    parallel: bool = False,
    price_hourly: Optional[float] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    env = environment()
    kernels = run_kernel_suite(iterations, progress)
    pipeline = run_pipeline_bench(iterations)
    scaling = run_parallel_scaling() if parallel else None
    return {
        "schema": "guardian-cv/benchmark/2",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": env,
        "frame_size": "480x360",
        "kernels": kernels,
        "pipeline": pipeline,
        "parallel": scaling,
        "finops": finops(pipeline, scaling, env["cpu_count"], TARGET_FPS, price_hourly),
        "notes": [
            "All latencies measured on this machine on real video frames.",
            "Cross-architecture comparison requires running this script on each machine.",
        ],
    }


def compare_reports(a: dict, b: dict) -> List[dict]:
    rows = []
    for name in a["kernels"]:
        if name in b["kernels"]:
            la, lb = a["kernels"][name]["mean_ms"], b["kernels"][name]["mean_ms"]
            rows.append({"kernel": name, "a_ms": la, "b_ms": lb,
                         "b_vs_a": round(la / lb, 2) if lb else None})
    la, lb = a["pipeline"]["mean_ms"], b["pipeline"]["mean_ms"]
    rows.append({"kernel": "FULL PIPELINE", "a_ms": la, "b_ms": lb,
                 "b_vs_a": round(la / lb, 2) if lb else None})
    return rows


def _print_report(report: dict) -> None:
    env = report["environment"]
    print(f"\nGuardianCV benchmark  |  {env['architecture']}  |  {env['cpu_count']} vCPU  |  OpenCV {env['opencv']}")
    print(f"{'kernel':<16}{'mean ms':>10}{'p50':>10}{'p95':>10}{'fps':>10}")
    for name, k in {**report["kernels"], "FULL PIPELINE": report["pipeline"]}.items():
        print(f"{name:<16}{k['mean_ms']:>10.2f}{k['p50_ms']:>10.2f}{k['p95_ms']:>10.2f}{k['fps']:>10.1f}")
    fo = report["finops"]
    print(f"\nstreams per node @ {fo['target_fps_per_stream']:.0f} fps: {fo['streams_per_node']}  ({fo['basis']})")
    print(f"{'streams':>8}{'nodes':>8}{'$/hour':>10}{'$/stream-hr':>14}")
    for row in fo["table"]:
        print(f"{row['streams']:>8}{str(row['nodes_required']):>8}"
              f"{row.get('usd_per_hour', '-'):>10}{row.get('usd_per_stream_hour', '-'):>14}")
    if fo["price_hourly_usd"] is None:
        print(f"cost: {fo['price_note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GuardianCV hardware profiler")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--parallel", action="store_true",
                        help="measure real multi-process scaling (one worker per vCPU)")
    parser.add_argument("--price-hourly", type=float, default=None,
                        help="your instance price in USD/hour, enables cost columns")
    parser.add_argument("--compare", metavar="OTHER_JSON", default=None,
                        help="compare this run against a report from another machine")
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()

    report = build_report(args.iterations, args.parallel, args.price_hourly)
    Path(args.out).write_text(json.dumps(report, indent=2))
    _print_report(report)

    if args.compare:
        other = json.loads(Path(args.compare).read_text())
        print(f"\nthis run ({report['environment']['architecture']}) vs "
              f"{args.compare} ({other['environment']['architecture']}) — ratio = other is N x faster")
        print(f"{'kernel':<16}{'this ms':>10}{'other ms':>10}{'other/this':>12}")
        for r in compare_reports(report, other):
            print(f"{r['kernel']:<16}{r['a_ms']:>10.2f}{r['b_ms']:>10.2f}{r['b_vs_a']:>12}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()