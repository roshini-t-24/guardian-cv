from __future__ import annotations

import asyncio
import base64
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.agent_core import (
    ActivePerceptionAgent,
    HITLLedger,
    MCP_TOOL_SCHEMAS,
    build_ehr_rx,
)
from utils.gait_analysis import GaitAnalyzer
from utils.kalman_tracker import JointKalmanBank
from utils.telemetry import compute_visual_telemetry
from utils.video_generator import ensure_clips, list_clips
from utils.vision_pipeline import (
    ClinicalVisionPipeline,
    apply_privacy_shield,
    compute_dense_flow,
    flow_sway_magnitude,
)

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "videos"
STATIC_DIR = ROOT / "static"
BENCH_REPORT = ROOT / "cool_benchmark_report.json"

app = FastAPI(title="GuardianCV", version="1.0.0")

STATE: dict = {
    "hitl": HITLLedger(),
    "last_run": None,
    "decision_log": [],
}


class Session:
    """One independent stream: owns its pipeline, agent, gait and Kalman state."""

    def __init__(self, night: bool = False, agent_on: bool = True) -> None:
        self.night, self.agent_on = night, agent_on
        self.pipeline = ClinicalVisionPipeline()
        self.agent = ActivePerceptionAgent()
        self.gait = GaitAnalyzer()
        self.kalman = JointKalmanBank()
        for obj in (self.pipeline, self.agent, self.gait, self.kalman):
            obj.reset()
        self.prev_mask = None
        self.prev_gray = None
        self.idx = 0
        self.core_s = 0.0
        self.valid = 0
        self.conf: list[float] = []
        self.sway: list[float] = []
        self.gains: list[float] = []
        self.interventions = 0
        self.log: list[str] = []

    def step(self, raw: np.ndarray, pii: bool = True, mode: str = "silhouette"):
        t0 = time.perf_counter()
        if self.night:
            raw = np.clip(raw.astype(np.float32) * 0.32, 0, 255).astype(np.uint8)
            noise = np.random.default_rng(self.idx).integers(0, 14, raw.shape, np.uint8)
            raw = cv2.add(raw, noise)

        decision, committed = None, raw
        if self.agent_on:
            decision, committed = self.agent.act(raw, mask=self.prev_mask)

        mask = self.pipeline.segment_foreground(committed)
        joints, bbox = self.pipeline.extract_joints(mask)
        g = self.gait.update(committed, joints, bbox)
        kin = self.kalman.update(g.get("joints") or {})

        gray = cv2.cvtColor(committed, cv2.COLOR_BGR2GRAY)
        tel = compute_visual_telemetry(gray, mask)

        sway = None
        if self.prev_gray is not None:
            mag, _ang = compute_dense_flow(self.prev_gray, gray)
            sway = flow_sway_magnitude(mag, bbox)
            self.sway.append(sway)

        display = (
            apply_privacy_shield(committed, gray, mask, joints, mode=mode)
            if pii else committed.copy()
        )
        for name, (jx, jy) in kin["smoothed_joints"].items():
            if g.get("valid"):
                cv2.circle(display, (int(jx), int(jy)), 3, (120, 235, 190), -1, cv2.LINE_AA)

        if decision and decision.get("triggered"):
            self.interventions += 1
            self.gains.append(float(decision["gain"]))
            self.log.append(
                f"f{self.idx:04d} INTERVENE {decision['reason']} -> "
                f"{decision['committed_policy']} C {decision['pre_confidence']:.1f}"
                f"->{decision['post_confidence']:.1f} (d {decision['gain']:+.2f})"
            )
        elif self.idx % 25 == 0:
            self.log.append(
                f"f{self.idx:04d} NOMINAL C={tel['snr_confidence']:.1f} "
                f"mu_L={tel['luminance']:.0f}"
            )
        self.log = self.log[-200:]

        self.conf.append(tel["snr_confidence"])
        self.valid += int(bool(g.get("valid")))
        self.prev_mask, self.prev_gray = mask, gray
        self.core_s += time.perf_counter() - t0

        payload = {
            "frame": self.idx,
            "snr_confidence": tel["snr_confidence"],
            "luminance": tel["luminance"],
            "entropy": tel["entropy"],
            "hull_solidity": tel["hull_solidity"],
            "innovation_px": kin["innovation_error"],
            "valid": bool(g.get("valid")),
            "iou": g.get("iou"),
            "flex_a": g.get("flex_a"),
            "flex_b": g.get("flex_b"),
            "lean_deg": g.get("lean_deg"),
            "sway_px": sway,
            "decision": decision if decision and decision.get("triggered") else None,
        }
        self.idx += 1
        return display, payload

    def finalize(self, clip: str) -> dict:
        summary = self.gait.summary()
        run = {
            "clip": clip,
            "frames": self.idx,
            "valid_gait_frames": self.valid,
            "mean_confidence": round(float(np.mean(self.conf)), 2) if self.conf else 0.0,
            "mean_latency_ms": round(self.core_s / max(self.idx, 1) * 1000.0, 2),
            "mean_sway_px": round(float(np.mean(self.sway)), 3) if self.sway else 0.0,
            "agent_interventions": self.interventions,
            "mean_agent_gain": round(float(np.mean(self.gains)), 2) if self.gains else 0.0,
            "gait": summary,
        }
        STATE["last_run"] = run
        STATE["decision_log"] = (STATE["decision_log"] + self.agent.decision_log)[-128:]
        return run


def _safe_clip(name: str) -> Path | None:
    for p in list_clips(VIDEO_DIR):
        if p.name == name:
            return p
    return None


def _flag(params, key: str, default: bool) -> bool:
    raw = params.get(key)
    return default if raw is None else raw.lower() in ("1", "true", "yes", "on")


def _encode_jpeg_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _run_headless(path: Path, night: bool, agent_on: bool, max_frames: int) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path.name}")
    sess = Session(night=night, agent_on=agent_on)
    while sess.idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        sess.step(frame, pii=False)
    cap.release()
    return sess.finalize(path.name)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "architecture": platform.machine(),
        "opencv": cv2.__version__,
        "python": platform.python_version(),
        "clips": len(list_clips(VIDEO_DIR)),
    }


@app.get("/api/clips")
def clips():
    ensure_clips(VIDEO_DIR)
    return [{"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
            for p in list_clips(VIDEO_DIR)]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "upload.mp4").name)
    if not name.lower().endswith((".mp4", ".avi")):
        return JSONResponse({"error": "only .mp4 or .avi accepted"}, status_code=400)
    VIDEO_DIR.mkdir(exist_ok=True)
    (VIDEO_DIR / name).write_bytes(await file.read())
    return {"saved": name}


@app.post("/api/evaluate")
async def evaluate(request: Request):
    body = await request.json()
    path = _safe_clip(str(body.get("clip", "")))
    if path is None:
        return JSONResponse({"error": "unknown clip"}, status_code=404)
    try:
        run = await asyncio.to_thread(
            _run_headless, path,
            bool(body.get("night", False)), bool(body.get("agent", True)),
            int(body.get("max_frames", 150)),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    return run


@app.post("/api/benchmark")
async def benchmark(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    iters = max(10, min(int(body.get("iters", 100)), 500))
    proc = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(ROOT / "benchmark_cool.py"), "--iterations", str(iters)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if proc.returncode != 0 or not BENCH_REPORT.is_file():
        return JSONResponse({"error": "benchmark failed", "stderr": proc.stderr[-800:]},
                            status_code=500)
    return json.loads(BENCH_REPORT.read_text())


@app.get("/api/benchmark/latest")
def benchmark_latest():
    if not BENCH_REPORT.is_file():
        return JSONResponse({"error": "no report yet"}, status_code=404)
    return json.loads(BENCH_REPORT.read_text())


@app.get("/api/ehr")
def ehr():
    run = STATE["last_run"]
    if run is None:
        return JSONResponse({"error": "run an evaluation first"}, status_code=409)
    return build_ehr_rx(run, STATE["hitl"])


@app.post("/api/hitl")
async def hitl(request: Request):
    body = await request.json()
    action = str(body.get("action", "")).lower()
    if action not in ("approve", "dismiss"):
        return JSONResponse({"error": "action must be approve|dismiss"}, status_code=400)
    event = STATE["hitl"].record(action, {"clip": body.get("clip")})
    return {"event": event, "ledger": STATE["hitl"].snapshot()}


@app.get("/api/audit")
def audit():
    return {
        "architecture": platform.machine(),
        "opencv": cv2.__version__,
        "last_run": STATE["last_run"],
        "agent_decision_log": STATE["decision_log"][-32:],
        "hitl": STATE["hitl"].snapshot(),
    }


@app.get("/api/mcp")
def mcp():
    return MCP_TOOL_SCHEMAS


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    q = ws.query_params
    path = _safe_clip(q.get("clip", ""))
    if path is None:
        await ws.send_json({"error": "unknown clip"})
        await ws.close()
        return

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    pii = _flag(q, "pii", True)
    mode = q.get("mode", "silhouette")
    sess = Session(night=_flag(q, "night", False), agent_on=_flag(q, "agent", True))
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            display, payload = await asyncio.to_thread(sess.step, frame, pii, mode)
            payload["jpeg_b64"] = _encode_jpeg_b64(display)
            payload["agent_log_tail"] = sess.log[-6:]
            await ws.send_json(payload)
            await asyncio.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))
        run = sess.finalize(path.name)
        await ws.send_json({"done": True, "summary": run})
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()


@app.get("/")
def index():
    page = STATIC_DIR / "index.html"
    if page.is_file():
        return FileResponse(page)
    return JSONResponse({"status": "GuardianCV API up", "docs": "/docs"})


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)