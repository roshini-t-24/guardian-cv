# GuardianCV

Privacy-first mobility and fall-risk monitoring for hospital wards and home care, built on OpenCV pixel mathematics and a closed-loop perception agent.

## Why this exists

1. **Raw patient video is a liability.** GuardianCV renders every frame through a Zero-PII shield on the device: structural edges and silhouette contours on a black canvas, with the head redacted. Raw frames are never sent anywhere.
2. **Vision pipelines fail quietly at night.** Dark, noisy corridors make passive trackers lose the patient without warning. The Active Perception Agent watches frame quality, tries three restoration policies on the actual pixels, and commits the one that measurably improves confidence. If none helps, it holds the raw frame.

## No mock data

Joint positions, knee flexion, sway and confidence all come from video pixels: contours, optical flow, Shannon entropy, Laplacian variance, Canny edge density and hull solidity. Nothing is generated with sine waves and no filename is ever inspected.

When the legs cannot be told apart in a clip, the tier is reported as `INDETERMINATE` instead of guessing `GREEN`.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest test_pipeline.py -q
python validate_metrics.py
python server.py
```

Open http://127.0.0.1:8000/. Three procedural clips are generated into `videos/` on first run if the folder is empty; your own `.mp4` files can be uploaded from the sidebar.

## Docker

```powershell
docker build -t guardian-cv .
docker run -d --name guardian-cv -p 8000:8000 guardian-cv
```

The image builds natively on linux/amd64 and linux/arm64 (AWS Graviton).

## Layout

| Path | Role |
| --- | --- |
| `utils/telemetry.py` | Frame-quality math and SNR confidence |
| `utils/kalman_tracker.py` | Six-joint Kalman filter bank |
| `utils/vision_pipeline.py` | Segmentation, joint extraction, privacy shield, optical flow |
| `utils/gait_analysis.py` | Two-leg tracking, symmetry, sway, risk tier |
| `utils/agent_core.py` | Active perception agent, HITL ledger, MCP schemas, EHR prescription |
| `utils/video_generator.py` | Procedural fallback clips |
| `server.py` | FastAPI backend with WebSocket live stream |
| `static/index.html` | Dashboard |
| `benchmark_cool.py` | Kernel and pipeline profiler (x86_64 / Graviton) |
| `validate_metrics.py` | Ground-truth and tier validation |

## Benchmarks

`python benchmark_cool.py --iters 100` measures the kernels and the full pipeline on the host. Run it on both an x86_64 and a Graviton instance and compare with `--compare`; cost figures are derived only from those measurements.

## Limits

Risk tier thresholds are not clinically validated. Output is decision support only and needs a licensed clinician's co-signature.

## License

MIT, see `LICENSE`.