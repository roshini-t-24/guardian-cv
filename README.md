# GuardianCV

Privacy-first gait and fall-risk monitoring for hospital wards and home care. It runs on plain OpenCV 5 pixel mathematics, a six-joint Kalman filter bank, and a closed-loop perception agent that repairs bad frames before tracking continues.

> **Decision support only.** Risk-tier thresholds are not clinically validated. Every prescription needs a licensed clinician's co-signature.

## The two problems

1. **Raw patient video is a liability.** Every frame passes through a Zero-PII shield on the device: structural edges and silhouette contours on a black canvas, with the head redacted. Raw frames are never sent anywhere.
2. **Vision pipelines fail quietly in the dark.** Underexposed, noisy corridors make passive trackers lose the patient without warning. The Active Perception Agent measures frame quality, tries three restoration policies on the actual pixels, and commits the one that raises confidence. If none helps, it holds the raw frame.

## No mock data

Joint positions, knee flexion, sway and confidence are all computed from video pixels: contours, Farneback optical flow, Shannon entropy, Laplacian variance, Canny edge density and convex-hull solidity. No sine-wave motion, no hardcoded confidence, no filename checks.

When the two legs cannot be told apart in a clip, the tier is reported as `INDETERMINATE` rather than guessing `GREEN`.

## How a frame flows

~~~text
raw frame
   |
   v
agent.act()  -- quality gate (mean luminance < 85 or confidence < 72 %)
   |            tries Gamma LUT / LAB-CLAHE / Hybrid, commits argmax gain or holds raw
   v
MOG2 + Canny segmentation (once, on the committed frame)
   |
   v
anatomical band slicing -> GaitAnalyzer (two-leg model) -> Kalman bank
   |
   v
Zero-PII shield + optical-flow sway -> WebSocket -> dashboard
~~~

## Quickstart

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest test_pipeline.py -q
python validate_metrics.py
python server.py
~~~

Open http://127.0.0.1:8000/. If `videos/` is empty, three procedural clips are generated on first run. Your own `.mp4` files can be uploaded from the sidebar.

## Dashboard

- **Live Kalman Stream:** Zero-PII viewport, confidence and sway metrics, bilateral knee-flexion chart, agent reasoning log, run summary with risk tier.
- **Graviton Arm64:** run the kernel benchmark or load the latest report.
- **Agent and MCP:** the objective function, latest candidate policy scores, MCP tool schemas.
- **Clinical EHR Rx:** FHIR-style prescription bundle, JSON download, printable view.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | architecture, OpenCV version, clip count |
| GET | `/api/clips` | list available clips |
| POST | `/api/upload` | upload an `.mp4` |
| POST | `/api/evaluate` | headless run: `{clip, night, agent, max_frames}` |
| WS | `/ws/stream` | live stream: `?clip=&night=&agent=&pii=&mode=` |
| POST | `/api/benchmark` | run the kernel benchmark: `{iters}` |
| GET | `/api/benchmark/latest` | last benchmark report |
| GET | `/api/ehr` | prescription for the latest run |
| POST | `/api/hitl` | log clinician `approve` or `dismiss` |
| GET | `/api/audit` | audit log (last run, agent decisions, HITL ledger) |
| GET | `/api/mcp` | MCP tool schemas |

Interactive docs are at `/docs`.

## Validation

`python validate_metrics.py` checks recovered knee-flexion peaks against ground truth on the procedural clips and checks the tier on every clip in `videos/`.

| Clip | Tier |
| --- | --- |
| clip1_asymmetric_gait | RED |
| clip2_symmetric_baseline | GREEN |
| clip3_postural_sway | RED |
| clip1_mixamo_limp_benchmark | AMBER |
| clip3_acute_stumble_event | AMBER |
| clip2_healthy_corridor_walk | INDETERMINATE (one leg hidden in the drawn figure) |

## Docker

~~~powershell
docker build -t guardian-cv .
docker run -d --name guardian-cv -p 8000:8000 guardian-cv
~~~

The image uses `python:3.12-slim`, which ships linux/amd64 and linux/arm64, so it builds natively on x86_64 and AWS Graviton. For a cross-arch image, build on an arm64 host or in CI rather than emulating.

## Benchmarking

~~~powershell
python benchmark_cool.py --iterations 100
python benchmark_cool.py --compare arm64_report.json
~~~

It times MOG2, CLAHE, Canny, Farneback and the full pipeline, optionally with `--parallel` scaling. Cost figures are derived only from measurements; pass `--price-hourly` to use your own instance price. Run it on both an x86_64 and a Graviton instance to compare; the CI workflow uploads a report from each.

## Layout

| Path | Role |
| --- | --- |
| `utils/telemetry.py` | frame-quality math, SNR confidence, quality gate |
| `utils/kalman_tracker.py` | six-joint Kalman filter bank |
| `utils/vision_pipeline.py` | segmentation, joint extraction, privacy shield, optical flow |
| `utils/gait_analysis.py` | two-leg tracking, symmetry, sway, risk tier |
| `utils/agent_core.py` | perception agent, HITL ledger, MCP schemas, EHR prescription |
| `utils/video_generator.py` | procedural fallback clips |
| `server.py` | FastAPI backend with WebSocket streaming |
| `static/index.html` | dashboard |
| `benchmark_cool.py` | kernel and pipeline profiler |
| `validate_metrics.py` | ground-truth and tier validation |
| `test_pipeline.py` | pytest suite (unit, end-to-end, API, WebSocket) |

## Known limits

- Tier thresholds are not clinically validated.
- Standing subjects are absorbed by the background model, so leg tracking needs a walking subject.
- Single-camera sagittal view only.
- Benchmark numbers depend on the host; compare runs from the same machine class.

## License

MIT, see `LICENSE`.
## Test clips

The three procedural clips in `videos/` are committed as lossless FFV1 `.avi` files. Lossy `.mp4` clips decoded to different pixels on x86_64 and aarch64, which changed the recovered knee-flexion peaks on one clip; lossless files decode identically on both. Risk tiers were stable either way, but the exact asymmetry magnitude from lossy footage can vary by decoder, so prefer lossless or high-bitrate input when comparing numbers across machines.

## CI

Every push runs the tests, metric validation and a 30-iteration kernel benchmark on an x86_64 runner and an aarch64 runner. The benchmark report is uploaded as an artifact per runner. On matched 4 vCPU GitHub runners the full pipeline measured about 24.5 ms on x86_64 and 35 ms on aarch64 (Azure Arm, not Graviton); a real Graviton comparison is still pending.

## Kalman tracking notes

The Kalman bank is fed only joints from frames where the two-leg fit is valid, and never the first warm-up frames where the background model is still empty and the whole frame reads as foreground. Without that gate, warm-up produced innovation spikes of about 270 px; with it the maximum was about 21 to 38 px on the procedural clips.

Process noise is 1.0. It was chosen from a sweep (0.01, 0.1, 1.0, 5.0) on two procedural walking clips: median innovation fell from about 14 px to about 6 px, at the cost of slightly less smoothing. It has not been validated on real ward footage, so retune it on your own data. About a quarter of frames in the walking clips are not valid fits and are skipped, so the filter coasts through them.

## Active perception agent: measured results

`evaluate_agent.py` runs every clip at four brightness levels (nominal, dim, night, severe), 60 frames each, with the agent off and on.

- Confidence rises with the agent on (night: 55.9 to 88.9).
- Restoring pixels before segmentation did not improve leg tracking. Valid-fit rate at night fell from 53.6% to 28.9% when tracking ran on the restored frame, because the confidence score rewards brightness and the restored frames were worse for background subtraction.
- Design change: segmentation and gait analysis run on the stressed frame by default (`track_on_restored=False`); restored frames are for display and the confidence readout. Valid-fit rates are then identical with the agent on or off.
- The agent therefore does not currently improve tracking. Both modes are reproducible: `python evaluate_agent.py --max-frames 60` and `--track-restored`. Reports: `eval_track_raw.json`, `eval_track_restored.json`.
- Limits: 60 frames per run, one seed, six clips (three procedural, three drawn figures), and many reference tiers are INDETERMINATE, so the tier column is weak evidence.

### Attempts to make restoration help tracking (all negative)

Four designs were evaluated on 6 clips x 4 brightness levels (60 frames, valid-fit rate with the agent on vs off, night level): confidence-scored policies (53.6 to 28.9), denoised policies (53.6 to 28.9), single forced policies (best 49.7 vs 53.6 for no restoration), and a shadow-segmentation tracking objective (53.6 to 22.2). None improved leg-fit validity, so tracking runs on the stressed frame and the agent's restoration is display-only. Single seed, short runs; the conclusion is for this segmentation pipeline.
