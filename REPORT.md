# GuardianCV: project report

Privacy-first gait and fall-risk screening from ordinary camera video, built on OpenCV 5 pixel mathematics. Repository: https://github.com/roshini-t-24/guardian-cv

> Decision support only. Risk tiers are not clinically validated, and every prescription requires a licensed clinician's co-signature.

## 1. Problem

1. Raw patient video is a privacy liability. GuardianCV renders each frame through an on-device Zero-PII shield (structural edges and silhouette on a black canvas, head redacted).
2. Vision pipelines degrade quietly in dim corridors. GuardianCV measures frame quality per frame and logs when the pipeline is operating on poor input.

## 2. What was built

| Component | Role |
| --- | --- |
| `utils/telemetry.py` | Per-frame quality from luminance, Laplacian variance, Shannon entropy, Canny edge density, hull solidity; composite confidence 12 to 99.4 % |
| `utils/vision_pipeline.py` | MOG2 + Canny segmentation, joint extraction, Zero-PII shield, Farneback optical flow |
| `utils/gait_analysis.py` | Two-leg tracking, knee flexion, symmetry, sway; tiers GREEN / AMBER / RED / INDETERMINATE |
| `utils/kalman_tracker.py` | Six-joint constant-velocity Kalman bank |
| `utils/agent_core.py` | Perception agent (sense, trigger, evaluate three restoration policies, commit or hold), HITL ledger, MCP tool schemas, FHIR-style prescription bundle |
| `server.py`, `static/index.html` | FastAPI backend, WebSocket live stream, dashboard |
| `benchmark_cool.py`, `aws_bench.sh` | Kernel and pipeline profiler, AWS reproduction script |
| `validate_metrics.py`, `evaluate_agent.py`, `test_pipeline.py` | Validation, agent evaluation, 21 tests |

No metric is synthesized: angles, sway and confidence come from pixels. When the two legs cannot be told apart, the tier is `INDETERMINATE`, never a guessed `GREEN`.

## 3. Evaluation

### 3.1 Metric validation
Three procedural clips with known ground truth plus three drawn-figure clips. `validate_metrics.py` passes on all six. Recovered knee-flexion peaks on the asymmetric procedural clip were 13.3 / 55.5 degrees against a true 18.6 / 62.0 (frame RMSE 5.1), inside the 12-degree allowance but not close, so the exact asymmetry magnitude should be treated as approximate. The tier (RED) was correct on both architectures.

### 3.2 Cross-architecture reproducibility
CI runs on x86_64 and aarch64. Lossy `.mp4` clips decoded to different pixels on the two architectures, which changed recovered peaks (31.3 / 49.1 on arm64 against 13.3 / 55.5 on x86_64) and failed validation on arm64. Decoded-frame hashes confirmed the decoder as the cause. Committing the clips as lossless FFV1 `.avi` made hashes and validation results identical on both. Lesson: asymmetry magnitudes from lossy footage can depend on the decoder.

### 3.3 Kalman tracking
Feeding the filter every frame produced innovation spikes up to about 270 px, caused by the empty background model in the first frames. Gating on valid two-leg fits removed them (maximum 21 to 38 px). A sweep of process noise (0.01, 0.1, 1.0, 5.0) on two procedural walking clips cut median innovation from about 14 px to about 6 px at 1.0, at the cost of less smoothing. This was tuned on procedural clips only.

### 3.4 Active perception agent: measured results
Six clips, four brightness levels (nominal, dim, night, severe), 60 frames, agent off versus on, valid-fit rate:

| Design | Night valid-fit, off to on |
| --- | --- |
| Confidence-scored policies, tracking on restored frame | 53.6 % to 28.9 % |
| Same with denoising added to policies | 53.6 % to 28.9 % |
| Best single forced policy (gamma) vs no restoration | 49.7 % vs 53.6 % |
| Shadow-segmentation tracking objective | 53.6 % to 22.2 % |

Confidence rose from about 56 to 89 at night in every design, but leg-fit validity did not improve in any. The confidence score rewards brightness and does not measure tracking quality. Design change: segmentation and gait analysis run on the stressed frame by default, and restored frames are used for display and the confidence readout. Valid-fit rates are then identical with the agent on or off. **The agent does not currently improve tracking.** What it provides is quality sensing, an auditable decision log, a human approve/dismiss ledger, and a measured safeguard against its own failure mode. Reports: `eval_track_raw.json`, `eval_track_restored.json`.

The pipeline itself is also fragile below normal lighting: with the agent off, the tier matched the nominal reference on only 2 of 6 clips at dim and 1 of 6 at night. Many reference tiers are `INDETERMINATE`, so the tier column is weak evidence and valid-fit rate is the better metric.

### 3.5 Performance
Single stream, 100 iterations, 2 vCPU, Ubuntu 24.04, AWS ap-south-1:

| Instance | CPU | Full pipeline | Farneback | Linux on-demand |
| --- | --- | --- | --- | --- |
| c7i-flex.large | Intel x86_64 | 32.7 ms (30.5 fps) | 29.4 ms | $0.08479/h |
| t4g.small | AWS Graviton2 (Neoverse-N1) | 58.3 ms (17.2 fps) | 53.2 ms | $0.0112/h |

The Intel instance is about 1.8x faster per frame. Graviton2 is about 7.6x cheaper per hour and, by arithmetic on these two runs, roughly 4x cheaper per processed frame. t4g.small sustains one 15 fps stream with little headroom. Optical flow is about 90 % of frame time on both. Raw reports: `benchmarks/`; reproduce with `aws_bench.sh`.

On matched 4 vCPU GitHub runners (Azure, not Graviton), the full pipeline measured about 24.5 ms on x86_64 and 35 ms on aarch64.

## 4. Human oversight and responsible use
- Every approve or dismiss is written to an audit ledger and exported in the JSON audit log.
- Prescriptions carry a disclaimer and require clinician co-signature.
- The Zero-PII shield is a design measure; it was not audited against any regulation.
- The agent's MCP integration is limited to published tool schemas (`/api/mcp`); no MCP server executes them.
- The three drawn-figure test clips are local only and not published in the repository.

## 5. Limitations
- Tier thresholds are not clinically validated, and no real ward footage was used; every tuned value comes from procedural clips.
- Standing subjects are absorbed by the background model, so leg tracking needs a walking subject; a quarter or more of frames in walking clips are not valid fits.
- Single camera, sagittal view.
- Agent: no tracking benefit measured (see 3.4). Evaluations used 60 frames and one noise seed.
- AWS results: Graviton2 (not Graviton4), burstable instance types, one run per instance, streams-per-node not measured with `--parallel`, prices as entered at test time.

## 6. Reproduce
~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest test_pipeline.py -q
python validate_metrics.py
python evaluate_agent.py --max-frames 60
python server.py
~~~

Author: YOUR NAME. Demo video: LINK.