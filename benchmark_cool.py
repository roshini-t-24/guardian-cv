import cv2
import numpy as np
import time
import platform
import json

def run_benchmark(iterations=150):
    print("==========================================================")
    print(" GuardianCV: OpenCV 5 + AWS Graviton COOL Kernel Benchmark")
    print("==========================================================")
    print(f"Architecture : {platform.machine()} ({platform.processor() or 'Arm64/x86_64'})")
    print(f"OpenCV Build : v{cv2.__version__}")
    print(f"Iterations   : {iterations} frames (640x420 RGB)")
    print("----------------------------------------------------------")

    img = np.random.randint(0, 255, (420, 640, 3), dtype=np.uint8)
    gray_a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_b = np.roll(gray_a, 3, axis=1)

    mog2 = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=20, detectShadows=False)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))

    # 1. MOG2 Silhouette Segmentation
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = mog2.apply(img)
    mog2_ms = ((time.perf_counter() - t0) / iterations) * 1000.0

    # 2. LAB-Space CLAHE Contrast Recovery
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = clahe.apply(gray_a)
    clahe_ms = ((time.perf_counter() - t0) / iterations) * 1000.0

    # 3. Zero-PII Canny Edge Shield
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = cv2.Canny(gray_a, 35, 115)
    canny_ms = ((time.perf_counter() - t0) / iterations) * 1000.0

    # 4. Farneback Dense Optical Flow
    flow_iters = max(20, iterations // 5)
    t0 = time.perf_counter()
    for _ in range(flow_iters):
        _ = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 2, 13, 2, 5, 1.1, 0)
    flow_ms = ((time.perf_counter() - t0) / flow_iters) * 1000.0

    total_ms = mog2_ms + clahe_ms + canny_ms + flow_ms
    fps = 1000.0 / max(total_ms, 0.001)

    results = {
        "architecture": platform.machine(),
        "opencv_version": cv2.__version__,
        "kernels_ms": {
            "mog2_segmentation_ms": round(mog2_ms, 3),
            "clahe_recovery_ms": round(clahe_ms, 3),
            "canny_privacy_shield_ms": round(canny_ms, 3),
            "farneback_optical_flow_ms": round(flow_ms, 3),
        },
        "total_pipeline_latency_ms": round(total_ms, 3),
        "sustained_throughput_fps": round(fps, 1),
    }

    print(json.dumps(results, indent=2))
    with open("cool_benchmark_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("----------------------------------------------------------")
    print("Saved reproducible telemetry to cool_benchmark_report.json")

if __name__ == "__main__":
    run_benchmark()