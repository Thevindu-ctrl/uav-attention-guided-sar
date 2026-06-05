#  UAV Attention-Guided Small-Object Detection for Search-and-Rescue (SAR)

[![Live App](https://img.shields.io/badge/Live-Telemetry_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://uav-attention-guided-sar-yeyixe2zgqcbt98vb2466u.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](#)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-006400?style=for-the-badge&logo=analytics&logoColor=white)](#)

---

## Project and scope Overview

This repository encapsulates an edge-optimized computer vision infrastructure engineered to preserve, isolate, and track sub-32-pixel human targets from high-altitude aerial payloads. Real-time rescue operations frequently stumble during post-disaster scenarios because standard down-sampling paths in traditional convolutional networks tend to completely destroy tiny structural features. Furthermore, deploying heavy transformer networks onto Unmanned Aerial Vehicles (UAVs) introduces high processing latencies that drain batteries and overload Size, Weight, and Power (SWaP)-constrained onboard processors.

By deploying a UAV-based small-object human detection system using lightweight attention-guided models (ECA-Net) and YOLO detectors for search-and-rescue applications, this framework bridges the gap between processing speed and tracking precision. The framework supports both Optical RGB and Thermal Infrared streams, ensuring continuous tracking capability regardless of weather anomalies or dark environmental conditions.

---

## Architectural Strategy & Training Core

The primary engine optimizes feature scaling by computing adaptive, localized cross-channel interactions without any dimensionality reduction. This allows the system to hold onto critical spatial details that are typically lost during deeper pooling operations.

```text
[Input Aerial Frame] ──> [Slicing Window (SAHI)] ──> [ECA-Net Feature Map Enrichment] ──> [YOLOv8 Regression Head] ──> [Target Isolation]
```

### Efficient Channel Attention Mechanism

Instead of relying on heavy spatial mapping channels, the engine calculates an optimized, adaptive 1D convolution kernel width `k` based on the total channel scale matrix `C`:

> **k = | log₂(C) / γ + b / γ |** *(evaluated to the nearest odd integer)*

This formula allows the feature extraction layers to dynamically re-weight channel importance with virtually zero parameter weight penalty, preserving fluid frame-rate delivery on embedded hardware.

### Training Lifecycle & Optimization

* **Optimization Base:** PyTorch Core utilizing the Adam optimizer.
* **Epoch Scale:** 50 optimization cycles over high-density tensors.
* **Hardware Profile:** Compiled and optimized using GPU-accelerated computing nodes to accurately map small gradient adjustments.
* **Inference Enhancement:** Integrates **SAHI (Slicing Aided Hyper Inference)** at runtime, breaking down ultra-high-definition inputs into localized patches to capture tiny targets without forcing image downscaling.

---

## Dataset Distribution Matrix

The models were systematically trained and validated on the benchmark **VisDrone-DET Dataset** from Tianjin University. The images reflect realistic aerial challenges, capturing heavily clustered human targets under diverse illumination conditions and variable look-down angles.

| Partition Target | Image Count | Primary Function in Pipeline |
| --- | --- | --- |
| **Training Phase (`train`)** | 6,471 | Optimizes base convolutional weights and local attention filters |
| **Validation Phase (`val`)** | 548 | Tunes model hyperparameters and monitors gradient loss behavior |
| **Testing Isolation (`test`)** | 1,610 | Conducts objective benchmarking (mAP_50 and AP_small isolation) |
| **Total Ecosystem Matrix** | **8,629** | **High-Density Aerial Telemetry Footprint** |

###  Recreating the Experiment

If you want to retrain the network or extend the pipeline, you can access the raw benchmark data here:

* **Dataset Download Repository:** [VisDrone Dataset Source](https://github.com/VisDrone/VisDrone-Dataset)

---

## Repository Layout

This repository utilizes a clean, flat directory layout to eliminate path confusion and ensure that scripts run predictably out of the box.

```text
└── uav-attention-guided-sar/
    ├── .gitignore                               # Excludes system runtimes and cache configurations
    ├── README.md                                # Professional system display and protocol guide
    ├── app.py                                   # Interactive Streamlit companion dashboard
    ├── requirements.txt                         # Managed dependencies listing (with headless OpenCV & SAHI)
    ├── Visdrone.ipynb                           # Notebook mapping out the training and validation loops
    ├── visdrone_human.yaml                      # Dataset anchors and tracking parameter paths
    ├── SARreport.pdf                            # Complete Academic Thesis & Research Report
    ├── best.pt                                  # Optimized, attention-guided production model weights
    ├── yolov8n.pt                               # Baseline benchmark model (Nano profile)
    ├── rtdetr-l.pt                               # Baseline benchmark model (Small profile)
    ├── data/                                  # Default ingestion directory for testing assets
    │   ├──  demo_rgb/                          # Built-in sample optical drone frames
    │   └──  demo_thermal/                      # Built-in sample thermal infrared profiles
    └── outputs/                               # Storage destination for exported logs and tracking logs
```

---

##  Live System Deployment & Simulation Protocol

### 1. Operating the Live Cloud Dashboard

You can immediately evaluate the network's capabilities without installing any local code by accessing the live cloud deployment:

* **Live Web App Link:** [uav-attention-guided-sar.streamlit.app](https://uav-attention-guided-sar-yeyixe2zgqcbt98vb2466u.streamlit.app/)
* **Testing with Custom Data:** You can upload your own high-resolution drone imagery and video streams directly using the drag-and-drop interface elements in the application.
* **Testing with Included Data:** If you do not have custom aerial footage on hand, you can immediately test the pipeline using the sample assets pre-packaged in the repository. Simply use the interface file browser to select frames from the `data/demo_rgb/` or `data/demo_thermal/` folders to see the attention filters instantly map bounding boxes over small targets.

### 2. Launching the Local Workspace

If you wish to run inference simulations locally on your own GPU/CPU hardware, execute this command sequence inside your terminal:

```bash
# Clone the repository and navigate into it
git clone https://github.com/YOUR_GITHUB_USERNAME/uav-attention-guided-sar.git
cd uav-attention-guided-sar

# Install the verified library stack
pip install -r requirements.txt

# Run the live telemetry controller
streamlit run app.py
```

### 3. Terminal Telemetry Preview

When running, the system dashboard simulates a real-time rescue operations command center:

```text
 🧭 UAV SEARCH & RESCUE TELEMETRY CONTROLLER | ENGINE: STREAMLIT ACTIVE [best.pt]
 ════════════════════════════════════════════════════════════════════════════════════
  [STREAM 01: OPTICAL RGB VIEW]               [STREAM 02: INFRARED THERMAL VIEW]
  ┌──────────────────────────────┐            ┌──────────────────────────────┐
  │  (Alt: 45m - Live Stream)    │            │  (Thermal Mode: Ironbow)     │
  │     ┌───┐                    │            │     ⚡                       │
  │     │ 👤│ [human_target 92%] │            │    🔥 [target detected]      │
  │     └───┘                    │            │                              │
  │                              │            │                              │
  └──────────────────────────────┘            └──────────────────────────────┘
  Telemetry Stream: 1080p @ 42 FPS             Telemetry Stream: FLIR LWIR Core
 ────────────────────────────────────────────────────────────────────────────────────
  LIVE INTERACTION PANEL:
  > Source Target: [X] Upload Custom Media   [ ] Read Built-in Sample Data Assets
  > Active Model:  (•) attention_guided_best  ( ) standard_yolov8_baseline
  
  SYSTEM INTERFACE OUTPUT LOG ENGINE:
  SYSTEM: Loading ECA-Net Layer attention modulation filters...
  INFERENCE: SAHI slicing complete. Bounding box locked at index [X:412, Y:892]
  METRIC: Target classified with high confidence footprint (92.4%)
 ════════════════════════════════════════════════════════════════════════════════════
```

---

## Academic Thesis Citation

The comprehensive research, mathematical validations, performance analysis curves, and system performance evaluations supporting this implementation are fully documented in the attached project thesis.

* **File:** `SARreport.pdf`
* **Title:** *Enhancing Small-Object Human Detection in UAV Imagery Using Lightweight Attention-Guided Models for Search-and-Rescue Applications in Sri Lanka*
* **Author:** Thevindu Nimesh Wijayasinghe (W.L.P.T.N. Wijayasinghe)
* **Affiliation:** NSBM Green University and the University of Plymouth
