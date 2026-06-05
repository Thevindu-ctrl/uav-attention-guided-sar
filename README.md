# UAV Attention-Guided Small-Object Detection for Search-and-Rescue (SAR)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLOv8-006400?style=for-the-badge&logo=analytics&logoColor=white)

An edge-optimized computer vision infrastructure engineered to preserve and isolate sub-32-pixel human targets within high-altitude aerial payload matrices. This framework integrates structural attention profiling directly into real-time object detection backbones to optimize resource management on Size, Weight, and Power (SWaP)-constrained embedded flight devices.

---

## Technical Architecture & Pipeline 

The core engine overcomes high-altitude data down-sampling degradation by incorporating an **ECA-Net (Efficient Channel Attention)** architecture layer into a lightweight convolutional detector footprint.

* **Dimensionality Preservation:** Captures local cross-channel interaction variables without reducing dimensional parameters, maintaining high structural feature density.
* **Dual-Modal Stream Processing:** Provisions separate tracking paths for **Optical RGB** telemetry and **Thermal Infrared** signatures.
* **Edge Acceleration Framework:** Configured to compile down into hardware-accelerated tensor runtimes for smooth edge integration.

<details>
<summary><b>📐Efficient Channel Attention Mathematical Mechanism</b></summary>

The engine bypasses heavy spatial transformation modules by computing adaptive 1D convolution kernels $k$, dynamically adjusted according to channel scale configurations $C$:

$$k = \psi(C) = \left| \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} \right|_{odd}$$

This mapping ensures high frame-rate rendering capabilities by scaling features with negligible parameter weight penalties.

</details>

---

## Production Repository Structure

```text
├── Visdrone.ipynb              # Production network training pipeline
├── visdrone_human.yaml         # Dataset path configuration map
├── best.pt                     # Optimized attention-guided production model
├── yolov8n.pt                  # Standard comparison benchmark model 1
├── rtdetr-l.pt                  # Standard comparison benchmark model 2
├── app.py                      # Streamlit companion app dashboard controller
├── requirements.txt            # System dependencies tracking array
├── data/                       # Ingestion target for validation telemetry
└── outputs/                    # Export destination for tracking anomalies 
and more...