# AI-Powered Driver Monitoring System (DMS) 🚗

A robust, real-time fatigue and distraction detection system designed for low-light automotive environments. This project uses a hybrid approach, combining Deep Learning (CNN) with Geometric Analysis (EAR) and Head Pose Estimation.



## 🌟 Key Features
- **Hybrid Eye Tracking**: Combines CNN-based texture analysis with Eye Aspect Ratio (EAR) geometry.
- **Sensor Fusion Scoring**: A unique weighted scoring system that reduces false positives.
- **Low-Light Robustness**: Implements CLAHE (Histogram Equalization) for enhanced eyelid visibility in dark cabins.
- **Head Pose Estimation**: 3D Pitch and Yaw detection to identify microsleep-induced head droops or phone-use distraction.
- **Performance Optimized**: Multi-threaded vision pipeline achieving 25+ FPS on standard CPUs.

## 📊 System Architecture


The system is built on a modular architecture:
- `vision/`: MediaPipe-based landmark extraction and Head Pose calculation.
- `model/`: Custom CNN Eye Classifier (24x24 grayscale input).
- `logic/`: The "Scoring Engine" that fuses data from all sensors.
- `utils/`: Alarm triggers and real-time telemetry logging.

## 📈 Performance & Accuracy
- **CNN Validation Accuracy**: ~98.3%
- **Detection Latency**: < 40ms
- **Robustness**: High stability in varying head orientations (Yaw ±25°, Pitch ±15°).



## 🛠️ Installation & Usage
1. Clone the repository.
2. Create a Python 3.10 environment: `py -3.10 -m venv venv`.
3. Activate environment: `.\venv\Scripts\Activate.ps1`.
4. Install dependencies: `pip install -r requirements.txt`.
5. Run the system: `python main.py`.

## 📂 Repository Structure
```text
├── main.py                # Main execution loop
├── config.py              # Threshold configurations
├── vision/                # Face tracking & Geometry
├── logic/                 # Decision making logic
├── model/                 # AI model and predictors
└── utils/                 # Audio alerts and logging