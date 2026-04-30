<div align="center">

# ◉ Activity 3: Python Streamlit + ML Model

### Real-Time Object Detection and Tracking using AI and Webcam

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Overview

In this hands-on activity, learners will build an interactive web application using Streamlit that
integrates a real-time video stream from a webcam and applies Artificial Intelligence for object
detection and tracking.

The system uses the YOLOv8 model to identify everyday objects (such as people, phones, or
bottles) and display them with bounding boxes and labels directly on the video feed. Through this
activity, students will explore how computer vision works in real-world applications, understand
frame-by-frame image processing, and experience how AI models are deployed in live
environments.

---

## 🎯 Learning Outcomes

By completing this activity, learners will:

| # | Outcome | Description |
|---|---------|-------------|
| 1 | 🖥️ **Computer Vision** | Understand the basics of real-time computer vision |
| 2 | 🧠 **AI Processing** | Learn how AI models process video frames |
| 3 | 🌐 **Web App Development** | Build a simple yet powerful AI-powered web app |
| 4 | 🔄 **Object Tracking** | Explore object tracking across multiple frames |

---

## ✅ Expected Software-based Results

### 1️⃣ Functional Web App Interface
A browser-based app displaying:
- **Title**: Live Object Detection & Tracing
- **Webcam feed** embedded in the page

### 2️⃣ Live Camera Detection
When the camera is turned on:
- ✦ Objects are **detected instantly**
- ✦ **Bounding boxes** appear around objects
- ✦ **Labels** (e.g., person, cell phone, bottle) are shown

### 3️⃣ Object Tracking Behavior
- ✦ Moving objects are **continuously tracked**
- ✦ The same object keeps its **identity across frames** (smooth tracking)

---

## 🖼️ Application Design

### Idle State (Before Detection)
```
┌─────────────────────────────────────────────────────────────────────┐
│  ◉ Live Object Detection & Tracing                    ● Idle      │
├─────────────────────────────────────────────────────────────────────┤
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│ │ ⬡ Model  │ │ ⚡Compute│ │ 📊 Dets  │ │ ◎ Camera │               │
│ │  YOLOV8N │ │   CPU    │ │    0     │ │  640×480 │               │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
│                                                                     │
│          ┌─────────────────────────────────┐                       │
│          │              ◎                  │                       │
│          │     Ready to Detect             │                       │
│          │                                 │                       │
│          │  Configure your model and camera │                       │
│          │  in the sidebar, then press     │                       │
│          │      ▶ Start Detection          │                       │
│          │                                 │                       │
│          │  ◈ Real-time  ◎ 80+ classes     │                       │
│          │  ⬡ Human alert ▦ Analytics       │                       │
│          └─────────────────────────────────┘                       │
│                                                                     │
│              [ ▶ Start Detection ]                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Live Detection State
```
┌─────────────────────────────────────────────────────────────────────┐
│  ◉ Live Object Detection & Tracing                   🔴 Live       │
├─────────────────────────────────────────────────────────────────────┤
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│ │ ⬡ Model  │ │ ⚡Compute│ │ 📊 Dets  │ │ ◎ Camera │               │
│ │  YOLOV8N │ │   CUDA   │ │    3     │ │  640×480 │               │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
│                                                                     │
│  ┌─────────────────────────────┐  ┌──────────────────┐             │
│  │                             │  │ FPS    30.2 fps  │             │
│  │   ┌─────────┐              │  │                  │             │
│  │   │ person  │              │  │ ● person     ×2  │             │
│  │   │  92%    │              │  │ ● cell phone ×1  │             │
│  │   └─────────┘              │  │                  │             │
│  │         ┌──────────┐       │  │ ✋ Open Palm     │             │
│  │         │cell phone│       │  │ ☺ Neutral        │             │
│  │         │   78%    │       │  │                  │             │
│  │         └──────────┘       │  │ Session          │             │
│  │                             │  │ Frames:  450    │             │
│  │    ✦✦ particle effects ✦✦   │  │ Elapsed: 15.0s  │             │
│  └─────────────────────────────┘  └──────────────────┘             │
│                                                                     │
│              [ ⏹ Stop Detection ]                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Sidebar Configuration Panel
```
┌──────────────────────┐
│ ◉ VisionAI           │
│ Configuration Panel   │
├──────────────────────┤
│ ⬡ Model              │
│ ┌──────────────────┐  │
│ │ Nano — Fastest ▼ │  │
│ └──────────────────┘  │
│                       │
│ ≡ Detection Thresholds│
│ Confidence  ━━━●━━ 0.50│
│ NMS IoU     ━━●━━━ 0.45│
│                       │
│ ◎ Camera             │
│ Device index   [0]   │
│ Resolution  640×480 ▼ │
│                       │
│ ⚙ Display            │
│ ☑ FPS overlay        │
│ ☑ Mirror camera      │
│ Frame-rate  ━━●━━ 30 │
│                       │
│ ✋ Hand Gestures      │
│ ☐ Enable hand detect │
│                       │
│ ☺ Face Expressions   │
│ ☐ Enable face detect │
│                       │
│ ✦ Particle Effects   │
│ ☐ Enable particles   │
│                       │
│ 💾 Save Frames       │
│ ☐ Save detected frames│
│                       │
│ 🔊 Audio Alerts      │
│ ☐ Enable TTS alerts  │
│                       │
│ ℹ System             │
│ Compute  [CPU]       │
│ GPU      —           │
│ PyTorch  2.x.x       │
└──────────────────────┘
```

---

## ⚡ Features

| Feature | Icon | Description |
|---------|------|-------------|
| Real-time Object Detection | 🔍 | YOLOv8 detects 80+ COCO object classes |
| Hand Gesture Recognition | ✋ | Detects 7 gestures: Fist, Open Palm, Pointing, Peace, Thumbs Up, Hang Loose, Rock |
| Face Expression Detection | ☺ | Recognizes Smiling, Angry, Surprised, Neutral |
| Gesture Particle Effects | ✦ | Interactive particle system responding to hand gestures |
| Audio Alerts | 🔊 | Text-to-speech announcements for detections |
| Frame Saving | 💾 | Save detected frames as images with configurable frequency |
| Multiple Model Variants | ⬡ | Nano → XLarge (speed vs accuracy trade-off) |
| GPU Acceleration | ⚡ | Automatic CUDA detection and utilization |
| Responsive Dark UI | 🌙 | Modern dark-themed interface with sidebar controls |

---

## 📁 File Structure

```
act_3/
├── 📄 main.py              # Main application (Streamlit UI + detection logic)
├── 📄 requirements.txt     # Python dependencies
├── 📄 README.md            # Project documentation
├── 📄 .gitignore           # Git ignore rules
├── 🤖 yolov8n.pt           # YOLOv8 Nano model (~6MB, fastest)
├── 🤖 yolov8x.pt           # YOLOv8 XLarge model (~131MB, most accurate)
└── 📁 saved_frames/        # Auto-created directory for saved frames
    ├── frame_0001.jpg
    ├── frame_0002.jpg
    └── ...
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Webcam (built-in or external)
- Windows OS (for audio alerts)

### Setup

```bash
# 1. Clone or download the project
cd act_3

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run main.py
```

The app will automatically open in your default browser at `http://localhost:8501`

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | >=1.36.0 | Web application framework |
| `opencv-python` | >=4.8.0 | Computer vision & video processing |
| `numpy` | >=2.0.0 | Numerical computing & array operations |
| `ultralytics` | >=8.0.0 | YOLOv8 object detection model |
| `mediapipe` | >=0.10.0 | Hand & face landmark detection |
| `torch` | auto | PyTorch deep learning (installed with ultralytics) |
| `pyttsx3` | >=2.90 | Text-to-speech audio alerts (Windows) |

---

## 🎮 Usage

### Starting the Application

1. Run `streamlit run main.py` in your terminal
2. The app opens in your default web browser
3. Configure settings in the **sidebar** on the left
4. Click **▶ Start Detection** to begin real-time inference

### Sidebar Controls

<details>
<summary>🔧 Model & Detection Settings</summary>

- **Model Variant**: 
  - 🟢 Nano: Fastest inference, lower accuracy (~6MB)
  - 🔵 Small: Balanced speed and accuracy
  - 🟡 Medium: Good accuracy, moderate speed
  - 🟠 Large: High accuracy, slower
  - 🔴 XLarge: Maximum accuracy, slowest (~131MB)

- **Detection Thresholds**:
  - Confidence: Minimum confidence for detection (lower = more detections)
  - NMS IoU: Non-maximum suppression threshold (remove overlapping boxes)

</details>

<details>
<summary>📷 Camera & Display Settings</summary>

- **Camera Settings**:
  - Device Index: Camera number (0 = default webcam)
  - Resolution: 640×480 / 800×600 / 1280×720

- **Display Options**:
  - FPS Overlay: Show real-time frame rate
  - Mirror Camera: Flip video horizontally
  - Frame-rate Cap: Limit maximum FPS (5-60)

</details>

<details>
<summary>✋ Hand & Face Detection</summary>

- **Hand Gestures**:
  - Enable hand detection toggle
  - Hand confidence threshold (0.30-1.0)

- **Face Expressions**:
  - Enable face detection toggle
  - Face confidence threshold (0.30-1.0)

</details>

<details>
<summary>✦ Effects & Extras</summary>

- **Particle Effects**:
  - Enable particle effects toggle
  - Max particles slider (50-400)

- **Save Frames**:
  - Enable frame saving toggle
  - Save folder path (default: ./saved_frames)
  - Save every N frames slider (1-60)

- **Audio Alerts**:
  - Enable text-to-speech alerts toggle

</details>

---

## 🏷️ Detectable Classes (COCO Dataset — 80 Classes)

<details>
<summary>👥 People & Animals</summary>

| Category | Classes |
|----------|---------|
| People | `person` |
| Pets | `cat`, `dog`, `bird` |
| Farm | `horse`, `sheep`, `cow`, `chicken` |
| Wild | `elephant`, `bear`, `zebra`, `giraffe` |

</details>

<details>
<summary>🚗 Vehicles & Outdoor</summary>

| Category | Classes |
|----------|---------|
| Road | `bicycle`, `car`, `motorcycle`, `bus`, `truck` |
| Other | `airplane`, `train`, `boat` |
| Street | `traffic light`, `fire hydrant`, `stop sign`, `parking meter` |
| Rest | `bench` |

</details>

<details>
<summary>🏠 Household & Kitchen</summary>

| Category | Classes |
|----------|---------|
| Dining | `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl` |
| Food | `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake` |
| Kitchen | `microwave`, `oven`, `toaster`, `sink`, `refrigerator` |
| Furniture | `chair`, `couch`, `bed`, `dining table`, `potted plant` |
| Bath | `toilet`, `toothbrush`, `hair drier` |

</details>

<details>
<summary>💻 Electronics & Personal</summary>

| Category | Classes |
|----------|---------|
| Tech | `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone` |
| Personal | `backpack`, `umbrella`, `handbag`, `suitcase`, `tie` |
| Decor | `book`, `clock`, `vase`, `scissors`, `teddy bear` |
| Sports | `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`, `frisbee`, `snowboard` |

</details>

---

## ✋ Hand Gestures

| Gesture | Visual | Particle Effect |
|---------|--------|-----------------|
| ✊ Fist | Closed hand | Particles pulled inward (gravity) |
| 🖐️ Open Palm | Open hand, fingers extended | Particles explode outward |
| ☝️ Pointing | Index finger pointing | Particles shoot in pointing direction |
| ✌️ Peace | Two fingers (V sign) | Particles fly in V pattern |
| 👍 Thumbs Up | Thumb raised | Particles float upward |
| 🤙 Hang Loose | Shaka sign | Particles spiral outward |
| 🤘 Rock | Index + pinky extended | Particles pulse at center |

---

## ☺ Face Expressions

| Expression | Description | Detection Method |
|------------|-------------|------------------|
| 😊 Smiling | Happy expression | Mouth corners raised |
| 😠 Angry | Angry expression | Eyebrows lowered, mouth tense |
| 😮 Surprised | Surprised expression | Eyes wide, mouth open |
| 😐 Neutral | Calm/resting face | Default facial state |

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit Server                │
│                                                   │
│  ┌───────────┐  ┌───────────┐  ┌──────────────┐ │
│  │  YOLOv8   │  │ MediaPipe │  │   Particle   │ │
│  │  Object   │  │  Hands +  │  │   System     │ │
│  │ Detection │  │Face Mesh  │  │  (NumPy)     │ │
│  └─────┬─────┘  └─────┬─────┘  └──────┬───────┘ │
│        │               │               │         │
│        └───────────────┬┴───────────────┘         │
│                        │                          │
│              ┌─────────▼──────────┐               │
│              │   Frame Processor  │               │
│              │   (OpenCV + NumPy) │               │
│              └─────────┬──────────┘               │
│                        │                          │
│              ┌─────────▼──────────┐               │
│              │   Streamlit Render  │               │
│              │   (Web UI + TTS)    │               │
│              └────────────────────┘               │
└─────────────────────────────────────────────────┘
```

### Model Details
| Component | Model | Landmarks |
|-----------|-------|-----------|
| Object Detection | YOLOv8 (COCO) | Bounding boxes + labels |
| Hand Detection | MediaPipe Hands | 21 landmarks per hand |
| Face Detection | MediaPipe Face Mesh | 468 landmarks per face |

### Performance Optimizations
- ✦ OpenCL disabled for CPU compatibility
- ✦ Frame skipping for hand/face detection (every 3-4 frames)
- ✦ Particle system with NumPy vectorization
- ✦ Model caching with `@st.cache_resource`
- ✦ Camera buffer size set to 1 (reduces latency)

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| 📷 Camera not detected | Try device index 0, 1, or 2. Close other apps using camera. Check browser permissions. |
| 🎯 Low detection accuracy | Use Medium/Large model. Lower confidence to 0.25-0.30. Improve lighting. Hold objects closer. |
| 🐌 Slow performance | Use Nano model. Lower resolution to 640×480. Disable particles/hand/face. Reduce FPS cap. |
| 🔊 Audio alerts not working | Windows only. Check speakers. Check audio settings. Does not work when hosted. |
| 📂 Sidebar disappeared | Click the ☰ hamburger icon in the top-left corner to toggle sidebar. |

---

## ⚠️ Important Notes

> **Object Tracking**: The current implementation detects objects frame-by-frame. For true persistent identity tracking across frames, additional algorithms (e.g., DeepSORT, ByteTrack) would need to be integrated.

> **Hosting Limitations**: 
> - Audio alerts (text-to-speech) only work locally on Windows
> - For hosted deployments, browser-based TTS requires WebSocket architecture
> - Camera access requires HTTPS in production environments

> **Model Files**: Pre-trained YOLOv8 models are included. For custom object detection, train your own model using the Ultralytics framework.

---

## 📄 License

This project is for educational purposes. YOLOv8 is licensed under [AGPL-3.0](https://ultralytics.com/license).

---

<div align="center">

**Built with ❤️ using Streamlit, YOLOv8, and MediaPipe**

</div> 
