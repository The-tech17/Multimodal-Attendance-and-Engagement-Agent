# 🎓 Multimodal Attendance & Engagement Agent
### *Skill Wallet Capstone Project 5 • Production-Ready AI System*

An ethical, offline edge computing platform and pedagogical intelligence system designed for modern physical classrooms. 
Built in Python and Streamlit, it integrates computer vision, acoustic feature extraction, multimodal fusion, and a heuristic instructional advisory agent to deliver continuous attendance verification and non-intrusive pedagogical feedback without cloud video transmission.

---

## 🌟 Core System Pillars

### 1. Privacy-First Edge Processing
- **100% Offline Edge Computation:** Video and audio feeds are ingested, processed, and evaluated strictly in local volatile RAM. No video frames, audio recordings, or biometric face vectors are ever transmitted to external cloud servers or persisted to disk.
- **Air-Gapped Operation:** Operates without external API keys or cloud database dependencies, enabling seamless 1-click deployment on Streamlit Community Cloud or institutional local servers.

### 2. Ethical, Non-Punitive Telemetry
- **Collective Feedback over Surveillance:** The platform is purpose-built to provide *collective pedagogical feedback* (pacing recommendations, modality switch prompts, collaborative momentum alerts) rather than punitive individual student policing.
- **FERPA, COPPA & GDPR Compliance:** Features participant anonymization, session token hashing, ephemeral cache purging, and aggregated cohort telemetry.

---

## 🏛️ Architecture & Module Breakdown

```
Multimodal Attendance and Engagement Agent/
├── app.py                            # Streamlit Master Application & Interactive Dashboard
├── requirements.txt                  # Production dependencies
├── README.md                         # Comprehensive architecture & operational documentation
├── assets/                           # Local Edge Detection Models & Cascades
│   ├── blaze_face_short_range.tflite # MediaPipe Edge Face Detector
│   ├── haarcascade_frontalface_default.xml # OpenCV Frontal Face Cascade (Offline Fallback)
│   ├── haarcascade_eye.xml          # OpenCV Eye Cascade for Gaze Orientation
│   ├── haarcascade_profileface.xml  # Profile Detection Cascade
│   └── samples/                      # Sample classroom image & synthetic audio assets
├── core/
│   ├── __init__.py
│   ├── privacy.py                    # Privacy engine, FERPA anonymization & memory flush
│   ├── vision_tracker.py             # Face detector, Centroid Tracker, Sustained Presence logic
│   ├── audio_analyzer.py             # Windowing, RMS, ZCR, Spectral Centroid, VAD, 4-Class Classifier
│   ├── fusion_engine.py              # Multimodal fusion matrix, Engagement Index (0-100)
│   ├── pedagogical_agent.py          # AI Teaching Advisory Agent & Pedagogical Playbook
│   └── report_generator.py           # Executive Markdown/HTML educator reports & CSV generator
├── utils/
│   ├── __init__.py
│   ├── sample_data.py                # Pre-packaged classroom datasets (4 scenarios)
│   └── ui_components.py              # Glassmorphic UI cards, Plotly charts, and custom CSS
└── tests/
    └── test_multimodal_agent.py      # Automated unittest suite across all modules
```

---

## 📑 Feature Walkthrough (The 5 Core Tabs)

### 📸 Tab 1: Automated Presence-Based Attendance (Epic 2)
- **Multi-Source Ingestion:** Live webcam (`st.camera_input`), uploaded classroom video/photos (PNG, JPG, MP4), or pre-loaded simulated classroom scenarios.
- **Dual-Engine Face Detection:** High-speed MediaPipe BlazeFace (TFLite edge task) with seamless fallback to OpenCV Haar Cascades.
- **Centroid & Spatial Tracking:** Maintains persistent participant tracking IDs across frames using Euclidean distance and bounding box association.
- **Sustained Presence Logic:** Requires participants to maintain a cumulative presence threshold (**>70% of session duration**, adjustable via sidebar slider from 50% to 90%).
  - Prevents false attendance marks from temporary hallway pass-throughs, cleaning staff, or transitory visitors (<40s or <18% presence).
- **Exportable Attendance Register:** Interactive search and status filtering (`Present (Verified)`, `Partial / Tardy`, `Hallway Pass-Through (Filtered)`). Downloadable as CSV with First Seen, Last Seen, Active Duration, and Attention Score.

### 🎙️ Tab 2: Acoustic Interaction & Noise Dynamics (Epic 3)
- **Sliding-Window Signal Processor:** Segments audio streams into 1.0s sliding windows with 50% overlap.
- **Acoustic Feature Extraction:** Computes Short-Time RMS energy (dBFS), Zero-Crossing Rate (ZCR), Spectral Centroid (Hz), Spectral Rolloff, and Voice Activity Detection (VAD).
- **4-Class Pedagogical Acoustic Classifier:**
  1. `Silence / Quiet Reflection` (RMS < -42 dB, minimal ZCR): Independent problem-solving, reflection, quiz.
  2. `Instructor Lecture` (Centroid 650-2300 Hz, steady RMS): Direct instruction, concept delivery.
  3. `Student Discussion / Verbal Interaction` (Dynamic multi-talker formants): Collaborative peer discourse, Q&A.
  4. `High Ambient Classroom Distraction` (RMS > -14 dB or loud bursts): Unruly noise, dropped objects, door slams.
- **Interactive Visualizations:** 2D STFT Spectrogram (0 - 5 kHz vocal formants), ambient dB volume gauge, and soundscape distribution donut chart.

### 🧬 Tab 3: Multimodal Fusion & Engagement Scoring (Epic 4)
- **Mathematical Fusion Formulation:**
  $$\text{Engagement Index}(t) = \alpha \cdot \text{VisualFocus}(t) + \beta \cdot \text{PostureStability}(t) + \gamma \cdot \text{AcousticResonance}(t)$$
  - *Visual Focus:* Proportion of learners maintaining forward head orientation.
  - *Posture Stability:* Ratio of learners maintaining upright posture vs desk slumping.
  - *Acoustic Resonance:* Active discussion provides high engagement resonance; distraction penalizes; lecture receives steady baseline.
  - *Normalized Weights:* User-tunable in sidebar (default $\alpha=0.50, \beta=0.20, \gamma=0.30$).
- **Normalized Timeline Curve (0 - 100):** Real-time timeline curve with shaded pedagogical zones:
  - 🟢 **Optimal Zone** ($\ge 70\%$)
  - 🟡 **Moderate / Reflective Zone** ($45\% - 69\%$)
  - 🔴 **Attention Flag / Fatigue Zone** ($< 45\%$)
- **KPI Summary:** Average Engagement, Attention Stability Index (/100), Optimal Engagement Time %, and Flagged Dip duration.

### 🤖 Tab 4: AI Teaching Suggestion Engine (Epic 5)
- **Heuristic Advisory Rules:**
  - *Prolonged Attention Dip:* Fires when collective engagement stays below 45% for $>3$ consecutive minutes $\to$ Prompts rapid modality reset (e.g. 60-second peer poll or kinesthetic stretch).
  - *Monologue Fatigue:* Fires when instructor lecture exceeds 12 minutes uninterrupted $\to$ Recommends transitioning to Think-Pair-Share.
  - *Acoustic Distraction Spike:* Sudden environmental noise $\to$ Suggests rhythmic refocusing cue.
  - *Silent Study Stall:* Extended quiet with slouching $\to$ Recommends checkpoint hint.
  - *Peak Collaborative Flow:* Engagement $>82\%$ in group work $\to$ Suggests whiteboard consolidation wrap-up.
- **Pedagogical Playbook Catalog:** Micro-Interventions (<1 min), Active Learning (2-5 mins), Energy Resetters (1-2 mins), and Formative Checks (1-3 mins).
- **Interactive Simulator Sandbox:** Allows educators to adjust engagement, lecture minutes, and acoustic states to observe AI advisory outputs in real time.

### 📊 Tab 5: Session Analytics & Educator Report (Epic 5 & 6)
- **Executive Instructional Briefing:** Formatted briefing evaluating lesson rhythm, pedagogical balance, and student presence.
- **Actionable Professional Development (PD):** Highlights strengths and targeted pacing tweaks for subsequent classes.
- **1-Click Artifact Downloads:**
  - 📄 Executive Briefing Report (`.md` Markdown)
  - 🌐 Formatted Executive Briefing (`.html`)
  - 📥 Verified Attendance Register (`.csv`)
  - 📦 Full Multimodal Time-Series Dataset (`.csv`)

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10+ (Fully tested and compatible up to Python 3.14 on Windows, macOS, and Linux)

### 1. Clone & Install Dependencies
```bash
git clone <repository-url>
cd "Multimodal Attendance and Engagement Agent"
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
The browser will automatically open to `http://localhost:8501`.

### 3. Run Automated Unit Tests
```bash
python -m unittest tests/test_multimodal_agent.py
```

---

## 🔒 Privacy & Compliance Attestation

| Standard | Implementation |
| :--- | :--- |
| **FERPA (34 CFR Part 99)** | Zero student PII persisted; student identifiers hashed or pseudonymized; aggregated cohort reporting. |
| **GDPR (Article 25 - Privacy by Design)** | 100% local edge processing; zero cloud egress; ephemeral buffers wiped on command. |
| **COPPA Safe Harbor** | No commercial telemetry, profiling, or persistent biometric biometric templates stored. |

---

## 🎓 Capstone Project Attributions
- **Course / Track:** Skill Wallet Capstone Project 5 (Multimodal AI & Computer Vision)
- **License:** MIT License
- **Deployment Target:** 1-Click Streamlit Community Cloud (Zero external DB dependencies)
