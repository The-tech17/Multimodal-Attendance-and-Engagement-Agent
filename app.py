"""
Multimodal Attendance & Engagement Agent
Skill Wallet Capstone Project 5

Core Principles:
1. Privacy-First Edge Processing: 100% offline processing, zero cloud video transmission.
2. Ethical Non-Punitive Telemetry: Collective pedagogical optimization over individual surveillance.
3. Modular Architecture: Presence tracking, acoustic dynamics, multimodal fusion, AI advisory agent, and educator reporting.
"""

import os
import time
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Core Modules
from core.privacy import PrivacyEngine
from core.vision_tracker import VisionAttendanceTracker, CentroidTracker
from core.audio_analyzer import AudioAcousticAnalyzer
from core.fusion_engine import MultimodalFusionEngine
from core.pedagogical_agent import PedagogicalAdvisoryAgent
from core.report_generator import EducatorReportGenerator

# Utility Modules
from utils.sample_data import SampleDataGenerator
from utils.ui_components import (
    CUSTOM_CSS,
    render_privacy_banner,
    render_kpi_card,
    create_engagement_timeline_chart,
    create_acoustic_pie_chart,
    create_attendance_distribution_chart,
    create_spectrogram_plot
)


# --- Page Configuration ---
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Session State Initialization ---
if "privacy_engine" not in st.session_state:
    st.session_state.privacy_engine = PrivacyEngine(anonymize_participants=True)

if "vision_tracker" not in st.session_state:
    st.session_state.vision_tracker = VisionAttendanceTracker()

if "centroid_tracker" not in st.session_state:
    st.session_state.centroid_tracker = CentroidTracker()

if "audio_analyzer" not in st.session_state:
    st.session_state.audio_analyzer = AudioAcousticAnalyzer()

if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = "Scenario 1: Interactive STEM Lecture & Problem Solving"

if "attendance_df" not in st.session_state:
    # Load default initial scenario
    att_df, track_hist = SampleDataGenerator.generate_scenario_attendance(st.session_state.active_scenario)
    st.session_state.attendance_df = att_df
    st.session_state.tracking_history = track_hist

if "fused_df" not in st.session_state:
    vis_df, ac_df = SampleDataGenerator.generate_scenario_multimodal_timeline(st.session_state.active_scenario)
    st.session_state.visual_df = vis_df
    st.session_state.acoustic_df = ac_df
    fusion_engine = MultimodalFusionEngine(0.50, 0.20, 0.30)
    st.session_state.fused_df = fusion_engine.fuse_streams(vis_df, ac_df)
    advisor = PedagogicalAdvisoryAgent(dip_threshold=45.0)
    st.session_state.advisory_events = advisor.scan_timeline(st.session_state.fused_df)


# --- Sidebar Controls & Edge Architecture Telemetry ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("Edge Attendance & Engagement Agent")
    st.caption("Skill Wallet Capstone Project 5 • Production Release")

    st.markdown("---")
    st.subheader("🛡️ Edge Privacy & Compliance")
    
    st.markdown("""
    - **Engine Status:** 🟢 `100% Local Edge Active`
    - **Cloud Transmission:** 🔒 `0 Bytes (Air-gapped)`
    - **Telemetry Mode:** 👥 `Cohort Pedagogical Aggregation`
    - **Compliance:** FERPA • GDPR Art. 25 • COPPA
    """)

    anonymize_toggle = st.toggle(
        "Participant Anonymization (FERPA)",
        value=st.session_state.privacy_engine.anonymize_participants,
        help="Replaces raw tracking identifiers with pseudorandom learner tokens."
    )
    if anonymize_toggle != st.session_state.privacy_engine.anonymize_participants:
        st.session_state.privacy_engine.anonymize_participants = anonymize_toggle
        st.rerun()

    if st.button("🧹 Flush Ephemeral Memory Buffers", use_container_width=True):
        res = st.session_state.privacy_engine.flush_memory_buffers()
        st.session_state.centroid_tracker = CentroidTracker()
        st.success(res["message"])

    st.markdown("---")
    st.subheader("📚 Quick Scenario Loader")
    selected_scenario = st.selectbox(
        "Load Classroom Dataset:",
        options=list(SampleDataGenerator.SCENARIOS.keys()),
        index=list(SampleDataGenerator.SCENARIOS.keys()).index(st.session_state.active_scenario)
    )

    if selected_scenario != st.session_state.active_scenario:
        st.session_state.active_scenario = selected_scenario
        att_df, track_hist = SampleDataGenerator.generate_scenario_attendance(selected_scenario)
        st.session_state.attendance_df = att_df
        st.session_state.tracking_history = track_hist
        vis_df, ac_df = SampleDataGenerator.generate_scenario_multimodal_timeline(selected_scenario)
        st.session_state.visual_df = vis_df
        st.session_state.acoustic_df = ac_df
        fusion_engine = MultimodalFusionEngine(0.50, 0.20, 0.30)
        st.session_state.fused_df = fusion_engine.fuse_streams(vis_df, ac_df)
        advisor = PedagogicalAdvisoryAgent(dip_threshold=45.0)
        st.session_state.advisory_events = advisor.scan_timeline(st.session_state.fused_df)
        st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Sustained Presence & Fusion")
    sustained_threshold = st.slider(
        "Sustained Presence Threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05,
        help="Participants must be detected for >70% of session time to receive verified attendance."
    )

    with st.expander("⚖️ Multimodal Fusion Weights", expanded=False):
        w_visual = st.slider("Visual Focus Weight (α)", 0.0, 1.0, 0.50, 0.05)
        w_posture = st.slider("Posture Stability Weight (β)", 0.0, 1.0, 0.20, 0.05)
        w_acoustic = st.slider("Acoustic Dynamics Weight (γ)", 0.0, 1.0, 0.30, 0.05)
        
        if st.button("Apply New Fusion Weights", use_container_width=True):
            fusion_engine = MultimodalFusionEngine(w_visual, w_posture, w_acoustic)
            st.session_state.fused_df = fusion_engine.fuse_streams(
                st.session_state.visual_df, st.session_state.acoustic_df
            )
            advisor = PedagogicalAdvisoryAgent(dip_threshold=45.0)
            st.session_state.advisory_events = advisor.scan_timeline(st.session_state.fused_df)
            st.success("Multimodal timeline recalculated!")
            st.rerun()

    st.markdown("---")
    st.caption("Multimodal Attendance & Engagement Agent v2.5 • Edge AI System")


# --- Main Application Header ---
st.markdown(render_privacy_banner(), unsafe_allow_html=True)

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Multimodal Attendance & Engagement Agent")
    st.markdown(
        f"**Active Session:** `{st.session_state.active_scenario}` • "
        f"**Detector:** `{st.session_state.vision_tracker.detector_type}`"
    )
with col_h2:
    st.metric(
        label="Edge Privacy Boundary",
        value="Air-Gapped",
        delta="0 Cloud Egress",
        delta_color="normal"
    )

st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)


# --- 5 Primary Modules Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Tab 1: Presence Attendance (Epic 2)",
    "🎙️ Tab 2: Acoustic Dynamics (Epic 3)",
    "🧬 Tab 3: Multimodal Fusion (Epic 4)",
    "🤖 Tab 4: AI Teaching Engine (Epic 5)",
    "📊 Tab 5: Session Analytics & Report (Epic 5 & 6)"
])


# ==============================================================================
# TAB 1: AUTOMATED PRESENCE-BASED ATTENDANCE (EPIC 2)
# ==============================================================================
with tab1:
    st.header("📸 Automated Presence-Based Attendance & Tracking")
    st.markdown("""
    **Zero-Surveillance Presence Logic:** Employs edge face tracking to monitor continuous classroom attendance. 
    Participants must maintain **sustained presence (>70% session time)** to be marked **Present**, ensuring temporary hallway pass-throughs and visitors are automatically filtered out.
    """)

    col_t1_input, col_t1_viz = st.columns([1, 1])

    with col_t1_input:
        st.subheader("1. Video Feed & Detector Pipeline")
        input_mode = st.radio(
            "Select Processing Input Source:",
            ["Pre-loaded Classroom Scenario", "Live Webcam Feed", "Upload Classroom Photo/Video"],
            horizontal=True
        )

        annotated_frame = None
        current_detections = []

        if input_mode == "Pre-loaded Classroom Scenario":
            st.info("💡 Processing sample classroom frame with edge face detector & tracker.")
            sample_img_path = "assets/samples/classroom_sample.jpg"
            if os.path.exists(sample_img_path):
                raw_frame = cv2.imread(sample_img_path)
                current_detections = st.session_state.vision_tracker.detect_and_estimate_attentiveness(raw_frame)
                
                # Update centroid tracker
                rects = [d["bbox"] for d in current_detections]
                st.session_state.centroid_tracker.update(rects, timestamp=time.time(), metadatas=current_detections)
                
                annotated_frame = st.session_state.vision_tracker.annotate_frame(
                    raw_frame,
                    st.session_state.centroid_tracker.bboxes,
                    st.session_state.centroid_tracker.history,
                    sustained_threshold=sustained_threshold,
                    session_duration=60.0
                )
            else:
                st.warning("Sample classroom frame not found.")

        elif input_mode == "Live Webcam Feed":
            st.markdown("Enable camera for live edge face detection and gaze tracking.")
            camera_img = st.camera_input("Capture Classroom Webcam Frame")
            if camera_img is not None:
                bytes_data = camera_img.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                current_detections = st.session_state.vision_tracker.detect_and_estimate_attentiveness(cv_img)
                
                rects = [d["bbox"] for d in current_detections]
                st.session_state.centroid_tracker.update(rects, timestamp=time.time(), metadatas=current_detections)
                
                annotated_frame = st.session_state.vision_tracker.annotate_frame(
                    cv_img,
                    st.session_state.centroid_tracker.bboxes,
                    st.session_state.centroid_tracker.history,
                    sustained_threshold=sustained_threshold,
                    session_duration=60.0
                )

        else: # Upload Classroom Photo/Video
            uploaded_file = st.file_uploader("Upload Classroom Image or Video Frame", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                current_detections = st.session_state.vision_tracker.detect_and_estimate_attentiveness(cv_img)
                rects = [d["bbox"] for d in current_detections]
                st.session_state.centroid_tracker.update(rects, timestamp=time.time(), metadatas=current_detections)
                
                annotated_frame = st.session_state.vision_tracker.annotate_frame(
                    cv_img,
                    st.session_state.centroid_tracker.bboxes,
                    st.session_state.centroid_tracker.history,
                    sustained_threshold=sustained_threshold,
                    session_duration=60.0
                )

    with col_t1_viz:
        st.subheader("2. Edge Visual Telemetry & Tracking Overlay")
        if annotated_frame is not None:
            rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_annotated, caption=f"Active Detections: {len(current_detections)} Participants Tracked", use_container_width=True)
        else:
            st.image("https://placehold.co/600x400/222b3a/ffffff?text=Video+Feed+Standby", caption="Waiting for video source", use_container_width=True)

    st.markdown("---")
    st.subheader("3. Sustained Presence Attendance Register")
    st.caption(f"Sustained Presence Threshold: **>{int(sustained_threshold * 100)}% of class duration**. Transitory visitors (<40s or <18%) are automatically filtered.")

    # Re-evaluate attendance with active threshold slider
    active_att_df = st.session_state.attendance_df.copy()
    
    # Update threshold logic if slider moved
    threshold_pct = sustained_threshold * 100.0
    for idx, row in active_att_df.iterrows():
        pct = row["Presence Rate (%)"]
        dur_s = row.get("Raw Duration (s)", 100.0)
        if pct >= threshold_pct:
            active_att_df.at[idx, "Attendance Status"] = "Present (Verified)"
            active_att_df.at[idx, "Status Badge"] = "✅ Present"
            active_att_df.at[idx, "Eligible Attendance"] = True
        elif dur_s < 40.0 or pct < 18.0:
            active_att_df.at[idx, "Attendance Status"] = "Hallway Pass-Through (Filtered)"
            active_att_df.at[idx, "Status Badge"] = "🚪 Filtered (Pass-through)"
            active_att_df.at[idx, "Eligible Attendance"] = False
        else:
            active_att_df.at[idx, "Attendance Status"] = "Partial / Tardy"
            active_att_df.at[idx, "Status Badge"] = "⚠️ Partial Attendance"
            active_att_df.at[idx, "Eligible Attendance"] = False

    # KPI row for attendance
    total_tracked = len(active_att_df)
    verified_present = sum(active_att_df["Eligible Attendance"])
    filtered_passers = sum(active_att_df["Attendance Status"].str.contains("Filtered", na=False))
    partial_count = total_tracked - verified_present - filtered_passers

    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.markdown(render_kpi_card("Total Tracked", str(total_tracked), "All detected entities"), unsafe_allow_html=True)
    with kcol2:
        st.markdown(render_kpi_card("Verified Present", f"{verified_present}", f"≥ {int(threshold_pct)}% Presence", "#059669"), unsafe_allow_html=True)
    with kcol3:
        st.markdown(render_kpi_card("Hallway Pass-Throughs", f"{filtered_passers}", "Safely filtered out", "#64748b"), unsafe_allow_html=True)
    with kcol4:
        st.markdown(render_kpi_card("Partial / Tardy", f"{partial_count}", "Below threshold", "#d97706"), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Search and Filter
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    with f_col1:
        search_query = st.text_input("🔍 Search Participant ID:", placeholder="e.g. Learner-001")
    with f_col2:
        status_filter = st.selectbox("Filter Status:", ["All", "Present (Verified)", "Partial / Tardy", "Hallway Pass-Through (Filtered)"])
    with f_col3:
        csv_data = active_att_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Register (CSV)",
            data=csv_data,
            file_name=f"Attendance_Register_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Filter DataFrame
    filtered_df = active_att_df.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df["Participant ID"].str.contains(search_query, case=False, na=False)]
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["Attendance Status"] == status_filter]

    st.dataframe(
        filtered_df[[
            "Participant ID", "First Seen", "Last Seen", "Active Duration",
            "Presence Rate (%)", "Avg Attention (%)", "Status Badge"
        ]],
        use_container_width=True,
        hide_index=True
    )


# ==============================================================================
# TAB 2: ACOUSTIC INTERACTION & NOISE DYNAMICS (EPIC 3)
# ==============================================================================
with tab2:
    st.header("🎙️ Acoustic Interaction & Noise Dynamics")
    st.markdown("""
    **Privacy-Preserving Soundscape Telemetry:** Classifies audio streams into distinct pedagogical states using spectral energy, 
    spectral centroid, and Voice Activity Detection (VAD). **Zero speech-to-text or conversation recording is ever executed.**
    """)

    a_col_input, a_col_stat = st.columns([1, 1])

    with a_col_input:
        st.subheader("1. Audio Signal Stream Input")
        audio_source_type = st.radio(
            "Select Acoustic Stream Source:",
            ["Pre-loaded Scenario Soundscape", "Upload Classroom Audio (.WAV / .MP3)", "Live Acoustic Sandbox Simulator"],
            horizontal=True
        )

        active_audio_df = st.session_state.acoustic_df.copy()

        if audio_source_type == "Upload Classroom Audio (.WAV / .MP3)":
            uploaded_audio = st.file_uploader("Upload Audio Stream", type=["wav", "mp3", "ogg", "flac"])
            if uploaded_audio is not None:
                with st.spinner("Analyzing audio signal locally..."):
                    audio_sig, sr = st.session_state.audio_analyzer.load_audio_file(uploaded_audio)
                    active_audio_df = st.session_state.audio_analyzer.analyze_stream(audio_sig)
                    st.session_state.acoustic_df = active_audio_df
                    st.success(f"Analyzed {len(active_audio_df)} audio time-windows!")

        elif audio_source_type == "Live Acoustic Sandbox Simulator":
            st.markdown("Generate synthetic classroom acoustic conditions:")
            sim_scenario = st.selectbox(
                "Simulated Acoustic Scenario:",
                ["Standard Active Lecture", "Collaborative Group Discussion", "Quiet Study & Assessment", "Noisy Distraction Spike"]
            )
            sim_duration = st.slider("Signal Duration (Seconds):", 10, 60, 30, 5)
            if st.button("Generate & Process Audio Signal", use_container_width=True):
                with st.spinner("Synthesizing and classifying classroom soundscape..."):
                    syn_audio = AudioAcousticAnalyzer.generate_synthetic_classroom_audio(sim_duration, sim_scenario)
                    active_audio_df = st.session_state.audio_analyzer.analyze_stream(syn_audio)
                    st.session_state.acoustic_df = active_audio_df
                    st.success(f"Generated {len(active_audio_df)} classified soundscape windows!")

    with a_col_stat:
        st.subheader("2. Pedagogical Acoustic States")
        st.markdown("""
        | Acoustic State | Frequency / Energy Characteristics | Classroom Meaning |
        | :--- | :--- | :--- |
        | **Silence / Quiet Reflection** | RMS < -42 dB, minimal ZCR | Focused problem solving, quiz, reading |
        | **Instructor Lecture** | Centroid 650-2300 Hz, steady RMS | Direct presentation, concept delivery |
        | **Student Discussion** | Dynamic pitch bursts, multi-talker | Active collaborative breakouts, Q&A |
        | **High Distraction** | RMS > -14 dB, high spectral rolloff | Dropped objects, door slams, loud clatter |
        """)

    st.markdown("---")
    st.subheader("3. Soundscape Analytics & Spectrogram")

    # Acoustic KPI cards
    avg_db = round(float(active_audio_df["RMS Energy (dB)"].mean()), 1) if not active_audio_df.empty else -30.0
    active_vad_pct = round(float(active_audio_df["VAD Active"].mean() * 100.0), 1) if not active_audio_df.empty else 60.0
    distraction_count = sum(active_audio_df["Acoustic Class"] == AudioAcousticAnalyzer.CLASS_DISTRACTION) if not active_audio_df.empty else 0

    akcol1, akcol2, akcol3, akcol4 = st.columns(4)
    with akcol1:
        st.markdown(render_kpi_card("Mean Ambient dB", f"{avg_db} dBFS", "Average energy level"), unsafe_allow_html=True)
    with akcol2:
        st.markdown(render_kpi_card("Speech Activity (VAD)", f"{active_vad_pct}%", "Active vocal presence", "#059669"), unsafe_allow_html=True)
    with akcol3:
        st.markdown(render_kpi_card("Distraction Spikes", f"{distraction_count}", "Noise > -14 dB", "#ef4444" if distraction_count > 0 else "#059669"), unsafe_allow_html=True)
    with akcol4:
        dominant_state = active_audio_df["Acoustic Class"].mode()[0] if not active_audio_df.empty else "Instructor Lecture"
        st.markdown(render_kpi_card("Dominant Mode", dominant_state.split("/")[0].strip(), "Pedagogical rhythm", "#3b82f6"), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Donut Chart & Spectrogram
    chart_col1, chart_col2 = st.columns([1, 1])
    with chart_col1:
        pie_fig = create_acoustic_pie_chart(active_audio_df)
        st.plotly_chart(pie_fig, use_container_width=True)

    with chart_col2:
        # Synthetic / real spectrogram
        test_audio = AudioAcousticAnalyzer.generate_synthetic_classroom_audio(20, "Standard Active Lecture")
        t_spec, f_spec, Sxx_db = st.session_state.audio_analyzer.compute_spectrogram_matrix(test_audio)
        spec_fig = create_spectrogram_plot(t_spec, f_spec, Sxx_db)
        st.plotly_chart(spec_fig, use_container_width=True)

    # Timeline table
    with st.expander("🔍 Inspect Acoustic Time-Window Log (CSV)", expanded=False):
        st.dataframe(active_audio_df.head(50), use_container_width=True)
        csv_audio = active_audio_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Acoustic Windows (CSV)",
            data=csv_audio,
            file_name="Acoustic_Timeline.csv",
            mime="text/csv"
        )


# ==============================================================================
# TAB 3: MULTIMODAL FUSION & ENGAGEMENT SCORING (EPIC 4)
# ==============================================================================
with tab3:
    st.header("🧬 Multimodal Fusion & Engagement Scoring")
    st.markdown("""
    **Unified Engagement Index Formulation:** Harmonizes visual attentiveness (forward gaze, head pose), 
    posture dynamics (upright vs slumped), and acoustic resonance into a continuous normalized Engagement Index (0-100).
    """)

    # Fusion formula reminder
    st.latex(r"Engagement(t) = \alpha \cdot Visual(t) + \beta \cdot Posture(t) + \gamma \cdot Acoustic(t)")

    fused_df = st.session_state.fused_df.copy()
    fusion_kpis = MultimodalFusionEngine.extract_summary_kpis(fused_df)

    # KPI Banner
    fkcol1, fkcol2, fkcol3, fkcol4 = st.columns(4)
    with fkcol1:
        st.markdown(render_kpi_card(
            "Average Engagement",
            f"{fusion_kpis['avg_engagement']}%",
            "Collective cohort index",
            "#059669" if fusion_kpis['avg_engagement'] >= 70 else "#d97706"
        ), unsafe_allow_html=True)
    with fkcol2:
        st.markdown(render_kpi_card(
            "Attention Stability",
            f"{fusion_kpis['attention_stability_index']}/100",
            "Low variance indicator",
            "#3b82f6"
        ), unsafe_allow_html=True)
    with fkcol3:
        st.markdown(render_kpi_card(
            "Optimal Focus Time",
            f"{fusion_kpis['optimal_time_pct']}%",
            "Time spent ≥ 70% engagement",
            "#059669"
        ), unsafe_allow_html=True)
    with fkcol4:
        st.markdown(render_kpi_card(
            "Flagged Attention Dips",
            f"{fusion_kpis['dip_count']} episodes",
            f"{fusion_kpis['dip_duration_sec']}s total below 45%",
            "#ef4444" if fusion_kpis['dip_count'] > 0 else "#059669"
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Master Timeline Curve
    timeline_fig = create_engagement_timeline_chart(fused_df, st.session_state.advisory_events)
    st.plotly_chart(timeline_fig, use_container_width=True)

    # Component correlation breakdown
    with st.expander("📊 Component Breakdown & Matrix Synchronization", expanded=False):
        st.markdown("""
        The table below shows the synchronized multimodal time-step matrix used to calculate the real-time timeline engagement curves.
        """)
        st.dataframe(fused_df, use_container_width=True, hide_index=True)
        csv_fused = fused_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Fused Engagement Matrix (CSV)",
            data=csv_fused,
            file_name="Multimodal_Engagement_Matrix.csv",
            mime="text/csv"
        )


# ==============================================================================
# TAB 4: AI TEACHING SUGGESTION ENGINE (EPIC 5)
# ==============================================================================
with tab4:
    st.header("🤖 AI Teaching Suggestion Engine")
    st.markdown("""
    **Non-Intrusive Pedagogical Advisory Agent:** Monitors collective classroom dynamics and automatically fires 
    empathetic, non-punitive instructional recommendations when engagement dips or when direct lecturing continues too long.
    """)

    col_t4_events, col_t4_sim = st.columns([1, 1])

    with col_t4_events:
        st.subheader("1. Active Pedagogical Prompts Feed")
        events = st.session_state.advisory_events

        if events:
            for ev in events:
                sev = ev["severity"]
                css_cls = "critical" if "Urgent" in sev else "action" if "Action" in sev else "info"
                
                st.markdown(f"""
                <div class="prompt-card {css_cls}">
                    <div class="prompt-header">
                        <span class="prompt-title">{ev['headline']}</span>
                        <span class="prompt-time">{ev['timestamp_str']} • {ev['engagement']}% Eng.</span>
                    </div>
                    <div class="prompt-body">{ev['prompt']}</div>
                    <div>
                        <span class="prompt-play">🎯 Suggested Play: {ev['recommended_action']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🌟 Classroom engagement is thriving! No critical pedagogical interventions triggered.")

    with col_t4_sim:
        st.subheader("2. Interactive Teaching Simulation Sandbox")
        st.markdown("Test the advisory agent's heuristic triggers under custom simulated conditions:")

        sim_eng = st.slider("Simulated Engagement Index (%):", 10.0, 100.0, 38.0, 1.0)
        sim_dip_mins = st.slider("Duration of Attention Dip (Minutes):", 0.5, 8.0, 3.5, 0.5)
        sim_ac_state = st.selectbox(
            "Classroom Acoustic Mode:",
            [
                "Instructor Lecture",
                "Student Discussion / Verbal Interaction",
                "Silence / Quiet Reflection",
                "High Ambient Classroom Distraction"
            ]
        )
        sim_lecture_mins = st.slider("Continuous Lecture Time (Minutes):", 0.0, 30.0, 14.0, 1.0)
        sim_slump = st.checkbox("Slumping / Downward Posture Detected", value=True)

        advisor = PedagogicalAdvisoryAgent(dip_threshold=45.0)
        sim_prompt = advisor.evaluate_live_state(
            current_engagement=sim_eng,
            recent_dip_duration_sec=sim_dip_mins * 60.0,
            acoustic_state=sim_ac_state,
            continuous_lecture_mins=sim_lecture_mins,
            posture_slump=sim_slump
        )

        st.markdown("---")
        st.markdown("**AI Advisory Output:**")
        if sim_prompt:
            st.error(f"**{sim_prompt['headline']}**\n\n{sim_prompt['prompt']}")
            st.info(f"💡 **Recommended Play:** `{sim_prompt['recommended_action']}`\n\n*Rationale: {sim_prompt['pedagogical_rationale']}*")
        else:
            st.success("✅ Nominal Conditions — Telemetry indicates healthy lesson progression. No prompt required.")

    st.markdown("---")
    st.subheader("3. Actionable Pedagogical Playbook Catalog")

    playbook_tabs = st.tabs(list(PedagogicalAdvisoryAgent.PLAYBOOK.keys()))
    for p_tab, (category_name, activities) in zip(playbook_tabs, PedagogicalAdvisoryAgent.PLAYBOOK.items()):
        with p_tab:
            for act in activities:
                with st.expander(f"📌 {act['title']} ({act['duration']})", expanded=False):
                    st.markdown(f"**Instructions:** {act['description']}")
                    st.markdown(f"**Pedagogical Rationale:** *{act['pedagogical_benefit']}*")


# ==============================================================================
# TAB 5: SESSION ANALYTICS & EDUCATOR REPORT (EPIC 5 & 6)
# ==============================================================================
with tab5:
    st.header("📊 Post-Session Analytics & Educator Report")
    st.markdown("""
    **Comprehensive Pedagogical Debrief:** Generates executive metrics, soundscape distribution, 
    and exportable reports for teacher professional development (PD). **Zero external database dependencies.**
    """)

    # Generate Markdown and HTML reports
    report_md = EducatorReportGenerator.generate_executive_markdown(
        session_name=st.session_state.active_scenario,
        attendance_df=st.session_state.attendance_df,
        fused_df=st.session_state.fused_df,
        fusion_kpis=fusion_kpis,
        advisory_events=st.session_state.advisory_events,
        sustained_threshold=sustained_threshold
    )
    report_html = EducatorReportGenerator.generate_html_report(
        report_md, st.session_state.active_scenario
    )

    # Action buttons
    r_col1, r_col2, r_col3 = st.columns([1, 1, 1])
    with r_col1:
        st.download_button(
            label="📄 Download Briefing Report (Markdown)",
            data=report_md,
            file_name=f"Session_Report_{time.strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with r_col2:
        st.download_button(
            label="🌐 Download Executive Briefing (HTML)",
            data=report_html,
            file_name=f"Session_Briefing_{time.strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
    with r_col3:
        # Full ZIP or Combined CSV download
        combined_csv = pd.concat([
            st.session_state.attendance_df.add_prefix("Att_"),
            st.session_state.fused_df.add_prefix("Fusion_")
        ], axis=1).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📦 Download Complete Session Dataset (CSV)",
            data=combined_csv,
            file_name=f"Full_Session_Dataset_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    # Render Report in UI
    st.markdown(report_md)
