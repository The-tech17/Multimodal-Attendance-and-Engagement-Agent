"""
UI Components & Plotly Data Visualizations
Provides sleek, modern educational styling, glassmorphic metric cards, and responsive interactive charts.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CUSTOM_CSS = """
<style>
    /* Modern Educational Glassmorphism Styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Privacy Banner */
    .privacy-badge-container {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.15));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 20px;
    }
    .privacy-badge-title {
        font-weight: 700;
        color: #059669;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .privacy-badge-desc {
        color: #374151;
        font-size: 0.82rem;
        margin: 0;
    }

    /* KPI Glassmorphic Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.08);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Pedagogical Prompt Cards */
    .prompt-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border-left: 5px solid #3b82f6;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }
    .prompt-card.critical {
        border-left-color: #ef4444;
        background: #fff5f5;
    }
    .prompt-card.action {
        border-left-color: #f59e0b;
        background: #fffdf5;
    }
    .prompt-card.info {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    .prompt-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .prompt-title {
        font-weight: 700;
        font-size: 0.98rem;
        color: #0f172a;
    }
    .prompt-time {
        font-size: 0.75rem;
        background: rgba(15, 23, 42, 0.06);
        padding: 2px 8px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .prompt-body {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .prompt-play {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        color: #2563eb;
        background: #eff6ff;
        padding: 3px 10px;
        border-radius: 6px;
    }

    /* Playbook Card */
    .play-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .play-card-title {
        font-weight: 700;
        color: #1e293b;
        font-size: 0.95rem;
    }
    .play-card-dur {
        font-size: 0.75rem;
        color: #64748b;
        margin-left: 8px;
        font-weight: normal;
    }
</style>
"""


def render_privacy_banner():
    """Renders the FERPA/GDPR edge-first privacy guarantee banner."""
    return """
    <div class="privacy-badge-container">
        <div style="font-size: 1.8rem;">🛡️</div>
        <div>
            <div class="privacy-badge-title">100% Offline Edge Architecture & Ethical Telemetry</div>
            <p class="privacy-badge-desc">
                <strong>Zero Cloud Video Transmission:</strong> All face tracking, head pose estimation, and audio classification run strictly within local device memory. 
                Focus is calibrated on <em>collective pedagogical feedback</em> (modality switches, lesson pacing) rather than punitive individual surveillance.
            </p>
        </div>
    </div>
    """


def render_kpi_card(label: str, value: str, subtext: str, color_hex: str = "#0f172a") -> str:
    """Renders an individual glassmorphic KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color_hex};">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """


def create_engagement_timeline_chart(fused_df: pd.DataFrame, advisory_events: List[Dict[str, Any]] = None) -> go.Figure:
    """
    Renders the unified multimodal engagement curve with colored pedagogical zones
    and annotated AI teaching prompt markers.
    """
    fig = go.Figure()

    if fused_df.empty:
        fig.add_annotation(text="No timeline data available", showarrow=False)
        return fig

    time_mins = fused_df["Time (s)"] / 60.0

    # 1. Background Zone Shading
    # Green Optimal Zone (>70)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0,
                  annotation_text="Optimal Engagement Zone (>70%)", annotation_position="top right",
                  annotation_font_size=10, annotation_font_color="#059669")
    # Yellow Moderate Zone (45-70)
    fig.add_hrect(y0=45, y1=70, fillcolor="rgba(245, 158, 11, 0.06)", line_width=0,
                  annotation_text="Moderate Focus / Reflection (45-70%)", annotation_position="top right",
                  annotation_font_size=10, annotation_font_color="#d97706")
    # Red Flagged Dip Zone (<45)
    fig.add_hrect(y0=0, y1=45, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0,
                  annotation_text="Attention Flag / Fatigue Zone (<45%)", annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color="#dc2626")

    # 2. Component Reference Lines (Visual Attention & Acoustic Resonance)
    if "Visual Attention (%)" in fused_df:
        fig.add_trace(go.Scatter(
            x=time_mins, y=fused_df["Visual Attention (%)"],
            name="Visual Focus (%)",
            line=dict(color="rgba(59, 130, 246, 0.4)", width=1.5, dash="dot"),
            hovertemplate="Time: %{x:.1f}m<br>Visual Focus: %{y:.1f}%<extra></extra>"
        ))

    if "Acoustic Resonance (%)" in fused_df:
        fig.add_trace(go.Scatter(
            x=time_mins, y=fused_df["Acoustic Resonance (%)"],
            name="Acoustic Resonance (%)",
            line=dict(color="rgba(139, 92, 246, 0.4)", width=1.5, dash="dash"),
            hovertemplate="Time: %{x:.1f}m<br>Acoustic Resonance: %{y:.1f}%<extra></extra>"
        ))

    # 3. Main Fused Engagement Curve
    fig.add_trace(go.Scatter(
        x=time_mins, y=fused_df["Engagement Index"],
        name="Fused Engagement Index",
        line=dict(color="#0f172a", width=3.2),
        mode="lines",
        hovertemplate="<b>Session Time:</b> %{x:.1f} mins<br><b>Engagement Index:</b> %{y:.1f}%<extra></extra>"
    ))

    # 4. Highlight Advisory Events / Prompts
    if advisory_events:
        event_times = [ev["timestamp_sec"] / 60.0 for ev in advisory_events]
        event_scores = [ev["engagement"] for ev in advisory_events]
        event_texts = [f"<b>{ev['headline']}</b><br>{ev.get('recommended_action', '')}" for ev in advisory_events]

        fig.add_trace(go.Scatter(
            x=event_times, y=event_scores,
            mode="markers",
            name="AI Teaching Advisory Triggers",
            marker=dict(
                size=12,
                symbol="diamond",
                color="#ef4444",
                line=dict(width=2, color="#ffffff")
            ),
            hovertext=event_texts,
            hoverinfo="text"
        ))

    fig.update_layout(
        title="<b>Real-Time Timeline Engagement Curve & Pedagogical Resonance</b>",
        xaxis_title="Lesson Timeline (Minutes)",
        yaxis_title="Engagement Score (0 - 100)",
        yaxis=dict(range=[0, 105], gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )

    return fig


def create_acoustic_pie_chart(acoustic_df: pd.DataFrame) -> go.Figure:
    """Renders acoustic state percentage distribution as a modern donut chart."""
    if acoustic_df.empty or "Acoustic Class" not in acoustic_df:
        return go.Figure()

    counts = acoustic_df["Acoustic Class"].value_counts()
    labels = list(counts.index)
    values = list(counts.values)

    color_map = {
        "Silence / Quiet Reflection": "#38bdf8",
        "Instructor Lecture": "#10b981",
        "Student Discussion / Verbal Interaction": "#f59e0b",
        "High Ambient Classroom Distraction": "#ef4444"
    }
    colors = [color_map.get(label, "#94a3b8") for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="percent+label",
        hoverinfo="label+value+percent"
    )])

    fig.update_layout(
        title="<b>Classroom Soundscape Breakdown</b>",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_attendance_distribution_chart(attendance_df: pd.DataFrame) -> go.Figure:
    """Renders attendance status distribution (Present vs Tardy vs Hallway Pass-through Filtered)."""
    if attendance_df.empty or "Attendance Status" not in attendance_df:
        return go.Figure()

    status_counts = attendance_df["Attendance Status"].value_counts()
    
    color_map = {
        "Present (Verified)": "#10b981",
        "Partial / Tardy": "#f59e0b",
        "Hallway Pass-Through (Filtered)": "#64748b",
        "Absent": "#ef4444"
    }

    fig = go.Figure(data=[go.Bar(
        x=list(status_counts.index),
        y=list(status_counts.values),
        marker_color=[color_map.get(k, "#3b82f6") for k in status_counts.index],
        text=list(status_counts.values),
        textposition="auto"
    )])

    fig.update_layout(
        title="<b>Sustained Presence Status Distribution</b>",
        xaxis_title="",
        yaxis_title="Participant Count",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#f1f5f9")
    )
    return fig


def create_spectrogram_plot(times: np.ndarray, freqs: np.ndarray, spec_db: np.ndarray) -> go.Figure:
    """Renders interactive 2D spectrogram heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=spec_db,
        x=times,
        y=freqs,
        colorscale="Viridis",
        colorbar=dict(title="dB")
    ))

    fig.update_layout(
        title="<b>Interactive Acoustic Spectrogram (Vocal Formants 0 - 5kHz)</b>",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
