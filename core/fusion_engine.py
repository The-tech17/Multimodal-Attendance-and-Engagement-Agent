"""
Multimodal Fusion & Engagement Scoring Engine
Epic 4 Multimodal Fusion

Fuses:
1. Visual Attentiveness Indicators (Forward head pose, direct eye contact)
2. Posture Dynamics (Upright posture vs slumping/drowsiness)
3. Acoustic Interaction Resonance (Lecture vs Collaborative Discussion vs Distraction)

Produces:
- Synchronized unified time-step matrix
- Normalized Engagement Index (0 - 100)
- Timeline metrics with peak/dip event detection and pedagogical zone flags
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class MultimodalFusionEngine:
    """
    Fuses edge visual and acoustic telemetry into an ethical, collective Engagement Index.
    """

    ZONE_OPTIMAL = "High Engagement (Optimal)"
    ZONE_MODERATE = "Moderate / Reflective"
    ZONE_FLAGGED = "Attention Flag / Low Focus"

    def __init__(self, weight_visual: float = 0.50, weight_posture: float = 0.20, weight_acoustic: float = 0.30):
        # Normalize weights so sum == 1.0
        total_w = weight_visual + weight_posture + weight_acoustic
        if total_w <= 0:
            total_w = 1.0
        self.w_visual = weight_visual / total_w
        self.w_posture = weight_posture / total_w
        self.w_acoustic = weight_acoustic / total_w

    def map_acoustic_to_resonance(self, acoustic_class: str, rms_db: float) -> float:
        """
        Maps acoustic classification to pedagogical interaction score (0 - 100).
        """
        if "Discussion" in acoustic_class:
            # Active collaborative discourse
            return min(95.0, 80.0 + max(0.0, (rms_db + 40.0) * 0.5))
        elif "Lecture" in acoustic_class:
            # Attentive listening to instructor
            return 75.0
        elif "Silence" in acoustic_class:
            # Quiet study, contemplation, or independent problem solving
            return 68.0
        elif "Distraction" in acoustic_class:
            # Environmental disruption / loud noise
            return 25.0
        else:
            return 60.0

    def fuse_streams(self, visual_timeline: pd.DataFrame, acoustic_timeline: pd.DataFrame) -> pd.DataFrame:
        """
        Merges visual attention time-series and acoustic timeline on time-step.
        Computes normalized Engagement Index (0-100) and pedagogical zone categorization.
        """
        # Ensure we have common time index
        if visual_timeline.empty and acoustic_timeline.empty:
            return pd.DataFrame(columns=[
                "Time (s)", "Visual Attention (%)", "Posture Stability (%)",
                "Acoustic State", "Acoustic Resonance (%)", "Raw Engagement",
                "Engagement Index", "Engagement Zone", "Flagged Dip"
            ])

        # If visual or acoustic is missing, generate harmonized time base
        if visual_timeline.empty:
            time_steps = acoustic_timeline["Time (s)"].values
            v_att = np.full(len(time_steps), 75.0)
            p_stab = np.full(len(time_steps), 85.0)
            vis_df = pd.DataFrame({"Time (s)": time_steps, "Visual Attention (%)": v_att, "Posture Stability (%)": p_stab})
        else:
            vis_df = visual_timeline.copy()

        if acoustic_timeline.empty:
            time_steps = vis_df["Time (s)"].values
            ac_df = pd.DataFrame({
                "Time (s)": time_steps,
                "Acoustic Class": ["Instructor Lecture"] * len(time_steps),
                "RMS Energy (dB)": [-25.0] * len(time_steps)
            })
        else:
            ac_df = acoustic_timeline.copy()

        # Merge on nearest time
        merged = pd.merge_asof(vis_df.sort_values("Time (s)"),
                               ac_df.sort_values("Time (s)"),
                               on="Time (s)", direction="nearest")

        # Compute components
        visual_scores = merged.get("Visual Attention (%)", pd.Series(75.0, index=merged.index)).clip(0, 100)
        posture_scores = merged.get("Posture Stability (%)", pd.Series(85.0, index=merged.index)).clip(0, 100)

        # Map acoustic resonance
        acoustic_classes = merged.get("Acoustic Class", pd.Series(["Instructor Lecture"] * len(merged)))
        rms_values = merged.get("RMS Energy (dB)", pd.Series([-25.0] * len(merged)))
        
        acoustic_resonance = [
            self.map_acoustic_to_resonance(cls_name, rms)
            for cls_name, rms in zip(acoustic_classes, rms_values)
        ]
        acoustic_resonance = pd.Series(acoustic_resonance, index=merged.index)

        # Fused Engagement Calculation: Weighted Combination
        raw_engagement = (self.w_visual * visual_scores +
                          self.w_posture * posture_scores +
                          self.w_acoustic * acoustic_resonance)

        # Apply moving average filter (window of 3) for smooth, non-jittery pedagogical trends
        smooth_engagement = raw_engagement.rolling(window=3, min_periods=1, center=True).mean()
        smooth_engagement = smooth_engagement.clip(0, 100).round(1)

        # Pedagogical Zone classification
        zones = []
        flagged_dips = []
        for val in smooth_engagement:
            if val >= 70.0:
                zones.append(self.ZONE_OPTIMAL)
                flagged_dips.append(False)
            elif val >= 45.0:
                zones.append(self.ZONE_MODERATE)
                flagged_dips.append(False)
            else:
                zones.append(self.ZONE_FLAGGED)
                flagged_dips.append(True)

        fused_df = pd.DataFrame({
            "Time (s)": merged["Time (s)"].round(1),
            "Visual Attention (%)": visual_scores.round(1),
            "Posture Stability (%)": posture_scores.round(1),
            "Acoustic State": acoustic_classes,
            "Acoustic Resonance (%)": acoustic_resonance.round(1),
            "Raw Engagement": raw_engagement.round(1),
            "Engagement Index": smooth_engagement,
            "Engagement Zone": zones,
            "Flagged Dip": flagged_dips
        })

        return fused_df

    @staticmethod
    def extract_summary_kpis(fused_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts high-level statistical summary and attention dip metrics.
        """
        if fused_df.empty:
            return {
                "avg_engagement": 0.0,
                "peak_engagement": 0.0,
                "peak_time_sec": 0.0,
                "lowest_engagement": 0.0,
                "lowest_time_sec": 0.0,
                "dip_count": 0,
                "dip_duration_sec": 0.0,
                "optimal_time_pct": 0.0,
                "attention_stability_index": 100.0
            }

        scores = fused_df["Engagement Index"]
        avg_eng = float(scores.mean())
        peak_idx = int(scores.idxmax())
        min_idx = int(scores.idxmin())

        peak_val = float(scores.loc[peak_idx])
        peak_time = float(fused_df.loc[peak_idx, "Time (s)"])
        lowest_val = float(scores.loc[min_idx])
        lowest_time = float(fused_df.loc[min_idx, "Time (s)"])

        # Count consecutive dip episodes
        is_dip = fused_df["Flagged Dip"].values
        dip_count = 0
        in_dip = False
        dip_steps = 0

        for flag in is_dip:
            if flag:
                dip_steps += 1
                if not in_dip:
                    dip_count += 1
                    in_dip = True
            else:
                in_dip = False

        step_duration = 1.0
        if len(fused_df) > 1:
            step_duration = float(np.mean(np.diff(fused_df["Time (s)"])))
        total_dip_sec = dip_steps * step_duration

        total_steps = len(fused_df)
        optimal_steps = sum(fused_df["Engagement Zone"] == MultimodalFusionEngine.ZONE_OPTIMAL)
        optimal_pct = (optimal_steps / total_steps) * 100.0 if total_steps > 0 else 0.0

        # Stability index: inverse of standard deviation
        score_std = float(scores.std()) if len(scores) > 1 else 0.0
        stability_index = max(10.0, min(100.0, 100.0 - score_std * 2.2))

        return {
            "avg_engagement": round(avg_eng, 1),
            "peak_engagement": round(peak_val, 1),
            "peak_time_sec": round(peak_time, 1),
            "lowest_engagement": round(lowest_val, 1),
            "lowest_time_sec": round(lowest_time, 1),
            "dip_count": dip_count,
            "dip_duration_sec": round(total_dip_sec, 1),
            "optimal_time_pct": round(optimal_pct, 1),
            "attention_stability_index": round(stability_index, 1)
        }
