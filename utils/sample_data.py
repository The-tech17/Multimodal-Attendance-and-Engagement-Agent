"""
Pre-packaged Realistic Classroom Datasets & Synthetic Scenario Generators
Provides out-of-the-box rich pedagogical test sessions for 1-click exploration.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


class SampleDataGenerator:
    """
    Generates rich, pedagogically realistic classroom datasets without external dependencies.
    """

    SCENARIOS = {
        "Scenario 1: Interactive STEM Lecture & Problem Solving": {
            "description": "Balanced 45-minute lesson with direct instruction, 2 lively Q&A breakouts, and a brief silent review.",
            "duration_mins": 45,
            "total_students": 24,
            "pass_through_count": 2,
            "lecture_pct": 50,
            "discussion_pct": 35,
            "silence_pct": 12,
            "distraction_pct": 3,
            "base_engagement": 78
        },
        "Scenario 2: Collaborative Project Studio (High Interaction)": {
            "description": "Hands-on team workshop with dominant peer discussions, constant movement, and sustained high engagement.",
            "duration_mins": 40,
            "total_students": 28,
            "pass_through_count": 3,
            "lecture_pct": 15,
            "discussion_pct": 72,
            "silence_pct": 8,
            "distraction_pct": 5,
            "base_engagement": 86
        },
        "Scenario 3: Afternoon Monologue & Fatigue Slump": {
            "description": "Prolonged teacher lecture where passive listening leads to a visible mid-session attention drop around minute 20.",
            "duration_mins": 50,
            "total_students": 22,
            "pass_through_count": 1,
            "lecture_pct": 80,
            "discussion_pct": 10,
            "silence_pct": 7,
            "distraction_pct": 3,
            "base_engagement": 58
        },
        "Scenario 4: Exam & Silent Individual Assessment": {
            "description": "Quiet diagnostic quiz period. Sustained low acoustic levels, high presence, and focused upright postures.",
            "duration_mins": 30,
            "total_students": 25,
            "pass_through_count": 2,
            "lecture_pct": 8,
            "discussion_pct": 4,
            "silence_pct": 86,
            "distraction_pct": 2,
            "base_engagement": 72
        }
    }

    @classmethod
    def generate_scenario_attendance(cls, scenario_key: str, sustained_threshold: float = 0.70) -> Tuple[pd.DataFrame, Dict[int, List[Dict[str, Any]]]]:
        """
        Generates realistic participant tracking history including true students and transitory hallway passers-by.
        """
        config = cls.SCENARIOS.get(scenario_key, cls.SCENARIOS["Scenario 1: Interactive STEM Lecture & Problem Solving"])
        total_learners = config["total_students"]
        pass_throughs = config["pass_through_count"]
        session_duration_sec = config["duration_mins"] * 60.0

        records = []
        tracking_history = {}
        np.random.seed(42 + hash(scenario_key) % 1000)

        obj_id = 1

        # 1. Generate Verified Active Students
        for i in range(total_learners):
            # Arrival time: mostly between 0 and 180s into class (some arrive slightly late)
            first_seen = np.random.uniform(10.0, 180.0)
            
            # Active duration: either full session (>85%) or occasional early/partial departure
            is_full = np.random.rand() > 0.12
            if is_full:
                last_seen = session_duration_sec - np.random.uniform(5.0, 90.0)
                active_sec = last_seen - first_seen
                presence_pct = min(100.0, (active_sec / session_duration_sec) * 100.0)
            else:
                # Partial attendee (e.g. left early or arrived 20 mins late)
                active_sec = session_duration_sec * np.random.uniform(0.45, 0.68)
                last_seen = first_seen + active_sec
                presence_pct = (active_sec / session_duration_sec) * 100.0

            # Attention score
            base_att = config["base_engagement"] + np.random.normal(0, 7)
            avg_att = max(40.0, min(96.0, base_att))

            # Status determination based on sustained presence (>70%)
            threshold_pct = sustained_threshold * 100.0
            if presence_pct >= threshold_pct:
                status = "Present (Verified)"
                badge = "✅ Present"
                is_eligible = True
            else:
                status = "Partial / Tardy"
                badge = "⚠️ Partial Attendance"
                is_eligible = False

            mins = int(active_sec // 60)
            secs = int(active_sec % 60)
            dur_str = f"{mins:02d}m {secs:02d}s"

            records.append({
                "Participant ID": f"Learner-{obj_id:03d}",
                "First Seen": time.strftime("%H:%M:%S", time.gmtime(first_seen)),
                "Last Seen": time.strftime("%H:%M:%S", time.gmtime(last_seen)),
                "Active Duration": dur_str,
                "Raw Duration (s)": round(active_sec, 1),
                "Presence Rate (%)": round(presence_pct, 1),
                "Avg Attention (%)": round(avg_att, 1),
                "Attendance Status": status,
                "Status Badge": badge,
                "Eligible Attendance": is_eligible
            })

            # Create sample tracking entries for visualization
            num_samples = int(min(60, active_sec / 30))
            track_points = []
            for s in range(num_samples):
                t_sample = first_seen + (s / max(1, num_samples)) * active_sec
                track_points.append({
                    "timestamp": t_sample,
                    "attentiveness_score": round(max(30.0, min(98.0, avg_att + np.random.normal(0, 10))), 1),
                    "head_pose": "Forward (Engaged)" if np.random.rand() > 0.25 else "Looking Away (Distracted)",
                    "posture": "Upright" if np.random.rand() > 0.15 else "Slumped"
                })
            tracking_history[obj_id] = track_points
            obj_id += 1

        # 2. Generate Transitory Hallway Pass-Throughs (Crucial test for Sustained Presence Logic!)
        for p in range(pass_throughs):
            # Only visible for 10 - 25 seconds walking past door/webcam
            entry_time = np.random.uniform(300.0, session_duration_sec - 300.0)
            pass_duration = np.random.uniform(8.0, 24.0)
            exit_time = entry_time + pass_duration
            presence_pct = (pass_duration / session_duration_sec) * 100.0

            dur_str = f"00m {int(pass_duration):02d}s"

            records.append({
                "Participant ID": f"Learner-{obj_id:03d} (Hallway)",
                "First Seen": time.strftime("%H:%M:%S", time.gmtime(entry_time)),
                "Last Seen": time.strftime("%H:%M:%S", time.gmtime(exit_time)),
                "Active Duration": dur_str,
                "Raw Duration (s)": round(pass_duration, 1),
                "Presence Rate (%)": round(presence_pct, 1),
                "Avg Attention (%)": round(np.random.uniform(30.0, 55.0), 1),
                "Attendance Status": "Hallway Pass-Through (Filtered)",
                "Status Badge": "🚪 Filtered (Pass-through)",
                "Eligible Attendance": False
            })

            # Add short track
            tracking_history[obj_id] = [
                {"timestamp": entry_time, "attentiveness_score": 45.0, "head_pose": "Looking Away", "posture": "Walking"},
                {"timestamp": exit_time, "attentiveness_score": 40.0, "head_pose": "Looking Away", "posture": "Walking"}
            ]
            obj_id += 1

        df = pd.DataFrame(records).sort_values(by="Presence Rate (%)", ascending=False).reset_index(drop=True)
        return df, tracking_history

    @classmethod
    def generate_scenario_multimodal_timeline(cls, scenario_key: str, step_sec: float = 30.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates synchronized visual attention and acoustic time series matching the pedagogical narrative.
        """
        config = cls.SCENARIOS.get(scenario_key, cls.SCENARIOS["Scenario 1: Interactive STEM Lecture & Problem Solving"])
        total_duration_sec = config["duration_mins"] * 60.0
        n_steps = int(total_duration_sec / step_sec)
        time_axis = np.arange(0, n_steps) * step_sec

        np.random.seed(101 + hash(scenario_key) % 1000)

        visual_records = []
        acoustic_records = []

        base_eng = config["base_engagement"]

        for idx, t in enumerate(time_axis):
            progress = t / total_duration_sec

            # Create scenario-specific dynamics
            if scenario_key == "Scenario 1: Interactive STEM Lecture & Problem Solving":
                # 0-35% Lecture, 35-55% Q&A/Discussion, 55-80% Group Activity, 80-100% Summary
                if progress < 0.35:
                    ac_state = "Instructor Lecture"
                    rms_db = -24.0 + np.random.normal(0, 2)
                    centroid = 1200 + np.random.normal(0, 100)
                    v_att = max(60, min(95, 82 + np.random.normal(0, 4) - progress * 20))
                    p_stab = max(70, min(98, 88 + np.random.normal(0, 3)))
                elif progress < 0.55:
                    ac_state = "Student Discussion / Verbal Interaction"
                    rms_db = -20.0 + np.random.normal(0, 3)
                    centroid = 2400 + np.random.normal(0, 200)
                    v_att = max(70, min(98, 88 + np.random.normal(0, 4)))
                    p_stab = max(75, min(95, 90 + np.random.normal(0, 3)))
                elif progress < 0.80:
                    ac_state = "Student Discussion / Verbal Interaction"
                    rms_db = -22.0 + np.random.normal(0, 2)
                    centroid = 2100 + np.random.normal(0, 150)
                    v_att = max(72, min(96, 85 + np.random.normal(0, 5)))
                    p_stab = max(75, min(95, 86 + np.random.normal(0, 4)))
                else:
                    ac_state = "Silence / Quiet Reflection"
                    rms_db = -46.0 + np.random.normal(0, 2)
                    centroid = 600 + np.random.normal(0, 50)
                    v_att = max(65, min(90, 76 + np.random.normal(0, 5)))
                    p_stab = max(70, min(95, 84 + np.random.normal(0, 3)))

            elif scenario_key == "Scenario 2: Collaborative Project Studio (High Interaction)":
                # High energy, frequent laughter, discussion dominates
                ac_state = "Student Discussion / Verbal Interaction" if np.random.rand() > 0.18 else "Instructor Lecture"
                rms_db = -19.0 + np.random.normal(0, 3)
                centroid = 2500 + np.random.normal(0, 250)
                v_att = max(75, min(99, 90 + np.random.normal(0, 4)))
                p_stab = max(70, min(95, 88 + np.random.normal(0, 4)))

            elif scenario_key == "Scenario 3: Afternoon Monologue & Fatigue Slump":
                # Continuous lecture. After 35% time, sharp drop down to 38-42% attention!
                ac_state = "Instructor Lecture"
                rms_db = -25.0 + np.random.normal(0, 1.5)
                centroid = 1150 + np.random.normal(0, 80)
                if 0.35 <= progress <= 0.65:
                    # The Afternoon Slump! Triggers AI Advisory Prompt!
                    v_att = max(25, min(48, 38 + np.random.normal(0, 4)))
                    p_stab = max(35, min(55, 45 + np.random.normal(0, 5)))
                else:
                    v_att = max(50, min(75, 62 + np.random.normal(0, 5)))
                    p_stab = max(60, min(80, 70 + np.random.normal(0, 4)))

            else:  # Scenario 4: Exam & Silent Individual Assessment
                ac_state = "Silence / Quiet Reflection" if np.random.rand() > 0.08 else "High Ambient Classroom Distraction"
                rms_db = -48.0 if "Silence" in ac_state else -18.0
                centroid = 550 + np.random.normal(0, 50)
                v_att = max(65, min(92, 78 + np.random.normal(0, 4)))
                p_stab = max(70, min(96, 88 + np.random.normal(0, 3)))

            # Random brief noise spike in 1% of frames
            if np.random.rand() < 0.02 and ac_state != "Silence / Quiet Reflection":
                ac_state = "High Ambient Classroom Distraction"
                rms_db = -12.0

            visual_records.append({
                "Time (s)": round(t, 1),
                "Visual Attention (%)": round(v_att, 1),
                "Posture Stability (%)": round(p_stab, 1),
                "Active Faces Detected": max(5, int(config["total_students"] * (0.85 + np.random.normal(0, 0.05))))
            })

            acoustic_records.append({
                "Time (s)": round(t, 1),
                "RMS Energy (dB)": round(rms_db, 1),
                "ZCR": round(0.12 + np.random.normal(0, 0.03), 3),
                "Spectral Centroid (Hz)": round(centroid, 1),
                "Spectral Rolloff (Hz)": round(centroid * 1.8, 1),
                "VAD Active": (rms_db > -40.0),
                "Acoustic Class": ac_state,
                "Confidence": round(0.85 + np.random.uniform(0, 0.1), 2),
                "Energy Index": round(min(100.0, max(0.0, (rms_db + 60.0) * 1.66)), 1)
            })

        vis_df = pd.DataFrame(visual_records)
        ac_df = pd.DataFrame(acoustic_records)
        return vis_df, ac_df
