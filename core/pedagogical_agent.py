"""
AI Teaching Suggestion Engine
Epic 5 Heuristic Advisory Agent

Features:
- Non-intrusive, real-time pedagogical prompt generator
- Contextual heuristic triggers:
    1. Collective Attention Dip (Prolonged low engagement)
    2. Monologue Fatigue (Excessive direct instruction without dialogue)
    3. Acoustic Distraction Spike (Environmental noise interference)
    4. Silent Task Stall (Extended quiet with physical slumping)
    5. Peak Momentum Reinforcement (Capitalizing on collaborative flow)
- Pedagogical Playbook Catalog (Micro-Interventions, Active Learning, Energy Resetters)
- Interactive Teacher Scenario Simulator
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


class PedagogicalAdvisoryAgent:
    """
    Heuristic pedagogical agent providing empathetic, non-punitive instructional recommendations.
    """

    SEVERITY_INFO = "💡 Pedagogical Opportunity"
    SEVERITY_ACTION = "⚡ Action Recommended"
    SEVERITY_INTERVENTION = "🚨 Urgent Pedagogical Pivot"

    PLAYBOOK = {
        "Micro-Interventions (< 1 min)": [
            {
                "title": "Fist-to-Five Confidence Check",
                "duration": "30-45 sec",
                "description": "Ask students to hold up fingers (0 to 5) indicating their understanding of the current concept.",
                "pedagogical_benefit": "Immediate tactile feedback without public embarrassment; resets physical posture."
            },
            {
                "title": "60-Second Lightning Write",
                "duration": "1 min",
                "description": "Prompt: 'Write down the single most important word from what we just covered.'",
                "pedagogical_benefit": "Transitions brain from passive listening to active conceptual retrieval."
            },
            {
                "title": "Choral Call & Response",
                "duration": "30 sec",
                "description": "Have the entire class complete a key definition in unison.",
                "pedagogical_benefit": "Resets auditory attention and synchronizes classroom energy."
            }
        ],
        "Active Learning Activities (2-5 mins)": [
            {
                "title": "Think-Pair-Share",
                "duration": "3 mins",
                "description": "1 min silent thought -> 1 min partner exchange -> 1 min whole-room brief.",
                "pedagogical_benefit": "Breaks lecture monologue, guarantees 100% verbal student participation."
            },
            {
                "title": "Stand Up, Hand Up, Pair Up",
                "duration": "3-4 mins",
                "description": "Students stand, raise a hand, high-five a classmate from another table, and answer a targeted prompt.",
                "pedagogical_benefit": "Combines physical mobility with peer-to-peer cognitive processing."
            },
            {
                "title": "Devil's Advocate Challenge",
                "duration": "4 mins",
                "description": "Present a common misconception and ask pairs to find the hidden logical flaw.",
                "pedagogical_benefit": "Spikes analytical engagement and higher-order critical thinking."
            }
        ],
        "Energy Resetters (1-2 mins)": [
            {
                "title": "Kinesthetic Cross-Body Stretch",
                "duration": "90 sec",
                "description": "Lead 3 deep breaths with overhead reach and cross-body shoulder stretch.",
                "pedagogical_benefit": "Oxygenates blood flow, combats afternoon circadian slump and slouching."
            },
            {
                "title": "Eye-Rest 20-20-20 Shift",
                "duration": "60 sec",
                "description": "Instruct students to look out the window at an object 20 feet away for 20 seconds.",
                "pedagogical_benefit": "Relieves screen/notebook visual strain and optical fatigue."
            }
        ],
        "Formative Checks (1-3 mins)": [
            {
                "title": "One-Question Digital / Paper Poll",
                "duration": "2 mins",
                "description": "Display a multiple-choice diagnostic question with plausible distractors.",
                "pedagogical_benefit": "Surfaces misconceptions before moving to subsequent complex topics."
            },
            {
                "title": "Traffic Light Reflection",
                "duration": "90 sec",
                "description": "Students mark green (solid grasp), yellow (some doubts), or red (totally confused).",
                "pedagogical_benefit": "Helps instructor calibrate pacing for the second half of class."
            }
        ]
    }

    def __init__(self, dip_threshold: float = 45.0, dip_window_seconds: float = 120.0):
        self.dip_threshold = dip_threshold
        self.dip_window_seconds = dip_window_seconds

    def evaluate_live_state(self, current_engagement: float, recent_dip_duration_sec: float,
                            acoustic_state: str, continuous_lecture_mins: float,
                            posture_slump: bool = False) -> Optional[Dict[str, Any]]:
        """
        Evaluates current real-time or simulated telemetry and generates a targeted teaching prompt.
        """
        # Trigger 1: Prolonged Attention Dip
        if current_engagement < self.dip_threshold and recent_dip_duration_sec >= self.dip_window_seconds:
            mins_str = round(recent_dip_duration_sec / 60.0, 1)
            return {
                "rule_id": "RULE_PROLONGED_DIP",
                "severity": self.SEVERITY_INTERVENTION,
                "headline": f"Collective Attention Dip Detected ({int(current_engagement)}% for {mins_str}m)",
                "prompt": (f"Collective engagement has stayed below {int(self.dip_threshold)}% for {mins_str} minutes. "
                           "Students are experiencing cognitive saturation. Recommend a rapid modality reset: "
                           "Launch a 60-second peer poll, pose an open-ended puzzle question, or initiate a 90-second stretch break."),
                "recommended_action": "Kinesthetic Cross-Body Stretch or Think-Pair-Share",
                "pedagogical_rationale": "Attention spans naturally reset when physical state or cognitive modality shifts."
            }

        # Trigger 2: Prolonged Monologue Fatigue
        if continuous_lecture_mins >= 12.0 and current_engagement < 62.0:
            return {
                "rule_id": "RULE_PROLONGED_MONOLOGUE",
                "severity": self.SEVERITY_ACTION,
                "headline": f"Prolonged Monologue ({round(continuous_lecture_mins, 1)}m Uninterrupted)",
                "prompt": (f"Direct instruction has continued uninterrupted for {round(continuous_lecture_mins, 1)} minutes. "
                           "Visual focus is beginning to taper. Recommend transitioning to a 2-minute Think-Pair-Share "
                           "or cold-calling a student for conceptual synthesis."),
                "recommended_action": "Think-Pair-Share",
                "pedagogical_rationale": "Direct teacher talk time exceeding 10-12 minutes leads to steep listening decay."
            }

        # Trigger 3: Acoustic Classroom Distraction
        if "Distraction" in acoustic_state:
            return {
                "rule_id": "RULE_ACOUSTIC_DISTRACTION",
                "severity": self.SEVERITY_ACTION,
                "headline": "High Ambient Classroom Distraction Spike",
                "prompt": ("Acoustic sensor detected sharp ambient background noise/clatter. "
                           "Recommend a rhythmic refocusing cue ('If you can hear my voice, clap once') "
                           "or pausing for 5 seconds of intentional silence."),
                "recommended_action": "Choral Call & Response or Silent Pause",
                "pedagogical_rationale": "Sudden environmental noise fractures auditory attention; rhythmic cues re-anchor focus."
            }

        # Trigger 4: Silent Task Stall
        if "Silence" in acoustic_state and posture_slump:
            return {
                "rule_id": "RULE_SILENT_STALL",
                "severity": self.SEVERITY_ACTION,
                "headline": "Silent Independent Work Stalling",
                "prompt": ("Extended quiet study accompanied by physical slumping and downward head tilt. "
                           "Learners may be stuck on current exercise. Recommend a whole-class checkpoint or releasing a hint."),
                "recommended_action": "Fist-to-Five Confidence Check",
                "pedagogical_rationale": "Passive struggle without intervention leads to disengagement."
            }

        # Trigger 5: Peak Collaborative Momentum
        if current_engagement >= 82.0 and "Discussion" in acoustic_state:
            return {
                "rule_id": "RULE_PEAK_MOMENTUM",
                "severity": self.SEVERITY_INFO,
                "headline": f"Peak Collaborative Flow ({int(current_engagement)}% Engagement)",
                "prompt": ("High collaborative energy and active dialogue detected across the room! Great momentum. "
                           "Prepare a whiteboard consolidation wrap-up to crystallize key takeaways before moving on."),
                "recommended_action": "Whiteboard Synthesis Wrap-up",
                "pedagogical_rationale": "Anchoring discoveries right after active peer discourse solidifies retention."
            }

        return None

    def scan_timeline(self, fused_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scans a complete session timeline to produce an annotated educator event log.
        """
        if fused_df.empty:
            return []

        events = []
        lecture_run_start = None
        dip_run_start = None

        for idx, row in fused_df.iterrows():
            t = row["Time (s)"]
            eng = row["Engagement Index"]
            ac_state = row["Acoustic State"]
            is_slump = (row.get("Posture Stability (%)", 80.0) < 55.0)

            # Track lecture runs
            if "Lecture" in ac_state:
                if lecture_run_start is None:
                    lecture_run_start = t
            else:
                lecture_run_start = None

            lecture_duration_mins = (t - lecture_run_start) / 60.0 if lecture_run_start is not None else 0.0

            # Track dip runs
            if eng < self.dip_threshold:
                if dip_run_start is None:
                    dip_run_start = t
            else:
                dip_run_start = None

            dip_duration_sec = (t - dip_run_start) if dip_run_start is not None else 0.0

            # Fire evaluation
            prompt_data = self.evaluate_live_state(
                current_engagement=eng,
                recent_dip_duration_sec=dip_duration_sec,
                acoustic_state=ac_state,
                continuous_lecture_mins=lecture_duration_mins,
                posture_slump=is_slump
            )

            if prompt_data:
                # Deduplicate: only add if at least 90s since previous same event
                last_event_time = events[-1]["timestamp_sec"] if events else -999.0
                if (t - last_event_time) >= 90.0:
                    events.append({
                        "timestamp_sec": round(t, 1),
                        "timestamp_str": f"{int(t // 60):02d}m {int(t % 60):02d}s",
                        "engagement": eng,
                        **prompt_data
                    })

        return events
