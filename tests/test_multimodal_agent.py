"""
Automated Comprehensive Test Suite for Multimodal Attendance & Engagement Agent
Tests all 5 Epics and Core Modules.
"""

import os
import unittest
import numpy as np
import pandas as pd

from core.privacy import PrivacyEngine
from core.vision_tracker import VisionAttendanceTracker, CentroidTracker
from core.audio_analyzer import AudioAcousticAnalyzer
from core.fusion_engine import MultimodalFusionEngine
from core.pedagogical_agent import PedagogicalAdvisoryAgent
from core.report_generator import EducatorReportGenerator
from utils.sample_data import SampleDataGenerator


class TestMultimodalAgent(unittest.TestCase):

    def setUp(self):
        self.privacy = PrivacyEngine(anonymize_participants=True)
        self.vision = VisionAttendanceTracker()
        self.tracker = CentroidTracker()
        self.audio = AudioAcousticAnalyzer()
        self.fusion = MultimodalFusionEngine(0.5, 0.2, 0.3)
        self.advisor = PedagogicalAdvisoryAgent(dip_threshold=45.0)

    def test_privacy_principles(self):
        """1. Privacy-First System Principles: offline edge, hashing, zero persistence."""
        manifest = self.privacy.get_privacy_manifest()
        self.assertEqual(manifest["cloud_transmission"], "DISABLED (0 bytes transmitted)")
        
        # Test anonymization
        raw_id = "Student_007"
        anon_id = self.privacy.anonymize_id(raw_id)
        self.assertTrue(anon_id.startswith("Learner-"))
        self.assertNotEqual(raw_id, anon_id)

        # Test buffer flush
        flush_res = self.privacy.flush_memory_buffers()
        self.assertEqual(flush_res["status"], "success")

    def test_sustained_presence_attendance(self):
        """2. Tab 1 - Automated Presence-Based Attendance & Hallway Pass-Through Filtering."""
        # Create a mock tracking history
        # Student 1: 50 minutes out of 60 minutes (>80% -> Present)
        # Student 2: 15 seconds (<40s -> Hallway Pass-through Filtered)
        history = {
            1: [{"timestamp": t, "attentiveness_score": 85.0} for t in range(0, 3000, 60)],
            2: [{"timestamp": 500, "attentiveness_score": 50.0}, {"timestamp": 515, "attentiveness_score": 50.0}]
        }
        att_df = VisionAttendanceTracker.calculate_sustained_attendance(
            history, total_session_seconds=3600.0, sustained_threshold=0.70
        )
        self.assertEqual(len(att_df), 2)
        
        # Student 1 must be Present
        stu1 = att_df[att_df["Participant ID"].str.contains("001")].iloc[0]
        self.assertEqual(stu1["Attendance Status"], "Present (Verified)")
        self.assertTrue(stu1["Eligible Attendance"])

        # Student 2 must be Hallway Pass-Through (Filtered)
        stu2 = att_df[att_df["Participant ID"].str.contains("002")].iloc[0]
        self.assertEqual(stu2["Attendance Status"], "Hallway Pass-Through (Filtered)")
        self.assertFalse(stu2["Eligible Attendance"])

    def test_acoustic_interaction_engine(self):
        """3. Tab 2 - Acoustic Interaction & Noise Dynamics (VAD, Spectral, 4-Classes)."""
        syn_audio = AudioAcousticAnalyzer.generate_synthetic_classroom_audio(15, "Standard Active Lecture")
        self.assertEqual(len(syn_audio), 15 * 16000)

        ac_df = self.audio.analyze_stream(syn_audio)
        self.assertFalse(ac_df.empty)
        self.assertIn("Acoustic Class", ac_df.columns)
        self.assertIn("RMS Energy (dB)", ac_df.columns)
        self.assertIn("VAD Active", ac_df.columns)
        
        # Verify 4 classes are supported
        valid_classes = {
            AudioAcousticAnalyzer.CLASS_SILENCE,
            AudioAcousticAnalyzer.CLASS_LECTURE,
            AudioAcousticAnalyzer.CLASS_DISCUSSION,
            AudioAcousticAnalyzer.CLASS_DISTRACTION
        }
        found_classes = set(ac_df["Acoustic Class"].unique())
        self.assertTrue(found_classes.issubset(valid_classes))

    def test_multimodal_fusion_and_scoring(self):
        """4. Tab 3 - Multimodal Fusion & Engagement Scoring (0-100 index)."""
        vis_df, ac_df = SampleDataGenerator.generate_scenario_multimodal_timeline("Scenario 1: Interactive STEM Lecture & Problem Solving")
        fused_df = self.fusion.fuse_streams(vis_df, ac_df)

        self.assertFalse(fused_df.empty)
        self.assertIn("Engagement Index", fused_df.columns)
        self.assertIn("Engagement Zone", fused_df.columns)

        # Check engagement index bounds
        scores = fused_df["Engagement Index"]
        self.assertTrue((scores >= 0).all() and (scores <= 100).all())

        kpis = MultimodalFusionEngine.extract_summary_kpis(fused_df)
        self.assertIn("avg_engagement", kpis)
        self.assertIn("optimal_time_pct", kpis)

    def test_pedagogical_advisory_agent(self):
        """5. Tab 4 - AI Teaching Suggestion Engine heuristics."""
        # Trigger prolonged dip (<45% for 3 mins)
        prompt = self.advisor.evaluate_live_state(
            current_engagement=38.0,
            recent_dip_duration_sec=200.0,
            acoustic_state="Instructor Lecture",
            continuous_lecture_mins=8.0
        )
        self.assertIsNotNone(prompt)
        self.assertEqual(prompt["rule_id"], "RULE_PROLONGED_DIP")
        self.assertIn("Collective Attention Dip", prompt["headline"])

        # Trigger monologue fatigue (>12 mins lecture)
        prompt_mono = self.advisor.evaluate_live_state(
            current_engagement=58.0,
            recent_dip_duration_sec=0.0,
            acoustic_state="Instructor Lecture",
            continuous_lecture_mins=15.0
        )
        self.assertIsNotNone(prompt_mono)
        self.assertEqual(prompt_mono["rule_id"], "RULE_PROLONGED_MONOLOGUE")

    def test_educator_report_generator(self):
        """6. Tab 5 - Session Analytics & Educator Report."""
        att_df, _ = SampleDataGenerator.generate_scenario_attendance("Scenario 1: Interactive STEM Lecture & Problem Solving")
        vis_df, ac_df = SampleDataGenerator.generate_scenario_multimodal_timeline("Scenario 1: Interactive STEM Lecture & Problem Solving")
        fused_df = self.fusion.fuse_streams(vis_df, ac_df)
        kpis = MultimodalFusionEngine.extract_summary_kpis(fused_df)
        events = self.advisor.scan_timeline(fused_df)

        report_md = EducatorReportGenerator.generate_executive_markdown(
            "Test Session", att_df, fused_df, kpis, events
        )
        self.assertIn("# 🎓 Classroom Pedagogical & Attendance Briefing", report_md)
        self.assertIn("Average Engagement Index", report_md)

        report_html = EducatorReportGenerator.generate_html_report(report_md, "Test Session")
        self.assertIn("<!DOCTYPE html>", report_html)


if __name__ == "__main__":
    unittest.main()
