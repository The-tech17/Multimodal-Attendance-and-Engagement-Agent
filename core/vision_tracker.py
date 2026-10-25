"""
Automated Presence-Based Attendance & Visual Attention Tracker
Epic 2 & Epic 4 Computer Vision Pipeline

Features:
- Dual-Engine Face Detection (MediaPipe Task API with OpenCV Haar Cascade fallback)
- Real-time Centroid & Spatial Tracking with persistent Track IDs
- Sustained Presence Attendance Logic (>70% session duration threshold)
- Hallway Pass-through Filtering (prevents transitory visitors from getting false attendance marks)
- Head Pose & Posture Slump Evaluation (Forward Attention vs Distraction/Drowsiness)
"""

import os
import math
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
import pandas as pd


class CentroidTracker:
    """
    Associates bounding boxes across frames to maintain persistent participant IDs.
    """
    def __init__(self, max_disappeared: int = 30, distance_threshold: float = 80.0):
        self.next_object_id = 1
        self.objects: Dict[int, Tuple[int, int]] = {}  # id -> (cX, cY)
        self.bboxes: Dict[int, Tuple[int, int, int, int]] = {}  # id -> (x, y, w, h)
        self.disappeared: Dict[int, int] = {}
        self.history: Dict[int, List[Dict[str, Any]]] = {}
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], timestamp: float, metadata: Dict[str, Any]):
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.history[obj_id] = [{
            "timestamp": timestamp,
            "centroid": centroid,
            "bbox": bbox,
            **metadata
        }]
        self.next_object_id += 1
        return obj_id

    def deregister(self, object_id: int):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]], timestamp: float, metadatas: List[Dict[str, Any]] = None) -> Dict[int, Tuple[int, int, int, int]]:
        if metadatas is None:
            metadatas = [{} for _ in rects]

        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.bboxes

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(tuple(input_centroids[i]), rects[i], timestamp, metadatas[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute Euclidean distances between existing and input centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            assigned_rows, assigned_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in assigned_rows or col in assigned_cols:
                    continue
                if D[row, col] > self.distance_threshold:
                    continue

                object_id = object_ids[row]
                centroid = tuple(input_centroids[col])
                bbox = rects[col]
                self.objects[object_id] = centroid
                self.bboxes[object_id] = bbox
                self.disappeared[object_id] = 0
                self.history[object_id].append({
                    "timestamp": timestamp,
                    "centroid": centroid,
                    "bbox": bbox,
                    **metadatas[col]
                })

                assigned_rows.add(row)
                assigned_cols.add(col)

            unused_rows = set(range(len(object_centroids))) - assigned_rows
            unused_cols = set(range(len(input_centroids))) - assigned_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(tuple(input_centroids[col]), rects[col], timestamp, metadatas[col])

        return self.bboxes


class VisionAttendanceTracker:
    """
    Multimodal presence and attentiveness tracker operating 100% on the local edge.
    """
    def __init__(self, model_asset_path: str = "assets/blaze_face_short_range.tflite",
                 haar_cascade_path: str = "assets/haarcascade_frontalface_default.xml",
                 eye_cascade_path: str = "assets/haarcascade_eye.xml"):
        self.model_asset_path = model_asset_path
        self.haar_cascade_path = haar_cascade_path
        self.eye_cascade_path = eye_cascade_path
        self.mp_detector = None
        self.face_cascade = None
        self.eye_cascade = None
        self.detector_type = "None"
        self._init_detectors()

    def _init_detectors(self):
        # 1. Attempt MediaPipe Face Detector initialization
        if os.path.exists(self.model_asset_path):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
                options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.45)
                self.mp_detector = vision.FaceDetector.create_from_options(options)
                self.detector_type = "MediaPipe BlazeFace (Edge TFLite)"
                return
            except Exception:
                self.mp_detector = None

        # 2. Fallback to OpenCV Haar Cascade
        if os.path.exists(self.haar_cascade_path):
            try:
                self.face_cascade = cv2.CascadeClassifier(self.haar_cascade_path)
                if os.path.exists(self.eye_cascade_path):
                    self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade_path)
                self.detector_type = "OpenCV Haar Cascade (Offline Edge)"
                return
            except Exception:
                self.face_cascade = None

        self.detector_type = "Heuristic Geometric Simulator"

    def detect_and_estimate_attentiveness(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in frame and infers head orientation & posture attentiveness.
        Returns list of detection dicts: bbox, centroid, attentiveness_score (0-100),
        head_pose ('Forward', 'Looking Away', 'Slumped'), confidence.
        """
        h, w = frame_bgr.shape[:2]
        detections = []

        # Strategy A: MediaPipe Detector
        if self.mp_detector is not None:
            try:
                import mediapipe as mp
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = self.mp_detector.detect(mp_img)

                for det in results.detections:
                    box = det.bounding_box
                    bx = max(0, int(box.origin_x))
                    by = max(0, int(box.origin_y))
                    bw = min(w - bx, int(box.width))
                    bh = min(h - by, int(box.height))
                    conf = float(det.categories[0].score) if det.categories else 0.85

                    # Keypoints: 0=right eye, 1=left eye, 2=nose tip, 3=mouth, 4=right ear, 5=left ear
                    attentiveness_score = 80.0
                    head_pose = "Forward (Engaged)"
                    posture = "Upright"

                    if len(det.keypoints) >= 4:
                        re = det.keypoints[0]  # right eye (normalized x,y)
                        le = det.keypoints[1]  # left eye
                        nose = det.keypoints[2]
                        
                        # Yaw estimation via eye-to-nose symmetry
                        eye_dist = abs(le.x - re.x) + 1e-6
                        nose_offset = (nose.x - min(re.x, le.x)) / eye_dist
                        yaw_deviation = abs(nose_offset - 0.5)

                        # Pitch estimation via vertical eye-to-nose vs nose-to-mouth
                        mouth = det.keypoints[3]
                        eye_y = (re.y + le.y) / 2.0
                        total_span = abs(mouth.y - eye_y) + 1e-6
                        nose_y_ratio = (nose.y - eye_y) / total_span

                        if yaw_deviation > 0.22:
                            head_pose = "Looking Away (Distracted)"
                            attentiveness_score = max(20.0, 75.0 - yaw_deviation * 160.0)
                        elif nose_y_ratio > 0.70 or by > h * 0.65:
                            head_pose = "Downward Tilt / Slumped"
                            posture = "Slumping"
                            attentiveness_score = 35.0
                        else:
                            head_pose = "Forward (Engaged)"
                            attentiveness_score = min(98.0, 85.0 + (0.22 - yaw_deviation) * 50.0)

                    detections.append({
                        "bbox": (bx, by, bw, bh),
                        "centroid": (int(bx + bw / 2), int(by + bh / 2)),
                        "confidence": conf,
                        "attentiveness_score": round(attentiveness_score, 1),
                        "head_pose": head_pose,
                        "posture": posture
                    })
                if detections:
                    return detections
            except Exception:
                pass

        # Strategy B: OpenCV Haar Cascade
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40))
            for (bx, by, bw, bh) in faces:
                face_roi_gray = gray[by:by+bh, bx:bx+bw]
                eyes = []
                if self.eye_cascade is not None:
                    eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))

                # Heuristic: 2 detected eyes indicates direct forward orientation
                if len(eyes) >= 2:
                    head_pose = "Forward (Engaged)"
                    attentiveness_score = 88.0
                    posture = "Upright"
                elif len(eyes) == 1:
                    head_pose = "Slight Turn (Partial Focus)"
                    attentiveness_score = 65.0
                    posture = "Upright"
                else:
                    head_pose = "Averted Gaze / Tilted"
                    attentiveness_score = 42.0
                    posture = "Slumped" if (by + bh) > h * 0.75 else "Neutral"

                detections.append({
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "centroid": (int(bx + bw / 2), int(by + bh / 2)),
                    "confidence": 0.80,
                    "attentiveness_score": attentiveness_score,
                    "head_pose": head_pose,
                    "posture": posture
                })
            if detections:
                return detections

        # Strategy C: Geometric Fallback for synthetic/classroom diagram images
        # If no real photographic face is found (e.g. synthetic drawing), detect skin-tone circles/contours
        if not detections:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            # Detect face-like skin tones or head regions
            mask = cv2.inRange(hsv, np.array([0, 15, 60]), np.array([25, 180, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 1200 < area < 40000:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect = float(bw) / float(bh)
                    if 0.6 <= aspect <= 1.5:
                        detections.append({
                            "bbox": (int(bx), int(by), int(bw), int(bh)),
                            "centroid": (int(bx + bw / 2), int(by + bh / 2)),
                            "confidence": 0.75,
                            "attentiveness_score": 82.0,
                            "head_pose": "Forward (Engaged)",
                            "posture": "Upright"
                        })
            if detections:
                return detections

        return detections

    def annotate_frame(self, frame_bgr: np.ndarray, tracked_objects: Dict[int, Tuple[int, int, int, int]],
                       history: Dict[int, List[Dict[str, Any]]], sustained_threshold: float = 0.70,
                       session_duration: float = 60.0) -> np.ndarray:
        """
        Draws educational visual telemetry on frame: tracking bounding boxes, status pills,
        sustained presence indicators, and gaze orientation arrows.
        """
        annotated = frame_bgr.copy()
        for obj_id, (x, y, w, h) in tracked_objects.items():
            hist = history.get(obj_id, [])
            last_entry = hist[-1] if hist else {}
            score = last_entry.get("attentiveness_score", 75.0)
            pose = last_entry.get("head_pose", "Forward")
            
            # Color coding: Green = Engaged, Amber = Partial, Red = Distracted/Slumped
            if score >= 70:
                color = (46, 204, 113)  # Bright Emerald
            elif score >= 50:
                color = (241, 196, 15)  # Amber
            else:
                color = (231, 76, 60)   # Coral Red

            # Modern rounded corner box simulation
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Corner accents
            c_len = max(10, int(min(w, h) * 0.2))
            cv2.line(annotated, (x, y), (x + c_len, y), color, 4)
            cv2.line(annotated, (x, y), (x, y + c_len), color, 4)
            cv2.line(annotated, (x + w, y), (x + w - c_len, y), color, 4)
            cv2.line(annotated, (x + w, y), (x + w, y + c_len), color, 4)
            
            # Header pill with Student ID & Score
            label = f"ID #{obj_id} | {int(score)}% Attentive"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - th - 10), (x + tw + 10, y), color, -1)
            cv2.putText(annotated, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Footer pill with Head Pose & Sustained Status
            active_sec = len(hist) * 0.5  # frame delta estimation
            pct = min(100.0, (active_sec / max(1.0, session_duration)) * 100.0)
            status_text = f"{pose} | Pres: {int(pct)}%"
            (sw, sh), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x, y + h), (x + sw + 8, y + h + sh + 8), (20, 24, 33), -1)
            cv2.putText(annotated, status_text, (x + 4, y + h + sh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 225, 235), 1, cv2.LINE_AA)

        return annotated

    @staticmethod
    def calculate_sustained_attendance(history: Dict[int, List[Dict[str, Any]]],
                                      total_session_seconds: float,
                                      sustained_threshold: float = 0.70,
                                      anonymizer=None) -> pd.DataFrame:
        """
        Sustained Presence Logic (Epic 2):
        Evaluates cumulative presence duration against threshold (>70% session time)
        to prevent false attendance marks from temporary hallway pass-throughs.
        """
        records = []
        threshold_pct = sustained_threshold * 100.0

        for obj_id, track in history.items():
            if not track:
                continue

            timestamps = [t.get("timestamp", 0.0) for t in track]
            first_seen = min(timestamps)
            last_seen = max(timestamps)
            
            # Active duration: span between first and last detection
            span_duration = max(1.0, last_seen - first_seen)
            total_active_seconds = span_duration
            presence_ratio = total_active_seconds / max(1.0, total_session_seconds)
            presence_pct = min(100.0, round(presence_ratio * 100.0, 1))

            scores = [t.get("attentiveness_score", 75.0) for t in track]
            avg_attention = round(float(np.mean(scores)), 1) if scores else 0.0

            # Sustained Presence Classification
            if presence_pct >= threshold_pct:
                status = "Present (Verified)"
                badge = "✅ Present"
            elif total_active_seconds < 40.0 or presence_pct < 18.0:
                status = "Hallway Pass-Through (Filtered)"
                badge = "🚪 Filtered (Pass-through)"
            else:
                status = "Partial / Tardy"
                badge = "⚠️ Partial Attendance"

            raw_id = f"Student_{obj_id:03d}"
            display_id = anonymizer.anonymize_id(raw_id) if anonymizer else raw_id

            first_str = time.strftime("%H:%M:%S", time.gmtime(first_seen))
            last_str = time.strftime("%H:%M:%S", time.gmtime(last_seen))
            mins = int(total_active_seconds // 60)
            secs = int(total_active_seconds % 60)
            dur_str = f"{mins:02d}m {secs:02d}s"

            records.append({
                "Participant ID": display_id,
                "First Seen": first_str,
                "Last Seen": last_str,
                "Active Duration": dur_str,
                "Raw Duration (s)": round(total_active_seconds, 1),
                "Presence Rate (%)": presence_pct,
                "Avg Attention (%)": avg_attention,
                "Attendance Status": status,
                "Status Badge": badge,
                "Eligible Attendance": (presence_pct >= threshold_pct)
            })

        if not records:
            return pd.DataFrame(columns=[
                "Participant ID", "First Seen", "Last Seen", "Active Duration",
                "Raw Duration (s)", "Presence Rate (%)", "Avg Attention (%)",
                "Attendance Status", "Status Badge", "Eligible Attendance"
            ])

        df = pd.DataFrame(records)
        df = df.sort_values(by="Presence Rate (%)", ascending=False).reset_index(drop=True)
        return df
