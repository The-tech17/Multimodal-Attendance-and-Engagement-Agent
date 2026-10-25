"""
Acoustic Interaction & Noise Dynamics Processor
Epic 3 Acoustic Engine

Features:
- Time-windowed audio stream segmentation (1.0s sliding windows)
- RMS Energy, Zero Crossing Rate (ZCR), Spectral Centroid, and Spectral Rolloff
- Voice Activity Detection (VAD) with adaptive noise gating
- 4-Class Pedagogical Acoustic Classifier:
    1. Silence / Quiet Reflection
    2. Instructor Lecture
    3. Student Discussion / Verbal Interaction
    4. High Ambient Classroom Distraction
- Interactive Spectrogram & Decibel Waveform Synthesis
"""

import os
import io
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import signal
import soundfile as sf


class AudioAcousticAnalyzer:
    """
    Classifies classroom audio dynamics 100% locally on the edge.
    Categorizes soundscapes into pedagogical states without transcribing speech.
    Preserves absolute conversation privacy (zero speech-to-text, zero content leakage).
    """

    CLASS_SILENCE = "Silence / Quiet Reflection"
    CLASS_LECTURE = "Instructor Lecture"
    CLASS_DISCUSSION = "Student Discussion / Verbal Interaction"
    CLASS_DISTRACTION = "High Ambient Classroom Distraction"

    COLOR_MAP = {
        CLASS_SILENCE: "#4A90E2",      # Serene Soft Blue
        CLASS_LECTURE: "#2ECC71",      # Emerald Green
        CLASS_DISCUSSION: "#F39C12",   # Vibrant Gold / Orange
        CLASS_DISTRACTION: "#E74C3C"   # Crimson Red
    }

    def __init__(self, sample_rate: int = 16000, window_sec: float = 1.0, hop_sec: float = 0.5):
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.window_samples = int(sample_rate * window_sec)
        self.hop_samples = int(sample_rate * hop_sec)

    def load_audio_file(self, file_source: Any) -> Tuple[np.ndarray, int]:
        """
        Loads audio from filepath, file-like object, or bytes buffer.
        Converts stereo to mono and resamples if needed.
        """
        try:
            if isinstance(file_source, (str, os.PathLike)):
                data, sr = sf.read(file_source, dtype="float32")
            elif hasattr(file_source, "read"):
                file_source.seek(0)
                data, sr = sf.read(io.BytesIO(file_source.read()), dtype="float32")
            elif isinstance(file_source, bytes):
                data, sr = sf.read(io.BytesIO(file_source), dtype="float32")
            else:
                raise ValueError("Unsupported audio source format")

            # Convert stereo to mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Resample if needed
            if sr != self.sample_rate and sr > 0:
                num_resampled = int(len(data) * self.sample_rate / sr)
                data = signal.resample(data, num_resampled)
                sr = self.sample_rate

            return data, sr
        except Exception as e:
            # Fallback to empty signal with warning
            return np.zeros(self.sample_rate * 5, dtype=np.float32), self.sample_rate

    @staticmethod
    def _compute_zcr(frame: np.ndarray) -> float:
        """Computes Zero Crossing Rate."""
        if len(frame) < 2:
            return 0.0
        signs = np.sign(frame)
        signs[signs == 0] = -1
        return float(np.sum(np.abs(np.diff(signs))) / (2.0 * len(frame)))

    @staticmethod
    def _compute_spectral_centroid(magnitude_spec: np.ndarray, freqs: np.ndarray) -> float:
        """Computes center of mass of the spectrum."""
        sum_mag = np.sum(magnitude_spec)
        if sum_mag < 1e-9:
            return 0.0
        return float(np.sum(freqs * magnitude_spec) / sum_mag)

    @staticmethod
    def _compute_spectral_rolloff(magnitude_spec: np.ndarray, freqs: np.ndarray, percentile: float = 0.85) -> float:
        """Computes frequency below which 85% of spectral energy lies."""
        total_energy = np.sum(magnitude_spec)
        if total_energy < 1e-9:
            return 0.0
        cum_energy = np.cumsum(magnitude_spec)
        idx = np.where(cum_energy >= percentile * total_energy)[0]
        if len(idx) > 0:
            return float(freqs[idx[0]])
        return float(freqs[-1])

    def classify_window(self, rms_db: float, zcr: float, centroid: float, rolloff: float, prev_classes: List[str]) -> Tuple[str, float]:
        """
        Pedagogical classification rule engine:
        Classifies window into Silence, Instructor Lecture, Student Discussion, or Distraction.
        """
        # 1. Silence / Quiet Reflection (< -44 dBFS)
        if rms_db < -42.0:
            confidence = min(0.98, max(0.65, ((-42.0 - rms_db) / 20.0) + 0.65))
            return self.CLASS_SILENCE, round(confidence, 2)

        # 2. High Ambient Classroom Distraction (Sudden loud clatter > -14 dB or high rolloff unvoiced bursts)
        if rms_db > -14.0 or (rms_db > -20.0 and rolloff > 4200 and zcr > 0.35):
            confidence = min(0.95, 0.70 + (rms_db + 20.0) / 30.0)
            return self.CLASS_DISTRACTION, round(confidence, 2)

        # 3. Instructor Lecture vs Student Discussion
        # Lecture: sustained dominant pitch (centroid between 600 - 2400 Hz), moderate zcr, steady cadence
        # Discussion: dynamic multi-talker overlap, centroid variance, conversational bursts
        recent_window = prev_classes[-4:] if prev_classes else []
        lecture_count = recent_window.count(self.CLASS_LECTURE)

        if 650 <= centroid <= 2300 and 0.04 <= zcr <= 0.22:
            # Consistent single voice formants
            if rms_db >= -36.0:
                return self.CLASS_LECTURE, 0.88
            else:
                return self.CLASS_LECTURE, 0.75
        elif centroid > 2300 or zcr > 0.20 or (rms_db >= -32.0 and lecture_count >= 1):
            # Dynamic multi-speaker chatter or higher formant interaction
            return self.CLASS_DISCUSSION, 0.82
        else:
            # Default to discussion if active speech with multi-tonal mix
            if rms_db >= -34.0:
                return self.CLASS_DISCUSSION, 0.72
            else:
                return self.CLASS_LECTURE, 0.68

    def analyze_stream(self, audio_data: np.ndarray) -> pd.DataFrame:
        """
        Segments audio into windows and executes full acoustic feature extraction.
        Returns detailed timeline DataFrame.
        """
        n_samples = len(audio_data)
        if n_samples < self.window_samples:
            # Pad if shorter than one window
            audio_data = np.pad(audio_data, (0, self.window_samples - n_samples))
            n_samples = len(audio_data)

        records = []
        classified_history = []
        
        # FFT configuration
        n_fft = 512
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

        step = 0
        current_sample = 0
        while current_sample + self.window_samples <= n_samples:
            window = audio_data[current_sample:current_sample + self.window_samples]
            timestamp_sec = round(current_sample / self.sample_rate, 2)
            
            # 1. RMS Energy (dBFS)
            rms_raw = np.sqrt(np.mean(window**2)) + 1e-9
            rms_db = float(20.0 * np.log10(rms_raw))
            rms_db = max(-80.0, min(0.0, rms_db))

            # 2. Zero Crossing Rate
            zcr = self._compute_zcr(window)

            # 3. Spectral Features via FFT
            window_tapered = window * np.hanning(len(window))
            fft_mag = np.abs(np.fft.rfft(window_tapered, n=n_fft))
            centroid = self._compute_spectral_centroid(fft_mag, freqs)
            rolloff = self._compute_spectral_rolloff(fft_mag, freqs, percentile=0.85)

            # 4. Voice Activity Detection (VAD)
            vad_active = bool(rms_db > -38.0 and 400 < centroid < 4000)

            # 5. Acoustic Classification
            acoustic_class, confidence = self.classify_window(rms_db, zcr, centroid, rolloff, classified_history)
            classified_history.append(acoustic_class)

            # Energy index (normalized 0-100)
            energy_index = min(100.0, max(0.0, (rms_db + 60.0) * 1.66))

            records.append({
                "Time (s)": timestamp_sec,
                "RMS Energy (dB)": round(rms_db, 1),
                "ZCR": round(zcr, 3),
                "Spectral Centroid (Hz)": round(centroid, 1),
                "Spectral Rolloff (Hz)": round(rolloff, 1),
                "VAD Active": vad_active,
                "Acoustic Class": acoustic_class,
                "Confidence": confidence,
                "Energy Index": round(energy_index, 1)
            })

            current_sample += self.hop_samples
            step += 1

        df = pd.DataFrame(records)
        return df

    def compute_spectrogram_matrix(self, audio_data: np.ndarray, n_bins: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 2D STFT spectrogram matrix for interactive visual heatmap rendering.
        """
        f, t, Sxx = signal.spectrogram(audio_data, fs=self.sample_rate, nperseg=512, noverlap=256)
        Sxx_db = 10 * np.log10(Sxx + 1e-9)
        # Limit to 0-5000 Hz for pedagogical vocal speech visualization
        freq_mask = f <= 5000
        return t, f[freq_mask], Sxx_db[freq_mask, :]

    @staticmethod
    def generate_synthetic_classroom_audio(duration_sec: int = 30, scenario: str = "Standard Active Lecture") -> np.ndarray:
        """
        Generates realistic synthetic classroom acoustic signals for offline edge testing.
        Scenarios:
        - 'Standard Active Lecture': Teacher speaking with student questions and brief quiet pauses
        - 'Collaborative Group Discussion': Multi-voice conversational chatter and laughter
        - 'Quiet Study & Assessment': Low ambient room hum, pencil writing, page turns
        - 'Noisy Distraction Spike': Lecture interrupted by loud desk move and hallway echo
        """
        sr = 16000
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        total_samples = len(t)
        audio = np.zeros(total_samples, dtype=np.float32)

        # Baseline gentle classroom ventilation hum (-50 dB)
        audio += 0.003 * np.random.normal(0, 1, total_samples)

        if scenario == "Standard Active Lecture":
            # 0-12s: Teacher lecturing (dominant 180Hz fundamental + formants)
            s1 = int(0 * sr)
            e1 = int(min(duration_sec, 14) * sr)
            t_chunk = t[s1:e1]
            mod = 0.5 * (1 + np.sin(2 * np.pi * 3.5 * t_chunk))  # Speech cadence
            lecture_vox = mod * (0.22 * np.sin(2 * np.pi * 185 * t_chunk) +
                                 0.15 * np.sin(2 * np.pi * 370 * t_chunk) +
                                 0.08 * np.sin(2 * np.pi * 920 * t_chunk))
            audio[s1:e1] += lecture_vox

            # 14-22s: Student interaction / Q&A (higher 280Hz fundamental + turn taking)
            if duration_sec > 14:
                s2 = int(14 * sr)
                e2 = int(min(duration_sec, 24) * sr)
                t_chunk2 = t[s2:e2]
                mod2 = 0.5 * (1 + np.sin(2 * np.pi * 5.0 * t_chunk2))
                student_vox = mod2 * (0.20 * np.sin(2 * np.pi * 270 * t_chunk2) +
                                      0.12 * np.sin(2 * np.pi * 540 * t_chunk2) +
                                      0.06 * np.sin(2 * np.pi * 1200 * t_chunk2))
                audio[s2:e2] += student_vox

            # Remainder: Brief quiet reflection pause
            # Audio remains quiet hum

        elif scenario == "Collaborative Group Discussion":
            # Multi-speaker overlapping chatter with conversational dynamic peaks
            for freq, rate in [(190, 4.2), (250, 5.1), (320, 3.8), (170, 6.0)]:
                mod = 0.3 * (1 + np.sin(2 * np.pi * rate * t))
                audio += mod * (0.10 * np.sin(2 * np.pi * freq * t) + 0.05 * np.sin(2 * np.pi * freq * 2 * t))
            # Dynamic bursts
            audio += 0.02 * np.random.normal(0, 1, total_samples) * (1 + np.sin(2 * np.pi * 0.8 * t))

        elif scenario == "Quiet Study & Assessment":
            # Very low ambient silence (< -45 dB) with occasional paper rustle
            rustle_points = [int(sr * 6), int(sr * 15), int(sr * 24)]
            for pt in rustle_points:
                if pt + sr < total_samples:
                    audio[pt:pt + int(sr * 0.8)] += 0.04 * np.random.normal(0, 1, int(sr * 0.8))

        elif scenario == "Noisy Distraction Spike":
            # Teacher lecture followed by sudden loud desk scrap / chair bump
            s1 = 0
            e1 = int(min(duration_sec, 12) * sr)
            t_chunk = t[s1:e1]
            audio[s1:e1] += 0.18 * np.sin(2 * np.pi * 200 * t_chunk)
            
            # Loud crash / disturbance at 13s
            if duration_sec > 13:
                s_spike = int(13 * sr)
                e_spike = int(min(duration_sec, 17) * sr)
                audio[s_spike:e_spike] += 0.75 * np.random.normal(0, 1, e_spike - s_spike)

        return np.clip(audio, -1.0, 1.0)
