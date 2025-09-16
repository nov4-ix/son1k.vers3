"""
Advanced Musical Analysis System
Provides comprehensive audio analysis including key detection, tempo analysis, and structural analysis
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

# Optional imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MusicalKey(Enum):
    """Musical keys"""
    C_MAJOR = "C major"
    C_SHARP_MAJOR = "C# major"
    D_MAJOR = "D major"
    E_FLAT_MAJOR = "Eb major"
    E_MAJOR = "E major"
    F_MAJOR = "F major"
    F_SHARP_MAJOR = "F# major"
    G_MAJOR = "G major"
    A_FLAT_MAJOR = "Ab major"
    A_MAJOR = "A major"
    B_FLAT_MAJOR = "Bb major"
    B_MAJOR = "B major"
    
    C_MINOR = "C minor"
    C_SHARP_MINOR = "C# minor"
    D_MINOR = "D minor"
    E_FLAT_MINOR = "Eb minor"
    E_MINOR = "E minor"
    F_MINOR = "F minor"
    F_SHARP_MINOR = "F# minor"
    G_MINOR = "G minor"
    A_FLAT_MINOR = "Ab minor"
    A_MINOR = "A minor"
    B_FLAT_MINOR = "Bb minor"
    B_MINOR = "B minor"

class TimeSignature(Enum):
    """Common time signatures"""
    FOUR_FOUR = "4/4"
    THREE_FOUR = "3/4"
    TWO_FOUR = "2/4"
    SIX_EIGHT = "6/8"
    NINE_EIGHT = "9/8"
    TWELVE_EIGHT = "12/8"
    SEVEN_EIGHT = "7/8"
    FIVE_FOUR = "5/4"

class MusicalGenre(Enum):
    """Musical genres for classification"""
    POP = "pop"
    ROCK = "rock"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    ELECTRONIC = "electronic"
    BLUES = "blues"
    COUNTRY = "country"
    REGGAE = "reggae"
    FUNK = "funk"
    METAL = "metal"
    FOLK = "folk"
    R_AND_B = "r&b"
    HIP_HOP = "hip_hop"
    AMBIENT = "ambient"

@dataclass
class StructuralSegment:
    """Represents a structural segment of a song"""
    start_time: float
    end_time: float
    label: str  # verse, chorus, bridge, intro, outro
    confidence: float
    features: Dict[str, Any]

@dataclass
class TempoAnalysis:
    """Tempo analysis results"""
    bpm: float
    confidence: float
    tempo_stability: float
    beat_times: List[float]
    downbeat_times: List[float]
    time_signature: TimeSignature
    time_signature_confidence: float

@dataclass
class KeyAnalysis:
    """Key analysis results"""
    key: MusicalKey
    confidence: float
    key_strength: float
    chromagram: np.ndarray
    key_changes: List[Tuple[float, MusicalKey, float]]  # time, key, confidence

@dataclass
class HarmonicAnalysis:
    """Harmonic analysis results"""
    chord_progression: List[Tuple[float, str, float]]  # time, chord, confidence
    harmonic_rhythm: float
    tension_curve: np.ndarray
    dissonance_levels: np.ndarray
    modulations: List[Tuple[float, str, str]]  # time, from_key, to_key

@dataclass
class StructuralAnalysis:
    """Structural analysis results"""
    segments: List[StructuralSegment]
    song_structure: str  # e.g., "ABABCB" where A=verse, B=chorus, C=bridge
    repetition_score: float
    novelty_curve: np.ndarray
    boundaries: List[float]  # segment boundary times

@dataclass
class SpectralAnalysis:
    """Spectral analysis results"""
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    spectral_bandwidth: np.ndarray
    zero_crossing_rate: np.ndarray
    mfcc: np.ndarray
    chroma: np.ndarray
    tonnetz: np.ndarray
    spectral_contrast: np.ndarray

@dataclass
class RhythmicAnalysis:
    """Rhythmic analysis results"""
    onset_times: List[float]
    onset_strength: np.ndarray
    rhythmic_pattern: np.ndarray
    syncopation_score: float
    rhythmic_complexity: float
    pulse_clarity: float

@dataclass
class ComprehensiveAnalysis:
    """Complete musical analysis results"""
    tempo: TempoAnalysis
    key: KeyAnalysis
    harmony: HarmonicAnalysis
    structure: StructuralAnalysis
    spectral: SpectralAnalysis
    rhythm: RhythmicAnalysis
    genre_prediction: Dict[str, float]
    overall_complexity: float
    energy_profile: np.ndarray
    mood_analysis: Dict[str, float]

class KeyDetector:
    """Advanced key detection using multiple algorithms"""
    
    def __init__(self):
        # Krumhansl-Schmuckler key profiles
        self.major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        self.major_profile = self.major_profile / np.linalg.norm(self.major_profile)
        self.minor_profile = self.minor_profile / np.linalg.norm(self.minor_profile)
    
    def detect_key(self, audio: np.ndarray, sr: int = 44100) -> KeyAnalysis:
        """Detect musical key using multiple methods"""
        try:
            if not LIBROSA_AVAILABLE:
                logger.warning("Librosa not available - using simplified key detection")
                return self._get_default_key_analysis()
                
            # Extract chromagram
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Normalize chromagram
            if np.sum(chroma_mean) > 0:
                chroma_mean = chroma_mean / np.sum(chroma_mean)
            
            # Calculate correlations with key profiles
            key_scores = []
            
            # Major keys
            for i in range(12):
                shifted_major = np.roll(self.major_profile, i)
                correlation = np.corrcoef(chroma_mean, shifted_major)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                key_scores.append((correlation, i, True))  # True for major
            
            # Minor keys
            for i in range(12):
                shifted_minor = np.roll(self.minor_profile, i)
                correlation = np.corrcoef(chroma_mean, shifted_minor)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                key_scores.append((correlation, i, False))  # False for minor
            
            # Find best match
            key_scores.sort(key=lambda x: x[0], reverse=True)
            best_score, best_root, is_major = key_scores[0]
            
            # Convert to MusicalKey enum
            key_names_major = [
                MusicalKey.C_MAJOR, MusicalKey.C_SHARP_MAJOR, MusicalKey.D_MAJOR,
                MusicalKey.E_FLAT_MAJOR, MusicalKey.E_MAJOR, MusicalKey.F_MAJOR,
                MusicalKey.F_SHARP_MAJOR, MusicalKey.G_MAJOR, MusicalKey.A_FLAT_MAJOR,
                MusicalKey.A_MAJOR, MusicalKey.B_FLAT_MAJOR, MusicalKey.B_MAJOR
            ]
            
            key_names_minor = [
                MusicalKey.C_MINOR, MusicalKey.C_SHARP_MINOR, MusicalKey.D_MINOR,
                MusicalKey.E_FLAT_MINOR, MusicalKey.E_MINOR, MusicalKey.F_MINOR,
                MusicalKey.F_SHARP_MINOR, MusicalKey.G_MINOR, MusicalKey.A_FLAT_MINOR,
                MusicalKey.A_MINOR, MusicalKey.B_FLAT_MINOR, MusicalKey.B_MINOR
            ]
            
            detected_key = key_names_major[best_root] if is_major else key_names_minor[best_root]
            
            # Calculate key strength
            key_strength = np.max(chroma_mean) - np.mean(chroma_mean)
            
            # Detect key changes (simplified)
            key_changes = self._detect_key_changes(chroma, sr)
            
            return KeyAnalysis(
                key=detected_key,
                confidence=float(max(0.0, min(1.0, best_score))),
                key_strength=float(key_strength),
                chromagram=chroma_mean,
                key_changes=key_changes
            )
            
        except Exception as e:
            logger.error(f"Error in key detection: {e}")
            return self._get_default_key_analysis()
    
    def _get_default_key_analysis(self) -> KeyAnalysis:
        """Return default key analysis when librosa is not available"""
        return KeyAnalysis(
            key=MusicalKey.C_MAJOR,
            confidence=0.0,
            key_strength=0.0,
            chromagram=np.zeros(12),
            key_changes=[]
        )
    
    def _detect_key_changes(self, chroma: np.ndarray, sr: int) -> List[Tuple[float, MusicalKey, float]]:
        """Detect key changes throughout the audio"""
        try:
            # Simplified key change detection
            hop_length = 512
            frame_rate = sr / hop_length
            
            # Analyze in windows
            window_size = int(frame_rate * 10)  # 10-second windows
            key_changes = []
            
            for i in range(0, chroma.shape[1] - window_size, window_size // 2):
                window_chroma = chroma[:, i:i+window_size]
                window_mean = np.mean(window_chroma, axis=1)
                
                if np.sum(window_mean) > 0:
                    window_mean = window_mean / np.sum(window_mean)
                    
                    # Find best key for this window
                    best_score = -1
                    best_key = MusicalKey.C_MAJOR
                    
                    for root in range(12):
                        for is_major in [True, False]:
                            profile = self.major_profile if is_major else self.minor_profile
                            shifted_profile = np.roll(profile, root)
                            correlation = np.corrcoef(window_mean, shifted_profile)[0, 1]
                            
                            if not np.isnan(correlation) and correlation > best_score:
                                best_score = correlation
                                key_names = [
                                    MusicalKey.C_MAJOR, MusicalKey.C_SHARP_MAJOR, MusicalKey.D_MAJOR,
                                    MusicalKey.E_FLAT_MAJOR, MusicalKey.E_MAJOR, MusicalKey.F_MAJOR,
                                    MusicalKey.F_SHARP_MAJOR, MusicalKey.G_MAJOR, MusicalKey.A_FLAT_MAJOR,
                                    MusicalKey.A_MAJOR, MusicalKey.B_FLAT_MAJOR, MusicalKey.B_MAJOR
                                ] if is_major else [
                                    MusicalKey.C_MINOR, MusicalKey.C_SHARP_MINOR, MusicalKey.D_MINOR,
                                    MusicalKey.E_FLAT_MINOR, MusicalKey.E_MINOR, MusicalKey.F_MINOR,
                                    MusicalKey.F_SHARP_MINOR, MusicalKey.G_MINOR, MusicalKey.A_FLAT_MINOR,
                                    MusicalKey.A_MINOR, MusicalKey.B_FLAT_MINOR, MusicalKey.B_MINOR
                                ]
                                best_key = key_names[root]
                    
                    time = i / frame_rate
                    key_changes.append((time, best_key, best_score))
            
            return key_changes[:5]  # Return first 5 key changes
            
        except Exception as e:
            logger.error(f"Error detecting key changes: {e}")
            return []

class TempoAnalyzer:
    """Advanced tempo and beat tracking"""
    
    def detect_tempo(self, audio: np.ndarray, sr: int = 44100) -> TempoAnalysis:
        """Comprehensive tempo analysis"""
        try:
            if not LIBROSA_AVAILABLE:
                logger.warning("Librosa not available - using simplified tempo detection")
                return self._get_default_tempo_analysis()
                
            # Basic tempo detection
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=512)
            
            # Beat times
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
            
            # Downbeat detection (simplified)
            downbeat_times = self._detect_downbeats(audio, sr, beat_times)
            
            # Tempo stability
            tempo_stability = self._calculate_tempo_stability(beat_times)
            
            # Time signature detection
            time_sig, time_sig_confidence = self._detect_time_signature(beat_times, downbeat_times)
            
            return TempoAnalysis(
                bpm=float(tempo),
                confidence=0.8,  # Simplified confidence
                tempo_stability=tempo_stability,
                beat_times=beat_times.tolist(),
                downbeat_times=downbeat_times,
                time_signature=time_sig,
                time_signature_confidence=time_sig_confidence
            )
            
        except Exception as e:
            logger.error(f"Error in tempo analysis: {e}")
            return self._get_default_tempo_analysis()
    
    def _get_default_tempo_analysis(self) -> TempoAnalysis:
        """Return default tempo analysis when librosa is not available"""
        return TempoAnalysis(
            bpm=120.0,
            confidence=0.0,
            tempo_stability=0.0,
            beat_times=[],
            downbeat_times=[],
            time_signature=TimeSignature.FOUR_FOUR,
            time_signature_confidence=0.0
        )
    
    def _detect_downbeats(self, audio: np.ndarray, sr: int, beat_times: np.ndarray) -> List[float]:
        """Detect downbeats (simplified approach)"""
        try:
            # Simplified: assume every 4th beat is a downbeat
            downbeats = []
            for i in range(0, len(beat_times), 4):
                if i < len(beat_times):
                    downbeats.append(float(beat_times[i]))
            return downbeats
        except:
            return []
    
    def _calculate_tempo_stability(self, beat_times: np.ndarray) -> float:
        """Calculate tempo stability"""
        try:
            if len(beat_times) < 3:
                return 0.0
            
            intervals = np.diff(beat_times)
            if len(intervals) == 0:
                return 0.0
            
            stability = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-8))
            return max(0.0, min(1.0, stability))
        except:
            return 0.0
    
    def _detect_time_signature(self, beat_times: np.ndarray, downbeat_times: List[float]) -> Tuple[TimeSignature, float]:
        """Detect time signature (simplified)"""
        try:
            if len(downbeat_times) < 2:
                return TimeSignature.FOUR_FOUR, 0.5
            
            # Calculate average beats per measure
            downbeat_intervals = np.diff(downbeat_times)
            if len(beat_times) > 0 and len(downbeat_intervals) > 0:
                avg_beat_interval = np.mean(np.diff(beat_times))
                avg_measure_interval = np.mean(downbeat_intervals)
                beats_per_measure = avg_measure_interval / avg_beat_interval
                
                # Classify time signature
                if 3.5 <= beats_per_measure <= 4.5:
                    return TimeSignature.FOUR_FOUR, 0.8
                elif 2.5 <= beats_per_measure <= 3.5:
                    return TimeSignature.THREE_FOUR, 0.7
                elif 1.5 <= beats_per_measure <= 2.5:
                    return TimeSignature.TWO_FOUR, 0.6
            
            return TimeSignature.FOUR_FOUR, 0.5
            
        except:
            return TimeSignature.FOUR_FOUR, 0.5

class StructuralAnalyzer:
    """Analyze song structure and segments"""
    
    def analyze_structure(self, audio: np.ndarray, sr: int = 44100) -> StructuralAnalysis:
        """Analyze song structure"""
        try:
            # Extract features for structure analysis
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=512, n_mfcc=13)
            
            # Combine features
            features = np.vstack([chroma, mfcc])
            
            # Self-similarity matrix
            similarity_matrix = self._compute_self_similarity(features)
            
            # Detect boundaries
            boundaries = self._detect_boundaries(similarity_matrix, sr)
            
            # Create segments
            segments = self._create_segments(boundaries, features, sr)
            
            # Analyze song structure pattern
            song_structure = self._analyze_song_pattern(segments)
            
            # Calculate repetition score
            repetition_score = self._calculate_repetition_score(similarity_matrix)
            
            # Novelty curve
            novelty_curve = self._compute_novelty_curve(features)
            
            return StructuralAnalysis(
                segments=segments,
                song_structure=song_structure,
                repetition_score=repetition_score,
                novelty_curve=novelty_curve,
                boundaries=boundaries
            )
            
        except Exception as e:
            logger.error(f"Error in structural analysis: {e}")
            return StructuralAnalysis(
                segments=[],
                song_structure="UNKNOWN",
                repetition_score=0.0,
                novelty_curve=np.array([]),
                boundaries=[]
            )
    
    def _compute_self_similarity(self, features: np.ndarray) -> np.ndarray:
        """Compute self-similarity matrix"""
        try:
            # Normalize features
            features = features / (np.linalg.norm(features, axis=0, keepdims=True) + 1e-8)
            
            # Compute similarity matrix
            similarity_matrix = np.dot(features.T, features)
            return similarity_matrix
        except:
            return np.eye(features.shape[1])
    
    def _detect_boundaries(self, similarity_matrix: np.ndarray, sr: int) -> List[float]:
        """Detect segment boundaries"""
        try:
            # Compute novelty function
            novelty = np.zeros(similarity_matrix.shape[0])
            kernel_size = min(16, similarity_matrix.shape[0] // 4)
            
            for i in range(kernel_size, similarity_matrix.shape[0] - kernel_size):
                local_sim = similarity_matrix[i-kernel_size:i+kernel_size, i-kernel_size:i+kernel_size]
                novelty[i] = -np.mean(local_sim)
            
            # Find peaks in novelty function
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(novelty, height=np.percentile(novelty, 70))
            
            # Convert to time
            hop_length = 512
            frame_rate = sr / hop_length
            boundary_times = peaks / frame_rate
            
            return boundary_times.tolist()
            
        except Exception as e:
            logger.error(f"Error detecting boundaries: {e}")
            return [0.0, 30.0, 60.0, 90.0]  # Default boundaries
    
    def _create_segments(self, boundaries: List[float], features: np.ndarray, sr: int) -> List[StructuralSegment]:
        """Create structural segments"""
        try:
            segments = []
            segment_labels = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
            
            for i in range(len(boundaries) - 1):
                start_time = boundaries[i]
                end_time = boundaries[i + 1] if i + 1 < len(boundaries) else boundaries[-1] + 30
                
                label = segment_labels[i % len(segment_labels)]
                confidence = 0.7  # Simplified confidence
                
                # Extract segment features
                hop_length = 512
                frame_rate = sr / hop_length
                start_frame = int(start_time * frame_rate)
                end_frame = int(end_time * frame_rate)
                
                segment_features = {}
                if start_frame < features.shape[1] and end_frame <= features.shape[1]:
                    segment_data = features[:, start_frame:end_frame]
                    segment_features = {
                        "mean_chroma": np.mean(segment_data[:12], axis=1).tolist(),
                        "mean_mfcc": np.mean(segment_data[12:], axis=1).tolist()
                    }
                
                segments.append(StructuralSegment(
                    start_time=start_time,
                    end_time=end_time,
                    label=label,
                    confidence=confidence,
                    features=segment_features
                ))
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating segments: {e}")
            return []
    
    def _analyze_song_pattern(self, segments: List[StructuralSegment]) -> str:
        """Analyze song structure pattern"""
        try:
            if not segments:
                return "UNKNOWN"
            
            # Create pattern string
            pattern = ""
            label_map = {"intro": "I", "verse": "A", "chorus": "B", "bridge": "C", "outro": "O"}
            
            for segment in segments:
                pattern += label_map.get(segment.label, "X")
            
            return pattern
            
        except:
            return "UNKNOWN"
    
    def _calculate_repetition_score(self, similarity_matrix: np.ndarray) -> float:
        """Calculate overall repetition score"""
        try:
            # Calculate mean similarity (excluding diagonal)
            mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
            repetition_score = np.mean(similarity_matrix[mask])
            return max(0.0, min(1.0, repetition_score))
        except:
            return 0.5
    
    def _compute_novelty_curve(self, features: np.ndarray) -> np.ndarray:
        """Compute novelty curve"""
        try:
            novelty = np.zeros(features.shape[1])
            for i in range(1, features.shape[1]):
                novelty[i] = np.linalg.norm(features[:, i] - features[:, i-1])
            return novelty
        except:
            return np.zeros(100)

class ComprehensiveMusicalAnalyzer:
    """Main analyzer combining all analysis components"""
    
    def __init__(self):
        self.key_detector = KeyDetector()
        self.tempo_analyzer = TempoAnalyzer()
        self.structural_analyzer = StructuralAnalyzer()
        logger.info("ComprehensiveMusicalAnalyzer initialized")
    
    def analyze(self, audio: np.ndarray, sr: int = 44100) -> ComprehensiveAnalysis:
        """Perform comprehensive musical analysis"""
        try:
            logger.info("Starting comprehensive musical analysis")
            
            # Tempo analysis
            tempo_analysis = self.tempo_analyzer.detect_tempo(audio, sr)
            
            # Key analysis
            key_analysis = self.key_detector.detect_key(audio, sr)
            
            # Spectral analysis
            spectral_analysis = self._analyze_spectral_features(audio, sr)
            
            # Harmonic analysis
            harmonic_analysis = self._analyze_harmony(audio, sr, key_analysis)
            
            # Structural analysis
            structural_analysis = self.structural_analyzer.analyze_structure(audio, sr)
            
            # Rhythmic analysis
            rhythmic_analysis = self._analyze_rhythm(audio, sr, tempo_analysis)
            
            # Genre prediction
            genre_prediction = self._predict_genre(spectral_analysis, tempo_analysis, key_analysis)
            
            # Overall complexity
            overall_complexity = self._calculate_overall_complexity(
                harmonic_analysis, rhythmic_analysis, structural_analysis
            )
            
            # Energy profile
            energy_profile = self._calculate_energy_profile(audio, sr)
            
            # Mood analysis
            mood_analysis = self._analyze_mood(spectral_analysis, key_analysis, tempo_analysis)
            
            logger.info("Comprehensive musical analysis completed")
            
            return ComprehensiveAnalysis(
                tempo=tempo_analysis,
                key=key_analysis,
                harmony=harmonic_analysis,
                structure=structural_analysis,
                spectral=spectral_analysis,
                rhythm=rhythmic_analysis,
                genre_prediction=genre_prediction,
                overall_complexity=overall_complexity,
                energy_profile=energy_profile,
                mood_analysis=mood_analysis
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return self._get_default_analysis()
    
    def _analyze_spectral_features(self, audio: np.ndarray, sr: int) -> SpectralAnalysis:
        """Analyze spectral features"""
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            return SpectralAnalysis(
                spectral_centroid=spectral_centroid,
                spectral_rolloff=spectral_rolloff,
                spectral_bandwidth=spectral_bandwidth,
                zero_crossing_rate=zero_crossing_rate,
                mfcc=mfcc,
                chroma=chroma,
                tonnetz=tonnetz,
                spectral_contrast=spectral_contrast
            )
            
        except Exception as e:
            logger.error(f"Error in spectral analysis: {e}")
            return SpectralAnalysis(
                spectral_centroid=np.array([1000.0]),
                spectral_rolloff=np.array([2000.0]),
                spectral_bandwidth=np.array([1500.0]),
                zero_crossing_rate=np.array([0.1]),
                mfcc=np.zeros((13, 100)),
                chroma=np.zeros((12, 100)),
                tonnetz=np.zeros((6, 100)),
                spectral_contrast=np.zeros((7, 100))
            )
    
    def _analyze_harmony(self, audio: np.ndarray, sr: int, key_analysis: KeyAnalysis) -> HarmonicAnalysis:
        """Analyze harmonic content"""
        try:
            # Simplified chord detection
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Basic chord progression detection (simplified)
            chord_progression = []
            hop_length = 512
            frame_rate = sr / hop_length
            
            for i in range(0, chroma.shape[1], int(frame_rate * 2)):  # Every 2 seconds
                if i + int(frame_rate * 2) <= chroma.shape[1]:
                    chord_chroma = np.mean(chroma[:, i:i + int(frame_rate * 2)], axis=1)
                    chord_name = self._detect_chord(chord_chroma)
                    time = i / frame_rate
                    chord_progression.append((time, chord_name, 0.7))
            
            # Calculate harmonic rhythm
            harmonic_rhythm = len(chord_progression) / (len(audio) / sr) if len(chord_progression) > 0 else 0
            
            # Tension curve (simplified)
            tension_curve = self._calculate_tension_curve(chroma)
            
            # Dissonance levels
            dissonance_levels = self._calculate_dissonance(chroma)
            
            return HarmonicAnalysis(
                chord_progression=chord_progression,
                harmonic_rhythm=harmonic_rhythm,
                tension_curve=tension_curve,
                dissonance_levels=dissonance_levels,
                modulations=[]  # Simplified
            )
            
        except Exception as e:
            logger.error(f"Error in harmonic analysis: {e}")
            return HarmonicAnalysis(
                chord_progression=[],
                harmonic_rhythm=0.0,
                tension_curve=np.array([]),
                dissonance_levels=np.array([]),
                modulations=[]
            )
    
    def _detect_chord(self, chroma: np.ndarray) -> str:
        """Detect chord from chromagram (simplified)"""
        try:
            # Find dominant notes
            top_notes = np.argsort(chroma)[-3:]  # Top 3 notes
            
            # Map to note names
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Simple chord detection
            root = top_notes[-1]  # Strongest note as root
            chord_name = note_names[root]
            
            # Determine if major or minor (simplified)
            if len(top_notes) >= 3:
                third = (root + 4) % 12  # Major third
                minor_third = (root + 3) % 12  # Minor third
                
                if minor_third in top_notes:
                    chord_name += "m"
                # else: assume major (default)
            
            return chord_name
            
        except:
            return "C"
    
    def _calculate_tension_curve(self, chroma: np.ndarray) -> np.ndarray:
        """Calculate harmonic tension curve"""
        try:
            tension = np.zeros(chroma.shape[1])
            for i in range(chroma.shape[1]):
                # Calculate dissonance as deviation from pure intervals
                frame = chroma[:, i]
                tension[i] = np.std(frame)  # Simplified tension measure
            return tension
        except:
            return np.zeros(100)
    
    def _calculate_dissonance(self, chroma: np.ndarray) -> np.ndarray:
        """Calculate dissonance levels"""
        try:
            # Simplified dissonance calculation
            dissonance = np.zeros(chroma.shape[1])
            for i in range(chroma.shape[1]):
                frame = chroma[:, i]
                # Count active notes
                active_notes = np.sum(frame > 0.1)
                dissonance[i] = active_notes / 12.0  # Normalize
            return dissonance
        except:
            return np.zeros(100)
    
    def _analyze_rhythm(self, audio: np.ndarray, sr: int, tempo_analysis: TempoAnalysis) -> RhythmicAnalysis:
        """Analyze rhythmic features"""
        try:
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=512)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            onset_strength = librosa.onset.onset_strength(y=audio, sr=sr)
            
            # Rhythmic pattern analysis
            rhythmic_pattern = self._extract_rhythmic_pattern(onset_times, tempo_analysis.bpm)
            
            # Syncopation score
            syncopation_score = self._calculate_syncopation(onset_times, tempo_analysis.beat_times)
            
            # Rhythmic complexity
            rhythmic_complexity = self._calculate_rhythmic_complexity(rhythmic_pattern)
            
            # Pulse clarity
            pulse_clarity = tempo_analysis.tempo_stability
            
            return RhythmicAnalysis(
                onset_times=onset_times.tolist(),
                onset_strength=onset_strength,
                rhythmic_pattern=rhythmic_pattern,
                syncopation_score=syncopation_score,
                rhythmic_complexity=rhythmic_complexity,
                pulse_clarity=pulse_clarity
            )
            
        except Exception as e:
            logger.error(f"Error in rhythmic analysis: {e}")
            return RhythmicAnalysis(
                onset_times=[],
                onset_strength=np.array([]),
                rhythmic_pattern=np.array([]),
                syncopation_score=0.0,
                rhythmic_complexity=0.0,
                pulse_clarity=0.0
            )
    
    def _extract_rhythmic_pattern(self, onset_times: np.ndarray, bpm: float) -> np.ndarray:
        """Extract rhythmic pattern"""
        try:
            if len(onset_times) == 0 or bpm == 0:
                return np.zeros(16)
            
            # Convert to beat positions
            beat_duration = 60.0 / bpm
            beat_positions = (onset_times % (4 * beat_duration)) / beat_duration
            
            # Create histogram
            pattern, _ = np.histogram(beat_positions, bins=16, range=(0, 4))
            
            # Normalize
            if np.sum(pattern) > 0:
                pattern = pattern / np.sum(pattern)
            
            return pattern
            
        except:
            return np.zeros(16)
    
    def _calculate_syncopation(self, onset_times: List[float], beat_times: List[float]) -> float:
        """Calculate syncopation score"""
        try:
            if len(onset_times) == 0 or len(beat_times) == 0:
                return 0.0
            
            syncopated_onsets = 0
            total_onsets = len(onset_times)
            
            for onset in onset_times:
                # Check if onset is close to a beat
                distances = [abs(onset - beat) for beat in beat_times]
                min_distance = min(distances) if distances else float('inf')
                
                # If onset is not close to a beat, it's syncopated
                if min_distance > 0.1:  # 100ms tolerance
                    syncopated_onsets += 1
            
            return syncopated_onsets / total_onsets if total_onsets > 0 else 0.0
            
        except:
            return 0.0
    
    def _calculate_rhythmic_complexity(self, rhythmic_pattern: np.ndarray) -> float:
        """Calculate rhythmic complexity"""
        try:
            if len(rhythmic_pattern) == 0:
                return 0.0
            
            # Use entropy as complexity measure
            pattern = rhythmic_pattern + 1e-8  # Avoid log(0)
            entropy = -np.sum(pattern * np.log2(pattern))
            
            # Normalize to 0-1 range
            max_entropy = np.log2(len(pattern))
            complexity = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return complexity
            
        except:
            return 0.0
    
    def _predict_genre(self, spectral: SpectralAnalysis, tempo: TempoAnalysis, key: KeyAnalysis) -> Dict[str, float]:
        """Predict musical genre (simplified rule-based)"""
        try:
            scores = {}
            
            # Initialize all genres with base score
            for genre in MusicalGenre:
                scores[genre.value] = 0.1
            
            # Tempo-based rules
            bpm = tempo.bpm
            if 60 <= bpm <= 80:
                scores["blues"] += 0.3
                scores["jazz"] += 0.2
            elif 80 <= bpm <= 100:
                scores["jazz"] += 0.3
                scores["r&b"] += 0.2
            elif 100 <= bpm <= 130:
                scores["pop"] += 0.3
                scores["rock"] += 0.2
            elif 130 <= bpm <= 180:
                scores["electronic"] += 0.3
                scores["metal"] += 0.2
            
            # Spectral centroid rules
            avg_centroid = np.mean(spectral.spectral_centroid)
            if avg_centroid < 1500:
                scores["classical"] += 0.2
                scores["jazz"] += 0.1
            elif avg_centroid > 3000:
                scores["electronic"] += 0.2
                scores["metal"] += 0.1
            
            # Normalize scores
            total = sum(scores.values())
            if total > 0:
                scores = {k: v/total for k, v in scores.items()}
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in genre prediction: {e}")
            return {"pop": 1.0}
    
    def _calculate_overall_complexity(self, harmonic: HarmonicAnalysis, 
                                    rhythmic: RhythmicAnalysis, structural: StructuralAnalysis) -> float:
        """Calculate overall musical complexity"""
        try:
            complexity_factors = [
                harmonic.harmonic_rhythm / 2.0,  # Normalize chord changes per second
                rhythmic.rhythmic_complexity,
                rhythmic.syncopation_score,
                min(1.0, len(structural.segments) / 8.0),  # Normalize segment count
                min(1.0, np.mean(harmonic.dissonance_levels) if len(harmonic.dissonance_levels) > 0 else 0)
            ]
            
            return np.mean(complexity_factors)
            
        except:
            return 0.5
    
    def _calculate_energy_profile(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Calculate energy profile over time"""
        try:
            # RMS energy
            rms = librosa.feature.rms(y=audio, hop_length=512)[0]
            
            # Smooth and normalize
            from scipy import ndimage
            smoothed = ndimage.gaussian_filter1d(rms, sigma=2)
            
            # Normalize to 0-1 range
            if np.max(smoothed) > np.min(smoothed):
                normalized = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
            else:
                normalized = smoothed
            
            return normalized
            
        except:
            return np.ones(100) * 0.5
    
    def _analyze_mood(self, spectral: SpectralAnalysis, key: KeyAnalysis, tempo: TempoAnalysis) -> Dict[str, float]:
        """Analyze mood characteristics"""
        try:
            mood_scores = {
                "happy": 0.0,
                "sad": 0.0,
                "energetic": 0.0,
                "calm": 0.0,
                "aggressive": 0.0,
                "romantic": 0.0,
                "melancholic": 0.0,
                "uplifting": 0.0
            }
            
            # Tempo influence
            bpm = tempo.bpm
            if bpm > 120:
                mood_scores["energetic"] += 0.4
                mood_scores["happy"] += 0.3
            else:
                mood_scores["calm"] += 0.3
                mood_scores["melancholic"] += 0.2
            
            # Key influence
            if "minor" in key.key.value.lower():
                mood_scores["sad"] += 0.3
                mood_scores["melancholic"] += 0.3
            else:
                mood_scores["happy"] += 0.3
                mood_scores["uplifting"] += 0.2
            
            # Spectral features influence
            avg_centroid = np.mean(spectral.spectral_centroid)
            if avg_centroid > 2500:
                mood_scores["aggressive"] += 0.2
                mood_scores["energetic"] += 0.1
            
            # Normalize
            total = sum(mood_scores.values())
            if total > 0:
                mood_scores = {k: v/total for k, v in mood_scores.items()}
            
            return mood_scores
            
        except Exception as e:
            logger.error(f"Error in mood analysis: {e}")
            return {"neutral": 1.0}
    
    def _get_default_analysis(self) -> ComprehensiveAnalysis:
        """Return default analysis on error"""
        return ComprehensiveAnalysis(
            tempo=TempoAnalysis(
                bpm=120.0, confidence=0.0, tempo_stability=0.0,
                beat_times=[], downbeat_times=[],
                time_signature=TimeSignature.FOUR_FOUR, time_signature_confidence=0.0
            ),
            key=KeyAnalysis(
                key=MusicalKey.C_MAJOR, confidence=0.0, key_strength=0.0,
                chromagram=np.zeros(12), key_changes=[]
            ),
            harmony=HarmonicAnalysis(
                chord_progression=[], harmonic_rhythm=0.0,
                tension_curve=np.array([]), dissonance_levels=np.array([]),
                modulations=[]
            ),
            structure=StructuralAnalysis(
                segments=[], song_structure="UNKNOWN", repetition_score=0.0,
                novelty_curve=np.array([]), boundaries=[]
            ),
            spectral=SpectralAnalysis(
                spectral_centroid=np.array([1000.0]), spectral_rolloff=np.array([2000.0]),
                spectral_bandwidth=np.array([1500.0]), zero_crossing_rate=np.array([0.1]),
                mfcc=np.zeros((13, 100)), chroma=np.zeros((12, 100)),
                tonnetz=np.zeros((6, 100)), spectral_contrast=np.zeros((7, 100))
            ),
            rhythm=RhythmicAnalysis(
                onset_times=[], onset_strength=np.array([]),
                rhythmic_pattern=np.array([]), syncopation_score=0.0,
                rhythmic_complexity=0.0, pulse_clarity=0.0
            ),
            genre_prediction={"unknown": 1.0},
            overall_complexity=0.5,
            energy_profile=np.ones(100) * 0.5,
            mood_analysis={"neutral": 1.0}
        )

# Convenience functions
def analyze_audio_file(file_path: str) -> ComprehensiveAnalysis:
    """Analyze audio file and return comprehensive analysis"""
    try:
        import soundfile as sf
        
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        
        analyzer = ComprehensiveMusicalAnalyzer()
        return analyzer.analyze(audio, sr)
        
    except Exception as e:
        logger.error(f"Error analyzing audio file {file_path}: {e}")
        analyzer = ComprehensiveMusicalAnalyzer()
        return analyzer._get_default_analysis()

def analyze_audio_data(audio: np.ndarray, sr: int = 44100) -> ComprehensiveAnalysis:
    """Analyze audio data and return comprehensive analysis"""
    try:
        analyzer = ComprehensiveMusicalAnalyzer()
        return analyzer.analyze(audio, sr)
        
    except Exception as e:
        logger.error(f"Error analyzing audio data: {e}")
        analyzer = ComprehensiveMusicalAnalyzer()
        return analyzer._get_default_analysis()