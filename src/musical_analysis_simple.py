"""
Simple Musical Analysis System
Basic audio analysis without external dependencies
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MusicalKey(Enum):
    """Musical keys"""
    C_MAJOR = "C major"
    C_MINOR = "C minor"
    D_MAJOR = "D major" 
    D_MINOR = "D minor"
    E_MAJOR = "E major"
    E_MINOR = "E minor"
    F_MAJOR = "F major"
    F_MINOR = "F minor"
    G_MAJOR = "G major"
    G_MINOR = "G minor"
    A_MAJOR = "A major"
    A_MINOR = "A minor"
    B_MAJOR = "B major"
    B_MINOR = "B minor"

class TimeSignature(Enum):
    """Common time signatures"""
    FOUR_FOUR = "4/4"
    THREE_FOUR = "3/4"
    TWO_FOUR = "2/4"
    SIX_EIGHT = "6/8"

@dataclass
class SimpleTempoAnalysis:
    """Simple tempo analysis results"""
    bpm: float
    confidence: float
    tempo_stability: float
    beat_count: int
    time_signature: TimeSignature
    time_signature_confidence: float

@dataclass
class SimpleKeyAnalysis:
    """Simple key analysis results"""
    key: MusicalKey
    confidence: float
    key_strength: float

@dataclass
class SimpleSpectralAnalysis:
    """Simple spectral analysis results"""
    spectral_centroid: float
    brightness: float
    warmth: float
    harshness: float

@dataclass
class SimpleRhythmicAnalysis:
    """Simple rhythmic analysis results"""
    pulse_clarity: float
    rhythmic_activity: float
    syncopation_estimate: float

@dataclass
class SimpleStructuralAnalysis:
    """Simple structural analysis results"""
    estimated_segments: int
    repetitiveness: float
    dynamic_variation: float

@dataclass
class SimpleComprehensiveAnalysis:
    """Simple comprehensive musical analysis results"""
    tempo: SimpleTempoAnalysis
    key: SimpleKeyAnalysis
    spectral: SimpleSpectralAnalysis
    rhythm: SimpleRhythmicAnalysis
    structure: SimpleStructuralAnalysis
    genre_prediction: Dict[str, float]
    energy_level: float
    mood_analysis: Dict[str, float]

class SimpleMusicalAnalyzer:
    """Simple musical analyzer using basic signal processing"""
    
    def __init__(self):
        logger.info("SimpleMusicalAnalyzer initialized")
    
    def analyze(self, audio: np.ndarray, sr: int = 44100) -> SimpleComprehensiveAnalysis:
        """Perform simple comprehensive musical analysis"""
        try:
            logger.info("Starting simple musical analysis")
            
            # Basic audio properties
            duration = len(audio) / sr
            rms_energy = np.sqrt(np.mean(audio ** 2))
            peak_energy = np.max(np.abs(audio))
            
            # Tempo analysis
            tempo_analysis = self._analyze_tempo(audio, sr)
            
            # Key analysis
            key_analysis = self._analyze_key(audio, sr)
            
            # Spectral analysis
            spectral_analysis = self._analyze_spectral(audio, sr)
            
            # Rhythmic analysis
            rhythmic_analysis = self._analyze_rhythm(audio, sr)
            
            # Structural analysis
            structural_analysis = self._analyze_structure(audio, sr)
            
            # Genre prediction
            genre_prediction = self._predict_genre(spectral_analysis, tempo_analysis)
            
            # Energy level
            energy_level = min(1.0, rms_energy * 3.0)  # Normalized estimate
            
            # Mood analysis
            mood_analysis = self._analyze_mood(spectral_analysis, tempo_analysis, key_analysis)
            
            logger.info("Simple musical analysis completed")
            
            return SimpleComprehensiveAnalysis(
                tempo=tempo_analysis,
                key=key_analysis,
                spectral=spectral_analysis,
                rhythm=rhythmic_analysis,
                structure=structural_analysis,
                genre_prediction=genre_prediction,
                energy_level=energy_level,
                mood_analysis=mood_analysis
            )
            
        except Exception as e:
            logger.error(f"Error in simple musical analysis: {e}")
            return self._get_default_analysis()
    
    def _analyze_tempo(self, audio: np.ndarray, sr: int) -> SimpleTempoAnalysis:
        """Simple tempo analysis using onset detection"""
        try:
            # Simple onset detection using energy differences
            frame_size = 1024
            hop_size = 512
            
            # Calculate frame energies
            num_frames = (len(audio) - frame_size) // hop_size + 1
            frame_energies = []
            
            for i in range(num_frames):
                start = i * hop_size
                end = start + frame_size
                frame = audio[start:end] if end <= len(audio) else audio[start:]
                energy = np.sum(frame ** 2)
                frame_energies.append(energy)
            
            frame_energies = np.array(frame_energies)
            
            # Find peaks in energy (simple onset detection)
            if len(frame_energies) > 1:
                energy_diff = np.diff(frame_energies)
                onset_threshold = np.mean(energy_diff) + 1.5 * np.std(energy_diff)
                onsets = np.where(energy_diff > onset_threshold)[0]
            else:
                onsets = np.array([])
            
            # Estimate tempo from onset intervals
            if len(onsets) > 1:
                onset_times = onsets * hop_size / sr
                intervals = np.diff(onset_times)
                
                if len(intervals) > 0:
                    # Filter reasonable intervals (0.3 to 2 seconds between beats)
                    valid_intervals = intervals[(intervals > 0.3) & (intervals < 2.0)]
                    
                    if len(valid_intervals) > 0:
                        avg_interval = np.median(valid_intervals)
                        bpm = 60.0 / avg_interval
                        confidence = 1.0 - (np.std(valid_intervals) / (avg_interval + 1e-8))
                        confidence = max(0.0, min(1.0, confidence))
                        
                        # Stability based on interval consistency
                        stability = 1.0 - (np.std(valid_intervals) / (np.mean(valid_intervals) + 1e-8))
                        stability = max(0.0, min(1.0, stability))
                    else:
                        bpm, confidence, stability = 120.0, 0.3, 0.5
                else:
                    bpm, confidence, stability = 120.0, 0.3, 0.5
            else:
                bpm, confidence, stability = 120.0, 0.3, 0.5
            
            # Estimate time signature based on tempo
            if bpm < 100:
                time_sig = TimeSignature.THREE_FOUR
                time_sig_conf = 0.6
            else:
                time_sig = TimeSignature.FOUR_FOUR
                time_sig_conf = 0.8
            
            return SimpleTempoAnalysis(
                bpm=float(bpm),
                confidence=float(confidence),
                tempo_stability=float(stability),
                beat_count=len(onsets),
                time_signature=time_sig,
                time_signature_confidence=float(time_sig_conf)
            )
            
        except Exception as e:
            logger.error(f"Error in tempo analysis: {e}")
            return SimpleTempoAnalysis(
                bpm=120.0, confidence=0.0, tempo_stability=0.0,
                beat_count=0, time_signature=TimeSignature.FOUR_FOUR,
                time_signature_confidence=0.0
            )
    
    def _analyze_key(self, audio: np.ndarray, sr: int) -> SimpleKeyAnalysis:
        """Simple key analysis using pitch content estimation"""
        try:
            # Very simplified key detection based on frequency content
            # This is not accurate but provides a basic estimate
            
            # Calculate simple frequency content
            fft = np.fft.fft(audio[:min(len(audio), sr * 4)])  # Analyze first 4 seconds max
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            # Focus on musical frequency range (80Hz - 2000Hz)
            musical_range = (freqs >= 80) & (freqs <= 2000)
            musical_freqs = freqs[musical_range]
            musical_mags = magnitude[musical_range]
            
            if len(musical_mags) > 0:
                # Find dominant frequencies
                peak_indices = np.argsort(musical_mags)[-10:]  # Top 10 peaks
                dominant_freqs = musical_freqs[peak_indices]
                
                # Very simple key estimation based on dominant frequency
                if len(dominant_freqs) > 0:
                    avg_freq = np.mean(dominant_freqs)
                    
                    # Map frequency ranges to keys (very approximate)
                    if avg_freq < 150:
                        key = MusicalKey.C_MINOR
                    elif avg_freq < 250:
                        key = MusicalKey.G_MINOR
                    elif avg_freq < 350:
                        key = MusicalKey.C_MAJOR
                    elif avg_freq < 500:
                        key = MusicalKey.G_MAJOR
                    elif avg_freq < 800:
                        key = MusicalKey.D_MAJOR
                    else:
                        key = MusicalKey.A_MAJOR
                    
                    # Confidence based on peak clarity
                    max_mag = np.max(musical_mags)
                    avg_mag = np.mean(musical_mags)
                    confidence = min(1.0, (max_mag / (avg_mag + 1e-8)) * 0.2)
                    
                    # Key strength based on harmonic content
                    key_strength = min(1.0, max_mag / (np.sum(musical_mags) + 1e-8) * 10)
                else:
                    key = MusicalKey.C_MAJOR
                    confidence = 0.3
                    key_strength = 0.5
            else:
                key = MusicalKey.C_MAJOR
                confidence = 0.3
                key_strength = 0.5
            
            return SimpleKeyAnalysis(
                key=key,
                confidence=float(confidence),
                key_strength=float(key_strength)
            )
            
        except Exception as e:
            logger.error(f"Error in key analysis: {e}")
            return SimpleKeyAnalysis(
                key=MusicalKey.C_MAJOR,
                confidence=0.0,
                key_strength=0.0
            )
    
    def _analyze_spectral(self, audio: np.ndarray, sr: int) -> SimpleSpectralAnalysis:
        """Simple spectral analysis"""
        try:
            # Calculate basic spectral features
            fft = np.fft.fft(audio[:min(len(audio), sr)])  # Analyze first second
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            # Only consider positive frequencies
            positive_freq_idx = freqs > 0
            freqs = freqs[positive_freq_idx]
            magnitude = magnitude[positive_freq_idx]
            
            if len(magnitude) > 0 and np.sum(magnitude) > 0:
                # Spectral centroid (center of mass)
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                spectral_centroid_norm = min(1.0, spectral_centroid / 2000)  # Normalize to 0-1
                
                # Brightness (high frequency content)
                high_freq_idx = freqs > 2000
                if np.any(high_freq_idx):
                    brightness = np.sum(magnitude[high_freq_idx]) / np.sum(magnitude)
                else:
                    brightness = 0.0
                
                # Warmth (low-mid frequency content)
                warm_freq_idx = (freqs > 200) & (freqs < 1000)
                if np.any(warm_freq_idx):
                    warmth = np.sum(magnitude[warm_freq_idx]) / np.sum(magnitude)
                else:
                    warmth = 0.5
                
                # Harshness (very high frequency content)
                harsh_freq_idx = freqs > 5000
                if np.any(harsh_freq_idx):
                    harshness = np.sum(magnitude[harsh_freq_idx]) / np.sum(magnitude)
                else:
                    harshness = 0.0
            else:
                spectral_centroid_norm = 0.5
                brightness = 0.5
                warmth = 0.5
                harshness = 0.1
            
            return SimpleSpectralAnalysis(
                spectral_centroid=float(spectral_centroid_norm),
                brightness=float(brightness),
                warmth=float(warmth),
                harshness=float(harshness)
            )
            
        except Exception as e:
            logger.error(f"Error in spectral analysis: {e}")
            return SimpleSpectralAnalysis(
                spectral_centroid=0.5,
                brightness=0.5,
                warmth=0.5,
                harshness=0.1
            )
    
    def _analyze_rhythm(self, audio: np.ndarray, sr: int) -> SimpleRhythmicAnalysis:
        """Simple rhythmic analysis"""
        try:
            # Calculate pulse clarity using autocorrelation
            max_lag = sr * 2  # 2 seconds max
            if len(audio) > max_lag:
                autocorr = np.correlate(audio[:max_lag], audio[:max_lag], mode='full')
                autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
                
                if len(autocorr) > 1:
                    # Pulse clarity is the maximum autocorrelation (excluding zero lag)
                    pulse_clarity = np.max(autocorr[1:]) / (autocorr[0] + 1e-8)
                    pulse_clarity = max(0.0, min(1.0, pulse_clarity))
                else:
                    pulse_clarity = 0.5
            else:
                pulse_clarity = 0.5
            
            # Rhythmic activity based on amplitude variations
            if len(audio) > 100:
                # Calculate envelope
                window_size = sr // 20  # 50ms windows
                envelope = []
                for i in range(0, len(audio) - window_size, window_size):
                    window = audio[i:i + window_size]
                    envelope.append(np.sqrt(np.mean(window ** 2)))
                
                envelope = np.array(envelope)
                if len(envelope) > 1:
                    rhythmic_activity = np.std(envelope) / (np.mean(envelope) + 1e-8)
                    rhythmic_activity = min(1.0, rhythmic_activity)
                else:
                    rhythmic_activity = 0.5
            else:
                rhythmic_activity = 0.5
            
            # Simple syncopation estimate based on off-beat energy
            syncopation_estimate = min(1.0, rhythmic_activity * 0.7)
            
            return SimpleRhythmicAnalysis(
                pulse_clarity=float(pulse_clarity),
                rhythmic_activity=float(rhythmic_activity),
                syncopation_estimate=float(syncopation_estimate)
            )
            
        except Exception as e:
            logger.error(f"Error in rhythmic analysis: {e}")
            return SimpleRhythmicAnalysis(
                pulse_clarity=0.5,
                rhythmic_activity=0.5,
                syncopation_estimate=0.3
            )
    
    def _analyze_structure(self, audio: np.ndarray, sr: int) -> SimpleStructuralAnalysis:
        """Simple structural analysis"""
        try:
            duration = len(audio) / sr
            
            # Estimate segments based on duration
            if duration < 30:
                estimated_segments = 2
            elif duration < 120:
                estimated_segments = 4
            elif duration < 240:
                estimated_segments = 6
            else:
                estimated_segments = 8
            
            # Calculate repetitiveness using simple self-similarity
            chunk_size = sr * 10  # 10-second chunks
            if len(audio) > chunk_size * 2:
                chunks = []
                for i in range(0, len(audio) - chunk_size, chunk_size):
                    chunk = audio[i:i + chunk_size]
                    chunks.append(chunk)
                
                if len(chunks) >= 2:
                    similarities = []
                    for i in range(len(chunks)):
                        for j in range(i + 1, len(chunks)):
                            corr = np.corrcoef(chunks[i], chunks[j])[0, 1]
                            if not np.isnan(corr):
                                similarities.append(abs(corr))
                    
                    if similarities:
                        repetitiveness = np.mean(similarities)
                    else:
                        repetitiveness = 0.5
                else:
                    repetitiveness = 0.5
            else:
                repetitiveness = 0.5
            
            # Dynamic variation based on amplitude changes
            window_size = sr  # 1-second windows
            if len(audio) > window_size * 2:
                amplitudes = []
                for i in range(0, len(audio) - window_size, window_size):
                    window = audio[i:i + window_size]
                    amplitude = np.sqrt(np.mean(window ** 2))
                    amplitudes.append(amplitude)
                
                amplitudes = np.array(amplitudes)
                if len(amplitudes) > 1 and np.mean(amplitudes) > 0:
                    dynamic_variation = np.std(amplitudes) / np.mean(amplitudes)
                    dynamic_variation = min(1.0, dynamic_variation)
                else:
                    dynamic_variation = 0.5
            else:
                dynamic_variation = 0.5
            
            return SimpleStructuralAnalysis(
                estimated_segments=estimated_segments,
                repetitiveness=float(repetitiveness),
                dynamic_variation=float(dynamic_variation)
            )
            
        except Exception as e:
            logger.error(f"Error in structural analysis: {e}")
            return SimpleStructuralAnalysis(
                estimated_segments=4,
                repetitiveness=0.5,
                dynamic_variation=0.5
            )
    
    def _predict_genre(self, spectral: SimpleSpectralAnalysis, tempo: SimpleTempoAnalysis) -> Dict[str, float]:
        """Simple genre prediction based on features"""
        try:
            scores = {}
            
            # Initialize with base scores
            genres = ["pop", "rock", "classical", "electronic", "jazz", "blues"]
            for genre in genres:
                scores[genre] = 0.1
            
            # Tempo-based scoring
            bpm = tempo.bpm
            if 60 <= bpm <= 80:
                scores["blues"] += 0.3
                scores["classical"] += 0.2
            elif 80 <= bpm <= 100:
                scores["jazz"] += 0.3
                scores["classical"] += 0.1
            elif 100 <= bpm <= 130:
                scores["pop"] += 0.3
                scores["rock"] += 0.2
            elif 130 <= bpm <= 180:
                scores["electronic"] += 0.3
                scores["rock"] += 0.1
            
            # Spectral-based scoring
            if spectral.brightness > 0.6:
                scores["electronic"] += 0.2
                scores["rock"] += 0.1
            elif spectral.warmth > 0.6:
                scores["jazz"] += 0.2
                scores["blues"] += 0.1
            
            if spectral.harshness > 0.3:
                scores["rock"] += 0.2
                scores["electronic"] += 0.1
            
            # Normalize scores
            total = sum(scores.values())
            if total > 0:
                scores = {k: v/total for k, v in scores.items()}
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in genre prediction: {e}")
            return {"pop": 1.0}
    
    def _analyze_mood(self, spectral: SimpleSpectralAnalysis, tempo: SimpleTempoAnalysis, 
                     key: SimpleKeyAnalysis) -> Dict[str, float]:
        """Simple mood analysis"""
        try:
            mood_scores = {
                "happy": 0.0,
                "sad": 0.0,
                "energetic": 0.0,
                "calm": 0.0,
                "aggressive": 0.0,
                "peaceful": 0.0
            }
            
            # Tempo influence
            bpm = tempo.bpm
            if bpm > 120:
                mood_scores["energetic"] += 0.4
                mood_scores["happy"] += 0.2
            else:
                mood_scores["calm"] += 0.3
                mood_scores["peaceful"] += 0.2
            
            # Key influence (simplified)
            if "minor" in key.key.value.lower():
                mood_scores["sad"] += 0.3
            else:
                mood_scores["happy"] += 0.3
            
            # Spectral influence
            if spectral.brightness > 0.6:
                mood_scores["energetic"] += 0.2
                mood_scores["happy"] += 0.1
            
            if spectral.harshness > 0.4:
                mood_scores["aggressive"] += 0.3
            
            if spectral.warmth > 0.6:
                mood_scores["peaceful"] += 0.2
                mood_scores["calm"] += 0.1
            
            # Normalize
            total = sum(mood_scores.values())
            if total > 0:
                mood_scores = {k: v/total for k, v in mood_scores.items()}
            
            return mood_scores
            
        except Exception as e:
            logger.error(f"Error in mood analysis: {e}")
            return {"neutral": 1.0}
    
    def _get_default_analysis(self) -> SimpleComprehensiveAnalysis:
        """Return default analysis on error"""
        return SimpleComprehensiveAnalysis(
            tempo=SimpleTempoAnalysis(
                bpm=120.0, confidence=0.0, tempo_stability=0.0,
                beat_count=0, time_signature=TimeSignature.FOUR_FOUR,
                time_signature_confidence=0.0
            ),
            key=SimpleKeyAnalysis(
                key=MusicalKey.C_MAJOR, confidence=0.0, key_strength=0.0
            ),
            spectral=SimpleSpectralAnalysis(
                spectral_centroid=0.5, brightness=0.5, warmth=0.5, harshness=0.1
            ),
            rhythm=SimpleRhythmicAnalysis(
                pulse_clarity=0.5, rhythmic_activity=0.5, syncopation_estimate=0.3
            ),
            structure=SimpleStructuralAnalysis(
                estimated_segments=4, repetitiveness=0.5, dynamic_variation=0.5
            ),
            genre_prediction={"pop": 1.0},
            energy_level=0.5,
            mood_analysis={"neutral": 1.0}
        )

# Convenience functions
def analyze_audio_data(audio: np.ndarray, sr: int = 44100) -> SimpleComprehensiveAnalysis:
    """Quick audio analysis function"""
    try:
        analyzer = SimpleMusicalAnalyzer()
        return analyzer.analyze(audio, sr)
        
    except Exception as e:
        logger.error(f"Error analyzing audio data: {e}")
        analyzer = SimpleMusicalAnalyzer()
        return analyzer._get_default_analysis()