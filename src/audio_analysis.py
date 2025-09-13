"""
Professional Audio Analysis Engine for Son1k v3.0
Basado en el sistema de Son1kVers3 - Análisis completo de audio
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available - using basic analysis")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.error("soundfile not available - audio analysis disabled")

try:
    from scipy import signal
    from scipy.stats import mode
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - advanced analysis disabled")

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Professional audio analysis engine"""
    
    def __init__(self, sample_rate: int = 32000):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_length = 2048
        self.fft_size = 4096
        
        # Key detection profiles (Krumhansl-Schmuckler)
        self.major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Note names
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Complete audio analysis pipeline
        Returns comprehensive analysis dictionary
        """
        logger.info(f"Starting audio analysis: {file_path}")
        
        if not SOUNDFILE_AVAILABLE:
            return self._basic_analysis(file_path)
        
        try:
            # Load audio file
            audio, orig_sr = sf.read(file_path)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if LIBROSA_AVAILABLE and orig_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
            
            # Basic file info
            file_info = self._get_file_info(file_path, audio, orig_sr)
            
            # Core analysis
            tempo_analysis = self._analyze_tempo(audio)
            key_analysis = self._analyze_key(audio)
            energy_analysis = self._analyze_energy(audio)
            vocal_analysis = self._analyze_vocals(audio)
            spectral_analysis = self._analyze_spectral(audio)
            
            # Compile results
            analysis = {
                "file_info": file_info,
                "tempo": tempo_analysis,
                "key_guess": key_analysis,
                "energy_structure": energy_analysis,
                "vocals": vocal_analysis,
                "spectral": spectral_analysis,
                "summary": self._generate_summary(
                    tempo_analysis, key_analysis, energy_analysis, vocal_analysis
                )
            }
            
            logger.info(f"Analysis complete: {tempo_analysis['bpm']:.1f}bpm, {key_analysis['root']}{key_analysis['scale']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return self._basic_analysis(file_path)

    def _basic_analysis(self, file_path: str) -> Dict[str, Any]:
        """Fallback basic analysis when libraries are missing"""
        try:
            if SOUNDFILE_AVAILABLE:
                info = sf.info(file_path)
                duration = info.duration
                sample_rate = info.samplerate
            else:
                # Ultra basic fallback
                file_size = Path(file_path).stat().st_size
                estimated_duration = file_size / (44100 * 2 * 2)  # Rough estimate
                duration = max(1.0, estimated_duration)
                sample_rate = 44100
            
            return {
                "file_info": {
                    "filename": Path(file_path).name,
                    "duration_s": duration,
                    "samplerate": sample_rate
                },
                "tempo": {"bpm": 120.0, "confidence": 0.0},
                "key_guess": {"root": "C", "scale": "major", "confidence": 0.0},
                "energy_structure": {"high_energy_percentage": 0.5},
                "vocals": {"has_vocals": False, "vocal_probability": 0.0},
                "spectral": {"brightness": 0.5},
                "summary": {
                    "tempo_category": "moderate",
                    "energy_level": "medium",
                    "characteristics": ["unknown"]
                }
            }
        except Exception as e:
            logger.error(f"Basic analysis failed: {e}")
            return {"error": str(e)}

    def _get_file_info(self, file_path: str, audio: np.ndarray, orig_sr: int) -> Dict[str, Any]:
        """Get basic file information"""
        file_path = Path(file_path)
        return {
            "filename": file_path.name,
            "format": file_path.suffix.lower(),
            "duration_s": len(audio) / self.sample_rate,
            "samplerate": orig_sr,
            "processed_samplerate": self.sample_rate,
            "channels": 1,  # We convert to mono
            "samples": len(audio),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0,
            "peak_level": float(np.max(np.abs(audio))),
            "rms_level": float(np.sqrt(np.mean(audio ** 2)))
        }

    def _analyze_tempo(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze tempo using multiple methods"""
        if not LIBROSA_AVAILABLE:
            return {"bpm": 120.0, "confidence": 0.0, "method": "fallback"}
        
        try:
            # Method 1: Beat tracking
            onset_envelope = librosa.onset.onset_strength(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )
            bpm_beats, beats = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Method 2: Onset detection
            onsets = librosa.onset.onset_detect(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length, units='time'
            )
            
            if len(onsets) > 2:
                onset_intervals = np.diff(onsets)
                avg_interval = np.median(onset_intervals)
                bpm_onsets = 60.0 / avg_interval if avg_interval > 0 else bpm_beats
            else:
                bpm_onsets = bpm_beats
            
            # Choose most reliable estimate
            bpms = [bpm_beats, bpm_onsets]
            bpms = [bpm for bpm in bpms if 60 <= bpm <= 200]  # Filter reasonable range
            
            if bpms:
                final_bpm = np.median(bpms)
            else:
                final_bpm = bpm_beats
            
            # Confidence based on beat consistency
            beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=self.hop_length)
            if len(beat_times) > 2:
                beat_intervals = np.diff(beat_times)
                tempo_consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
                tempo_consistency = max(0.0, min(1.0, tempo_consistency))
            else:
                tempo_consistency = 0.5
            
            return {
                "bpm": float(final_bpm),
                "confidence": float(tempo_consistency),
                "method": "librosa",
                "onset_count": len(onsets),
                "beat_positions": beat_times.tolist() if len(beat_times) < 100 else []
            }
            
        except Exception as e:
            logger.warning(f"Tempo analysis failed: {e}")
            return {"bpm": 120.0, "confidence": 0.0, "method": "error"}

    def _analyze_key(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze musical key using chromagram"""
        if not LIBROSA_AVAILABLE:
            return {"root": "C", "scale": "major", "confidence": 0.0, "method": "fallback"}
        
        try:
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            )
            
            # Average across time
            chroma_profile = np.mean(chroma, axis=1)
            chroma_profile = chroma_profile / np.sum(chroma_profile)
            
            # Calculate correlations for each key
            major_correlations = []
            minor_correlations = []
            
            for shift in range(12):
                shifted_chroma = np.roll(chroma_profile, shift)
                
                major_corr = np.corrcoef(shifted_chroma, self.major_profile)[0, 1]
                minor_corr = np.corrcoef(shifted_chroma, self.minor_profile)[0, 1]
                
                major_correlations.append(major_corr if not np.isnan(major_corr) else 0)
                minor_correlations.append(minor_corr if not np.isnan(minor_corr) else 0)
            
            # Find best matches
            best_major_idx = np.argmax(major_correlations)
            best_minor_idx = np.argmax(minor_correlations)
            best_major_corr = major_correlations[best_major_idx]
            best_minor_corr = minor_correlations[best_minor_idx]
            
            # Choose major or minor
            if best_major_corr > best_minor_corr:
                key_root = self.note_names[best_major_idx]
                key_scale = "major"
                confidence = best_major_corr
            else:
                key_root = self.note_names[best_minor_idx]
                key_scale = "minor"
                confidence = best_minor_corr
            
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                "root": key_root,
                "scale": key_scale,
                "confidence": float(confidence),
                "method": "chroma_correlation",
                "chroma_profile": chroma_profile.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Key analysis failed: {e}")
            return {"root": "C", "scale": "major", "confidence": 0.0, "method": "error"}

    def _analyze_energy(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze energy structure and dynamics"""
        try:
            if LIBROSA_AVAILABLE:
                # RMS energy
                rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
                times = librosa.frames_to_time(
                    np.arange(len(rms)), sr=self.sample_rate, hop_length=self.hop_length
                )
            else:
                # Basic RMS calculation
                frame_size = self.hop_length
                rms = []
                for i in range(0, len(audio) - frame_size, frame_size):
                    frame = audio[i:i + frame_size]
                    rms.append(np.sqrt(np.mean(frame ** 2)))
                rms = np.array(rms)
                times = np.arange(len(rms)) * frame_size / self.sample_rate
            
            # Energy statistics
            energy_stats = {
                "mean": float(np.mean(rms)),
                "std": float(np.std(rms)),
                "max": float(np.max(rms)),
                "min": float(np.min(rms)),
                "dynamic_range": float(np.max(rms) - np.min(rms))
            }
            
            # High energy percentage
            energy_threshold = np.percentile(rms, 60)
            high_energy_percentage = float(np.mean(rms > energy_threshold))
            
            return {
                "statistics": energy_stats,
                "high_energy_percentage": high_energy_percentage,
                "energy_curve": rms.tolist() if len(rms) < 1000 else rms[::10].tolist(),
                "energy_times": times.tolist() if len(times) < 1000 else times[::10].tolist()
            }
            
        except Exception as e:
            logger.warning(f"Energy analysis failed: {e}")
            return {"statistics": {}, "high_energy_percentage": 0.5}

    def _analyze_vocals(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect vocal presence using spectral features"""
        try:
            if LIBROSA_AVAILABLE:
                # Spectral centroid (frequency brightness)
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio, sr=self.sample_rate, hop_length=self.hop_length
                )[0]
                
                # MFCCs for vocal characteristics
                mfccs = librosa.feature.mfcc(
                    y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length
                )
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(
                    y=audio, hop_length=self.hop_length
                )[0]
                
                # Vocal indicators
                centroid_mean = np.mean(spectral_centroid)
                vocal_freq_indicator = 500 < centroid_mean < 4000  # Vocal frequency range
                
                mfcc_variance = np.var(mfccs[1:5], axis=1)
                mfcc_indicator = np.mean(mfcc_variance) > 0.1
                
                zcr_mean = np.mean(zcr)
                zcr_indicator = 0.01 < zcr_mean < 0.3
                
                # Harmonic vs percussive
                harmonic, percussive = librosa.effects.hpss(audio)
                harmonic_strength = np.mean(np.abs(harmonic))
                percussive_strength = np.mean(np.abs(percussive))
                
                if harmonic_strength + percussive_strength > 0:
                    harmonic_ratio = harmonic_strength / (harmonic_strength + percussive_strength)
                    harmonic_indicator = harmonic_ratio > 0.6
                else:
                    harmonic_ratio = 0.5
                    harmonic_indicator = False
                
                # Combine indicators
                indicators = [vocal_freq_indicator, mfcc_indicator, zcr_indicator, harmonic_indicator]
                weights = [0.3, 0.3, 0.2, 0.2]
                vocal_probability = sum(w * i for w, i in zip(weights, indicators))
                
            else:
                # Basic vocal detection
                rms = np.sqrt(np.mean(audio ** 2))
                vocal_probability = min(rms * 2, 1.0)  # Simple heuristic
                harmonic_ratio = 0.5
            
            has_vocals = vocal_probability > 0.5
            
            return {
                "has_vocals": bool(has_vocals),
                "vocal_probability": float(vocal_probability),
                "features": {
                    "harmonic_ratio": float(harmonic_ratio) if 'harmonic_ratio' in locals() else 0.5
                },
                "method": "librosa" if LIBROSA_AVAILABLE else "basic"
            }
            
        except Exception as e:
            logger.warning(f"Vocal analysis failed: {e}")
            return {"has_vocals": False, "vocal_probability": 0.0, "method": "error"}

    def _analyze_spectral(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral characteristics"""
        try:
            if LIBROSA_AVAILABLE:
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                    y=audio, sr=self.sample_rate, hop_length=self.hop_length
                ))
                
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
                    y=audio, sr=self.sample_rate, hop_length=self.hop_length
                ))
                
                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
                    y=audio, sr=self.sample_rate, hop_length=self.hop_length
                ))
                
                brightness = float(spectral_centroid / (self.sample_rate / 2))
                
            else:
                # Basic spectral analysis using FFT
                fft = np.fft.fft(audio)
                magnitude = np.abs(fft[:len(fft)//2])
                frequencies = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(magnitude)]
                
                # Spectral centroid (weighted average frequency)
                spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 1000
                brightness = float(spectral_centroid / (self.sample_rate / 2))
                spectral_bandwidth = spectral_rolloff = spectral_centroid
            
            return {
                "centroid_hz": float(spectral_centroid),
                "bandwidth_hz": float(spectral_bandwidth),
                "rolloff_hz": float(spectral_rolloff),
                "brightness": float(brightness),
                "method": "librosa" if LIBROSA_AVAILABLE else "fft"
            }
            
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            return {"brightness": 0.5, "method": "error"}

    def _generate_summary(self, tempo_analysis: Dict, key_analysis: Dict, 
                         energy_analysis: Dict, vocal_analysis: Dict) -> Dict[str, Any]:
        """Generate high-level summary"""
        
        # Categorize tempo
        bpm = tempo_analysis.get("bpm", 120)
        if bpm < 80:
            tempo_category = "slow"
        elif bpm < 120:
            tempo_category = "moderate"
        elif bpm < 160:
            tempo_category = "fast"
        else:
            tempo_category = "very_fast"
        
        # Energy level
        energy_stats = energy_analysis.get("statistics", {})
        energy_level = "medium"
        if energy_stats:
            dynamic_range = energy_stats.get("dynamic_range", 0.1)
            if dynamic_range > 0.2:
                energy_level = "high"
            elif dynamic_range < 0.05:
                energy_level = "low"
        
        # Characteristics
        characteristics = []
        
        if vocal_analysis.get("has_vocals", False):
            characteristics.append("vocal")
        
        if energy_level == "high":
            characteristics.append("energetic")
        elif energy_level == "low":
            characteristics.append("calm")
        
        if tempo_category in ["fast", "very_fast"]:
            characteristics.append("upbeat")
        elif tempo_category == "slow":
            characteristics.append("mellow")
        
        return {
            "tempo_category": tempo_category,
            "energy_level": energy_level,
            "key_signature": f"{key_analysis.get('root', 'C')} {key_analysis.get('scale', 'major')}",
            "has_vocals": vocal_analysis.get("has_vocals", False),
            "characteristics": characteristics,
            "overall_confidence": float(np.mean([
                tempo_analysis.get("confidence", 0),
                key_analysis.get("confidence", 0),
                vocal_analysis.get("vocal_probability", 0)
            ]))
        }

# Convenience function
def analyze_demo(file_path: str, sample_rate: int = 32000) -> Dict[str, Any]:
    """Convenience function to analyze a demo file"""
    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    return analyzer.analyze_audio_file(file_path)

# Export for backward compatibility
AudioAnalysisEngine = AudioAnalyzer
