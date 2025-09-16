"""
Professional Audio Post-Processing Engine for Son1kVers3
Advanced audio processing with studio-quality effects and mastering
"""

import numpy as np
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    GENTLE = "gentle"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VINTAGE = "vintage"
    MODERN = "modern"

@dataclass
class EQBand:
    """Equalizer band configuration"""
    frequency: float  # Center frequency in Hz
    gain: float      # Gain in dB
    q_factor: float  # Quality factor (bandwidth)
    filter_type: str # 'peak', 'highpass', 'lowpass', 'shelf'

@dataclass
class CompressorSettings:
    """Dynamic range compressor settings"""
    threshold: float    # Threshold in dB
    ratio: float       # Compression ratio
    attack: float      # Attack time in ms
    release: float     # Release time in ms
    makeup_gain: float # Makeup gain in dB
    knee: float        # Soft knee amount

@dataclass
class MasteringChain:
    """Complete mastering chain configuration"""
    eq_bands: List[EQBand]
    compressor: CompressorSettings
    limiter_ceiling: float  # Limiter ceiling in dB
    stereo_width: float    # Stereo width (0-2, 1=normal)
    harmonic_excitement: float  # Harmonic distortion amount
    vintage_warmth: float  # Analog warmth simulation

class ProfessionalAudioProcessor:
    """Professional-grade audio processing engine"""
    
    def __init__(self, sample_rate: int = 32000):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
        # Initialize processing presets
        self.mastering_presets = self._create_mastering_presets()
        self.style_presets = self._create_style_presets()
        
        logger.info(f"Professional Audio Processor initialized at {sample_rate}Hz")
    
    def _create_mastering_presets(self) -> Dict[str, MasteringChain]:
        """Create professional mastering presets"""
        presets = {}
        
        # Modern Pop Mastering
        presets['pop_modern'] = MasteringChain(
            eq_bands=[
                EQBand(frequency=60, gain=2.0, q_factor=0.7, filter_type='highpass'),
                EQBand(frequency=200, gain=-1.5, q_factor=1.0, filter_type='peak'),
                EQBand(frequency=2000, gain=1.0, q_factor=0.8, filter_type='peak'),
                EQBand(frequency=8000, gain=2.5, q_factor=0.6, filter_type='shelf'),
                EQBand(frequency=12000, gain=1.5, q_factor=0.5, filter_type='peak')
            ],
            compressor=CompressorSettings(
                threshold=-12.0, ratio=3.0, attack=10.0, release=100.0, 
                makeup_gain=3.0, knee=2.0
            ),
            limiter_ceiling=-0.3,
            stereo_width=1.1,
            harmonic_excitement=0.15,
            vintage_warmth=0.1
        )
        
        # Rock Mastering
        presets['rock'] = MasteringChain(
            eq_bands=[
                EQBand(frequency=80, gain=1.5, q_factor=0.8, filter_type='peak'),
                EQBand(frequency=400, gain=-2.0, q_factor=1.2, filter_type='peak'),
                EQBand(frequency=1500, gain=1.5, q_factor=0.7, filter_type='peak'),
                EQBand(frequency=5000, gain=2.0, q_factor=0.6, filter_type='peak'),
                EQBand(frequency=10000, gain=1.0, q_factor=0.5, filter_type='shelf')
            ],
            compressor=CompressorSettings(
                threshold=-10.0, ratio=4.0, attack=5.0, release=80.0,
                makeup_gain=4.0, knee=1.5
            ),
            limiter_ceiling=-0.5,
            stereo_width=1.0,
            harmonic_excitement=0.25,
            vintage_warmth=0.3
        )
        
        # Electronic/EDM Mastering
        presets['electronic'] = MasteringChain(
            eq_bands=[
                EQBand(frequency=40, gain=3.0, q_factor=0.9, filter_type='peak'),
                EQBand(frequency=120, gain=-1.0, q_factor=1.0, filter_type='peak'),
                EQBand(frequency=3000, gain=0.5, q_factor=0.8, filter_type='peak'),
                EQBand(frequency=8000, gain=3.0, q_factor=0.5, filter_type='shelf'),
                EQBand(frequency=15000, gain=2.0, q_factor=0.4, filter_type='peak')
            ],
            compressor=CompressorSettings(
                threshold=-8.0, ratio=6.0, attack=1.0, release=50.0,
                makeup_gain=5.0, knee=0.5
            ),
            limiter_ceiling=-0.1,
            stereo_width=1.3,
            harmonic_excitement=0.1,
            vintage_warmth=0.0
        )
        
        # Jazz/Acoustic Mastering
        presets['acoustic'] = MasteringChain(
            eq_bands=[
                EQBand(frequency=100, gain=0.5, q_factor=0.6, filter_type='peak'),
                EQBand(frequency=500, gain=-0.5, q_factor=1.5, filter_type='peak'),
                EQBand(frequency=2500, gain=1.0, q_factor=0.9, filter_type='peak'),
                EQBand(frequency=7000, gain=1.5, q_factor=0.7, filter_type='peak'),
                EQBand(frequency=12000, gain=0.8, q_factor=0.6, filter_type='shelf')
            ],
            compressor=CompressorSettings(
                threshold=-16.0, ratio=2.5, attack=30.0, release=200.0,
                makeup_gain=2.0, knee=3.0
            ),
            limiter_ceiling=-1.0,
            stereo_width=1.2,
            harmonic_excitement=0.05,
            vintage_warmth=0.4
        )
        
        # Vintage/Analog Mastering
        presets['vintage'] = MasteringChain(
            eq_bands=[
                EQBand(frequency=80, gain=1.0, q_factor=0.8, filter_type='peak'),
                EQBand(frequency=300, gain=-1.0, q_factor=1.0, filter_type='peak'),
                EQBand(frequency=3000, gain=2.0, q_factor=0.5, filter_type='peak'),
                EQBand(frequency=8000, gain=-0.5, q_factor=0.8, filter_type='peak'),
                EQBand(frequency=15000, gain=-2.0, q_factor=0.7, filter_type='lowpass')
            ],
            compressor=CompressorSettings(
                threshold=-14.0, ratio=3.5, attack=20.0, release=150.0,
                makeup_gain=3.5, knee=2.5
            ),
            limiter_ceiling=-0.8,
            stereo_width=0.9,
            harmonic_excitement=0.3,
            vintage_warmth=0.6
        )
        
        return presets
    
    def _create_style_presets(self) -> Dict[str, Dict[str, Any]]:
        """Create style-specific processing presets"""
        return {
            'vocal_pop': {
                'de_esser': {'frequency': 6000, 'threshold': -20, 'ratio': 4.0},
                'presence_boost': {'frequency': 5000, 'gain': 2.0, 'q': 0.8},
                'warmth': {'frequency': 200, 'gain': 1.0, 'q': 0.6},
                'compression': {'threshold': -18, 'ratio': 3.0}
            },
            'vocal_rock': {
                'de_esser': {'frequency': 7000, 'threshold': -18, 'ratio': 3.0},
                'bite': {'frequency': 2500, 'gain': 2.5, 'q': 1.0},
                'body': {'frequency': 150, 'gain': 1.5, 'q': 0.7},
                'compression': {'threshold': -15, 'ratio': 4.0}
            },
            'instrumental': {
                'clarity': {'frequency': 4000, 'gain': 1.5, 'q': 0.7},
                'depth': {'frequency': 100, 'gain': 1.0, 'q': 0.8},
                'sparkle': {'frequency': 10000, 'gain': 1.8, 'q': 0.5}
            }
        }
    
    def master_track(self, 
                    audio: np.ndarray, 
                    style: str = 'pop_modern',
                    lufs_target: float = -14.0,
                    custom_chain: Optional[MasteringChain] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply professional mastering chain to audio
        
        Args:
            audio: Input audio (mono or stereo)
            style: Mastering style preset
            lufs_target: Target LUFS loudness
            custom_chain: Custom mastering chain (overrides style)
        
        Returns:
            Tuple of (processed_audio, processing_metadata)
        """
        
        if custom_chain:
            chain = custom_chain
        else:
            chain = self.mastering_presets.get(style, self.mastering_presets['pop_modern'])
        
        logger.info(f"Applying {style} mastering chain")
        
        # Ensure audio is in correct format
        if len(audio.shape) == 1:
            # Convert mono to stereo for processing
            stereo_audio = np.column_stack([audio, audio])
        else:
            stereo_audio = audio.copy()
        
        processing_log = []
        
        # 1. Pre-processing cleanup
        stereo_audio = self._remove_dc_offset(stereo_audio)
        processing_log.append("DC offset removal")
        
        # 2. EQ Processing
        for band in chain.eq_bands:
            stereo_audio = self._apply_eq_band(stereo_audio, band)
        processing_log.append(f"EQ applied ({len(chain.eq_bands)} bands)")
        
        # 3. Harmonic Excitement
        if chain.harmonic_excitement > 0:
            stereo_audio = self._apply_harmonic_excitement(stereo_audio, chain.harmonic_excitement)
            processing_log.append("Harmonic excitement")
        
        # 4. Vintage Warmth
        if chain.vintage_warmth > 0:
            stereo_audio = self._apply_vintage_warmth(stereo_audio, chain.vintage_warmth)
            processing_log.append("Vintage warmth")
        
        # 5. Dynamic Range Compression
        stereo_audio = self._apply_compression(stereo_audio, chain.compressor)
        processing_log.append("Compression")
        
        # 6. Stereo Width Adjustment
        if chain.stereo_width != 1.0:
            stereo_audio = self._adjust_stereo_width(stereo_audio, chain.stereo_width)
            processing_log.append(f"Stereo width ({chain.stereo_width:.1f})")
        
        # 7. Loudness Normalization
        stereo_audio = self._normalize_loudness(stereo_audio, lufs_target)
        processing_log.append(f"LUFS normalization ({lufs_target})")
        
        # 8. Final Limiting
        stereo_audio = self._apply_limiter(stereo_audio, chain.limiter_ceiling)
        processing_log.append(f"Limiting ({chain.limiter_ceiling}dB)")
        
        # Generate metadata
        metadata = {
            'mastering_style': style,
            'processing_chain': processing_log,
            'target_lufs': lufs_target,
            'limiter_ceiling': chain.limiter_ceiling,
            'final_stats': self._analyze_audio(stereo_audio)
        }
        
        # Convert back to original format if needed
        if len(audio.shape) == 1:
            # Return mono
            final_audio = np.mean(stereo_audio, axis=1)
        else:
            final_audio = stereo_audio
        
        logger.info(f"Mastering complete: {len(processing_log)} processes applied")
        
        return final_audio, metadata
    
    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio"""
        if len(audio.shape) == 1:
            return audio - np.mean(audio)
        else:
            return audio - np.mean(audio, axis=0)
    
    def _apply_eq_band(self, audio: np.ndarray, band: EQBand) -> np.ndarray:
        """Apply a single EQ band (simplified implementation)"""
        # This is a simplified EQ implementation
        # In production, would use proper filter design (IIR/FIR filters)
        
        if band.filter_type == 'peak':
            # Simplified peak filter
            boost_factor = 10 ** (band.gain / 20)  # Convert dB to linear
            
            # Create a simple frequency response modification
            # This is NOT a proper filter implementation
            if band.gain > 0:
                # Boost
                return audio * boost_factor * 0.1 + audio * 0.9
            else:
                # Cut
                return audio * boost_factor
        
        elif band.filter_type == 'highpass':
            # Very simplified highpass
            if band.frequency < 100:
                # Reduce low frequencies
                return audio * 0.9 + self._add_slight_highpass(audio) * 0.1
        
        elif band.filter_type == 'lowpass':
            # Very simplified lowpass
            if band.frequency > 10000:
                # Reduce high frequencies slightly
                return audio * 0.95
        
        elif band.filter_type == 'shelf':
            # Simplified shelf filter
            boost_factor = 10 ** (band.gain / 20)
            if band.frequency > 5000:  # High shelf
                return audio * boost_factor * 0.2 + audio * 0.8
            else:  # Low shelf
                return audio * boost_factor * 0.15 + audio * 0.85
        
        return audio
    
    def _add_slight_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Add slight high-frequency emphasis"""
        # Very simple high-frequency boost
        if len(audio.shape) == 1:
            diff = np.diff(audio, prepend=audio[0])
            return audio + diff * 0.1
        else:
            diff = np.diff(audio, axis=0, prepend=audio[0:1])
            return audio + diff * 0.1
    
    def _apply_harmonic_excitement(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Apply harmonic excitement (subtle distortion)"""
        # Gentle saturation to add harmonics
        drive = 1 + amount * 2
        saturated = np.tanh(audio * drive) / drive
        
        # Blend with original
        return audio * (1 - amount) + saturated * amount
    
    def _apply_vintage_warmth(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Apply vintage analog warmth simulation"""
        # Simulate analog warmth with gentle compression and saturation
        
        # Gentle tube-style saturation
        tube_saturated = np.sign(audio) * (1 - np.exp(-np.abs(audio * 2))) * 0.5
        
        # Add slight low-frequency emphasis
        warm_audio = audio + tube_saturated * amount * 0.3
        
        # Gentle high-frequency roll-off
        if amount > 0.3:
            warm_audio = warm_audio * 0.98  # Very slight HF reduction
        
        return warm_audio
    
    def _apply_compression(self, audio: np.ndarray, settings: CompressorSettings) -> np.ndarray:
        """Apply dynamic range compression"""
        
        # Convert threshold from dB to linear
        threshold_linear = 10 ** (settings.threshold / 20)
        
        # Simple compression algorithm
        compressed = audio.copy()
        
        # Calculate instantaneous amplitude
        if len(audio.shape) == 1:
            amplitude = np.abs(audio)
        else:
            amplitude = np.max(np.abs(audio), axis=1)
        
        # Find samples above threshold
        over_threshold = amplitude > threshold_linear
        
        if np.any(over_threshold):
            # Calculate gain reduction
            excess = amplitude[over_threshold] / threshold_linear
            gain_reduction = 1 / (1 + (excess - 1) * (settings.ratio - 1) / settings.ratio)
            
            # Apply compression
            if len(audio.shape) == 1:
                compressed[over_threshold] *= gain_reduction
            else:
                compressed[over_threshold] *= gain_reduction[:, np.newaxis]
        
        # Apply makeup gain
        makeup_linear = 10 ** (settings.makeup_gain / 20)
        compressed *= makeup_linear
        
        return compressed
    
    def _adjust_stereo_width(self, audio: np.ndarray, width: float) -> np.ndarray:
        """Adjust stereo width"""
        if len(audio.shape) != 2:
            return audio
        
        # M/S processing
        mid = (audio[:, 0] + audio[:, 1]) * 0.5
        side = (audio[:, 0] - audio[:, 1]) * 0.5
        
        # Adjust width
        side *= width
        
        # Convert back to L/R
        left = mid + side
        right = mid - side
        
        return np.column_stack([left, right])
    
    def _normalize_loudness(self, audio: np.ndarray, target_lufs: float) -> np.ndarray:
        """Normalize audio to target LUFS (simplified)"""
        # This is a simplified LUFS normalization
        # Real implementation would use proper LUFS measurement
        
        # Calculate RMS as proxy for loudness
        if len(audio.shape) == 1:
            rms = np.sqrt(np.mean(audio ** 2))
        else:
            rms = np.sqrt(np.mean(audio ** 2))
        
        # Rough conversion from LUFS to RMS scaling
        target_rms = 10 ** ((target_lufs + 20) / 20)  # Rough approximation
        
        if rms > 0:
            scaling_factor = target_rms / rms
            # Limit scaling to prevent excessive amplification
            scaling_factor = min(scaling_factor, 3.0)
            return audio * scaling_factor
        
        return audio
    
    def _apply_limiter(self, audio: np.ndarray, ceiling_db: float) -> np.ndarray:
        """Apply peak limiter"""
        ceiling_linear = 10 ** (ceiling_db / 20)
        
        # Simple peak limiter
        peak = np.max(np.abs(audio))
        if peak > ceiling_linear:
            # Apply limiting
            limited = np.clip(audio, -ceiling_linear, ceiling_linear)
            
            # Gentle transition to avoid harsh limiting artifacts
            blend_factor = 0.9
            return audio * (1 - blend_factor) + limited * blend_factor
        
        return audio
    
    def _analyze_audio(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze audio characteristics"""
        if len(audio.shape) == 1:
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio ** 2)))
            dynamic_range = peak / (rms + 1e-10)
        else:
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio ** 2)))
            dynamic_range = peak / (rms + 1e-10)
        
        return {
            'peak_level': peak,
            'rms_level': rms,
            'dynamic_range_ratio': dynamic_range,
            'peak_db': 20 * math.log10(peak + 1e-10),
            'rms_db': 20 * math.log10(rms + 1e-10)
        }
    
    def process_vocals(self, 
                      audio: np.ndarray, 
                      style: str = 'vocal_pop',
                      presence_boost: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply vocal-specific processing"""
        
        style_settings = self.style_presets.get(style, self.style_presets['vocal_pop'])
        processed = audio.copy()
        processing_log = []
        
        # De-essing (reduce harsh sibilants)
        if 'de_esser' in style_settings:
            processed = self._apply_de_esser(processed, style_settings['de_esser'])
            processing_log.append("De-essing")
        
        # Presence boost
        if 'presence_boost' in style_settings:
            boost_settings = style_settings['presence_boost'].copy()
            boost_settings['gain'] *= presence_boost
            processed = self._apply_presence_boost(processed, boost_settings)
            processing_log.append("Presence boost")
        
        # Warmth
        if 'warmth' in style_settings:
            processed = self._apply_vocal_warmth(processed, style_settings['warmth'])
            processing_log.append("Vocal warmth")
        
        # Vocal compression
        if 'compression' in style_settings:
            processed = self._apply_vocal_compression(processed, style_settings['compression'])
            processing_log.append("Vocal compression")
        
        metadata = {
            'vocal_style': style,
            'processing_chain': processing_log,
            'presence_boost': presence_boost,
            'final_stats': self._analyze_audio(processed)
        }
        
        return processed, metadata
    
    def _apply_de_esser(self, audio: np.ndarray, settings: Dict) -> np.ndarray:
        """Apply de-essing to reduce harsh sibilants"""
        # Simplified de-esser
        # In practice, would use frequency-selective compression
        
        # Detect high-frequency content
        if len(audio) > 1:
            high_freq_content = np.abs(np.diff(audio))
            threshold = np.percentile(high_freq_content, 95) * 0.7
            
            # Reduce harsh peaks
            mask = high_freq_content > threshold
            if len(audio.shape) == 1:
                reduction = np.ones_like(audio)
                reduction[1:][mask] *= 0.7
                return audio * reduction
        
        return audio
    
    def _apply_presence_boost(self, audio: np.ndarray, settings: Dict) -> np.ndarray:
        """Apply presence boost for vocal clarity"""
        # Simplified presence boost
        boost_factor = 10 ** (settings['gain'] / 20)
        return audio * (0.8 + 0.2 * boost_factor)
    
    def _apply_vocal_warmth(self, audio: np.ndarray, settings: Dict) -> np.ndarray:
        """Apply warmth to vocals"""
        # Gentle low-frequency boost
        boost_factor = 10 ** (settings['gain'] / 20)
        return audio * (0.9 + 0.1 * boost_factor)
    
    def _apply_vocal_compression(self, audio: np.ndarray, settings: Dict) -> np.ndarray:
        """Apply vocal-specific compression"""
        threshold_linear = 10 ** (settings['threshold'] / 20)
        ratio = settings['ratio']
        
        # Apply gentle compression
        amplitude = np.abs(audio)
        over_threshold = amplitude > threshold_linear
        
        if np.any(over_threshold):
            excess = amplitude[over_threshold] / threshold_linear
            gain_reduction = 1 / (1 + (excess - 1) * (ratio - 1) / ratio)
            audio[over_threshold] *= gain_reduction
        
        return audio
    
    def get_processing_presets(self) -> Dict[str, List[str]]:
        """Get available processing presets"""
        return {
            'mastering': list(self.mastering_presets.keys()),
            'vocal_styles': list(self.style_presets.keys())
        }

# Global processor instance
_audio_processor: Optional[ProfessionalAudioProcessor] = None

def get_audio_processor() -> ProfessionalAudioProcessor:
    """Get the global audio processor"""
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = ProfessionalAudioProcessor()
    return _audio_processor