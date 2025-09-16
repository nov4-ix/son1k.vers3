"""
APLIO Voice System for Son1kVers3
Advanced voice synthesis with 7 distinct AI voices and expressiveness controls
"""

import numpy as np
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class VoiceType(Enum):
    ELENA = "elena"          # Warm, soulful female voice
    MARCUS = "marcus"        # Deep, rich male voice  
    SOFIA = "sofia"          # Bright, energetic female voice
    DAVID = "david"          # Smooth, sophisticated male voice
    LUNA = "luna"            # Ethereal, dreamy female voice
    CARLOS = "carlos"        # Powerful, dramatic male voice
    ARIA = "aria"            # Classical, operatic female voice

@dataclass
class VoiceCharacteristics:
    """Detailed voice characteristics and parameters"""
    name: str
    gender: str
    fundamental_freq: float  # Base frequency in Hz
    formant_freqs: List[float]  # Formant frequencies
    timbre_profile: Dict[str, float]  # Harmonic content
    vibrato_rate: float  # Vibrato frequency in Hz
    vibrato_depth: float  # Vibrato depth as percentage
    expressiveness: float  # Base expressiveness (0-1)
    vocal_range: Tuple[float, float]  # Min, max frequency range
    breathiness: float  # Amount of breath noise (0-1)
    resonance: float  # Vocal tract resonance (0-1)
    articulation: float  # Clarity of consonants (0-1)

class AplioVoiceEngine:
    """Advanced voice synthesis engine with multiple AI voices"""
    
    def __init__(self, sample_rate: int = 32000):
        self.sample_rate = sample_rate
        self.voices = self._initialize_voices()
        
        # Phoneme database for synthesis
        self.phonemes = self._initialize_phonemes()
        
        # Emotion mapping
        self.emotions = {
            'neutral': {'pitch_mod': 1.0, 'vibrato_mod': 1.0, 'breath_mod': 1.0},
            'happy': {'pitch_mod': 1.1, 'vibrato_mod': 1.2, 'breath_mod': 0.8},
            'sad': {'pitch_mod': 0.9, 'vibrato_mod': 0.7, 'breath_mod': 1.3},
            'angry': {'pitch_mod': 1.2, 'vibrato_mod': 1.5, 'breath_mod': 0.6},
            'calm': {'pitch_mod': 0.95, 'vibrato_mod': 0.8, 'breath_mod': 1.1},
            'energetic': {'pitch_mod': 1.15, 'vibrato_mod': 1.4, 'breath_mod': 0.7},
            'romantic': {'pitch_mod': 0.98, 'vibrato_mod': 1.3, 'breath_mod': 1.2}
        }
        
        logger.info(f"APLIO Voice Engine initialized with {len(self.voices)} voices")
    
    def _initialize_voices(self) -> Dict[VoiceType, VoiceCharacteristics]:
        """Initialize the 7 AI voices with unique characteristics"""
        
        voices = {}
        
        # ELENA - Warm, soulful female voice
        voices[VoiceType.ELENA] = VoiceCharacteristics(
            name="Elena",
            gender="female",
            fundamental_freq=220.0,  # A3
            formant_freqs=[800, 1150, 2900, 3300, 4950],
            timbre_profile={
                'fundamental': 1.0,
                'harmonic_2': 0.7,
                'harmonic_3': 0.5,
                'harmonic_4': 0.3,
                'harmonic_5': 0.2
            },
            vibrato_rate=5.5,
            vibrato_depth=0.08,
            expressiveness=0.8,
            vocal_range=(180.0, 400.0),
            breathiness=0.3,
            resonance=0.8,
            articulation=0.7
        )
        
        # MARCUS - Deep, rich male voice
        voices[VoiceType.MARCUS] = VoiceCharacteristics(
            name="Marcus",
            gender="male",
            fundamental_freq=120.0,  # Approximately B2
            formant_freqs=[730, 1090, 2440, 3200, 4500],
            timbre_profile={
                'fundamental': 1.0,
                'harmonic_2': 0.8,
                'harmonic_3': 0.6,
                'harmonic_4': 0.4,
                'harmonic_5': 0.25
            },
            vibrato_rate=4.8,
            vibrato_depth=0.06,
            expressiveness=0.7,
            vocal_range=(80.0, 250.0),
            breathiness=0.2,
            resonance=0.9,
            articulation=0.8
        )
        
        # SOFIA - Bright, energetic female voice
        voices[VoiceType.SOFIA] = VoiceCharacteristics(
            name="Sofia",
            gender="female",
            fundamental_freq=260.0,  # C4
            formant_freqs=[850, 1220, 3100, 3500, 5200],
            timbre_profile={
                'fundamental': 1.0,
                'harmonic_2': 0.6,
                'harmonic_3': 0.8,  # Brighter sound
                'harmonic_4': 0.5,
                'harmonic_5': 0.3
            },
            vibrato_rate=6.2,
            vibrato_depth=0.09,
            expressiveness=0.9,
            vocal_range=(200.0, 450.0),
            breathiness=0.1,
            resonance=0.7,
            articulation=0.9
        )
        
        # DAVID - Smooth, sophisticated male voice
        voices[VoiceType.DAVID] = VoiceCharacteristics(
            name="David",
            gender="male",
            fundamental_freq=140.0,  # Approximately C#3
            formant_freqs=[750, 1100, 2600, 3150, 4600],
            timbre_profile={
                'fundamental': 1.0,
                'harmonic_2': 0.9,
                'harmonic_3': 0.4,
                'harmonic_4': 0.3,
                'harmonic_5': 0.15
            },
            vibrato_rate=5.0,
            vibrato_depth=0.05,
            expressiveness=0.6,
            vocal_range=(100.0, 280.0),
            breathiness=0.15,
            resonance=0.85,
            articulation=0.85
        )
        
        # LUNA - Ethereal, dreamy female voice
        voices[VoiceType.LUNA] = VoiceCharacteristics(
            name="Luna",
            gender="female",
            fundamental_freq=200.0,  # G3
            formant_freqs=[780, 1180, 2800, 3100, 4800],
            timbre_profile={
                'fundamental': 1.0,
                'harmonic_2': 0.5,
                'harmonic_3': 0.7,
                'harmonic_4': 0.6,
                'harmonic_5': 0.4
            },
            vibrato_rate=4.5,
            vibrato_depth=0.12,
            expressiveness=0.75,
            vocal_range=(160.0, 380.0),
            breathiness=0.4,
            resonance=0.6,
            articulation=0.6
        )
        
        # CARLOS - Powerful, dramatic male voice
        voices[VoiceType.CARLOS] = VoiceCharacteristics(
            name="Carlos",
            gender="male",
            fundamental_freq=110.0,  # A2
            formant_freqs=[700, 1050, 2350, 3000, 4200],
            timbre_profile={
                'fundamental': 1.0,
                'harmonic_2': 1.0,
                'harmonic_3': 0.8,
                'harmonic_4': 0.6,
                'harmonic_5': 0.4
            },
            vibrato_rate=5.8,
            vibrato_depth=0.10,
            expressiveness=0.95,
            vocal_range=(70.0, 300.0),
            breathiness=0.1,
            resonance=1.0,
            articulation=0.9
        )
        
        # ARIA - Classical, operatic female voice
        voices[VoiceType.ARIA] = VoiceCharacteristics(
            name="Aria",
            gender="female",
            fundamental_freq=280.0,  # C#4
            formant_freqs=[900, 1300, 3200, 3800, 5500],
            timbre_profile={
                'fundamental': 1.0,
                'harmonic_2': 0.8,
                'harmonic_3': 0.9,
                'harmonic_4': 0.7,
                'harmonic_5': 0.5
            },
            vibrato_rate=6.5,
            vibrato_depth=0.15,
            expressiveness=1.0,
            vocal_range=(220.0, 500.0),
            breathiness=0.05,
            resonance=0.95,
            articulation=0.95
        )
        
        return voices
    
    def _initialize_phonemes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize phoneme database for voice synthesis"""
        return {
            'a': {'duration': 0.15, 'formant_shift': [0, 0, 0], 'closure': False},
            'e': {'duration': 0.12, 'formant_shift': [50, 0, 0], 'closure': False},
            'i': {'duration': 0.10, 'formant_shift': [100, 200, 0], 'closure': False},
            'o': {'duration': 0.14, 'formant_shift': [-50, -100, 0], 'closure': False},
            'u': {'duration': 0.16, 'formant_shift': [-100, -200, -100], 'closure': False},
            'la': {'duration': 0.08, 'formant_shift': [0, 0, 0], 'closure': True},
            'na': {'duration': 0.06, 'formant_shift': [20, 0, 0], 'closure': True},
            'da': {'duration': 0.05, 'formant_shift': [30, 0, 0], 'closure': True},
            'ma': {'duration': 0.07, 'formant_shift': [0, 0, 0], 'closure': True},
            'ra': {'duration': 0.04, 'formant_shift': [-10, 50, 0], 'closure': False}
        }
    
    def generate_vocals(self,
                       text: str,
                       voice_type: VoiceType,
                       melody: List[float],
                       durations: List[float],
                       emotion: str = 'neutral',
                       expressiveness: float = 1.0) -> np.ndarray:
        """
        Generate vocal audio from text, melody, and timing
        
        Args:
            text: Lyrics or syllables to sing
            voice_type: Which APLIO voice to use
            melody: List of frequencies for each syllable
            durations: List of durations for each syllable
            emotion: Emotional expression
            expressiveness: Expressiveness multiplier (0-2)
        """
        
        voice = self.voices[voice_type]
        emotion_params = self.emotions.get(emotion, self.emotions['neutral'])
        
        # Parse text into phonemes
        phonemes = self._text_to_phonemes(text)
        
        # Ensure melody and durations match phonemes
        if len(melody) != len(phonemes):
            melody = self._interpolate_melody(melody, len(phonemes))
        if len(durations) != len(phonemes):
            durations = self._interpolate_durations(durations, len(phonemes))
        
        # Generate audio for each phoneme
        vocal_audio = []
        
        for i, (phoneme, freq, duration) in enumerate(zip(phonemes, melody, durations)):
            phoneme_audio = self._synthesize_phoneme(
                phoneme=phoneme,
                frequency=freq,
                duration=duration,
                voice=voice,
                emotion_params=emotion_params,
                expressiveness=expressiveness
            )
            vocal_audio.append(phoneme_audio)
        
        # Concatenate and apply global effects
        complete_vocals = np.concatenate(vocal_audio)
        complete_vocals = self._apply_vocal_effects(complete_vocals, voice, emotion_params)
        
        return complete_vocals
    
    def _text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phoneme sequence (simplified)"""
        # This is a simplified phoneme conversion
        # In production, would use proper phonetic analysis
        
        phonemes = []
        words = text.lower().split()
        
        for word in words:
            if len(word) == 0:
                continue
                
            # Simple vowel/consonant pattern
            for i, char in enumerate(word):
                if char in 'aeiou':
                    phonemes.append(char)
                else:
                    # Add consonant + vowel combination
                    if i < len(word) - 1 and word[i + 1] in 'aeiou':
                        phonemes.append(char + word[i + 1])
                    else:
                        phonemes.append(char + 'a')  # Default vowel
        
        # Ensure we have at least some phonemes
        if not phonemes:
            phonemes = ['la', 'la', 'la']
        
        return phonemes
    
    def _interpolate_melody(self, melody: List[float], target_length: int) -> List[float]:
        """Interpolate melody to match phoneme count"""
        if len(melody) == target_length:
            return melody
        
        if len(melody) == 1:
            return melody * target_length
        
        # Linear interpolation
        result = []
        step = (len(melody) - 1) / (target_length - 1)
        
        for i in range(target_length):
            index = i * step
            lower_idx = int(index)
            upper_idx = min(lower_idx + 1, len(melody) - 1)
            
            if lower_idx == upper_idx:
                result.append(melody[lower_idx])
            else:
                weight = index - lower_idx
                interpolated = melody[lower_idx] * (1 - weight) + melody[upper_idx] * weight
                result.append(interpolated)
        
        return result
    
    def _interpolate_durations(self, durations: List[float], target_length: int) -> List[float]:
        """Interpolate durations to match phoneme count"""
        if len(durations) == target_length:
            return durations
        
        total_duration = sum(durations)
        avg_duration = total_duration / target_length
        
        return [avg_duration] * target_length
    
    def _synthesize_phoneme(self,
                           phoneme: str,
                           frequency: float,
                           duration: float,
                           voice: VoiceCharacteristics,
                           emotion_params: Dict[str, float],
                           expressiveness: float) -> np.ndarray:
        """Synthesize a single phoneme with voice characteristics"""
        
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Get phoneme characteristics
        phoneme_data = self.phonemes.get(phoneme, self.phonemes['la'])
        
        # Apply emotion to frequency
        actual_frequency = frequency * emotion_params['pitch_mod']
        
        # Ensure frequency is within voice range
        actual_frequency = np.clip(actual_frequency, voice.vocal_range[0], voice.vocal_range[1])
        
        # Generate fundamental frequency with vibrato
        vibrato_rate = voice.vibrato_rate * emotion_params['vibrato_mod']
        vibrato_depth = voice.vibrato_depth * expressiveness
        
        vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        instantaneous_freq = 2 * np.pi * actual_frequency * vibrato * t
        
        # Generate harmonic content
        audio = np.zeros_like(t)
        
        for harmonic, amplitude in voice.timbre_profile.items():
            harmonic_num = int(harmonic.split('_')[-1]) if '_' in harmonic else 1
            harmonic_freq = actual_frequency * harmonic_num * vibrato
            
            # Avoid aliasing
            if np.max(harmonic_freq) < self.sample_rate / 2:
                harmonic_audio = amplitude * np.sin(2 * np.pi * harmonic_freq * t)
                audio += harmonic_audio
        
        # Apply formant filtering (simplified)
        audio = self._apply_formants(audio, voice.formant_freqs, phoneme_data['formant_shift'])
        
        # Add breathiness
        breath_amount = voice.breathiness * emotion_params['breath_mod']
        if breath_amount > 0:
            breath_noise = np.random.normal(0, breath_amount * 0.1, len(audio))
            # Filter breath noise to vocal frequencies
            breath_noise = self._simple_bandpass_filter(breath_noise, 100, 3000)
            audio += breath_noise
        
        # Apply envelope
        audio = self._apply_phoneme_envelope(audio, phoneme_data, voice)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8
        
        return audio
    
    def _apply_formants(self, audio: np.ndarray, formant_freqs: List[float], shifts: List[float]) -> np.ndarray:
        """Apply formant filtering to audio (simplified implementation)"""
        # This is a simplified formant filter
        # In production, would use proper vocal tract modeling
        
        processed = audio.copy()
        
        # Apply emphasis at formant frequencies
        for i, (formant_freq, shift) in enumerate(zip(formant_freqs[:3], shifts[:3] + [0, 0, 0])):
            actual_freq = formant_freq + shift
            
            # Simple resonance boost (not physically accurate but effective)
            t = np.arange(len(processed)) / self.sample_rate
            resonance = 0.1 * np.sin(2 * np.pi * actual_freq * t)
            processed = processed + resonance * 0.1
        
        return processed
    
    def _simple_bandpass_filter(self, audio: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Simple bandpass filter (placeholder implementation)"""
        # This is a placeholder - would use proper filter design in production
        return audio * 0.5  # Simplified attenuation
    
    def _apply_phoneme_envelope(self, audio: np.ndarray, phoneme_data: Dict, voice: VoiceCharacteristics) -> np.ndarray:
        """Apply envelope based on phoneme characteristics"""
        
        envelope = np.ones_like(audio)
        
        if phoneme_data['closure']:
            # Consonant - sharp attack and decay
            attack_samples = min(len(audio) // 10, int(0.01 * self.sample_rate))
            decay_samples = min(len(audio) // 5, int(0.02 * self.sample_rate))
            
            # Attack
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            
            # Decay
            if len(audio) > attack_samples + decay_samples:
                envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, 0.3, decay_samples)
        else:
            # Vowel - sustained with slight fade
            fade_in = min(len(audio) // 8, int(0.02 * self.sample_rate))
            fade_out = min(len(audio) // 8, int(0.02 * self.sample_rate))
            
            envelope[:fade_in] = np.linspace(0, 1, fade_in)
            envelope[-fade_out:] = np.linspace(1, 0, fade_out)
        
        return audio * envelope
    
    def _apply_vocal_effects(self, audio: np.ndarray, voice: VoiceCharacteristics, emotion_params: Dict) -> np.ndarray:
        """Apply global vocal effects"""
        
        processed = audio.copy()
        
        # Apply resonance
        if voice.resonance > 0.5:
            # Add slight reverb effect (simplified)
            delay_samples = int(0.01 * self.sample_rate)  # 10ms delay
            if len(processed) > delay_samples:
                delayed = np.zeros_like(processed)
                delayed[delay_samples:] = processed[:-delay_samples] * voice.resonance * 0.2
                processed = processed + delayed
        
        # Dynamic range compression
        threshold = 0.7
        ratio = 3.0
        over_threshold = np.abs(processed) > threshold
        gain_reduction = 1 - (1 - threshold / np.abs(processed[over_threshold])) / ratio
        processed[over_threshold] *= gain_reduction
        
        # Final normalization
        max_val = np.max(np.abs(processed))
        if max_val > 0:
            processed = processed / max_val * 0.9
        
        return processed
    
    def get_voice_info(self, voice_type: VoiceType) -> Dict[str, Any]:
        """Get detailed information about a voice"""
        voice = self.voices[voice_type]
        
        return {
            'name': voice.name,
            'gender': voice.gender,
            'fundamental_frequency': voice.fundamental_freq,
            'vocal_range': {
                'min_hz': voice.vocal_range[0],
                'max_hz': voice.vocal_range[1]
            },
            'characteristics': {
                'breathiness': voice.breathiness,
                'resonance': voice.resonance,
                'articulation': voice.articulation,
                'expressiveness': voice.expressiveness
            },
            'vibrato': {
                'rate_hz': voice.vibrato_rate,
                'depth_percent': voice.vibrato_depth * 100
            }
        }
    
    def list_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """List all available voices with their characteristics"""
        return {voice_type.value: self.get_voice_info(voice_type) 
                for voice_type in VoiceType}
    
    def generate_vocal_demo(self, voice_type: VoiceType, emotion: str = 'neutral') -> np.ndarray:
        """Generate a demo of the voice singing a simple melody"""
        
        # Simple demo melody (C major scale)
        demo_melody = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
        demo_durations = [0.5] * 8
        demo_text = "la la la la la la la la"
        
        return self.generate_vocals(
            text=demo_text,
            voice_type=voice_type,
            melody=demo_melody,
            durations=demo_durations,
            emotion=emotion,
            expressiveness=1.0
        )

# Global voice engine instance
_voice_engine: Optional[AplioVoiceEngine] = None

def get_voice_engine() -> AplioVoiceEngine:
    """Get the global APLIO voice engine"""
    global _voice_engine
    if _voice_engine is None:
        _voice_engine = AplioVoiceEngine()
    return _voice_engine

def demo_all_voices() -> Dict[str, np.ndarray]:
    """Generate demos for all APLIO voices"""
    engine = get_voice_engine()
    demos = {}
    
    for voice_type in VoiceType:
        demos[voice_type.value] = engine.generate_vocal_demo(voice_type)
    
    return demos

if __name__ == "__main__":
    # Test the voice system
    engine = AplioVoiceEngine()
    
    print("🎤 APLIO Voice System Test")
    print(f"Available voices: {len(engine.voices)}")
    
    for voice_type in VoiceType:
        info = engine.get_voice_info(voice_type)
        print(f"  {info['name']}: {info['gender']}, {info['fundamental_frequency']:.1f}Hz")
    
    # Generate a test vocal
    test_vocal = engine.generate_vocal_demo(VoiceType.ELENA, 'happy')
    print(f"Generated test vocal: {len(test_vocal)} samples")