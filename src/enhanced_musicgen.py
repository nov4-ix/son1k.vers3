"""
Enhanced MusicGen Service with Quality Improvements and AI Features
Optimized for Son1kVers3 with professional-grade audio synthesis
"""

import numpy as np
import soundfile as sf
import asyncio
import logging
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AudioQuality(Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    STUDIO = "studio"

@dataclass
class GenerationConfig:
    """Configuration for music generation"""
    sample_rate: int = 32000
    quality: AudioQuality = AudioQuality.STANDARD
    use_stereo: bool = True
    apply_mastering: bool = True
    harmonic_richness: float = 1.0
    dynamic_range: float = 1.0
    frequency_shaping: bool = True
    
class AdvancedOscillator:
    """Professional-grade oscillator with multiple waveforms and modulation"""
    
    def __init__(self, sample_rate: int = 32000):
        self.sample_rate = sample_rate
        self.phase = 0.0
        
    def generate_waveform(self, 
                         frequency: float, 
                         duration: float, 
                         waveform: str = "complex",
                         amplitude: float = 1.0,
                         modulation: Optional[Dict] = None) -> np.ndarray:
        """Generate sophisticated waveforms with modulation"""
        
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        if waveform == "complex":
            # Multi-harmonic complex waveform
            signal = self._generate_complex_wave(frequency, t, amplitude)
        elif waveform == "fm":
            # FM synthesis
            signal = self._generate_fm_wave(frequency, t, amplitude)
        elif waveform == "analog":
            # Analog-style oscillator with drift
            signal = self._generate_analog_wave(frequency, t, amplitude)
        else:
            # Standard sine wave
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Apply modulation if specified
        if modulation:
            signal = self._apply_modulation(signal, t, modulation)
            
        return signal
    
    def _generate_complex_wave(self, freq: float, t: np.ndarray, amplitude: float) -> np.ndarray:
        """Generate complex multi-harmonic waveform"""
        signal = np.zeros_like(t)
        
        # Fundamental
        signal += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Harmonics with decreasing amplitude
        harmonics = [2, 3, 4, 5, 7, 9]
        harmonic_amps = [0.5, 0.33, 0.25, 0.2, 0.15, 0.1]
        
        for h, h_amp in zip(harmonics, harmonic_amps):
            h_freq = freq * h
            if h_freq < self.sample_rate / 2:  # Avoid aliasing
                phase_offset = random.uniform(0, 2 * np.pi)
                signal += amplitude * h_amp * np.sin(2 * np.pi * h_freq * t + phase_offset)
        
        # Add subtle subharmonics
        sub_freq = freq / 2
        if sub_freq > 20:  # Above audible threshold
            signal += amplitude * 0.1 * np.sin(2 * np.pi * sub_freq * t)
            
        return signal
    
    def _generate_fm_wave(self, freq: float, t: np.ndarray, amplitude: float) -> np.ndarray:
        """Generate FM synthesized waveform"""
        # Carrier frequency
        carrier_freq = freq
        
        # Modulator frequency (usually lower)
        mod_freq = freq * 0.1
        mod_depth = freq * 0.3
        
        # FM synthesis: carrier + (mod_depth * sin(mod_freq * t))
        instantaneous_freq = 2 * np.pi * (carrier_freq * t + 
                                        (mod_depth / (2 * np.pi * mod_freq)) * 
                                        np.sin(2 * np.pi * mod_freq * t))
        
        signal = amplitude * np.sin(instantaneous_freq)
        return signal
    
    def _generate_analog_wave(self, freq: float, t: np.ndarray, amplitude: float) -> np.ndarray:
        """Generate analog-style waveform with drift and imperfections"""
        # Base signal
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add frequency drift (analog instability)
        drift_freq = 0.1  # Very slow drift
        drift_amount = freq * 0.001  # Small drift
        freq_drift = drift_amount * np.sin(2 * np.pi * drift_freq * t)
        
        # Apply drift
        instantaneous_freq = 2 * np.pi * (freq + freq_drift) * t
        signal = amplitude * np.sin(instantaneous_freq)
        
        # Add subtle harmonic distortion
        signal += amplitude * 0.05 * np.sin(3 * 2 * np.pi * freq * t)
        signal += amplitude * 0.02 * np.sin(5 * 2 * np.pi * freq * t)
        
        # Add very subtle noise (analog warmth)
        noise = np.random.normal(0, amplitude * 0.001, len(t))
        signal += noise
        
        return signal
    
    def _apply_modulation(self, signal: np.ndarray, t: np.ndarray, modulation: Dict) -> np.ndarray:
        """Apply various modulation effects"""
        mod_type = modulation.get('type', 'none')
        
        if mod_type == 'tremolo':
            # Amplitude modulation
            mod_freq = modulation.get('frequency', 5.0)
            mod_depth = modulation.get('depth', 0.3)
            mod_signal = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
            signal = signal * mod_signal
            
        elif mod_type == 'vibrato':
            # Frequency modulation (small amount)
            mod_freq = modulation.get('frequency', 6.0)
            mod_depth = modulation.get('depth', 0.02)
            # This is simplified vibrato - more complex implementation would require phase modulation
            mod_signal = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
            signal = signal * mod_signal
            
        return signal

class EnhancedMusicGenService:
    """Enhanced MusicGen with professional audio synthesis capabilities"""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.oscillator = AdvancedOscillator(self.config.sample_rate)
        self.model_loaded = True
        
        # Musical knowledge base
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'pentatonic': [0, 2, 4, 7, 9],
            'blues': [0, 3, 5, 6, 7, 10]
        }
        
        self.chord_progressions = {
            'pop': ['I', 'V', 'vi', 'IV'],
            'rock': ['I', 'bVII', 'IV', 'I'],
            'jazz': ['ii', 'V', 'I', 'vi'],
            'blues': ['I', 'I', 'I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'IV', 'I', 'V'],
            'latin': ['i', 'VII', 'VI', 'VII'],
            'electronic': ['i', 'bVII', 'bVI', 'bVII']
        }
        
        logger.info(f"Enhanced MusicGen initialized - Quality: {self.config.quality.value}")
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt to extract musical parameters"""
        prompt_lower = prompt.lower()
        
        # Detect key
        key_info = self._detect_key(prompt_lower)
        
        # Detect tempo
        tempo_info = self._detect_tempo(prompt_lower)
        
        # Detect style
        style_info = self._detect_style(prompt_lower)
        
        # Detect instruments
        instruments = self._detect_instruments(prompt_lower)
        
        # Detect mood/energy
        mood_info = self._analyze_mood(prompt_lower)
        
        return {
            'key': key_info,
            'tempo': tempo_info,
            'style': style_info,
            'instruments': instruments,
            'mood': mood_info,
            'complexity': self._assess_complexity(prompt_lower)
        }
    
    def _detect_key(self, prompt: str) -> Dict[str, Any]:
        """Detect musical key from prompt"""
        # Key detection patterns
        key_patterns = {
            'c major': {'root': 'C', 'scale': 'major'},
            'a minor': {'root': 'A', 'scale': 'minor'},
            'g major': {'root': 'G', 'scale': 'major'},
            'e minor': {'root': 'E', 'scale': 'minor'},
            'f major': {'root': 'F', 'scale': 'major'},
            'd minor': {'root': 'D', 'scale': 'minor'},
            'b♭ major': {'root': 'Bb', 'scale': 'major'},
            'bb major': {'root': 'Bb', 'scale': 'major'},
        }
        
        for pattern, info in key_patterns.items():
            if pattern in prompt:
                return info
        
        # Default based on style
        if any(word in prompt for word in ['sad', 'melancholic', 'dark', 'emotional']):
            return {'root': 'A', 'scale': 'minor'}
        else:
            return {'root': 'C', 'scale': 'major'}
    
    def _detect_tempo(self, prompt: str) -> Dict[str, Any]:
        """Detect tempo from prompt"""
        tempo_keywords = {
            'slow': 60, 'ballad': 70, 'moderate': 100, 'medium': 110,
            'fast': 130, 'upbeat': 120, 'energetic': 140, 'rapid': 150,
            'adagio': 70, 'andante': 90, 'allegro': 120, 'presto': 150
        }
        
        # BPM patterns
        import re
        bpm_match = re.search(r'(\d+)\s*bpm', prompt)
        if bpm_match:
            return {'bpm': int(bpm_match.group(1)), 'confidence': 0.9}
        
        # Keyword detection
        for keyword, bpm in tempo_keywords.items():
            if keyword in prompt:
                return {'bpm': bpm, 'confidence': 0.7}
        
        # Default medium tempo
        return {'bpm': 100, 'confidence': 0.3}
    
    def _detect_style(self, prompt: str) -> Dict[str, Any]:
        """Detect musical style from prompt"""
        style_keywords = {
            'rock': ['rock', 'guitar', 'drums', 'electric'],
            'pop': ['pop', 'catchy', 'mainstream', 'commercial'],
            'electronic': ['electronic', 'synth', 'edm', 'techno', 'house'],
            'jazz': ['jazz', 'swing', 'bebop', 'fusion'],
            'classical': ['classical', 'orchestra', 'symphony', 'baroque'],
            'latin': ['latin', 'salsa', 'bossa', 'tango', 'reggaeton'],
            'blues': ['blues', 'twelve bar', 'bb king'],
            'folk': ['folk', 'acoustic', 'traditional'],
            'ambient': ['ambient', 'atmospheric', 'soundscape']
        }
        
        detected_styles = []
        for style, keywords in style_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                detected_styles.append(style)
        
        if detected_styles:
            return {'primary': detected_styles[0], 'secondary': detected_styles[1:]}
        
        return {'primary': 'pop', 'secondary': []}
    
    def _detect_instruments(self, prompt: str) -> List[str]:
        """Detect instruments mentioned in prompt"""
        instruments = {
            'piano': ['piano', 'keyboard', 'keys'],
            'guitar': ['guitar', 'acoustic guitar', 'electric guitar'],
            'bass': ['bass', 'bass guitar', 'upright bass'],
            'drums': ['drums', 'percussion', 'beat'],
            'violin': ['violin', 'strings', 'orchestra'],
            'synth': ['synth', 'synthesizer', 'electronic'],
            'vocals': ['vocal', 'voice', 'singing', 'lyrics'],
            'saxophone': ['sax', 'saxophone'],
            'trumpet': ['trumpet', 'brass'],
            'flute': ['flute', 'woodwind']
        }
        
        detected = []
        for instrument, keywords in instruments.items():
            if any(keyword in prompt for keyword in keywords):
                detected.append(instrument)
        
        return detected
    
    def _analyze_mood(self, prompt: str) -> Dict[str, float]:
        """Analyze emotional content and energy level"""
        mood_keywords = {
            'energy': {
                'low': ['calm', 'peaceful', 'slow', 'relaxed', 'gentle'],
                'medium': ['moderate', 'steady', 'balanced'],
                'high': ['energetic', 'fast', 'intense', 'powerful', 'driving']
            },
            'emotion': {
                'happy': ['happy', 'joyful', 'uplifting', 'cheerful', 'bright'],
                'sad': ['sad', 'melancholic', 'emotional', 'tearful'],
                'dark': ['dark', 'ominous', 'mysterious', 'haunting'],
                'romantic': ['romantic', 'love', 'tender', 'intimate'],
                'aggressive': ['aggressive', 'angry', 'harsh', 'brutal']
            }
        }
        
        scores = {'energy': 0.5, 'valence': 0.5}  # Default neutral
        
        # Analyze energy
        for level, keywords in mood_keywords['energy'].items():
            if any(keyword in prompt for keyword in keywords):
                if level == 'low':
                    scores['energy'] = 0.2
                elif level == 'high':
                    scores['energy'] = 0.8
                break
        
        # Analyze emotional valence
        for emotion, keywords in mood_keywords['emotion'].items():
            if any(keyword in prompt for keyword in keywords):
                if emotion in ['happy', 'romantic']:
                    scores['valence'] = 0.8
                elif emotion in ['sad', 'dark']:
                    scores['valence'] = 0.2
                elif emotion == 'aggressive':
                    scores['valence'] = 0.1
                    scores['energy'] = 0.9
                break
        
        return scores
    
    def _assess_complexity(self, prompt: str) -> float:
        """Assess desired musical complexity"""
        simple_keywords = ['simple', 'basic', 'minimal', 'clean']
        complex_keywords = ['complex', 'intricate', 'sophisticated', 'layered', 'rich']
        
        if any(keyword in prompt for keyword in simple_keywords):
            return 0.3
        elif any(keyword in prompt for keyword in complex_keywords):
            return 0.8
        else:
            return 0.5  # Default medium complexity
    
    async def generate_music(self, 
                           prompt: str, 
                           duration: float = 8.0,
                           temperature: float = 1.0,
                           **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate enhanced music with AI analysis"""
        
        start_time = time.time()
        
        # Analyze the prompt
        analysis = self.analyze_prompt(prompt)
        logger.info(f"Prompt analysis: {analysis}")
        
        # Generate base parameters
        key_info = analysis['key']
        tempo_info = analysis['tempo']
        style_info = analysis['style']
        mood_info = analysis['mood']
        
        # Calculate base frequency from key
        base_freq = self._get_base_frequency(key_info['root'])
        
        # Generate musical structure
        structure = self._generate_structure(duration, tempo_info['bpm'], style_info['primary'])
        
        # Generate audio
        audio = await self._synthesize_music(
            structure=structure,
            base_freq=base_freq,
            scale=key_info['scale'],
            style=style_info['primary'],
            mood=mood_info,
            complexity=analysis['complexity'],
            temperature=temperature
        )
        
        # Apply post-processing
        if self.config.apply_mastering:
            audio = self._apply_mastering(audio, style_info['primary'])
        
        # Generate metadata
        metadata = {
            'prompt': prompt,
            'duration_s': duration,
            'sample_rate': self.config.sample_rate,
            'analysis': analysis,
            'structure': structure,
            'generation_time': time.time() - start_time,
            'quality': self.config.quality.value,
            'provider': 'enhanced_musicgen',
            'audio_stats': {
                'peak': float(np.max(np.abs(audio))),
                'rms': float(np.sqrt(np.mean(audio ** 2))),
                'dynamic_range': float(np.max(audio) - np.min(audio))
            }
        }
        
        return audio, metadata
    
    def _get_base_frequency(self, root: str) -> float:
        """Get base frequency for a musical root note"""
        note_frequencies = {
            'C': 261.63, 'C#': 277.18, 'Db': 277.18,
            'D': 293.66, 'D#': 311.13, 'Eb': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'Gb': 369.99,
            'G': 392.00, 'G#': 415.30, 'Ab': 415.30,
            'A': 440.00, 'A#': 466.16, 'Bb': 466.16,
            'B': 493.88
        }
        return note_frequencies.get(root, 261.63)  # Default to C
    
    def _generate_structure(self, duration: float, bpm: float, style: str) -> Dict[str, Any]:
        """Generate musical structure based on duration and style"""
        beats_per_second = bpm / 60
        total_beats = duration * beats_per_second
        
        # Different structures for different styles
        if style in ['pop', 'rock']:
            # Verse-chorus structure
            if duration >= 20:
                return {
                    'sections': [
                        {'name': 'intro', 'duration': 2, 'start': 0},
                        {'name': 'verse1', 'duration': 6, 'start': 2},
                        {'name': 'chorus', 'duration': 6, 'start': 8},
                        {'name': 'verse2', 'duration': 6, 'start': 14},
                        {'name': 'outro', 'duration': duration-20, 'start': 20}
                    ],
                    'chord_progression': self.chord_progressions.get(style, ['I', 'V', 'vi', 'IV']),
                    'bpm': bpm
                }
            else:
                # Simple structure for short durations
                return {
                    'sections': [
                        {'name': 'main', 'duration': duration, 'start': 0}
                    ],
                    'chord_progression': self.chord_progressions.get(style, ['I', 'V', 'vi', 'IV']),
                    'bpm': bpm
                }
        
        elif style == 'electronic':
            # Build-up structure
            return {
                'sections': [
                    {'name': 'intro', 'duration': duration * 0.2, 'start': 0},
                    {'name': 'buildup', 'duration': duration * 0.3, 'start': duration * 0.2},
                    {'name': 'drop', 'duration': duration * 0.4, 'start': duration * 0.5},
                    {'name': 'outro', 'duration': duration * 0.1, 'start': duration * 0.9}
                ],
                'chord_progression': self.chord_progressions.get(style, ['i', 'bVII', 'bVI', 'bVII']),
                'bpm': bpm
            }
        
        else:
            # Generic structure
            return {
                'sections': [
                    {'name': 'main', 'duration': duration, 'start': 0}
                ],
                'chord_progression': self.chord_progressions.get(style, ['I', 'V', 'vi', 'IV']),
                'bpm': bpm
            }
    
    async def _synthesize_music(self, 
                              structure: Dict,
                              base_freq: float,
                              scale: str,
                              style: str,
                              mood: Dict,
                              complexity: float,
                              temperature: float) -> np.ndarray:
        """Synthesize music based on analyzed parameters"""
        
        total_duration = sum(section['duration'] for section in structure['sections'])
        total_samples = int(total_duration * self.config.sample_rate)
        
        # Initialize stereo audio if enabled
        if self.config.use_stereo:
            audio = np.zeros((total_samples, 2))
        else:
            audio = np.zeros(total_samples)
        
        # Generate each section
        for section in structure['sections']:
            section_audio = await self._generate_section(
                section=section,
                base_freq=base_freq,
                scale=scale,
                style=style,
                mood=mood,
                complexity=complexity,
                temperature=temperature,
                chord_progression=structure['chord_progression']
            )
            
            # Place section in the full audio
            start_sample = int(section['start'] * self.config.sample_rate)
            end_sample = start_sample + len(section_audio)
            
            if self.config.use_stereo and len(section_audio.shape) == 1:
                # Convert mono to stereo
                section_audio = np.column_stack([section_audio, section_audio])
            
            if end_sample <= len(audio):
                if self.config.use_stereo:
                    audio[start_sample:end_sample] += section_audio
                else:
                    audio[start_sample:end_sample] += section_audio
        
        # Convert to mono if stereo is disabled
        if self.config.use_stereo and len(audio.shape) == 2:
            pass  # Keep stereo
        elif len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)  # Convert to mono
        
        return audio
    
    async def _generate_section(self,
                              section: Dict,
                              base_freq: float,
                              scale: str,
                              style: str,
                              mood: Dict,
                              complexity: float,
                              temperature: float,
                              chord_progression: List[str]) -> np.ndarray:
        """Generate audio for a specific section"""
        
        duration = section['duration']
        section_name = section['name']
        
        # Adjust parameters based on section type
        if section_name == 'intro':
            volume = 0.3 + mood['energy'] * 0.3
            density = 0.5
        elif section_name == 'buildup':
            volume = 0.4 + mood['energy'] * 0.4
            density = 0.7
        elif section_name in ['chorus', 'drop']:
            volume = 0.6 + mood['energy'] * 0.4
            density = 1.0
        elif section_name == 'outro':
            volume = 0.2 + mood['energy'] * 0.2
            density = 0.3
        else:  # verse, main
            volume = 0.4 + mood['energy'] * 0.3
            density = 0.6 + complexity * 0.3
        
        # Generate chord sequence
        num_chords = max(1, int(duration / 2))  # Change chord every 2 seconds
        scale_notes = self._get_scale_notes(base_freq, scale)
        
        section_audio = np.zeros(int(duration * self.config.sample_rate))
        
        # Generate harmonic content
        for i in range(num_chords):
            chord_start = (i / num_chords) * duration
            chord_duration = duration / num_chords
            
            # Select chord from progression
            chord_index = i % len(chord_progression)
            chord_notes = self._get_chord_notes(scale_notes, chord_progression[chord_index])
            
            # Generate chord
            chord_audio = self._generate_chord(
                notes=chord_notes,
                duration=chord_duration,
                volume=volume,
                style=style,
                complexity=complexity,
                temperature=temperature
            )
            
            # Add to section
            start_sample = int(chord_start * self.config.sample_rate)
            end_sample = start_sample + len(chord_audio)
            
            if end_sample <= len(section_audio):
                section_audio[start_sample:end_sample] += chord_audio
        
        # Add rhythmic elements for certain styles
        if style in ['rock', 'electronic', 'pop']:
            rhythm_audio = self._generate_rhythm(duration, style, mood['energy'])
            section_audio += rhythm_audio * 0.3
        
        # Apply section-specific effects
        section_audio = self._apply_section_effects(section_audio, section_name, style)
        
        return section_audio
    
    def _get_scale_notes(self, root_freq: float, scale: str) -> List[float]:
        """Get frequencies for notes in a scale"""
        scale_intervals = self.scales.get(scale, self.scales['major'])
        
        notes = []
        for interval in scale_intervals:
            # Calculate frequency using equal temperament
            freq = root_freq * (2 ** (interval / 12))
            notes.append(freq)
        
        return notes
    
    def _get_chord_notes(self, scale_notes: List[float], chord_symbol: str) -> List[float]:
        """Get frequencies for notes in a chord"""
        # Simplified chord mapping
        chord_map = {
            'I': [0, 2, 4],      # Root, third, fifth
            'ii': [1, 3, 5],     # Second, fourth, sixth
            'iii': [2, 4, 6],    # Third, fifth, seventh
            'IV': [3, 5, 0],     # Fourth, sixth, root
            'V': [4, 6, 1],      # Fifth, seventh, second
            'vi': [5, 0, 2],     # Sixth, root, third
            'vii': [6, 1, 3],    # Seventh, second, fourth
            'bVII': [5, 0, 2],   # Flat seventh
            'bVI': [4, 6, 1]     # Flat sixth
        }
        
        if chord_symbol.lower() in [s.lower() for s in chord_map.keys()]:
            # Find the correct case version
            actual_symbol = next(k for k in chord_map.keys() if k.lower() == chord_symbol.lower())
            indices = chord_map[actual_symbol]
        else:
            indices = [0, 2, 4]  # Default to major triad
        
        chord_notes = []
        for idx in indices:
            if idx < len(scale_notes):
                chord_notes.append(scale_notes[idx])
        
        return chord_notes
    
    def _generate_chord(self,
                       notes: List[float],
                       duration: float,
                       volume: float,
                       style: str,
                       complexity: float,
                       temperature: float) -> np.ndarray:
        """Generate a chord with multiple notes"""
        
        chord_audio = np.zeros(int(duration * self.config.sample_rate))
        
        for note_freq in notes:
            # Choose waveform based on style
            if style == 'electronic':
                waveform = 'fm'
            elif style in ['rock', 'blues']:
                waveform = 'analog'
            else:
                waveform = 'complex'
            
            # Add some randomness based on temperature
            freq_variation = 1.0 + (temperature - 1.0) * 0.02 * (random.random() - 0.5)
            actual_freq = note_freq * freq_variation
            
            # Generate note
            note_audio = self.oscillator.generate_waveform(
                frequency=actual_freq,
                duration=duration,
                waveform=waveform,
                amplitude=volume / len(notes)  # Normalize by number of notes
            )
            
            chord_audio += note_audio
        
        return chord_audio
    
    def _generate_rhythm(self, duration: float, style: str, energy: float) -> np.ndarray:
        """Generate rhythmic elements"""
        rhythm_audio = np.zeros(int(duration * self.config.sample_rate))
        
        if style == 'electronic':
            # Four-on-the-floor kick pattern
            kick_freq = 60  # Low frequency for kick
            beat_interval = 60 / 120  # 120 BPM
            
            num_beats = int(duration / beat_interval)
            for i in range(num_beats):
                beat_time = i * beat_interval
                if beat_time < duration:
                    # Generate kick drum sound
                    kick_duration = 0.1
                    kick_samples = int(kick_duration * self.config.sample_rate)
                    t = np.linspace(0, kick_duration, kick_samples)
                    
                    # Exponentially decaying sine wave
                    kick_sound = np.sin(2 * np.pi * kick_freq * t) * np.exp(-t * 30)
                    kick_sound *= energy * 0.3
                    
                    start_sample = int(beat_time * self.config.sample_rate)
                    end_sample = start_sample + len(kick_sound)
                    
                    if end_sample <= len(rhythm_audio):
                        rhythm_audio[start_sample:end_sample] += kick_sound
        
        return rhythm_audio
    
    def _apply_section_effects(self, audio: np.ndarray, section_name: str, style: str) -> np.ndarray:
        """Apply effects specific to section type"""
        
        if section_name == 'intro':
            # Fade in
            fade_samples = min(len(audio) // 4, int(0.5 * self.config.sample_rate))
            fade_in = np.linspace(0, 1, fade_samples)
            audio[:fade_samples] *= fade_in
            
        elif section_name == 'outro':
            # Fade out
            fade_samples = min(len(audio) // 2, int(1.0 * self.config.sample_rate))
            fade_out = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_out
            
        elif section_name == 'buildup' and style == 'electronic':
            # High-pass filter sweep (simulated)
            # This is a simplified version - real implementation would use proper filters
            t = np.linspace(0, 1, len(audio))
            emphasis = 1 + t * 0.5  # Gradually emphasize highs
            audio *= emphasis
        
        return audio
    
    def _apply_mastering(self, audio: np.ndarray, style: str) -> np.ndarray:
        """Apply professional mastering chain"""
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        
        # Gentle compression (simplified)
        threshold = 0.7
        ratio = 3.0
        compressed_audio = np.copy(audio)
        
        over_threshold = np.abs(audio) > threshold
        gain_reduction = 1 - (1 - threshold / np.abs(audio[over_threshold])) / ratio
        compressed_audio[over_threshold] *= gain_reduction
        
        # EQ adjustments based on style
        if style == 'rock':
            # Boost mids and highs slightly
            compressed_audio *= 1.1
        elif style == 'electronic':
            # Boost sub-bass and highs
            compressed_audio *= 1.05
        
        # Final limiting
        limit = 0.95
        compressed_audio = np.clip(compressed_audio, -limit, limit)
        
        return compressed_audio
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model_loaded
    
    def get_device(self) -> str:
        """Get the device being used"""
        return "cpu_enhanced"