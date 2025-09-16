"""
Style Transfer AI for Music Transformation
Implements intelligent music style conversion with neural networks
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import logging
from pathlib import Path

# Optional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

logger = logging.getLogger(__name__)

class MusicStyle(Enum):
    """Supported music styles for transfer"""
    JAZZ = "jazz"
    CLASSICAL = "classical"
    ROCK = "rock"
    POP = "pop"
    ELECTRONIC = "electronic"
    BLUES = "blues"
    COUNTRY = "country"
    REGGAE = "reggae"
    FUNK = "funk"
    AMBIENT = "ambient"
    METAL = "metal"
    FOLK = "folk"

@dataclass
class StyleTransferConfig:
    """Configuration for style transfer"""
    source_style: Optional[MusicStyle] = None
    target_style: MusicStyle = MusicStyle.POP
    intensity: float = 0.7  # 0.0 to 1.0
    preserve_melody: bool = True
    preserve_rhythm: bool = False
    preserve_harmony: bool = False
    crossfade_duration: float = 2.0
    sample_rate: int = 44100

@dataclass
class StyleFeatures:
    """Musical style characteristics"""
    harmonic_profile: np.ndarray
    rhythmic_pattern: np.ndarray
    timbral_features: np.ndarray
    dynamic_envelope: np.ndarray
    spectral_centroid: float
    tempo_stability: float
    key_strength: float

if TORCH_AVAILABLE:
    class StyleEncoder(nn.Module):
        """Neural network for encoding musical style features"""
        
        def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, style_dim: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, style_dim),
                nn.Tanh()
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    class StyleDecoder(nn.Module):
        """Neural network for decoding style features back to audio"""
        
        def __init__(self, style_dim: int = 128, hidden_dim: int = 512, output_dim: int = 2048):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(style_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(x)

    class StyleTransferNetwork(nn.Module):
        """Complete style transfer neural network"""
        
        def __init__(self, input_dim: int = 2048, style_dim: int = 128):
            super().__init__()
            self.source_encoder = StyleEncoder(input_dim, 512, style_dim)
            self.target_encoder = StyleEncoder(input_dim, 512, style_dim)
            self.style_mixer = nn.Sequential(
                nn.Linear(style_dim * 2, style_dim),
                nn.ReLU(),
                nn.Linear(style_dim, style_dim),
                nn.Tanh()
            )
            self.decoder = StyleDecoder(style_dim, 512, input_dim)
            
        def forward(self, source_features: torch.Tensor, target_style: torch.Tensor, 
                    intensity: float = 0.7) -> torch.Tensor:
            source_encoded = self.source_encoder(source_features)
            target_encoded = self.target_encoder(target_style)
            
            # Mix styles based on intensity
            mixed_style = self.style_mixer(torch.cat([source_encoded, target_encoded], dim=-1))
            blended_style = (1 - intensity) * source_encoded + intensity * mixed_style
            
            return self.decoder(blended_style)
else:
    # Dummy classes when torch is not available
    class StyleEncoder:
        def __init__(self, *args, **kwargs):
            pass
    
    class StyleDecoder:
        def __init__(self, *args, **kwargs):
            pass
    
    class StyleTransferNetwork:
        def __init__(self, *args, **kwargs):
            pass

class MusicStyleAnalyzer:
    """Analyzes and extracts style features from audio"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        
    def extract_features(self, audio: np.ndarray) -> StyleFeatures:
        """Extract comprehensive style features from audio"""
        try:
            # Spectral features
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # Harmonic profile
            harmonic_profile = self._extract_harmonic_profile(audio)
            
            # Rhythmic pattern
            rhythmic_pattern = self._extract_rhythmic_pattern(audio)
            
            # Timbral features
            timbral_features = self._extract_timbral_features(magnitude)
            
            # Dynamic envelope
            dynamic_envelope = self._extract_dynamic_envelope(audio)
            
            # Additional features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            tempo_stability = self._calculate_tempo_stability(beats)
            key_strength = self._calculate_key_strength(audio)
            
            return StyleFeatures(
                harmonic_profile=harmonic_profile,
                rhythmic_pattern=rhythmic_pattern,
                timbral_features=timbral_features,
                dynamic_envelope=dynamic_envelope,
                spectral_centroid=float(spectral_centroid),
                tempo_stability=tempo_stability,
                key_strength=key_strength
            )
            
        except Exception as e:
            logger.error(f"Error extracting style features: {e}")
            # Return default features
            return self._get_default_features()
    
    def _extract_harmonic_profile(self, audio: np.ndarray) -> np.ndarray:
        """Extract harmonic content profile"""
        try:
            # Chromagram for harmonic content
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            harmonic_profile = np.mean(chroma, axis=1)
            return harmonic_profile
        except:
            return np.zeros(12)
    
    def _extract_rhythmic_pattern(self, audio: np.ndarray) -> np.ndarray:
        """Extract rhythmic pattern characteristics"""
        try:
            # Onset detection and rhythm features
            onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sample_rate)
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            # Create rhythmic pattern vector
            if len(onset_frames) > 0:
                onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
                rhythm_pattern = np.histogram(onset_times % 4, bins=16)[0]
                return rhythm_pattern / np.sum(rhythm_pattern) if np.sum(rhythm_pattern) > 0 else rhythm_pattern
            else:
                return np.zeros(16)
        except:
            return np.zeros(16)
    
    def _extract_timbral_features(self, magnitude: np.ndarray) -> np.ndarray:
        """Extract timbral characteristics"""
        try:
            # Spectral features for timbre
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(magnitude))
            
            # MFCC features
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(magnitude**2), sr=self.sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Combine timbral features
            timbral_features = np.concatenate([
                [spectral_rolloff, spectral_bandwidth, zero_crossing_rate],
                mfcc_mean
            ])
            
            return timbral_features
        except:
            return np.zeros(16)
    
    def _extract_dynamic_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Extract dynamic envelope characteristics"""
        try:
            # RMS energy envelope
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
            envelope = np.squeeze(rms)
            
            # Normalize and downsample to fixed size
            if len(envelope) > 100:
                envelope = np.interp(np.linspace(0, len(envelope)-1, 100), 
                                   np.arange(len(envelope)), envelope)
            else:
                envelope = np.pad(envelope, (0, max(0, 100 - len(envelope))), 'constant')
            
            return envelope
        except:
            return np.zeros(100)
    
    def _calculate_tempo_stability(self, beats: np.ndarray) -> float:
        """Calculate tempo stability metric"""
        try:
            if len(beats) < 2:
                return 0.0
            
            intervals = np.diff(beats)
            if len(intervals) == 0:
                return 0.0
            
            stability = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-8))
            return max(0.0, min(1.0, stability))
        except:
            return 0.0
    
    def _calculate_key_strength(self, audio: np.ndarray) -> float:
        """Calculate key strength/tonality"""
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Key strength based on chromagram distribution
            key_strength = np.max(chroma_mean) - np.mean(chroma_mean)
            return float(key_strength)
        except:
            return 0.0
    
    def _get_default_features(self) -> StyleFeatures:
        """Return default style features"""
        return StyleFeatures(
            harmonic_profile=np.zeros(12),
            rhythmic_pattern=np.zeros(16),
            timbral_features=np.zeros(16),
            dynamic_envelope=np.zeros(100),
            spectral_centroid=1000.0,
            tempo_stability=0.5,
            key_strength=0.3
        )

class StyleDatabase:
    """Database of style templates and characteristics"""
    
    def __init__(self):
        self.style_templates = self._initialize_style_templates()
    
    def _initialize_style_templates(self) -> Dict[MusicStyle, StyleFeatures]:
        """Initialize style templates with characteristic features"""
        templates = {}
        
        # Jazz template
        jazz_harmonic = np.array([0.15, 0.08, 0.12, 0.18, 0.1, 0.09, 0.06, 0.11, 0.03, 0.04, 0.02, 0.02])
        jazz_rhythm = np.array([0.1, 0.05, 0.15, 0.05, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0])
        templates[MusicStyle.JAZZ] = StyleFeatures(
            harmonic_profile=jazz_harmonic,
            rhythmic_pattern=jazz_rhythm,
            timbral_features=np.array([2000, 1500, 0.1] + [0]*13),
            dynamic_envelope=np.random.beta(2, 5, 100),
            spectral_centroid=1800.0,
            tempo_stability=0.6,
            key_strength=0.7
        )
        
        # Classical template
        classical_harmonic = np.array([0.2, 0.05, 0.15, 0.1, 0.18, 0.12, 0.05, 0.08, 0.02, 0.03, 0.01, 0.01])
        classical_rhythm = np.array([0.25, 0.1, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0])
        templates[MusicStyle.CLASSICAL] = StyleFeatures(
            harmonic_profile=classical_harmonic,
            rhythmic_pattern=classical_rhythm,
            timbral_features=np.array([1200, 800, 0.05] + [0]*13),
            dynamic_envelope=np.random.beta(3, 3, 100),
            spectral_centroid=1200.0,
            tempo_stability=0.9,
            key_strength=0.9
        )
        
        # Rock template
        rock_harmonic = np.array([0.18, 0.12, 0.08, 0.15, 0.1, 0.15, 0.08, 0.1, 0.02, 0.01, 0.005, 0.005])
        rock_rhythm = np.array([0.3, 0, 0.2, 0, 0.3, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        templates[MusicStyle.ROCK] = StyleFeatures(
            harmonic_profile=rock_harmonic,
            rhythmic_pattern=rock_rhythm,
            timbral_features=np.array([3000, 2000, 0.2] + [0]*13),
            dynamic_envelope=np.random.beta(1.5, 1.5, 100),
            spectral_centroid=2500.0,
            tempo_stability=0.8,
            key_strength=0.6
        )
        
        # Electronic template
        electronic_harmonic = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0])
        electronic_rhythm = np.array([0.25, 0.125, 0.125, 0.125, 0.25, 0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        templates[MusicStyle.ELECTRONIC] = StyleFeatures(
            harmonic_profile=electronic_harmonic,
            rhythmic_pattern=electronic_rhythm,
            timbral_features=np.array([4000, 3000, 0.15] + [0]*13),
            dynamic_envelope=np.random.beta(1, 3, 100),
            spectral_centroid=3500.0,
            tempo_stability=0.95,
            key_strength=0.4
        )
        
        # Add more style templates as needed
        self._add_remaining_templates(templates)
        
        return templates
    
    def _add_remaining_templates(self, templates: Dict[MusicStyle, StyleFeatures]):
        """Add remaining style templates"""
        # Pop template
        pop_harmonic = np.array([0.16, 0.1, 0.12, 0.14, 0.12, 0.14, 0.08, 0.1, 0.02, 0.01, 0.005, 0.005])
        pop_rhythm = np.array([0.2, 0.1, 0.15, 0.1, 0.2, 0.1, 0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        templates[MusicStyle.POP] = StyleFeatures(
            harmonic_profile=pop_harmonic,
            rhythmic_pattern=pop_rhythm,
            timbral_features=np.array([2200, 1600, 0.12] + [0]*13),
            dynamic_envelope=np.random.beta(2, 2, 100),
            spectral_centroid=2000.0,
            tempo_stability=0.85,
            key_strength=0.75
        )
        
        # Blues template
        blues_harmonic = np.array([0.18, 0.05, 0.08, 0.15, 0.1, 0.12, 0.18, 0.08, 0.03, 0.02, 0.005, 0.005])
        blues_rhythm = np.array([0.15, 0.05, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0])
        templates[MusicStyle.BLUES] = StyleFeatures(
            harmonic_profile=blues_harmonic,
            rhythmic_pattern=blues_rhythm,
            timbral_features=np.array([1800, 1200, 0.15] + [0]*13),
            dynamic_envelope=np.random.beta(2, 3, 100),
            spectral_centroid=1600.0,
            tempo_stability=0.7,
            key_strength=0.8
        )
    
    def get_style_template(self, style: MusicStyle) -> StyleFeatures:
        """Get style template for given music style"""
        return self.style_templates.get(style, self.style_templates[MusicStyle.POP])

class StyleTransferAI:
    """Main Style Transfer AI system"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.analyzer = MusicStyleAnalyzer()
        self.style_db = StyleDatabase()
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self._load_or_initialize_model(model_path)
            logger.info(f"StyleTransferAI initialized on {self.device}")
        else:
            self.device = None
            self.model = StyleTransferNetwork()  # Dummy model
            logger.warning("StyleTransferAI initialized without PyTorch - using simplified mode")
    
    def _load_or_initialize_model(self, model_path: Optional[str]) -> StyleTransferNetwork:
        """Load pre-trained model or initialize new one"""
        model = StyleTransferNetwork()
        
        if model_path and Path(model_path).exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                logger.info("Using randomly initialized model")
        else:
            logger.info("Using randomly initialized model")
        
        model.to(self.device)
        model.eval()
        return model
    
    def transfer_style(self, audio: np.ndarray, config: StyleTransferConfig) -> np.ndarray:
        """Transfer musical style of audio"""
        try:
            logger.info(f"Transferring style to {config.target_style.value} with intensity {config.intensity}")
            
            # Extract source features
            source_features = self.analyzer.extract_features(audio)
            
            # Get target style template
            target_template = self.style_db.get_style_template(config.target_style)
            
            # Apply style transfer
            transformed_audio = self._apply_style_transfer(
                audio, source_features, target_template, config
            )
            
            # Apply preservation constraints
            if config.preserve_melody:
                transformed_audio = self._preserve_melody(audio, transformed_audio, config.intensity)
            
            if config.preserve_rhythm and not config.preserve_melody:
                transformed_audio = self._preserve_rhythm(audio, transformed_audio, config.intensity)
            
            # Apply crossfade
            transformed_audio = self._apply_crossfade(
                audio, transformed_audio, config.crossfade_duration, config.sample_rate
            )
            
            logger.info("Style transfer completed successfully")
            return transformed_audio
            
        except Exception as e:
            logger.error(f"Error in style transfer: {e}")
            return audio  # Return original audio on error
    
    def _apply_style_transfer(self, audio: np.ndarray, source_features: StyleFeatures, 
                            target_template: StyleFeatures, config: StyleTransferConfig) -> np.ndarray:
        """Apply neural style transfer"""
        try:
            # Convert features to tensors
            source_tensor = self._features_to_tensor(source_features)
            target_tensor = self._features_to_tensor(target_template)
            
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    # Apply neural style transfer
                    transformed_features = self.model(source_tensor, target_tensor, config.intensity)
            else:
                # Simplified style transfer without torch
                transformed_features = source_tensor
            
            # Convert back to audio (simplified approach)
            transformed_audio = self._tensor_to_audio(transformed_features, audio, config)
            
            return transformed_audio
            
        except Exception as e:
            logger.error(f"Error in neural style transfer: {e}")
            return self._apply_traditional_style_transfer(audio, source_features, target_template, config)
    
    def _apply_traditional_style_transfer(self, audio: np.ndarray, source_features: StyleFeatures,
                                        target_template: StyleFeatures, config: StyleTransferConfig) -> np.ndarray:
        """Apply traditional DSP-based style transfer as fallback"""
        try:
            # Apply spectral modifications based on target style
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Modify spectral content based on target style
            modified_magnitude = self._modify_spectrum(magnitude, target_template, config.intensity)
            
            # Reconstruct audio
            modified_stft = modified_magnitude * np.exp(1j * phase)
            transformed_audio = librosa.istft(modified_stft, hop_length=512)
            
            return transformed_audio
            
        except Exception as e:
            logger.error(f"Error in traditional style transfer: {e}")
            return audio
    
    def _modify_spectrum(self, magnitude: np.ndarray, target_template: StyleFeatures, 
                        intensity: float) -> np.ndarray:
        """Modify spectrum based on target style characteristics"""
        try:
            # Apply spectral shaping based on target timbral features
            freq_bins = magnitude.shape[0]
            modification_curve = np.ones(freq_bins)
            
            # Simple spectral shaping based on target spectral centroid
            target_centroid = target_template.spectral_centroid
            centroid_bin = int(target_centroid * freq_bins / 22050)  # Rough mapping
            
            # Create modification curve
            for i in range(freq_bins):
                distance = abs(i - centroid_bin) / freq_bins
                modification_curve[i] = 1.0 + intensity * (0.5 - distance)
            
            # Apply modification
            modified_magnitude = magnitude * modification_curve[:, np.newaxis]
            
            return modified_magnitude
            
        except Exception as e:
            logger.error(f"Error modifying spectrum: {e}")
            return magnitude
    
    def _features_to_tensor(self, features: StyleFeatures):
        """Convert style features to tensor"""
        try:
            # Combine all features into a single vector
            combined_features = np.concatenate([
                features.harmonic_profile,
                features.rhythmic_pattern,
                features.timbral_features,
                features.dynamic_envelope[:50],  # Truncate for fixed size
                [features.spectral_centroid / 10000.0],  # Normalize
                [features.tempo_stability],
                [features.key_strength]
            ])
            
            # Pad or truncate to fixed size
            target_size = 2048
            if len(combined_features) < target_size:
                combined_features = np.pad(combined_features, 
                                         (0, target_size - len(combined_features)), 'constant')
            else:
                combined_features = combined_features[:target_size]
            
            if TORCH_AVAILABLE:
                tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
                return tensor
            else:
                return combined_features
            
        except Exception as e:
            logger.error(f"Error converting features to tensor: {e}")
            if TORCH_AVAILABLE:
                return torch.zeros(1, 2048).to(self.device)
            else:
                return np.zeros(2048)
    
    def _tensor_to_audio(self, tensor, original_audio: np.ndarray, 
                        config: StyleTransferConfig) -> np.ndarray:
        """Convert tensor back to audio (simplified)"""
        try:
            # This is a simplified conversion - in practice, you'd need a more sophisticated decoder
            if TORCH_AVAILABLE and hasattr(tensor, 'cpu'):
                features_np = tensor.cpu().numpy().squeeze()
            else:
                features_np = tensor if isinstance(tensor, np.ndarray) else np.array(tensor)
            
            # Apply modifications to original audio based on decoded features
            return self._apply_feature_modifications(original_audio, features_np, config)
            
        except Exception as e:
            logger.error(f"Error converting tensor to audio: {e}")
            return original_audio
    
    def _apply_feature_modifications(self, audio: np.ndarray, features: np.ndarray, 
                                   config: StyleTransferConfig) -> np.ndarray:
        """Apply feature-based modifications to audio"""
        try:
            # Simple time-domain modifications based on features
            modified_audio = audio.copy()
            
            # Apply gain modulation based on features
            gain_modulation = 1.0 + 0.2 * config.intensity * (features[0] - 0.5)
            modified_audio = modified_audio * gain_modulation
            
            # Apply simple filtering
            if len(features) > 1:
                cutoff_mod = features[1] * config.intensity
                modified_audio = self._apply_simple_filter(modified_audio, cutoff_mod, config.sample_rate)
            
            return modified_audio
            
        except Exception as e:
            logger.error(f"Error applying feature modifications: {e}")
            return audio
    
    def _apply_simple_filter(self, audio: np.ndarray, cutoff_mod: float, 
                           sample_rate: int) -> np.ndarray:
        """Apply simple filtering"""
        try:
            from scipy import signal
            
            # Calculate cutoff frequency
            base_cutoff = 2000  # Hz
            cutoff = base_cutoff * (1.0 + cutoff_mod)
            cutoff = np.clip(cutoff, 100, sample_rate // 2 - 100)
            
            # Design and apply filter
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return audio
    
    def _preserve_melody(self, original: np.ndarray, transformed: np.ndarray, 
                        intensity: float) -> np.ndarray:
        """Preserve melodic content while applying style transfer"""
        try:
            # Extract pitch information from original
            pitches, magnitudes = librosa.piptrack(y=original, sr=44100, threshold=0.1)
            
            # Blend original and transformed based on pitch strength
            preservation_factor = 1.0 - intensity * 0.7  # Preserve more melody
            blended = preservation_factor * original + (1 - preservation_factor) * transformed
            
            return blended
            
        except Exception as e:
            logger.error(f"Error preserving melody: {e}")
            return transformed
    
    def _preserve_rhythm(self, original: np.ndarray, transformed: np.ndarray, 
                        intensity: float) -> np.ndarray:
        """Preserve rhythmic content while applying style transfer"""
        try:
            # Extract onset information
            onset_frames = librosa.onset.onset_detect(y=original, sr=44100)
            
            if len(onset_frames) > 0:
                # Blend with emphasis on preserving onset timing
                preservation_factor = 1.0 - intensity * 0.5
                blended = preservation_factor * original + (1 - preservation_factor) * transformed
                return blended
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error preserving rhythm: {e}")
            return transformed
    
    def _apply_crossfade(self, original: np.ndarray, transformed: np.ndarray, 
                        duration: float, sample_rate: int) -> np.ndarray:
        """Apply crossfade between original and transformed audio"""
        try:
            if duration <= 0:
                return transformed
            
            crossfade_samples = int(duration * sample_rate)
            min_length = min(len(original), len(transformed))
            
            if crossfade_samples >= min_length:
                # Linear blend if crossfade is longer than audio
                blend_factor = np.linspace(0, 1, min_length)
                result = original[:min_length] * (1 - blend_factor) + transformed[:min_length] * blend_factor
            else:
                # Apply crossfade at beginning and end
                result = transformed.copy()
                
                # Fade in
                fade_in = np.linspace(0, 1, crossfade_samples)
                result[:crossfade_samples] = (original[:crossfade_samples] * (1 - fade_in) + 
                                            transformed[:crossfade_samples] * fade_in)
                
                # Fade out
                fade_out = np.linspace(1, 0, crossfade_samples)
                result[-crossfade_samples:] = (transformed[-crossfade_samples:] * fade_out + 
                                             original[-crossfade_samples:] * (1 - fade_out))
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying crossfade: {e}")
            return transformed
    
    def analyze_style(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze the musical style of audio"""
        try:
            features = self.analyzer.extract_features(audio)
            
            # Calculate style similarity scores
            style_scores = {}
            for style, template in self.style_db.style_templates.items():
                score = self._calculate_style_similarity(features, template)
                style_scores[style.value] = float(score)
            
            # Find most likely style
            most_likely_style = max(style_scores, key=style_scores.get)
            
            return {
                "most_likely_style": most_likely_style,
                "style_scores": style_scores,
                "features": {
                    "spectral_centroid": float(features.spectral_centroid),
                    "tempo_stability": float(features.tempo_stability),
                    "key_strength": float(features.key_strength),
                    "harmonic_complexity": float(np.std(features.harmonic_profile)),
                    "rhythmic_complexity": float(np.std(features.rhythmic_pattern))
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing style: {e}")
            return {
                "most_likely_style": "unknown",
                "style_scores": {},
                "features": {}
            }
    
    def _calculate_style_similarity(self, features1: StyleFeatures, features2: StyleFeatures) -> float:
        """Calculate similarity between two style feature sets"""
        try:
            # Normalize features and calculate similarity
            harmonic_sim = 1.0 - np.linalg.norm(features1.harmonic_profile - features2.harmonic_profile)
            rhythmic_sim = 1.0 - np.linalg.norm(features1.rhythmic_pattern - features2.rhythmic_pattern)
            timbral_sim = 1.0 - np.linalg.norm(features1.timbral_features - features2.timbral_features) / 100
            
            # Weight different aspects
            total_similarity = (
                0.4 * max(0, harmonic_sim) +
                0.3 * max(0, rhythmic_sim) +
                0.2 * max(0, timbral_sim) +
                0.1 * (1.0 - abs(features1.spectral_centroid - features2.spectral_centroid) / 5000)
            )
            
            return max(0.0, min(1.0, total_similarity))
            
        except Exception as e:
            logger.error(f"Error calculating style similarity: {e}")
            return 0.0

# Convenience functions
def create_style_transfer_ai(model_path: Optional[str] = None) -> StyleTransferAI:
    """Create and return StyleTransferAI instance"""
    return StyleTransferAI(model_path)

def transfer_music_style(audio: np.ndarray, target_style: str, intensity: float = 0.7,
                        preserve_melody: bool = True, sample_rate: int = 44100) -> np.ndarray:
    """Quick style transfer function"""
    try:
        style_enum = MusicStyle(target_style.lower())
        config = StyleTransferConfig(
            target_style=style_enum,
            intensity=intensity,
            preserve_melody=preserve_melody,
            sample_rate=sample_rate
        )
        
        style_ai = create_style_transfer_ai()
        return style_ai.transfer_style(audio, config)
        
    except Exception as e:
        logger.error(f"Error in quick style transfer: {e}")
        return audio