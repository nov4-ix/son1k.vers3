"""
Intelligent Music Generation Service with Suno Integration and Fallback
Combines Suno AI with Enhanced MusicGen for reliable, high-quality music generation
"""

import asyncio
import logging
import time
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .suno_client import SecureSunoClient, get_suno_client, SunoResponse
from .enhanced_musicgen import EnhancedMusicGenService, GenerationConfig, AudioQuality

logger = logging.getLogger(__name__)

class GenerationProvider(Enum):
    SUNO = "suno"
    MUSICGEN = "musicgen"
    HYBRID = "hybrid"

@dataclass
class GenerationRequest:
    """Standardized generation request"""
    prompt: str
    duration: float = 8.0
    temperature: float = 1.0
    top_k: int = 250
    apply_postprocessing: bool = True
    preferred_provider: GenerationProvider = GenerationProvider.SUNO
    quality: AudioQuality = AudioQuality.STANDARD
    user_tier: str = "free"

@dataclass
class GenerationResult:
    """Standardized generation result"""
    success: bool
    audio: Optional[np.ndarray] = None
    sample_rate: int = 32000
    metadata: Optional[Dict[str, Any]] = None
    provider_used: Optional[GenerationProvider] = None
    generation_time: float = 0.0
    error: Optional[str] = None
    fallback_used: bool = False

class IntelligentGenerationService:
    """
    Intelligent music generation service that automatically selects the best provider
    and implements sophisticated fallback mechanisms
    """
    
    def __init__(self):
        self.suno_client = get_suno_client()
        self.musicgen_service = EnhancedMusicGenService()
        self.generation_stats = {
            'suno_success': 0,
            'suno_failures': 0,
            'musicgen_used': 0,
            'total_generations': 0
        }
        
        # Provider selection logic
        self.provider_weights = {
            GenerationProvider.SUNO: 1.0,
            GenerationProvider.MUSICGEN: 0.8,
            GenerationProvider.HYBRID: 0.9
        }
        
        logger.info("Intelligent Generation Service initialized")
        logger.info(f"Suno available: {self.suno_client is not None}")
        logger.info(f"MusicGen available: {self.musicgen_service.is_loaded()}")
    
    async def generate_music(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate music using intelligent provider selection and fallback
        """
        start_time = time.time()
        self.generation_stats['total_generations'] += 1
        
        # Analyze request to determine best provider
        optimal_provider = await self._select_optimal_provider(request)
        
        logger.info(f"Selected provider: {optimal_provider.value} for prompt: {request.prompt[:50]}...")
        
        # Try primary provider
        result = await self._try_provider(optimal_provider, request)
        
        # If primary fails, try fallback
        if not result.success and optimal_provider != GenerationProvider.MUSICGEN:
            logger.warning(f"Primary provider {optimal_provider.value} failed, falling back to MusicGen")
            fallback_request = self._adapt_request_for_fallback(request)
            result = await self._try_provider(GenerationProvider.MUSICGEN, fallback_request)
            result.fallback_used = True
        
        # Final processing
        result.generation_time = time.time() - start_time
        
        # Update statistics
        self._update_stats(result)
        
        logger.info(f"Generation completed in {result.generation_time:.2f}s using {result.provider_used.value if result.provider_used else 'none'}")
        
        return result
    
    async def _select_optimal_provider(self, request: GenerationRequest) -> GenerationProvider:
        """
        Select the optimal provider based on request characteristics and system state
        """
        
        # Check user preferences
        if request.preferred_provider != GenerationProvider.SUNO:
            return request.preferred_provider
        
        # Check if Suno is available
        if not self.suno_client:
            return GenerationProvider.MUSICGEN
        
        # Quick health check for Suno
        suno_available = False
        try:
            async with self.suno_client as client:
                suno_available = await client.health_check()
        except Exception as e:
            logger.warning(f"Suno health check failed: {e}")
        
        if not suno_available:
            logger.info("Suno unavailable, using MusicGen")
            return GenerationProvider.MUSICGEN
        
        # Analyze request characteristics
        prompt_analysis = self._analyze_prompt_for_provider(request.prompt)
        
        # Decision logic based on prompt analysis
        if prompt_analysis['complexity'] > 0.8:
            # High complexity - prefer MusicGen for better control
            return GenerationProvider.MUSICGEN
        elif prompt_analysis['duration'] > 30:
            # Long duration - prefer MusicGen (Suno has limits)
            return GenerationProvider.MUSICGEN
        elif prompt_analysis['style_confidence'] > 0.8:
            # Well-defined style - Suno might be better
            return GenerationProvider.SUNO
        elif request.user_tier == "free" and self._get_suno_success_rate() < 0.7:
            # Low success rate for free users - use MusicGen for reliability
            return GenerationProvider.MUSICGEN
        else:
            # Default to Suno for general cases
            return GenerationProvider.SUNO
    
    def _analyze_prompt_for_provider(self, prompt: str) -> Dict[str, float]:
        """Analyze prompt to determine which provider would be optimal"""
        prompt_lower = prompt.lower()
        
        # Complexity analysis
        complexity_indicators = [
            'complex', 'intricate', 'layered', 'sophisticated', 'advanced',
            'polyrhythm', 'counterpoint', 'modulation', 'time signature'
        ]
        complexity = sum(1 for indicator in complexity_indicators if indicator in prompt_lower) / len(complexity_indicators)
        
        # Style confidence
        clear_styles = [
            'rock', 'pop', 'jazz', 'classical', 'electronic', 'blues',
            'country', 'reggae', 'hip hop', 'folk', 'ambient'
        ]
        style_mentions = sum(1 for style in clear_styles if style in prompt_lower)
        style_confidence = min(style_mentions / 2, 1.0)  # Cap at 1.0
        
        # Duration estimate from prompt
        duration_indicators = {
            'short': 0.2, 'quick': 0.2, 'brief': 0.2,
            'long': 0.8, 'extended': 0.9, 'epic': 1.0
        }
        duration_score = 0.5  # Default
        for indicator, score in duration_indicators.items():
            if indicator in prompt_lower:
                duration_score = score
                break
        
        return {
            'complexity': complexity,
            'style_confidence': style_confidence,
            'duration': duration_score
        }
    
    async def _try_provider(self, provider: GenerationProvider, request: GenerationRequest) -> GenerationResult:
        """Try generating music with a specific provider"""
        
        if provider == GenerationProvider.SUNO:
            return await self._generate_with_suno(request)
        elif provider == GenerationProvider.MUSICGEN:
            return await self._generate_with_musicgen(request)
        elif provider == GenerationProvider.HYBRID:
            return await self._generate_hybrid(request)
        else:
            return GenerationResult(
                success=False,
                error=f"Unknown provider: {provider.value}"
            )
    
    async def _generate_with_suno(self, request: GenerationRequest) -> GenerationResult:
        """Generate music using Suno AI"""
        if not self.suno_client:
            return GenerationResult(
                success=False,
                error="Suno client not available"
            )
        
        try:
            async with self.suno_client as client:
                suno_response = await client.generate_music(
                    prompt=request.prompt,
                    duration=request.duration,
                    temperature=request.temperature
                )
                
                if suno_response.success and suno_response.audio_url:
                    # Download and process the audio
                    audio_data = await self._download_suno_audio(suno_response.audio_url)
                    
                    if audio_data is not None:
                        self.generation_stats['suno_success'] += 1
                        return GenerationResult(
                            success=True,
                            audio=audio_data,
                            sample_rate=32000,  # Assuming Suno's sample rate
                            metadata=suno_response.metadata,
                            provider_used=GenerationProvider.SUNO
                        )
                
                self.generation_stats['suno_failures'] += 1
                return GenerationResult(
                    success=False,
                    error=suno_response.error or "Suno generation failed",
                    provider_used=GenerationProvider.SUNO
                )
                
        except Exception as e:
            self.generation_stats['suno_failures'] += 1
            logger.error(f"Suno generation error: {e}")
            return GenerationResult(
                success=False,
                error=f"Suno error: {str(e)}",
                provider_used=GenerationProvider.SUNO
            )
    
    async def _generate_with_musicgen(self, request: GenerationRequest) -> GenerationResult:
        """Generate music using Enhanced MusicGen"""
        try:
            audio, metadata = await self.musicgen_service.generate_music(
                prompt=request.prompt,
                duration=request.duration,
                temperature=request.temperature
            )
            
            self.generation_stats['musicgen_used'] += 1
            
            return GenerationResult(
                success=True,
                audio=audio,
                sample_rate=self.musicgen_service.config.sample_rate,
                metadata=metadata,
                provider_used=GenerationProvider.MUSICGEN
            )
            
        except Exception as e:
            logger.error(f"MusicGen generation error: {e}")
            return GenerationResult(
                success=False,
                error=f"MusicGen error: {str(e)}",
                provider_used=GenerationProvider.MUSICGEN
            )
    
    async def _generate_hybrid(self, request: GenerationRequest) -> GenerationResult:
        """Generate music using hybrid approach (both providers)"""
        # This could involve using Suno for melody and MusicGen for harmony,
        # or other sophisticated combinations
        # For now, we'll try Suno first then enhance with MusicGen
        
        suno_result = await self._generate_with_suno(request)
        
        if suno_result.success:
            # Enhance Suno result with MusicGen processing
            try:
                # Add some MusicGen-style processing/enhancement
                enhanced_audio = await self._enhance_with_musicgen(
                    suno_result.audio, 
                    request.prompt
                )
                
                suno_result.audio = enhanced_audio
                suno_result.provider_used = GenerationProvider.HYBRID
                suno_result.metadata['hybrid_processing'] = True
                
                return suno_result
                
            except Exception as e:
                logger.warning(f"Hybrid enhancement failed: {e}")
                return suno_result  # Return original Suno result
        else:
            # Fallback to pure MusicGen
            return await self._generate_with_musicgen(request)
    
    async def _download_suno_audio(self, audio_url: str) -> Optional[np.ndarray]:
        """Download and convert Suno audio to numpy array"""
        try:
            import aiohttp
            import io
            import soundfile as sf
            
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status == 200:
                        audio_bytes = await response.read()
                        
                        # Convert bytes to audio array
                        audio_io = io.BytesIO(audio_bytes)
                        audio_data, sample_rate = sf.read(audio_io)
                        
                        # Resample if necessary (simplified - would need proper resampling)
                        if sample_rate != 32000:
                            logger.warning(f"Sample rate {sample_rate} != 32000, using as-is")
                        
                        # Convert to mono if stereo
                        if len(audio_data.shape) > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        
                        return audio_data
                    else:
                        logger.error(f"Failed to download audio: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error downloading Suno audio: {e}")
            return None
    
    async def _enhance_with_musicgen(self, audio: np.ndarray, prompt: str) -> np.ndarray:
        """Enhance Suno audio using MusicGen techniques"""
        # This is a placeholder for sophisticated audio enhancement
        # Could involve:
        # - Adding harmonic richness
        # - Improving dynamics
        # - Style-specific processing
        
        # For now, just apply some basic enhancement
        enhanced = audio.copy()
        
        # Gentle compression
        peak = np.max(np.abs(enhanced))
        if peak > 0:
            enhanced = enhanced / peak * 0.9
        
        # Add slight harmonic emphasis based on prompt analysis
        analysis = self.musicgen_service.analyze_prompt(prompt)
        if analysis['style']['primary'] == 'rock':
            # Add slight saturation
            enhanced = np.tanh(enhanced * 1.1) * 0.9
        
        return enhanced
    
    def _adapt_request_for_fallback(self, request: GenerationRequest) -> GenerationRequest:
        """Adapt request parameters for fallback provider"""
        # Create a copy with adjusted parameters for MusicGen
        fallback_request = GenerationRequest(
            prompt=request.prompt,
            duration=min(request.duration, 30),  # MusicGen handles shorter durations better
            temperature=request.temperature * 0.9,  # Slightly less random for reliability
            top_k=request.top_k,
            apply_postprocessing=True,
            preferred_provider=GenerationProvider.MUSICGEN,
            quality=request.quality,
            user_tier=request.user_tier
        )
        return fallback_request
    
    def _get_suno_success_rate(self) -> float:
        """Calculate Suno success rate"""
        total_suno = self.generation_stats['suno_success'] + self.generation_stats['suno_failures']
        if total_suno == 0:
            return 1.0  # Assume good until proven otherwise
        return self.generation_stats['suno_success'] / total_suno
    
    def _update_stats(self, result: GenerationResult):
        """Update generation statistics"""
        # Statistics are already updated in the respective generation methods
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        suno_available = self.suno_client is not None
        suno_healthy = False
        
        if suno_available:
            try:
                # Quick sync health check (not ideal, but for status)
                import asyncio
                async def check():
                    async with self.suno_client as client:
                        return await client.health_check()
                
                # This is not ideal in production - should be cached
                try:
                    suno_healthy = asyncio.get_event_loop().run_until_complete(check())
                except:
                    suno_healthy = False
            except:
                suno_healthy = False
        
        return {
            'providers': {
                'suno': {
                    'available': suno_available,
                    'healthy': suno_healthy,
                    'success_rate': self._get_suno_success_rate()
                },
                'musicgen': {
                    'available': True,
                    'healthy': self.musicgen_service.is_loaded(),
                    'device': self.musicgen_service.get_device()
                }
            },
            'statistics': self.generation_stats.copy(),
            'intelligent_routing': True,
            'fallback_enabled': True
        }
    
    async def test_all_providers(self) -> Dict[str, Any]:
        """Test all available providers"""
        test_prompt = "a short peaceful melody with piano"
        test_request = GenerationRequest(
            prompt=test_prompt,
            duration=5.0,
            temperature=1.0
        )
        
        results = {}
        
        # Test Suno
        if self.suno_client:
            try:
                suno_result = await self._generate_with_suno(test_request)
                results['suno'] = {
                    'success': suno_result.success,
                    'error': suno_result.error,
                    'generation_time': suno_result.generation_time
                }
            except Exception as e:
                results['suno'] = {'success': False, 'error': str(e)}
        else:
            results['suno'] = {'success': False, 'error': 'Not configured'}
        
        # Test MusicGen
        try:
            musicgen_result = await self._generate_with_musicgen(test_request)
            results['musicgen'] = {
                'success': musicgen_result.success,
                'error': musicgen_result.error,
                'generation_time': musicgen_result.generation_time
            }
        except Exception as e:
            results['musicgen'] = {'success': False, 'error': str(e)}
        
        return results

# Global service instance
_intelligent_service: Optional[IntelligentGenerationService] = None

def get_intelligent_service() -> IntelligentGenerationService:
    """Get the global intelligent generation service"""
    global _intelligent_service
    if _intelligent_service is None:
        _intelligent_service = IntelligentGenerationService()
    return _intelligent_service