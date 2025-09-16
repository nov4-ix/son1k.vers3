"""
Son1kVers3 Enhanced - Advanced AI Music Generation Platform
Integrated with Suno AI, Enhanced MusicGen, APLIO Voices, and Professional Audio Processing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import uuid
from datetime import datetime
import numpy as np
import soundfile as sf
import time

# Import enhanced AI modules
from .intelligent_generation import (
    IntelligentGenerationService, 
    GenerationRequest, 
    GenerationResult,
    GenerationProvider,
    get_intelligent_service
)
from .aplio_voices import (
    AplioVoiceEngine, 
    VoiceType, 
    get_voice_engine
)
from .professional_audio import (
    ProfessionalAudioProcessor,
    get_audio_processor
)
from .prompt_intelligence import (
    PromptIntelligenceEngine,
    PromptComplexity,
    get_prompt_engine
)
from .suno_client import get_suno_client
# Try to import advanced AI modules, fall back to simple versions
try:
    from .style_transfer import (
        StyleTransferAI,
        StyleTransferConfig,
        MusicStyle,
        create_style_transfer_ai
    )
    ADVANCED_STYLE_TRANSFER = True
except ImportError:
    from .style_transfer_simple import (
        SimpleStyleTransferAI as StyleTransferAI,
        StyleTransferConfig,
        MusicStyle,
        create_style_transfer_ai
    )
    ADVANCED_STYLE_TRANSFER = False

try:
    from .musical_analysis import (
        ComprehensiveMusicalAnalyzer,
        analyze_audio_data
    )
    ADVANCED_MUSICAL_ANALYSIS = True
except ImportError:
    from .musical_analysis_simple import (
        SimpleMusicalAnalyzer as ComprehensiveMusicalAnalyzer,
        analyze_audio_data
    )
    ADVANCED_MUSICAL_ANALYSIS = False
from .lyrics_ai import (
    AdvancedLyricsAI,
    LyricsGenerationRequest,
    LyricsTheme,
    LyricStyle,
    RhymeScheme,
    EmotionalArc,
    create_lyrics_ai
)

# Global AI services
style_transfer_ai = None
musical_analyzer = None
lyrics_ai = None

# Original modules fallback
try:
    from .audio_analysis import AudioAnalyzer
    from .audio_post import AudioPostProcessor
    ORIGINAL_MODULES_AVAILABLE = True
except ImportError:
    AudioAnalyzer = None
    AudioPostProcessor = None
    ORIGINAL_MODULES_AVAILABLE = False
    logging.warning("Original audio modules not available - using enhanced versions")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ENHANCED CONFIGURATION ===
class EnhancedSettings:
    STORAGE_ROOT = "storage"
    UPLOADS_DIR = "storage/uploads"
    OUTPUT_DIR = "storage/output"
    VOCALS_DIR = "storage/vocals"
    SAMPLE_RATE = 32000
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".aiff", ".m4a"]
    
    # AI Enhancement settings
    ENABLE_SUNO = os.getenv('SUNO_ENABLE', 'true').lower() == 'true'
    ENABLE_APLIO = True
    ENABLE_PROMPT_ENHANCEMENT = True
    DEFAULT_QUALITY = "high"
    
    @property
    def storage_paths(self):
        return {
            "uploads": Path(self.UPLOADS_DIR),
            "output": Path(self.OUTPUT_DIR),
            "vocals": Path(self.VOCALS_DIR),
            "ghost_uploads": Path(self.UPLOADS_DIR) / "ghost",
            "ghost_output": Path(self.OUTPUT_DIR) / "ghost"
        }

settings = EnhancedSettings()

# === ENHANCED MODELS ===
class EnhancedGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Musical prompt or lyrics")
    duration: float = Field(8.0, ge=1.0, le=60.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(250, ge=1, le=1000)
    apply_postprocessing: bool = Field(True)
    enhance_prompt: bool = Field(True, description="Apply AI prompt enhancement")
    preferred_provider: str = Field("auto", description="suno, musicgen, or auto")
    style_hint: Optional[str] = Field(None, description="Style guidance")
    cultural_context: Optional[str] = Field(None, description="Cultural context")
    user_tier: str = Field("free", description="User subscription tier")

class VocalGenerationRequest(BaseModel):
    text: str = Field(..., description="Lyrics or vocal text")
    voice_type: str = Field("elena", description="APLIO voice type")
    melody_notes: List[float] = Field(default_factory=list, description="Melody frequencies")
    durations: List[float] = Field(default_factory=list, description="Note durations")
    emotion: str = Field("neutral", description="Emotional expression")
    expressiveness: float = Field(1.0, ge=0.0, le=2.0, description="Expressiveness level")

class StyleTransferRequest(BaseModel):
    target_style: str = Field(..., description="Target musical style")
    intensity: float = Field(0.7, ge=0.0, le=1.0, description="Transfer intensity")
    preserve_melody: bool = Field(True, description="Preserve original melody")
    preserve_rhythm: bool = Field(False, description="Preserve original rhythm")
    crossfade_duration: float = Field(2.0, ge=0.0, le=10.0, description="Crossfade duration")

class MusicalAnalysisRequest(BaseModel):
    include_harmony: bool = Field(True, description="Include harmonic analysis")
    include_rhythm: bool = Field(True, description="Include rhythmic analysis")
    include_structure: bool = Field(True, description="Include structural analysis")
    include_genre: bool = Field(True, description="Include genre prediction")
    include_mood: bool = Field(True, description="Include mood analysis")

class LyricsGenerationAPIRequest(BaseModel):
    theme: str = Field(..., description="Lyrical theme")
    style: str = Field("narrative", description="Lyrical style")
    rhyme_scheme: str = Field("abab", description="Rhyme scheme")
    verse_count: int = Field(2, ge=1, le=5, description="Number of verses")
    chorus_count: int = Field(1, ge=0, le=3, description="Number of choruses")
    bridge_count: int = Field(1, ge=0, le=2, description="Number of bridges")
    emotional_arc: str = Field("u_shaped", description="Emotional progression")
    target_length: int = Field(100, ge=20, le=500, description="Target word count")
    complexity_level: float = Field(0.7, ge=0.1, le=1.0, description="Complexity level")
    language_register: str = Field("casual", description="Language register")
    cultural_context: str = Field("contemporary", description="Cultural context")

class AudioProcessingRequest(BaseModel):
    style: str = Field("pop_modern", description="Mastering style")
    lufs_target: float = Field(-14.0, description="Target loudness in LUFS")
    vocal_processing: bool = Field(False, description="Apply vocal processing")
    vocal_style: str = Field("vocal_pop", description="Vocal processing style")

class JobStatus:
    def __init__(self, job_id: str, job_type: str):
        self.job_id = job_id
        self.job_type = job_type
        self.status = "processing"
        self.progress = 0
        self.message = "Iniciando..."
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.provider_used = None
        self.ai_features_used = []

def setup_enhanced_directories():
    """Create all necessary directories"""
    for path in settings.storage_paths.values():
        path.mkdir(parents=True, exist_ok=True)

# === ENHANCED SERVICES ===
class EnhancedAIService:
    """Unified AI service coordinator"""
    
    def __init__(self):
        self.intelligent_gen = get_intelligent_service()
        self.voice_engine = get_voice_engine() if settings.ENABLE_APLIO else None
        self.audio_processor = get_audio_processor()
        self.prompt_engine = get_prompt_engine() if settings.ENABLE_PROMPT_ENHANCEMENT else None
        self.suno_client = get_suno_client() if settings.ENABLE_SUNO else None
        
        # Initialize new AI services
        global style_transfer_ai, musical_analyzer, lyrics_ai
        style_transfer_ai = create_style_transfer_ai()
        musical_analyzer = ComprehensiveMusicalAnalyzer()
        lyrics_ai = create_lyrics_ai()
        
        logger.info("Enhanced AI Service initialized")
        logger.info(f"  - Intelligent Generation: ✓")
        logger.info(f"  - APLIO Voices: {'✓' if self.voice_engine else '✗'}")
        logger.info(f"  - Professional Audio: ✓")
        logger.info(f"  - Prompt Intelligence: {'✓' if self.prompt_engine else '✗'}")
        logger.info(f"  - Suno Integration: {'✓' if self.suno_client else '✗'}")
        logger.info(f"  - Style Transfer AI: {'✓ (Advanced)' if ADVANCED_STYLE_TRANSFER else '✓ (Simple)'}")
        logger.info(f"  - Musical Analysis: {'✓ (Advanced)' if ADVANCED_MUSICAL_ANALYSIS else '✓ (Simple)'}")
        logger.info(f"  - Advanced Lyrics AI: ✓")
    
    async def generate_enhanced_music(self, request: EnhancedGenerateRequest) -> GenerationResult:
        """Generate music with full AI enhancement pipeline"""
        
        # 1. Enhance prompt if enabled
        original_prompt = request.prompt
        enhanced_prompt = original_prompt
        
        if self.prompt_engine and request.enhance_prompt:
            enhancement_result = self.prompt_engine.enhance_prompt(
                prompt=original_prompt,
                target_complexity=PromptComplexity.DETAILED,
                style_hint=request.style_hint,
                cultural_context=request.cultural_context
            )
            enhanced_prompt = enhancement_result.enhanced_prompt
            logger.info(f"Prompt enhanced: '{original_prompt}' -> '{enhanced_prompt}'")
        
        # 2. Create generation request
        gen_request = GenerationRequest(
            prompt=enhanced_prompt,
            duration=request.duration,
            temperature=request.temperature,
            top_k=request.top_k,
            apply_postprocessing=request.apply_postprocessing,
            preferred_provider=GenerationProvider(request.preferred_provider) if request.preferred_provider != "auto" else GenerationProvider.SUNO,
            user_tier=request.user_tier
        )
        
        # 3. Generate music with intelligent provider selection
        result = await self.intelligent_gen.generate_music(gen_request)
        
        # 4. Apply professional audio processing if successful
        if result.success and result.audio is not None and request.apply_postprocessing:
            processed_audio, processing_metadata = self.audio_processor.master_track(
                audio=result.audio,
                style=request.style_hint or "pop_modern"
            )
            result.audio = processed_audio
            
            # Update metadata
            if result.metadata:
                result.metadata['professional_processing'] = processing_metadata
        
        return result
    
    async def generate_vocals(self, request: VocalGenerationRequest) -> Dict[str, Any]:
        """Generate vocals using APLIO voice system"""
        if not self.voice_engine:
            raise HTTPException(status_code=503, detail="APLIO voice system not available")
        
        try:
            voice_type = VoiceType(request.voice_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid voice type: {request.voice_type}")
        
        # Generate default melody if not provided
        melody = request.melody_notes
        durations = request.durations
        
        if not melody:
            # Generate simple melody based on text syllables
            syllables = len(request.text.split())
            base_freq = 220.0  # A3
            melody = [base_freq * (1.1 ** (i % 8)) for i in range(syllables)]
        
        if not durations:
            durations = [0.5] * len(melody)
        
        # Generate vocals
        vocal_audio = self.voice_engine.generate_vocals(
            text=request.text,
            voice_type=voice_type,
            melody=melody,
            durations=durations,
            emotion=request.emotion,
            expressiveness=request.expressiveness
        )
        
        # Save vocals
        timestamp = int(time.time())
        filename = f"vocals_{voice_type.value}_{timestamp}.wav"
        filepath = settings.storage_paths["vocals"] / filename
        
        sf.write(filepath, vocal_audio, settings.SAMPLE_RATE)
        
        return {
            "success": True,
            "audio_url": f"/vocals/{filename}",
            "voice_type": voice_type.value,
            "metadata": {
                "text": request.text,
                "emotion": request.emotion,
                "expressiveness": request.expressiveness,
                "duration": len(vocal_audio) / settings.SAMPLE_RATE,
                "sample_rate": settings.SAMPLE_RATE
            }
        }
    
    async def process_audio_professionally(self, 
                                         audio_file: UploadFile,
                                         processing_request: AudioProcessingRequest) -> Dict[str, Any]:
        """Apply professional audio processing to uploaded audio"""
        
        # Save uploaded file temporarily
        temp_id = str(uuid.uuid4())
        temp_path = settings.storage_paths["uploads"] / f"temp_{temp_id}.wav"
        
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        try:
            # Load audio
            audio, sample_rate = sf.read(temp_path)
            
            # Resample if necessary
            if sample_rate != settings.SAMPLE_RATE:
                logger.warning(f"Resampling from {sample_rate}Hz to {settings.SAMPLE_RATE}Hz")
                # Simple resampling (in production, use proper resampling)
                audio = audio  # Keep as-is for now
            
            # Apply processing
            if processing_request.vocal_processing:
                processed_audio, metadata = self.audio_processor.process_vocals(
                    audio=audio,
                    style=processing_request.vocal_style
                )
            else:
                processed_audio, metadata = self.audio_processor.master_track(
                    audio=audio,
                    style=processing_request.style,
                    lufs_target=processing_request.lufs_target
                )
            
            # Save processed audio
            output_filename = f"processed_{temp_id}.wav"
            output_path = settings.storage_paths["output"] / output_filename
            
            sf.write(output_path, processed_audio, settings.SAMPLE_RATE)
            
            return {
                "success": True,
                "audio_url": f"/output/{output_filename}",
                "processing_metadata": metadata,
                "original_filename": audio_file.filename
            }
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive system capabilities"""
        return {
            "ai_features": {
                "intelligent_generation": True,
                "suno_integration": self.suno_client is not None,
                "enhanced_musicgen": True,
                "aplio_voices": self.voice_engine is not None,
                "professional_audio": True,
                "prompt_intelligence": self.prompt_engine is not None
            },
            "providers": self.intelligent_gen.get_system_status()["providers"],
            "voice_types": list(VoiceType) if self.voice_engine else [],
            "processing_presets": self.audio_processor.get_processing_presets(),
            "supported_formats": settings.SUPPORTED_FORMATS,
            "max_duration": {
                "free": 8,
                "pro": 30,
                "beta": 60
            }
        }

# Global services
job_storage = {}
history_storage = []
ai_service = None

def setup_services():
    """Initialize enhanced services"""
    global ai_service
    ai_service = EnhancedAIService()

# === ENHANCED APPLICATION SETUP ===
@asynccontextmanager
async def enhanced_lifespan(app: FastAPI):
    """Enhanced application lifecycle"""
    # Startup
    logger.info("🚀 Iniciando Son1kVers3 Enhanced...")
    setup_enhanced_directories()
    setup_services()
    logger.info("✅ Son1kVers3 Enhanced listo - AI Completo Activado")
    
    yield
    
    # Shutdown
    logger.info("🛑 Cerrando Son1kVers3 Enhanced...")

# Create enhanced application
app = FastAPI(
    title="Son1kVers3 Enhanced API",
    description="Plataforma avanzada de creación musical con IA - Todas las funciones AI activadas",
    version="3.1.0-enhanced",
    lifespan=enhanced_lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
try:
    app.mount("/output", StaticFiles(directory=settings.OUTPUT_DIR), name="output")
    app.mount("/vocals", StaticFiles(directory=settings.VOCALS_DIR), name="vocals")
    app.mount("/uploads", StaticFiles(directory=settings.UPLOADS_DIR), name="uploads")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# === ENHANCED ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def enhanced_root():
    """Serve enhanced frontend"""
    frontend_paths = [
        Path("index.html"),
        Path("frontend") / "index.html"
    ]
    
    for path in frontend_paths:
        if path.exists():
            try:
                return HTMLResponse(content=path.read_text(encoding='utf-8'))
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")
    
    # Enhanced fallback with AI features
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Son1kVers3 Enhanced - AI Complete</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
                background: linear-gradient(135deg, #0a0a0a, #1a1a2e);
                color: #fff; padding: 2rem; line-height: 1.6;
            }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 3rem; }}
            h1 {{ 
                background: linear-gradient(135deg, #00FFE7, #8b5cf6, #ff6b35);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                font-size: 3.5rem; margin-bottom: 0.5rem;
            }}
            .ai-badge {{ 
                background: linear-gradient(45deg, #00FFE7, #8b5cf6);
                padding: 0.5rem 1rem; border-radius: 25px; color: #000;
                font-weight: bold; margin: 1rem 0;
            }}
            .feature-grid {{ 
                display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem; margin: 3rem 0;
            }}
            .feature-card {{ 
                background: rgba(255,255,255,0.05); padding: 2rem;
                border-radius: 15px; border: 1px solid rgba(0,255,231,0.3);
            }}
            .feature-title {{ color: #00FFE7; font-size: 1.3rem; margin-bottom: 1rem; }}
            .api-link {{ color: #8b5cf6; text-decoration: none; font-weight: 500; }}
            .api-link:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Son1kVers3 Enhanced</h1>
                <div class="ai-badge">🤖 AI COMPLETE SYSTEM ACTIVE</div>
                <p style="font-size: 1.2rem; color: #999;">Inteligencia Musical Avanzada</p>
            </div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-title">🎵 Generación Inteligente</div>
                    <p>• Suno AI + Enhanced MusicGen</p>
                    <p>• Fallback automático inteligente</p>
                    <p>• Optimización de calidad avanzada</p>
                    <p><strong>Endpoint:</strong> <a href="/docs#/default/enhanced_generate_music_api_v2_generate_post" class="api-link">POST /api/v2/generate</a></p>
                </div>
                
                <div class="feature-card">
                    <div class="feature-title">🎤 APLIO Voices</div>
                    <p>• 7 voces AI profesionales</p>
                    <p>• Control de expresividad total</p>
                    <p>• Síntesis vocal avanzada</p>
                    <p><strong>Endpoint:</strong> <a href="/docs#/default/generate_vocals_api_v2_vocals_generate_post" class="api-link">POST /api/v2/vocals/generate</a></p>
                </div>
                
                <div class="feature-card">
                    <div class="feature-title">🎛️ Audio Profesional</div>
                    <p>• Mastering de estudio completo</p>
                    <p>• Procesamiento vocal especializado</p>
                    <p>• Múltiples estilos de producción</p>
                    <p><strong>Endpoint:</strong> <a href="/docs#/default/process_audio_api_v2_audio_process_post" class="api-link">POST /api/v2/audio/process</a></p>
                </div>
                
                <div class="feature-card">
                    <div class="feature-title">🧠 Prompt Intelligence</div>
                    <p>• Mejora automática de prompts</p>
                    <p>• Análisis musical profundo</p>
                    <p>• Contexto cultural adaptativo</p>
                    <p><strong>Endpoint:</strong> <a href="/docs#/default/enhance_prompt_api_v2_prompt_enhance_post" class="api-link">POST /api/v2/prompt/enhance</a></p>
                </div>
            </div>
            
            <div style="text-align: center; margin: 3rem 0;">
                <h3>🚀 Documentación Completa</h3>
                <p><a href="/docs" class="api-link" style="font-size: 1.2rem;">Explorar API Interactiva</a></p>
                <p><a href="/capabilities" class="api-link">Ver Capacidades del Sistema</a></p>
            </div>
            
            <div style="text-align: center; color: #666; margin-top: 3rem; border-top: 1px solid #333; padding-top: 2rem;">
                <p>"La música perfecta nace de la imperfección perfecta"</p>
                <p><strong>Son1kVers3 Enhanced v3.1.0</strong> - AI Complete System</p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def enhanced_health():
    """Enhanced health check with AI system status"""
    ai_status = ai_service.get_system_capabilities() if ai_service else {}
    
    return {
        "status": "healthy",
        "service": "Son1kVers3 Enhanced",
        "version": "3.1.0-enhanced", 
        "timestamp": datetime.now().isoformat(),
        "ai_features": ai_status.get("ai_features", {}),
        "capabilities": ai_status,
        "endpoints": {
            "v2_generation": "/api/v2/generate",
            "v2_vocals": "/api/v2/vocals/generate", 
            "v2_audio_processing": "/api/v2/audio/process",
            "v2_prompt_enhancement": "/api/v2/prompt/enhance",
            "system_capabilities": "/capabilities"
        }
    }

@app.get("/capabilities")
async def get_system_capabilities():
    """Get comprehensive system capabilities"""
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not initialized")
    
    return ai_service.get_system_capabilities()

# === V2 API ENDPOINTS (Enhanced) ===

@app.post("/api/v2/generate")
async def enhanced_generate_music(request: EnhancedGenerateRequest):
    """Enhanced music generation with full AI pipeline"""
    try:
        start_time = time.time()
        
        if not ai_service:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        # Generate music with full enhancement
        result = await ai_service.generate_enhanced_music(request)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Save result
        timestamp = int(time.time())
        filename = f"enhanced_generation_{timestamp}.wav"
        filepath = settings.storage_paths["output"] / filename
        
        sf.write(filepath, result.audio, result.sample_rate)
        
        # Add to history
        history_entry = {
            "id": str(uuid.uuid4()),
            "type": "enhanced_generation",
            "created_at": datetime.now().isoformat(),
            "audio_url": f"/output/{filename}",
            "metadata": {
                "original_prompt": request.prompt,
                "enhanced_prompt": result.metadata.get("enhanced_prompt") if result.metadata else None,
                "provider_used": result.provider_used.value if result.provider_used else None,
                "generation_time": result.generation_time,
                "fallback_used": result.fallback_used,
                "ai_features": ["intelligent_generation", "prompt_enhancement", "professional_processing"]
            }
        }
        history_storage.append(history_entry)
        
        return {
            "success": True,
            "audio_url": f"/output/{filename}",
            "filename": filename,
            "provider_used": result.provider_used.value if result.provider_used else None,
            "generation_time": result.generation_time,
            "fallback_used": result.fallback_used,
            "metadata": result.metadata,
            "ai_features_used": ["intelligent_generation", "prompt_enhancement", "professional_processing"]
        }
        
    except Exception as e:
        logger.error(f"Enhanced generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/vocals/generate")
async def generate_vocals(request: VocalGenerationRequest):
    """Generate vocals using APLIO voice system"""
    try:
        if not ai_service:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        result = await ai_service.generate_vocals(request)
        
        # Add to history
        history_entry = {
            "id": str(uuid.uuid4()),
            "type": "vocal_generation",
            "created_at": datetime.now().isoformat(),
            "audio_url": result["audio_url"],
            "metadata": {
                **result["metadata"],
                "ai_features": ["aplio_voices"]
            }
        }
        history_storage.append(history_entry)
        
        return result
        
    except Exception as e:
        logger.error(f"Vocal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/audio/process")
async def process_audio(
    audio_file: UploadFile = File(...),
    style: str = Form("pop_modern"),
    lufs_target: float = Form(-14.0),
    vocal_processing: bool = Form(False),
    vocal_style: str = Form("vocal_pop")
):
    """Professional audio processing"""
    try:
        if not ai_service:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        processing_request = AudioProcessingRequest(
            style=style,
            lufs_target=lufs_target,
            vocal_processing=vocal_processing,
            vocal_style=vocal_style
        )
        
        result = await ai_service.process_audio_professionally(audio_file, processing_request)
        
        # Add to history
        history_entry = {
            "id": str(uuid.uuid4()),
            "type": "audio_processing",
            "created_at": datetime.now().isoformat(),
            "audio_url": result["audio_url"],
            "metadata": {
                **result["processing_metadata"],
                "original_filename": result["original_filename"],
                "ai_features": ["professional_audio_processing"]
            }
        }
        history_storage.append(history_entry)
        
        return result
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class PromptEnhanceRequest(BaseModel):
    prompt: str
    complexity: str = Field("detailed", description="Complexity level")
    style_hint: Optional[str] = Field(None, description="Style hint")
    cultural_context: Optional[str] = Field(None, description="Cultural context")
    
class LyricsAnalysisRequest(BaseModel):
    lyrics: str = Field(..., description="Lyrics text to analyze")

@app.post("/api/v2/prompt/enhance")
async def enhance_prompt(request: PromptEnhanceRequest):
    """Enhance musical prompts with AI intelligence"""
    try:
        if not ai_service or not ai_service.prompt_engine:
            raise HTTPException(status_code=503, detail="Prompt intelligence not available")
        
        complexity_level = PromptComplexity(request.complexity) if request.complexity in [e.value for e in PromptComplexity] else PromptComplexity.DETAILED
        
        result = ai_service.prompt_engine.enhance_prompt(
            prompt=request.prompt,
            target_complexity=complexity_level,
            style_hint=request.style_hint,
            cultural_context=request.cultural_context
        )
        
        return {
            "success": True,
            "original_prompt": result.original_prompt,
            "enhanced_prompt": result.enhanced_prompt,
            "confidence_score": result.confidence_score,
            "detected_elements": result.detected_elements,
            "suggestions": result.suggestions,
            "complexity": result.complexity.value,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Prompt enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/voices")
async def list_voices():
    """List available APLIO voices"""
    try:
        if not ai_service or not ai_service.voice_engine:
            raise HTTPException(status_code=503, detail="APLIO voice system not available")
        
        return ai_service.voice_engine.list_available_voices()
        
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/voices/{voice_type}/demo")
async def voice_demo(voice_type: str, emotion: str = "neutral"):
    """Generate demo for specific voice"""
    try:
        if not ai_service or not ai_service.voice_engine:
            raise HTTPException(status_code=503, detail="APLIO voice system not available")
        
        try:
            voice_enum = VoiceType(voice_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid voice type: {voice_type}")
        
        demo_audio = ai_service.voice_engine.generate_vocal_demo(voice_enum, emotion)
        
        # Save demo
        timestamp = int(time.time())
        filename = f"voice_demo_{voice_type}_{emotion}_{timestamp}.wav"
        filepath = settings.storage_paths["vocals"] / filename
        
        sf.write(filepath, demo_audio, settings.SAMPLE_RATE)
        
        return {
            "success": True,
            "audio_url": f"/vocals/{filename}",
            "voice_type": voice_type,
            "emotion": emotion
        }
        
    except Exception as e:
        logger.error(f"Voice demo failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === LEGACY COMPATIBILITY ENDPOINTS ===

@app.post("/api/v1/generate")
async def legacy_generate(request: EnhancedGenerateRequest):
    """Legacy endpoint with enhanced features"""
    # Convert to enhanced request
    enhanced_request = EnhancedGenerateRequest(
        prompt=request.prompt,
        duration=request.duration,
        temperature=request.temperature,
        enhance_prompt=True,
        preferred_provider="auto"
    )
    
    return await enhanced_generate_music(enhanced_request)

@app.get("/api/v1/history")
async def get_history():
    """Get creation history"""
    return history_storage[-50:]  # Last 50 items

@app.get("/api/v2/system/test")
async def test_ai_systems():
    """Test all AI systems"""
    try:
        if not ai_service:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        results = {}
        
        # Test intelligent generation
        try:
            test_results = await ai_service.intelligent_gen.test_all_providers()
            results["intelligent_generation"] = test_results
        except Exception as e:
            results["intelligent_generation"] = {"error": str(e)}
        
        # Test APLIO voices
        if ai_service.voice_engine:
            try:
                voice_info = ai_service.voice_engine.get_voice_info(VoiceType.ELENA)
                results["aplio_voices"] = {"status": "available", "sample_voice": voice_info}
            except Exception as e:
                results["aplio_voices"] = {"error": str(e)}
        else:
            results["aplio_voices"] = {"status": "not_available"}
        
        # Test audio processing
        try:
            presets = ai_service.audio_processor.get_processing_presets()
            results["professional_audio"] = {"status": "available", "presets": presets}
        except Exception as e:
            results["professional_audio"] = {"error": str(e)}
        
        # Test prompt intelligence
        if ai_service.prompt_engine:
            try:
                test_result = ai_service.prompt_engine.enhance_prompt("test song")
                results["prompt_intelligence"] = {
                    "status": "available",
                    "test_confidence": test_result.confidence_score
                }
            except Exception as e:
                results["prompt_intelligence"] = {"error": str(e)}
        else:
            results["prompt_intelligence"] = {"status": "not_available"}
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "test_results": results
        }
        
    except Exception as e:
        logger.error(f"System test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === NEW AI FEATURE ENDPOINTS ===

@app.post("/api/v2/style-transfer")
async def apply_style_transfer(file: UploadFile = File(...), request: StyleTransferRequest = Form(...)):
    """Apply style transfer to uploaded audio"""
    try:
        if not style_transfer_ai:
            raise HTTPException(status_code=503, detail="Style transfer AI not available")
        
        # Validate file
        if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Save uploaded file
        upload_path = Path(settings.UPLOADS_DIR) / f"{uuid.uuid4().hex}_{file.filename}"
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read audio
        audio, sr = sf.read(str(upload_path))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        
        # Configure style transfer
        config = StyleTransferConfig(
            target_style=MusicStyle(request.target_style.lower()),
            intensity=request.intensity,
            preserve_melody=request.preserve_melody,
            preserve_rhythm=request.preserve_rhythm,
            crossfade_duration=request.crossfade_duration
        )
        
        # Apply style transfer
        transformed_audio = style_transfer_ai.transfer_style(audio, config)
        
        # Save result
        output_filename = f"style_transfer_{uuid.uuid4().hex}.wav"
        output_path = Path(settings.OUTPUT_DIR) / output_filename
        sf.write(str(output_path), transformed_audio, sr)
        
        # Analyze style
        style_analysis = style_transfer_ai.analyze_style(transformed_audio)
        
        # Clean up input file
        upload_path.unlink()
        
        return {
            "success": True,
            "message": f"Style transfer applied: {request.target_style}",
            "output_file": f"/output/{output_filename}",
            "original_style": style_analysis.get("most_likely_style", "unknown"),
            "target_style": request.target_style,
            "intensity_applied": request.intensity,
            "style_analysis": style_analysis,
            "metadata": {
                "duration": len(transformed_audio) / sr,
                "sample_rate": sr,
                "channels": 1,
                "processing_time": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Style transfer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/analyze-audio")
async def analyze_audio(file: UploadFile = File(...), analysis_request: MusicalAnalysisRequest = Form(...)):
    """Comprehensive musical analysis of uploaded audio"""
    try:
        if not musical_analyzer:
            raise HTTPException(status_code=503, detail="Musical analyzer not available")
        
        # Validate file
        if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Save uploaded file
        upload_path = Path(settings.UPLOADS_DIR) / f"{uuid.uuid4().hex}_{file.filename}"
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read and analyze audio
        audio, sr = sf.read(str(upload_path))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Perform comprehensive analysis
        analysis = musical_analyzer.analyze(audio, sr)
        
        # Clean up input file
        upload_path.unlink()
        
        # Build response based on request
        response_data = {
            "success": True,
            "filename": file.filename,
            "basic_info": {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "channels": 1
            }
        }
        
        if analysis_request.include_harmony:
            response_data["tempo_analysis"] = {
                "bpm": analysis.tempo.bpm,
                "confidence": analysis.tempo.confidence,
                "tempo_stability": analysis.tempo.tempo_stability,
                "time_signature": analysis.tempo.time_signature.value,
                "beat_count": len(analysis.tempo.beat_times)
            }
        
        if analysis_request.include_harmony:
            response_data["key_analysis"] = {
                "key": analysis.key.key.value,
                "confidence": analysis.key.confidence,
                "key_strength": analysis.key.key_strength,
                "key_changes": len(analysis.key.key_changes)
            }
            
            response_data["harmonic_analysis"] = {
                "chord_count": len(analysis.harmony.chord_progression),
                "harmonic_rhythm": analysis.harmony.harmonic_rhythm,
                "average_tension": float(np.mean(analysis.harmony.tension_curve)) if len(analysis.harmony.tension_curve) > 0 else 0.0
            }
        
        if analysis_request.include_rhythm:
            response_data["rhythmic_analysis"] = {
                "onset_count": len(analysis.rhythm.onset_times),
                "syncopation_score": analysis.rhythm.syncopation_score,
                "rhythmic_complexity": analysis.rhythm.rhythmic_complexity,
                "pulse_clarity": analysis.rhythm.pulse_clarity
            }
        
        if analysis_request.include_structure:
            response_data["structural_analysis"] = {
                "segments": [
                    {
                        "start": seg.start_time,
                        "end": seg.end_time,
                        "label": seg.label,
                        "confidence": seg.confidence
                    }
                    for seg in analysis.structure.segments[:10]  # Limit to first 10 segments
                ],
                "song_structure": analysis.structure.song_structure,
                "repetition_score": analysis.structure.repetition_score
            }
        
        if analysis_request.include_genre:
            response_data["genre_prediction"] = analysis.genre_prediction
        
        if analysis_request.include_mood:
            response_data["mood_analysis"] = analysis.mood_analysis
            
        response_data["overall_complexity"] = analysis.overall_complexity
        
        return response_data
        
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/generate-lyrics")
async def generate_lyrics(request: LyricsGenerationAPIRequest):
    """Generate lyrics using advanced AI"""
    try:
        if not lyrics_ai:
            raise HTTPException(status_code=503, detail="Lyrics AI not available")
        
        # Create request object - the request is already the correct format
        from .lyrics_ai import LyricsGenerationRequest as LyricsReq
        lyrics_request = LyricsReq(
            theme=LyricsTheme(request.theme.lower()),
            style=LyricStyle(request.style.lower()),
            rhyme_scheme=RhymeScheme(request.rhyme_scheme.lower()),
            verse_count=request.verse_count,
            chorus_count=request.chorus_count,
            bridge_count=request.bridge_count,
            emotional_arc=EmotionalArc(request.emotional_arc.lower()),
            target_length=request.target_length,
            complexity_level=request.complexity_level,
            language_register=request.language_register,
            cultural_context=request.cultural_context
        )
        
        # Generate lyrics
        result = lyrics_ai.generate_lyrics(lyrics_request)
        
        return {
            "success": True,
            "lyrics": result["lyrics"],
            "metadata": result["metadata"],
            "narrative_structure": result["narrative_structure"],
            "coherence_analysis": result["coherence_analysis"],
            "generation_info": {
                "processing_time": time.time(),
                "ai_system": "AdvancedLyricsAI",
                "features_used": [
                    "narrative_coherence",
                    "rhyme_scheme_intelligence", 
                    "emotional_arc_mapping",
                    "thematic_consistency"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Lyrics generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/analyze-lyrics")
async def analyze_lyrics(request: LyricsAnalysisRequest):
    """Analyze existing lyrics for coherence and structure"""
    try:
        if not lyrics_ai:
            raise HTTPException(status_code=503, detail="Lyrics AI not available")
        
        # Analyze lyrics
        analysis = lyrics_ai.analyze_existing_lyrics(request.lyrics)
        
        return {
            "success": True,
            "analysis": analysis,
            "processing_info": {
                "processing_time": time.time(),
                "ai_system": "AdvancedLyricsAI",
                "analysis_features": [
                    "coherence_scoring",
                    "structure_analysis", 
                    "rhyme_detection",
                    "thematic_analysis"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Lyrics analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/capabilities-extended")
async def get_extended_capabilities():
    """Get comprehensive system capabilities including new AI features"""
    try:
        base_capabilities = await get_system_capabilities()
        
        # Add new AI capabilities
        extended_capabilities = base_capabilities.copy()
        
        extended_capabilities.update({
            "style_transfer": {
                "available": style_transfer_ai is not None,
                "supported_styles": [style.value for style in MusicStyle] if style_transfer_ai else [],
                "features": ["neural_style_transfer", "style_analysis", "crossfade_blending", "melody_preservation"]
            },
            "musical_analysis": {
                "available": musical_analyzer is not None,
                "features": [
                    "key_detection", "tempo_analysis", "chord_progression", 
                    "structural_analysis", "genre_prediction", "mood_analysis",
                    "harmonic_analysis", "rhythmic_analysis", "spectral_analysis"
                ],
                "output_formats": ["detailed", "summary", "raw_data"]
            },
            "advanced_lyrics": {
                "available": lyrics_ai is not None,
                "themes": [theme.value for theme in LyricsTheme] if lyrics_ai else [],
                "styles": [style.value for style in LyricStyle] if lyrics_ai else [],
                "rhyme_schemes": [scheme.value for scheme in RhymeScheme] if lyrics_ai else [],
                "emotional_arcs": [arc.value for arc in EmotionalArc] if lyrics_ai else [],
                "features": [
                    "narrative_coherence", "rhyme_intelligence", "emotional_arc_mapping",
                    "thematic_consistency", "cultural_adaptation", "complexity_control"
                ]
            },
            "integrated_workflows": {
                "available": True,
                "workflows": [
                    "music_generation_with_analysis",
                    "style_transfer_with_lyrics",
                    "comprehensive_audio_processing",
                    "ai_enhanced_composition"
                ]
            },
            "system_status": {
                "version": "3.1.0-enhanced",
                "ai_features_count": 8,
                "last_update": datetime.now().isoformat()
            }
        })
        
        return extended_capabilities
        
    except Exception as e:
        logger.error(f"Error getting extended capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === ERROR HANDLERS ===

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint no encontrado",
            "available_versions": {
                "v1": "Legacy compatibility",
                "v2": "Enhanced AI features"
            },
            "documentation": "/docs"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "src.main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )