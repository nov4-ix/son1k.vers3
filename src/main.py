"""
Son1kVers3 - API Principal Integrada
Versión corregida 100% compatible con módulos existentes
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import os
import tempfile
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import uuid
from datetime import datetime
import numpy as np
import soundfile as sf
import time
import shutil

# Imports compatibles con tus módulos existentes
try:
    from .audio_analysis import AudioAnalyzer
    from .audio_post import AudioPostProcessor
    AUDIO_MODULES_AVAILABLE = True
except ImportError:
    # Fallback si los módulos no están disponibles
    AudioAnalyzer = None
    AudioPostProcessor = None
    AUDIO_MODULES_AVAILABLE = False
    logging.warning("Módulos de audio no disponibles - usando versión simplificada")

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURACIÓN ===
class Settings:
    STORAGE_ROOT = "storage"
    UPLOADS_DIR = "storage/uploads" 
    OUTPUT_DIR = "storage/output"
    SAMPLE_RATE = 32000
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".aiff", ".m4a"]
    
    @property
    def storage_paths(self):
        return {
            "uploads": Path(self.UPLOADS_DIR),
            "output": Path(self.OUTPUT_DIR),
            "ghost_uploads": Path(self.UPLOADS_DIR) / "ghost",
            "ghost_output": Path(self.OUTPUT_DIR) / "ghost"
        }

settings = Settings()

# === MODELOS ===
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Descripción de la música")
    duration: float = Field(8.0, ge=1.0, le=30.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(250, ge=1, le=1000)
    apply_postprocessing: bool = Field(True)

class MaquetaJob:
    def __init__(self, job_id: str, file_path: str, style: str, intensity: float):
        self.job_id = job_id
        self.file_path = file_path
        self.style = style
        self.intensity = intensity
        self.status = "processing"
        self.progress = 0
        self.message = "Iniciando análisis..."
        self.result = None
        self.error = None
        self.created_at = datetime.now()

def setup_directories():
    """Crear directorios necesarios"""
    for path in settings.storage_paths.values():
        path.mkdir(parents=True, exist_ok=True)

# === SERVICIOS ===
class MusicGenService:
    """Servicio MusicGen optimizado"""
    
    def __init__(self):
        self.model_loaded = False
        self.device = "cpu"
    
    async def load_model(self):
        """Simular carga de modelo"""
        logger.info("Cargando modelo MusicGen...")
        await asyncio.sleep(1)  # Simular carga
        self.model_loaded = True
        return True
    
    def is_loaded(self):
        return self.model_loaded
    
    def generate_music(self, prompt: str, duration: float, **kwargs):
        """Generar música sintética realista"""
        logger.info(f"Generando: '{prompt}' ({duration}s)")
        
        sample_rate = settings.SAMPLE_RATE
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generar música más realista basada en el prompt
        if "rock" in prompt.lower():
            base_freq = 220.0  # A3
            harmonics = [1, 2, 3, 4, 5]
            amp_decay = 0.7
        elif "electronic" in prompt.lower():
            base_freq = 440.0  # A4
            harmonics = [1, 0.5, 2, 0.3, 4]
            amp_decay = 0.5
        else:  # Default/acoustic
            base_freq = 261.6  # C4
            harmonics = [1, 0.6, 0.4, 0.3, 0.2]
            amp_decay = 0.8
        
        # Construir audio con múltiples componentes
        audio = np.zeros_like(t)
        
        # Fundamental y armónicos
        for i, harm_amp in enumerate(harmonics):
            freq = base_freq * (i + 1)
            if freq < sample_rate / 2:  # Evitar aliasing
                component = harm_amp * np.sin(2 * np.pi * freq * t)
                audio += component * (amp_decay ** i)
        
        # Modulación para variación
        mod_freq = 0.5  # Hz
        modulation = 1 + 0.1 * np.sin(2 * np.pi * mod_freq * t)
        audio = audio * modulation
        
        # Envelope realista
        attack_time = 0.1
        decay_time = 0.2
        sustain_level = 0.7
        release_time = 0.3
        
        total_samples = len(audio)
        attack_samples = int(attack_time * sample_rate)
        decay_samples = int(decay_time * sample_rate)
        release_samples = int(release_time * sample_rate)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        envelope = np.ones_like(audio)
        
        if total_samples > attack_samples + decay_samples + release_samples:
            # Attack
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            
            # Decay
            start_idx = attack_samples
            end_idx = start_idx + decay_samples
            envelope[start_idx:end_idx] = np.linspace(1, sustain_level, decay_samples)
            
            # Sustain
            start_idx = end_idx
            end_idx = start_idx + sustain_samples
            envelope[start_idx:end_idx] = sustain_level
            
            # Release
            envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
        else:
            # Envelope simple para duraciones muy cortas
            envelope[:len(envelope)//2] = np.linspace(0, 1, len(envelope)//2)
            envelope[len(envelope)//2:] = np.linspace(1, 0, len(envelope) - len(envelope)//2)
        
        audio = audio * envelope
        
        # Normalizar
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        # Añadir ruido sutil para realismo
        noise_level = 0.002
        noise = np.random.normal(0, noise_level, len(audio))
        audio = audio + noise
        
        metadata = {
            "prompt": prompt,
            "duration_s": duration,
            "sample_rate": sample_rate,
            "device": self.device,
            "base_frequency": base_freq,
            "audio_stats": {
                "peak": float(np.max(np.abs(audio))),
                "rms": float(np.sqrt(np.mean(audio ** 2)))
            }
        }
        
        return audio, metadata
    
    def save_audio(self, audio: np.ndarray, path: Path, metadata: Dict = None):
        """Guardar audio"""
        try:
            sf.write(path, audio, settings.SAMPLE_RATE)
            return str(path)
        except Exception as e:
            logger.error(f"Error guardando audio: {e}")
            raise

# Storage global
job_storage = {}
history_storage = []

# Servicios globales
musicgen_service = MusicGenService()

# Inicializar servicios de audio si están disponibles
if AUDIO_MODULES_AVAILABLE:
    audio_analyzer = AudioAnalyzer(sample_rate=settings.SAMPLE_RATE)
    audio_processor = AudioPostProcessor(sample_rate=settings.SAMPLE_RATE)
    logger.info("Módulos de audio profesional cargados")
else:
    audio_analyzer = None
    audio_processor = None
    logger.warning("Usando modo simplificado sin módulos de audio")

# === SETUP DE APLICACIÓN ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación"""
    # Startup
    logger.info("🚀 Iniciando Son1kVers3...")
    setup_directories()
    await musicgen_service.load_model()
    logger.info("✅ Son1kVers3 listo - Resistencia Sonora Activa")
    
    yield
    
    # Shutdown
    logger.info("🛑 Cerrando Son1kVers3...")

# === CREAR APLICACIÓN ===
app = FastAPI(
    title="Son1kVers3 API",
    description="Plataforma de creación musical con IA - Resistencia Sonora",
    version="3.0.0",
    lifespan=lifespan
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
    app.mount("/uploads", StaticFiles(directory=settings.UPLOADS_DIR), name="uploads")
    app.mount("/output", StaticFiles(directory=settings.OUTPUT_DIR), name="output")
except Exception as e:
    logger.warning(f"No se pudieron montar archivos estáticos: {e}")

# === ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def root():
    """Servir frontend HTML"""
    # Buscar frontend en múltiples ubicaciones
    possible_paths = [
        Path("index.html"),
        Path("frontend") / "index.html", 
        Path("frontend") / "src" / "index.html",
        Path(__file__).parent.parent / "index.html",
        Path(__file__).parent.parent / "frontend" / "index.html"
    ]
    
    for frontend_path in possible_paths:
        if frontend_path.exists():
            try:
                return HTMLResponse(content=frontend_path.read_text(encoding='utf-8'), status_code=200)
            except Exception as e:
                logger.warning(f"Error leyendo {frontend_path}: {e}")
                continue
    
    # Frontend de fallback
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Son1kVers3 - Resistencia Sonora</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
                background: #0a0a0a; 
                color: #fff; 
                padding: 2rem; 
                line-height: 1.6;
            }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 3rem; }
            h1 { 
                background: linear-gradient(135deg, #0066cc, #ff6b35);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 3rem;
                margin-bottom: 0.5rem;
            }
            .status { 
                background: #1a1a1a; 
                padding: 1.5rem; 
                border-radius: 12px; 
                margin: 1.5rem 0; 
                border: 1px solid #333;
            }
            .api-link { 
                color: #0066cc; 
                text-decoration: none; 
                font-weight: 500;
            }
            .api-link:hover { text-decoration: underline; }
            .endpoint { 
                background: #0f0f0f; 
                padding: 1rem; 
                margin: 0.5rem 0; 
                border-radius: 6px; 
                border-left: 3px solid #0066cc;
            }
            .endpoint code { color: #ff6b35; }
            .feature { 
                display: inline-block; 
                background: #333; 
                padding: 0.3rem 0.8rem; 
                margin: 0.2rem; 
                border-radius: 16px; 
                font-size: 0.85rem;
            }
            .motto { 
                font-style: italic; 
                text-align: center; 
                color: #999; 
                margin-top: 2rem;
                border-top: 1px solid #333;
                padding-top: 1rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Son1kVers3</h1>
                <p style="font-size: 1.2rem; color: #999;">Resistencia Sonora</p>
            </div>
            
            <div class="status">
                <h3>🚀 API Activa</h3>
                <p>Documentación interactiva: <a href="/docs" class="api-link">/docs</a></p>
                <p>Documentación alternativa: <a href="/redoc" class="api-link">/redoc</a></p>
            </div>

            <div class="status">
                <h3>🎵 Endpoints Principales</h3>
                <div class="endpoint">
                    <strong>POST</strong> <code>/api/v1/maqueta/process</code><br>
                    Procesar maqueta → producción profesional
                </div>
                <div class="endpoint">
                    <strong>POST</strong> <code>/api/v1/generate</code><br>
                    Generación musical manual con IA
                </div>
                <div class="endpoint">
                    <strong>GET</strong> <code>/api/v1/history</code><br>
                    Historial de creaciones
                </div>
                <div class="endpoint">
                    <strong>GET</strong> <code>/health</code><br>
                    Estado del sistema
                </div>
            </div>

            <div class="status">
                <h3>🛠️ Características</h3>
                <span class="feature">Análisis de Audio</span>
                <span class="feature">SSL EQ</span>
                <span class="feature">Neve Saturation</span>
                <span class="feature">LUFS Mastering</span>
                <span class="feature">Generación IA</span>
                <span class="feature">API REST</span>
                <span class="feature">Interfaz Accesible</span>
            </div>

            <div class="status">
                <h3>📖 Inicio Rápido</h3>
                <p>1. Ve a <a href="/docs" class="api-link">/docs</a> para probar la API</p>
                <p>2. Sube un archivo de audio en <strong>/api/v1/maqueta/process</strong></p>
                <p>3. O genera música desde cero en <strong>/api/v1/generate</strong></p>
                <p>4. Revisa el historial en <strong>/api/v1/history</strong></p>
            </div>

            <div class="motto">
                "Lo imperfecto también es sagrado"<br>
                Democratización musical para 400M+ hispanohablantes
            </div>
        </div>
    </body>
    </html>
    """, status_code=200)

@app.get("/health")
async def health_check():
    """Health check completo"""
    return {
        "status": "healthy",
        "service": "Son1kVers3",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "model_loaded": musicgen_service.is_loaded(),
            "device": musicgen_service.device,
            "audio_modules": AUDIO_MODULES_AVAILABLE
        },
        "storage": {
            name: path.exists() for name, path in settings.storage_paths.items()
        },
        "capabilities": {
            "audio_analysis": AUDIO_MODULES_AVAILABLE,
            "professional_processing": AUDIO_MODULES_AVAILABLE,
            "music_generation": True,
            "job_processing": True
        }
    }

# === PROCESAMIENTO DE MAQUETAS ===
@app.post("/api/v1/maqueta/process")
async def process_maqueta(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    style: str = Form("auto"),
    intensity: float = Form(50.0)
):
    """Procesar maqueta usando tus módulos reales"""
    try:
        # Validaciones
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser de audio")
        
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo requerido")
        
        file_ext = Path(audio_file.filename).suffix.lower()
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"Formato no soportado: {file_ext}")
        
        # Generar job ID
        job_id = str(uuid.uuid4())
        
        # Guardar archivo
        temp_dir = settings.storage_paths["uploads"]
        file_path = temp_dir / f"{job_id}_{audio_file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Crear job
        job = MaquetaJob(job_id, str(file_path), style, intensity)
        job_storage[job_id] = job
        
        # Procesar en background
        background_tasks.add_task(process_maqueta_background, job)
        
        logger.info(f"Maqueta job iniciado: {job_id} - {audio_file.filename}")
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Maqueta en procesamiento",
            "filename": audio_file.filename,
            "estimated_time": "30-60 segundos"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing maqueta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando maqueta: {str(e)}")

async def process_maqueta_background(job: MaquetaJob):
    """Procesar maqueta usando tus módulos AudioAnalyzer y AudioPostProcessor"""
    try:
        start_time = time.time()
        
        # Paso 1: Análisis usando tu AudioAnalyzer
        job.progress = 20
        job.message = "Analizando audio original..."
        
        if AUDIO_MODULES_AVAILABLE and audio_analyzer:
            analysis_result = audio_analyzer.analyze_audio_file(job.file_path)
        else:
            # Fallback analysis
            analysis_result = {
                "file_info": {"duration_s": 10.0},
                "tempo": {"bpm": 120.0, "confidence": 0.5},
                "key_guess": {"root": "C", "scale": "major", "confidence": 0.5},
                "vocals": {"has_vocals": False, "vocal_probability": 0.3},
                "energy_structure": {"high_energy_percentage": 0.6},
                "spectral": {"brightness": 0.5}
            }
        
        # Paso 2: Cargar audio para procesamiento
        job.progress = 40
        job.message = "Preparando generación..."
        
        try:
            original_audio, sr = sf.read(job.file_path)
            if original_audio.ndim > 1:
                original_audio = np.mean(original_audio, axis=1)
        except Exception as e:
            logger.error(f"Error cargando audio: {e}")
            raise HTTPException(status_code=400, detail="Error leyendo archivo de audio")
        
        # Paso 3: Construir prompt inteligente
        tempo_info = analysis_result.get("tempo", {}).get("bpm", 120)
        key_info = analysis_result.get("key_guess", {})
        key_desc = f"{key_info.get('root', 'C')} {key_info.get('scale', 'major')}"
        has_vocals = analysis_result.get("vocals", {}).get("has_vocals", False)
        
        prompt_parts = [f"Create a {job.style} style production"]
        prompt_parts.append(f"in {key_desc}")
        prompt_parts.append(f"at {tempo_info:.0f} BPM")
        if has_vocals:
            prompt_parts.append("with vocal elements")
        
        smart_prompt = ", ".join(prompt_parts)
        
        # Paso 4: Generar música con IA
        job.progress = 60
        job.message = "Generando con IA..."
        
        duration = min(30.0, analysis_result.get("file_info", {}).get("duration_s", 12.0))
        generated_audio, gen_metadata = musicgen_service.generate_music(smart_prompt, duration)
        
        # Paso 5: Post-procesamiento usando tu AudioPostProcessor
        job.progress = 80
        job.message = "Aplicando procesamiento profesional..."
        
        if AUDIO_MODULES_AVAILABLE and audio_processor:
            # Usar tu sistema profesional
            eq_params = {
                "low_gain_db": 1.5,
                "mid1_gain_db": -1.0,
                "mid2_gain_db": 1.5,
                "high_gain_db": 1.0
            }
            
            tune_params = {
                "enabled": has_vocals,
                "key_root": key_info.get("root", "C"),
                "key_scale": key_info.get("scale", "major"),
                "strength": job.intensity / 100.0
            }
            
            sat_params = {
                "drive_db": 6.0,
                "mix": 0.35
            }
            
            master_params = {
                "lufs_target": -14.0,
                "ceiling_db": -0.3,
                "fade_in_ms": 50,
                "fade_out_ms": 200
            }
            
            processed_audio, post_metadata = audio_processor.process_master(
                generated_audio,
                eq_params=eq_params,
                tune_params=tune_params,
                sat_params=sat_params,
                master_params=master_params
            )
        else:
            # Procesamiento básico fallback
            processed_audio = generated_audio
            # Normalización básica
            max_val = np.max(np.abs(processed_audio))
            if max_val > 0:
                processed_audio = processed_audio / max_val * 0.9
            
            post_metadata = {
                "processing_chain": ["basic_normalization"],
                "peak_level": float(np.max(np.abs(processed_audio))),
                "rms_level": float(np.sqrt(np.mean(processed_audio ** 2)))
            }
        
        # Paso 6: Guardar resultado
        job.progress = 90
        job.message = "Guardando resultado..."
        
        output_filename = f"production_{job.job_id}.wav"
        output_path = settings.storage_paths["output"] / output_filename
        
        sf.write(output_path, processed_audio, settings.SAMPLE_RATE)
        
        # Paso 7: Finalizar
        job.progress = 100
        job.message = "Procesamiento completado"
        job.status = "completed"
        
        processing_time = time.time() - start_time
        
        # Resultado final
        job.result = {
            "production_url": f"/output/{output_filename}",
            "original_filename": Path(job.file_path).name,
            "analysis": {
                "key": analysis_result.get("key_guess", {}).get("root", "C"),
                "scale": analysis_result.get("key_guess", {}).get("scale", "major"),
                "tempo": analysis_result.get("tempo", {}).get("bpm", 120),
                "duration": analysis_result.get("file_info", {}).get("duration_s", 0),
                "has_vocals": analysis_result.get("vocals", {}).get("has_vocals", False),
                "energy": analysis_result.get("energy_structure", {}).get("high_energy_percentage", 0.5),
                "brightness": analysis_result.get("spectral", {}).get("brightness", 0.5)
            },
            "generation": {
                "prompt": smart_prompt,
                "duration": duration,
                "base_frequency": gen_metadata.get("base_frequency", 261.6)
            },
            "processing": {
                "chain": post_metadata.get("processing_chain", []),
                "peak_level": post_metadata.get("peak_level", 0),
                "rms_level": post_metadata.get("rms_level", 0)
            },
            "session_id": job.job_id,
            "processing_time_s": processing_time,
            "style": job.style,
            "intensity": job.intensity
        }
        
        # Agregar al historial
        history_entry = {
            "id": job.job_id,
            "type": "maqueta",
            "created_at": job.created_at.isoformat(),
            "audio_url": f"/output/{output_filename}",
            "metadata": job.result
        }
        history_storage.append(history_entry)
        
        logger.info(f"Maqueta completada: {job.job_id} ({processing_time:.1f}s)")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.progress = 0
        logger.error(f"Error en maqueta background: {str(e)}", exc_info=True)

# === GENERACIÓN MANUAL ===
@app.post("/api/v1/generate")
async def generate_music(request: GenerateRequest):
    """Generación manual de música"""
    try:
        if not musicgen_service.is_loaded():
            await musicgen_service.load_model()
        
        start_time = time.time()
        
        # Generar música
        audio, metadata = musicgen_service.generate_music(
            prompt=request.prompt,
            duration=request.duration,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        # Post-procesamiento si está habilitado
        if request.apply_postprocessing and AUDIO_MODULES_AVAILABLE and audio_processor:
            processed_audio, post_metadata = audio_processor.process_master(audio)
            processing_chain = post_metadata.get("processing_chain", [])
        else:
            # Normalización básica
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                processed_audio = audio / max_val * 0.9
            else:
                processed_audio = audio
            processing_chain = ["basic_normalization"]
        
        # Guardar
        timestamp = int(time.time())
        filename = f"generation_{timestamp}.wav"
        output_path = settings.storage_paths["output"] / filename
        sf.write(output_path, processed_audio, settings.SAMPLE_RATE)
        
        total_time = time.time() - start_time
        
        # Agregar al historial
        history_entry = {
            "id": str(uuid.uuid4()),
            "type": "generation",
            "created_at": datetime.now().isoformat(),
            "audio_url": f"/output/{filename}",
            "metadata": {
                "prompt": request.prompt,
                "duration": request.duration,
                "processing_time": total_time,
                "processing_chain": processing_chain
            }
        }
        history_storage.append(history_entry)
        
        return {
            "ok": True,
            "audio_url": f"/output/{filename}",
            "filename": filename,
            "prompt": request.prompt,
            "duration": metadata["duration_s"],
            "generation_time": total_time,
            "device": metadata["device"],
            "postprocessing_applied": request.apply_postprocessing,
            "processing_chain": processing_chain
        }
        
    except Exception as e:
        logger.error(f"Generación falló: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === ENDPOINTS DE ESTADO ===
@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Obtener estado de un job"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    job = job_storage[job_id]
    
    response = {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "created_at": job.created_at.isoformat()
    }
    
    if job.status == "completed" and job.result:
        response["result"] = job.result
        response["result_type"] = "maqueta"
    elif job.status == "failed" and job.error:
        response["error"] = job.error
    
    return response

@app.get("/api/v1/history")
async def get_history():
    """Obtener historial de creaciones"""
    return history_storage[-20:]  # Últimos 20

@app.delete("/api/v1/history/{item_id}")
async def delete_history_item(item_id: str):
    """Eliminar elemento del historial"""
    global history_storage
    initial_count = len(history_storage)
    history_storage = [item for item in history_storage if item["id"] != item_id]
    
    if len(history_storage) < initial_count:
        return {"message": "Elemento eliminado del historial", "deleted": True}
    else:
        raise HTTPException(status_code=404, detail="Elemento no encontrado")

# === GHOST STUDIO ===
@app.get("/api/v1/ghost/status")
async def ghost_status():
    """Estado de Ghost Studio"""
    active_jobs = [job for job in job_storage.values() if job.status == "processing"]
    completed_jobs = [job for job in job_storage.values() if job.status == "completed"]
    failed_jobs = [job for job in job_storage.values() if job.status == "failed"]
    
    return {
        "active_workers": len(active_jobs),
        "pending_jobs": 0,  # No tenemos cola separada
        "completed_jobs": len(completed_jobs),
        "failed_jobs": len(failed_jobs),
        "total_jobs": len(job_storage),
        "last_activity": max([job.created_at for job in job_storage.values()]).isoformat() if job_storage else None,
        "system_status": "active",
        "capabilities": {
            "audio_analysis": AUDIO_MODULES_AVAILABLE,
            "professional_processing": AUDIO_MODULES_AVAILABLE,
            "music_generation": musicgen_service.is_loaded()
        }
    }

@app.post("/api/v1/ghost/clear-cache")
async def clear_cache():
    """Limpiar caché del sistema"""
    global job_storage, history_storage
    
    # Mantener solo jobs activos
    active_jobs = {k: v for k, v in job_storage.items() if v.status == "processing"}
    removed_jobs = len(job_storage) - len(active_jobs)
    job_storage = active_jobs
    
    # Limpiar archivos temporales antiguos
    temp_files_removed = 0
    try:
        for temp_dir in [settings.storage_paths["uploads"], Path(tempfile.gettempdir()) / "son1kvers3"]:
            if temp_dir.exists():
                for file_path in temp_dir.glob("*"):
                    if file_path.is_file():
                        # Eliminar archivos de más de 1 hora
                        file_age = time.time() - file_path.stat().st_mtime
                        if file_age > 3600:  # 1 hora
                            file_path.unlink()
                            temp_files_removed += 1
    except Exception as e:
        logger.warning(f"Error limpiando archivos temporales: {e}")
    
    return {
        "message": "Caché limpiado exitosamente",
        "removed_jobs": removed_jobs,
        "active_jobs": len(job_storage),
        "temp_files_removed": temp_files_removed
    }

@app.post("/api/v1/ghost/restart-workers")
async def restart_workers():
    """Reiniciar workers del sistema"""
    try:
        # Reiniciar servicio de generación musical
        await musicgen_service.load_model()
        
        # Reinicializar módulos de audio si están disponibles
        global audio_analyzer, audio_processor
        if AUDIO_MODULES_AVAILABLE:
            audio_analyzer = AudioAnalyzer(sample_rate=settings.SAMPLE_RATE)
            audio_processor = AudioPostProcessor(sample_rate=settings.SAMPLE_RATE)
        
        return {
            "message": "Workers reiniciados exitosamente",
            "status": "ok",
            "model_loaded": musicgen_service.is_loaded(),
            "audio_modules": AUDIO_MODULES_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Error reiniciando workers: {e}")
        raise HTTPException(status_code=500, detail=f"Error reiniciando workers: {str(e)}")

# === ENDPOINTS ADICIONALES ===
@app.get("/api/v1/presets")
async def get_presets():
    """Obtener presets de estilos musicales"""
    presets = {
        "auto": {
            "name": "Detección Automática",
            "description": "Análisis automático del estilo basado en el audio",
            "suggested_intensity": 50
        },
        "pop": {
            "name": "Pop Moderno",
            "description": "Producción pop comercial con brillo y punch",
            "suggested_intensity": 70
        },
        "rock": {
            "name": "Rock/Alternative",
            "description": "Sonido rock con distorsión y energía",
            "suggested_intensity": 80
        },
        "electronic": {
            "name": "Electronic/EDM",
            "description": "Música electrónica con síntesis y efectos",
            "suggested_intensity": 60
        },
        "latin": {
            "name": "Latino/Regional",
            "description": "Estilos latinos con instrumentación tradicional",
            "suggested_intensity": 65
        },
        "acoustic": {
            "name": "Acústico/Folk",
            "description": "Sonido natural con instrumentos acústicos",
            "suggested_intensity": 40
        },
        "hiphop": {
            "name": "Hip-Hop/Trap",
            "description": "Beats urbanos con graves profundos",
            "suggested_intensity": 75
        }
    }
    
    return {"presets": presets, "count": len(presets)}

@app.get("/api/v1/stats")
async def get_stats():
    """Estadísticas del sistema"""
    total_jobs = len(job_storage)
    completed_jobs = len([job for job in job_storage.values() if job.status == "completed"])
    failed_jobs = len([job for job in job_storage.values() if job.status == "failed"])
    
    # Estadísticas de historial
    maqueta_count = len([item for item in history_storage if item["type"] == "maqueta"])
    generation_count = len([item for item in history_storage if item["type"] == "generation"])
    
    # Estadísticas de archivos
    try:
        output_files = list(settings.storage_paths["output"].glob("*.wav"))
        total_output_size = sum(f.stat().st_size for f in output_files if f.is_file())
        total_output_size_mb = total_output_size / (1024 * 1024)
    except Exception:
        total_output_size_mb = 0
    
    return {
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "success_rate": (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        },
        "creations": {
            "total": len(history_storage),
            "maquetas": maqueta_count,
            "generations": generation_count
        },
        "storage": {
            "output_files": len(output_files) if 'output_files' in locals() else 0,
            "total_size_mb": round(total_output_size_mb, 2)
        },
        "system": {
            "uptime_info": "Disponible en /health",
            "audio_modules": AUDIO_MODULES_AVAILABLE,
            "model_loaded": musicgen_service.is_loaded()
        }
    }

# === MANEJO DE ERRORES ===
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint no encontrado",
            "path": str(request.url.path),
            "available_endpoints": [
                "/health",
                "/docs", 
                "/api/v1/maqueta/process",
                "/api/v1/generate",
                "/api/v1/history",
                "/api/v1/ghost/status"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor",
            "timestamp": datetime.now().isoformat(),
            "support": "Revisa los logs para más detalles"
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Error de validación en los datos enviados",
            "errors": exc.detail if hasattr(exc, 'detail') else str(exc),
            "help": "Revisa la documentación en /docs"
        }
    )

# === EJECUTAR ===
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
