from celery import Celery
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "son1k",
    broker=os.getenv("REDIS_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("REDIS_BROKER_URL", "redis://redis:6379/0"),
    include=["src.celery_worker"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "src.celery_worker.generate_music_task": {"queue": os.getenv("CELERY_QUEUE_CPU", "cpu")},
        "src.celery_worker.generate_music_gpu_task": {"queue": os.getenv("CELERY_QUEUE_GPU", "gpu")},
    },
)

@celery_app.task(bind=True)
def generate_music_task(self, prompt: str, duration: int = 30, style: str = "default"):
    """CPU-based music generation task (stub implementation)"""
    try:
        logger.info(f"Starting music generation: {prompt}")
        
        # Update task progress
        self.update_state(
            state="PROGRESS",
            meta={"progress": 10, "message": "Initializing music generation..."}
        )
        time.sleep(2)
        
        self.update_state(
            state="PROGRESS", 
            meta={"progress": 30, "message": "Analyzing prompt..."}
        )
        time.sleep(3)
        
        self.update_state(
            state="PROGRESS",
            meta={"progress": 60, "message": "Generating audio..."}
        )
        time.sleep(5)
        
        self.update_state(
            state="PROGRESS",
            meta={"progress": 90, "message": "Finalizing output..."}
        )
        time.sleep(2)
        
        # Mock result
        result = {
            "prompt": prompt,
            "duration": duration,
            "style": style,
            "output_file": f"/outputs/generated_music_{self.request.id}.wav",
            "status": "completed",
            "message": "Music generation completed successfully"
        }
        
        logger.info(f"Completed music generation: {prompt}")
        return result
        
    except Exception as exc:
        logger.error(f"Music generation failed: {str(exc)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(exc)}
        )
        raise

@celery_app.task(bind=True)
def generate_music_gpu_task(self, prompt: str, duration: int = 30, style: str = "default"):
    """GPU-based music generation task (stub implementation)"""
    try:
        logger.info(f"Starting GPU music generation: {prompt}")
        
        # Simulate GPU processing
        self.update_state(
            state="PROGRESS",
            meta={"progress": 20, "message": "Loading AI model..."}
        )
        time.sleep(3)
        
        self.update_state(
            state="PROGRESS",
            meta={"progress": 50, "message": "Processing with neural network..."}
        )
        time.sleep(8)
        
        self.update_state(
            state="PROGRESS",
            meta={"progress": 80, "message": "Rendering high-quality audio..."}
        )
        time.sleep(4)
        
        result = {
            "prompt": prompt,
            "duration": duration,
            "style": style,
            "output_file": f"/outputs/gpu_generated_music_{self.request.id}.wav",
            "quality": "high",
            "gpu_used": True,
            "status": "completed",
            "message": "GPU music generation completed successfully"
        }
        
        logger.info(f"Completed GPU music generation: {prompt}")
        return result
        
    except Exception as exc:
        logger.error(f"GPU music generation failed: {str(exc)}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(exc)}
        )
        raise