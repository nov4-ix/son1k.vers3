# Import from src directory for compatibility
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.celery_worker import celery_app, generate_music_task, generate_music_gpu_task

__all__ = ["celery_app", "generate_music_task", "generate_music_gpu_task"]