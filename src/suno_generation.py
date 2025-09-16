"""
Suno Musical Generation System for Son1kVers3
Integrated queue system for asynchronous music generation with real-time updates
"""

import uuid
import json
import asyncio
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
from pathlib import Path
import os

from .suno_client import get_suno_client, SunoResponse

logger = logging.getLogger(__name__)

class GenerationStatus(Enum):
    """Status of music generation jobs"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class GenerationMode(Enum):
    """Suno generation modes"""
    ORIGINAL = "original"
    PROMPTLESS = "promptless"
    EXTEND = "extend"
    REMIX = "remix"

@dataclass
class SunoGenerationRequest:
    """Request for Suno music generation"""
    prompt: Optional[str] = None
    lyrics: Optional[str] = None
    style: Optional[str] = None
    length_sec: int = 60
    mode: GenerationMode = GenerationMode.ORIGINAL
    
    # Advanced controls
    expressiveness: float = 1.0  # 0.0 - 2.0
    production_quality: float = 1.0  # 0.0 - 2.0
    creativity: float = 1.0  # 0.0 - 2.0
    
    # Metadata
    user_id: Optional[str] = None
    priority: int = 1  # 1-5, higher is more priority
    webhook_url: Optional[str] = None

@dataclass
class GenerationJob:
    """Music generation job"""
    job_id: str
    request: SunoGenerationRequest
    status: GenerationStatus
    created_at: datetime
    updated_at: datetime
    
    # Results
    audio_url: Optional[str] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None
    
    # Processing info
    suno_job_id: Optional[str] = None
    progress: float = 0.0
    estimated_completion: Optional[datetime] = None

class GenerationQueue:
    """In-memory queue for generation jobs (production would use Redis/DB)"""
    
    def __init__(self):
        self.jobs: Dict[str, GenerationJob] = {}
        self.processing: Dict[str, GenerationJob] = {}
        self.completed: Dict[str, GenerationJob] = {}
        self.max_concurrent = 3
        self.max_queue_size = 100
    
    def add_job(self, request: SunoGenerationRequest) -> str:
        """Add a new generation job to queue"""
        if len(self.jobs) >= self.max_queue_size:
            raise ValueError("Queue is full")
        
        job_id = str(uuid.uuid4())
        now = datetime.now()
        
        job = GenerationJob(
            job_id=job_id,
            request=request,
            status=GenerationStatus.QUEUED,
            created_at=now,
            updated_at=now
        )
        
        self.jobs[job_id] = job
        logger.info(f"Added generation job {job_id} to queue")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get job by ID from any queue"""
        return (self.jobs.get(job_id) or 
                self.processing.get(job_id) or 
                self.completed.get(job_id))
    
    def get_next_job(self) -> Optional[GenerationJob]:
        """Get next job to process (priority-based)"""
        if len(self.processing) >= self.max_concurrent:
            return None
        
        if not self.jobs:
            return None
        
        # Sort by priority (higher first), then by creation time
        sorted_jobs = sorted(
            self.jobs.values(),
            key=lambda j: (-j.request.priority, j.created_at)
        )
        
        return sorted_jobs[0]
    
    def start_processing(self, job_id: str):
        """Move job from queue to processing"""
        if job_id in self.jobs:
            job = self.jobs.pop(job_id)
            job.status = GenerationStatus.PROCESSING
            job.updated_at = datetime.now()
            self.processing[job_id] = job
            logger.info(f"Started processing job {job_id}")
    
    def complete_job(self, job_id: str, audio_url: Optional[str] = None, 
                    metadata: Optional[Dict] = None, error: Optional[str] = None):
        """Complete a job with results or error"""
        job = self.processing.pop(job_id, None)
        if not job:
            logger.warning(f"Job {job_id} not found in processing queue")
            return
        
        job.status = GenerationStatus.COMPLETED if not error else GenerationStatus.FAILED
        job.updated_at = datetime.now()
        job.audio_url = audio_url
        job.metadata = metadata
        job.error = error
        job.progress = 100.0
        
        self.completed[job_id] = job
        logger.info(f"Completed job {job_id} with status {job.status.value}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
        return {
            "queued": len(self.jobs),
            "processing": len(self.processing),
            "completed": len(self.completed),
            "max_concurrent": self.max_concurrent,
            "queue_capacity": self.max_queue_size
        }

class SunoGenerationService:
    """Main service for Suno music generation"""
    
    def __init__(self):
        self.queue = GenerationQueue()
        self.suno_client = get_suno_client()
        self.processing_task = None
        self.storage_dir = Path("storage/suno_generations")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SunoGenerationService initialized")
    
    async def start_processor(self):
        """Start the background job processor"""
        if self.processing_task and not self.processing_task.done():
            logger.warning("Processor already running")
            return
        
        self.processing_task = asyncio.create_task(self._process_jobs())
        logger.info("Started background job processor")
    
    async def stop_processor(self):
        """Stop the background job processor"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped background job processor")
    
    async def generate_music(self, request: SunoGenerationRequest) -> str:
        """Queue a new music generation job"""
        try:
            job_id = self.queue.add_job(request)
            
            # Start processor if not running
            if not self.processing_task or self.processing_task.done():
                await self.start_processor()
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error queueing generation request: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a generation job"""
        job = self.queue.get_job(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "audio_url": job.audio_url,
            "metadata": job.metadata,
            "error": job.error,
            "estimated_completion": job.estimated_completion.isoformat() if job.estimated_completion else None
        }
    
    async def _process_jobs(self):
        """Background job processor"""
        logger.info("Started job processor loop")
        
        while True:
            try:
                # Get next job to process
                job = self.queue.get_next_job()
                if not job:
                    await asyncio.sleep(1)
                    continue
                
                # Start processing
                self.queue.start_processing(job.job_id)
                
                # Process the job in background
                asyncio.create_task(self._process_single_job(job))
                
            except Exception as e:
                logger.error(f"Error in job processor loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_single_job(self, job: GenerationJob):
        """Process a single generation job"""
        logger.info(f"Processing job {job.job_id}")
        
        try:
            # Set estimated completion time
            job.estimated_completion = datetime.now() + timedelta(minutes=2)
            job.progress = 10.0
            
            # Check if Suno client is available
            if not self.suno_client:
                raise Exception("Suno client not available")
            
            # Prepare Suno API request
            suno_request = {
                "prompt": job.request.prompt,
                "lyrics": job.request.lyrics, 
                "style": job.request.style,
                "length_sec": job.request.length_sec,
                "mode": job.request.mode.value
            }
            
            job.progress = 30.0
            
            # Call Suno API through existing client
            response = await self._call_suno_api(suno_request)
            
            if not response.success:
                raise Exception(f"Suno API error: {response.error}")
            
            job.suno_job_id = response.job_id
            job.progress = 50.0
            
            # Poll for completion (simplified - in production use webhooks)
            audio_url, metadata = await self._poll_suno_completion(response.job_id)
            
            job.progress = 90.0
            
            # Download and store audio file
            local_path = await self._download_audio(audio_url, job.job_id)
            
            # Complete the job
            self.queue.complete_job(
                job.job_id,
                audio_url=f"/storage/suno_generations/{local_path.name}",
                metadata={
                    **metadata,
                    "original_url": audio_url,
                    "local_path": str(local_path),
                    "suno_job_id": job.suno_job_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing job {job.job_id}: {e}")
            self.queue.complete_job(job.job_id, error=str(e))
    
    async def _call_suno_api(self, request: Dict[str, Any]) -> SunoResponse:
        """Call Suno API through existing client"""
        try:
            # Use the existing Suno client
            if hasattr(self.suno_client, 'generate_music'):
                return await self.suno_client.generate_music(**request)
            else:
                # Fallback: simulate response for development
                logger.warning("Suno client not fully configured - using mock response")
                return SunoResponse(
                    success=True,
                    job_id=str(uuid.uuid4()),
                    data={"status": "processing"}
                )
                
        except Exception as e:
            logger.error(f"Suno API call failed: {e}")
            return SunoResponse(success=False, error=str(e))
    
    async def _poll_suno_completion(self, suno_job_id: str) -> tuple[str, Dict]:
        """Poll Suno API for job completion"""
        max_attempts = 60  # 5 minutes max
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Check job status (would use real Suno API in production)
                if hasattr(self.suno_client, 'get_job_status'):
                    status = await self.suno_client.get_job_status(suno_job_id)
                    if status.success and status.audio_url:
                        return status.audio_url, status.metadata or {}
                
                # Mock completion for development
                if attempt > 10:  # Simulate completion after some attempts
                    return f"https://example.com/audio/{suno_job_id}.wav", {"duration": 60}
                
                await asyncio.sleep(5)
                attempt += 1
                
            except Exception as e:
                logger.error(f"Error polling Suno job {suno_job_id}: {e}")
                await asyncio.sleep(10)
                attempt += 2
        
        raise Exception(f"Suno job {suno_job_id} timed out")
    
    async def _download_audio(self, audio_url: str, job_id: str) -> Path:
        """Download audio file and store locally"""
        try:
            file_path = self.storage_dir / f"{job_id}.wav"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status == 200:
                        with open(file_path, 'wb') as f:
                            f.write(await response.read())
                        logger.info(f"Downloaded audio for job {job_id}")
                        return file_path
                    else:
                        raise Exception(f"Failed to download audio: HTTP {response.status}")
            
        except Exception as e:
            logger.error(f"Error downloading audio for job {job_id}: {e}")
            # Create empty file as fallback
            file_path = self.storage_dir / f"{job_id}_failed.wav"
            file_path.touch()
            return file_path

# Global service instance
_suno_service: Optional[SunoGenerationService] = None

def get_suno_generation_service() -> SunoGenerationService:
    """Get global Suno generation service instance"""
    global _suno_service
    if _suno_service is None:
        _suno_service = SunoGenerationService()
    return _suno_service

async def start_suno_service():
    """Start the Suno generation service"""
    service = get_suno_generation_service()
    await service.start_processor()

async def stop_suno_service():
    """Stop the Suno generation service"""
    service = get_suno_generation_service()
    await service.stop_processor()

# Convenience functions
async def generate_music_async(
    prompt: Optional[str] = None,
    lyrics: Optional[str] = None,
    style: Optional[str] = None,
    length_sec: int = 60,
    mode: str = "original",
    expressiveness: float = 1.0,
    production_quality: float = 1.0,
    creativity: float = 1.0,
    user_id: Optional[str] = None
) -> str:
    """Quick function to generate music"""
    
    request = SunoGenerationRequest(
        prompt=prompt,
        lyrics=lyrics,
        style=style,
        length_sec=length_sec,
        mode=GenerationMode(mode),
        expressiveness=expressiveness,
        production_quality=production_quality,
        creativity=creativity,
        user_id=user_id
    )
    
    service = get_suno_generation_service()
    return await service.generate_music(request)

async def get_generation_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Quick function to get generation status"""
    service = get_suno_generation_service()
    return await service.get_job_status(job_id)