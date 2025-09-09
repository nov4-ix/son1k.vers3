from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import asyncio

router = APIRouter()

# Mock job storage (in production, use Redis or database)
jobs = {}

class JobRequest(BaseModel):
    prompt: str
    duration: int = 30

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    result: str = None

@router.post("/process", response_model=JobResponse)
async def process_job(job: JobRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "prompt": job.prompt,
        "duration": job.duration,
        "result": None
    }
    
    # Start background processing (mock)
    asyncio.create_task(mock_process_job(job_id, job.duration))
    
    return JobResponse(
        job_id=job_id,
        status="processing",
        message="Job started successfully"
    )

@router.get("/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job.get("result")
    )

async def mock_process_job(job_id: str, duration: int):
    """Mock job processing with progress updates"""
    for i in range(10):
        await asyncio.sleep(duration / 10)
        jobs[job_id]["progress"] = (i + 1) * 10
    
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["result"] = f"Generated music for: {jobs[job_id]['prompt']}"
