from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..celery_worker import generate_music_task

router = APIRouter()

class MusicRequest(BaseModel):
    prompt: str
    duration: Optional[int] = 30
    use_gpu: Optional[bool] = False

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

@router.post("/generate-music", response_model=TaskResponse)
async def generate_music(request: MusicRequest):
    """Generate music based on text prompt"""
    try:
        if request.use_gpu:
            from ..celery_worker import generate_music_gpu_task
            task = generate_music_gpu_task.delay(
                prompt=request.prompt,
                duration=request.duration,
                style="gpu"
            )
        else:
            task = generate_music_task.delay(
                prompt=request.prompt,
                duration=request.duration,
                style="default"
            )
        return TaskResponse(
            task_id=task.id,
            status="pending",
            message="Music generation task started"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a music generation task"""
    try:
        from ..celery_worker import celery_app
        task = celery_app.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                'task_id': task_id,
                'state': task.state,
                'progress': 0,
                'message': 'Task is waiting to be processed'
            }
        elif task.state == 'PROGRESS':
            response = {
                'task_id': task_id,
                'state': task.state,
                'progress': task.info.get('progress', 0),
                'message': task.info.get('message', 'Processing...')
            }
        elif task.state == 'SUCCESS':
            response = {
                'task_id': task_id,
                'state': task.state,
                'progress': 100,
                'result': task.info
            }
        else:  # FAILURE
            response = {
                'task_id': task_id,
                'state': task.state,
                'error': str(task.info)
            }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))