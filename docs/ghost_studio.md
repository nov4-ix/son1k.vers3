# 🤖 Ghost Studio API Documentation

Ghost Studio is Son1k's automated music generation system that uses presets and job queues to create music with minimal user input.

## Overview

Ghost Studio provides a high-level interface for music generation through:

- **Presets**: Pre-configured music styles with optimized parameters
- **Job Queue**: Background processing with status tracking
- **Automated Workflows**: From prompt to finished audio

## API Endpoints

### Get Presets

```http
GET /api/v1/ghost/presets
```

Returns all available music generation presets.

**Response:**
```json
{
  "presets": {
    "latin_rock": {
      "name": "Latin Rock",
      "description": "Energetic Latin rock with guitars and percussion",
      "prompt_base": "latin rock with electric guitars, congas, timbales, energetic drums, 120 bpm, major key",
      "suggested_bpm": 120,
      "suggested_duration": 12,
      "seed": 42,
      "tags": ["rock", "latin", "energetic", "guitar"],
      "parameters": {
        "temperature": 1.0,
        "top_k": 250,
        "top_p": 0.9
      }
    }
  },
  "count": 5
}
```

### Create Job

```http
POST /api/v1/ghost/job
```

Creates a new music generation job.

**Request Body:**
```json
{
  "preset": "latin_rock",
  "prompt_extra": "with saxophone solo",
  "duration": 15.0
}
```

**Parameters:**
- `preset` (required): Preset identifier
- `prompt_extra` (optional): Additional prompt text
- `duration` (optional): Override preset duration (1-30 seconds)

**Response:**
```json
{
  "ok": true,
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "message": "Job created and processing started"
}
```

### Get Jobs

```http
GET /api/v1/ghost/jobs?limit=50&status=done
```

Lists jobs with optional filtering.

**Query Parameters:**
- `limit` (optional, default=50): Maximum number of jobs to return
- `status` (optional): Filter by status (`queued`, `running`, `done`, `error`)

**Response:**
```json
{
  "jobs": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "preset": "latin_rock",
      "prompt_extra": "with saxophone solo",
      "duration": 15.0,
      "status": "done",
      "output_url": "/output/son1k_20250101_143022.wav",
      "created_at": "2025-01-01T14:30:15.123456",
      "completed_at": "2025-01-01T14:30:45.654321"
    }
  ],
  "count": 1,
  "total_count": 25
}
```

### Get Job Status

```http
GET /api/v1/ghost/jobs/{job_id}
```

Get status of a specific job.

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "done",
  "output_url": "/output/son1k_20250101_143022.wav",
  "progress": "Job done"
}
```

### Delete Job

```http
DELETE /api/v1/ghost/jobs/{job_id}
```

Deletes a job from the queue (not allowed for running jobs).

**Response:**
```json
{
  "ok": true,
  "message": "Job 123e4567... deleted"
}
```

### Get Statistics

```http
GET /api/v1/ghost/stats
```

Returns Ghost Studio statistics.

**Response:**
```json
{
  "total_jobs": 25,
  "total_presets": 5,
  "status_breakdown": {
    "done": 20,
    "running": 1,
    "queued": 2,
    "error": 2
  },
  "available_presets": ["latin_rock", "trap_808", "ambient_cinematique"]
}
```

## Job Status Flow

```
queued → running → done
   ↓        ↓        ↑
   ↓        ↓    (success)
   ↓        ↓
   ↓    → error ←
   ↓       ↑
   ↓   (failure)
   ↓
→ error
(validation failure)
```

## Built-in Presets

### 🎸 Latin Rock (`latin_rock`)
- **Style**: Energetic Latin rock with guitars and percussion
- **BPM**: 120
- **Duration**: 12s
- **Seed**: 42
- **Tags**: rock, latin, energetic, guitar

### 🎤 Trap 808 (`trap_808`)
- **Style**: Heavy trap beat with 808 drums and dark atmosphere
- **BPM**: 140  
- **Duration**: 15s
- **Seed**: 808
- **Tags**: trap, hip-hop, 808, dark, urban

### 🎬 Ambient Cinematic (`ambient_cinematique`)
- **Style**: Atmospheric cinematic soundscape with orchestral elements
- **BPM**: 70
- **Duration**: 20s
- **Seed**: 2001
- **Tags**: ambient, cinematic, orchestral, atmospheric, emotional

### 🌆 Synthwave Retro (`synthwave_retro`)
- **Style**: 80s inspired synthwave with retro aesthetics
- **BPM**: 110
- **Duration**: 18s
- **Seed**: 1984
- **Tags**: synthwave, retro, 80s, electronic, cyberpunk

### 🎷 Jazz Fusion (`jazz_fusion`)
- **Style**: Complex jazz fusion with electric instruments
- **BPM**: 130
- **Duration**: 16s
- **Seed**: 1959
- **Tags**: jazz, fusion, complex, electric, sophisticated

## Usage Examples

### Python/Requests Example

```python
import requests
import time

API_BASE = "http://localhost:8000"

# 1. Get available presets
presets = requests.get(f"{API_BASE}/api/v1/ghost/presets").json()
print(f"Available presets: {list(presets['presets'].keys())}")

# 2. Create a job
job_data = {
    "preset": "latin_rock",
    "prompt_extra": "with trumpet section",
    "duration": 12
}

response = requests.post(f"{API_BASE}/api/v1/ghost/job", json=job_data)
job = response.json()
job_id = job["job_id"]

print(f"Created job: {job_id}")

# 3. Poll for completion
while True:
    status_response = requests.get(f"{API_BASE}/api/v1/ghost/jobs/{job_id}")
    status = status_response.json()
    
    print(f"Status: {status['status']}")
    
    if status["status"] == "done":
        print(f"Audio ready: {API_BASE}{status['output_url']}")
        break
    elif status["status"] == "error":
        print(f"Error: {status.get('error_message')}")
        break
    
    time.sleep(2)
```

### cURL Examples

```bash
# Get presets
curl http://localhost:8000/api/v1/ghost/presets

# Create job
curl -X POST http://localhost:8000/api/v1/ghost/job \
  -H "Content-Type: application/json" \
  -d '{"preset": "trap_808", "prompt_extra": "with heavy bass", "duration": 10}'

# Check job status
curl http://localhost:8000/api/v1/ghost/jobs/YOUR_JOB_ID

# List all jobs
curl "http://localhost:8000/api/v1/ghost/jobs?limit=10"

# Get statistics
curl http://localhost:8000/api/v1/ghost/stats
```

### JavaScript/Fetch Example

```javascript
// Create and monitor a Ghost Studio job
async function createGhostJob() {
  // Create job
  const jobResponse = await fetch('/api/v1/ghost/job', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      preset: 'ambient_cinematique',
      prompt_extra: 'with ethereal vocals',
      duration: 20
    })
  });
  
  const job = await jobResponse.json();
  console.log('Job created:', job.job_id);
  
  // Poll for completion
  while (true) {
    const statusResponse = await fetch(`/api/v1/ghost/jobs/${job.job_id}`);
    const status = await statusResponse.json();
    
    if (status.status === 'done') {
      console.log('Audio ready:', status.output_url);
      // Play audio
      const audio = new Audio(status.output_url);
      audio.play();
      break;
    }
    
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}
```

## Architecture Details

### File Storage

- **Jobs**: `ghost-studio/jobs.json` with file locking
- **Presets**: `ghost-studio/presets.json` (read-only)
- **Audio Output**: `output/` directory, served as static files

### Background Processing

Jobs are processed in background threads using Python's `threading` module. For production, consider:

- **Celery**: For distributed task processing
- **Redis**: For job queue and caching
- **Database**: PostgreSQL for job persistence

### Thread Safety

- File locking (`fcntl`) prevents job queue corruption
- Model cache is thread-safe
- Each job runs in isolated thread

## Error Handling

### Common Errors

**400 Bad Request**
```json
{
  "detail": "Preset 'invalid_preset' not found"
}
```

**404 Not Found**
```json
{
  "detail": "Job not found"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Music generation failed: CUDA out of memory"
}
```

### Job Errors

Jobs can fail during processing. Check the `error_message` field:

```json
{
  "id": "job-123",
  "status": "error",
  "error_message": "Model loading failed: insufficient memory"
}
```

## Performance Considerations

- **Model Loading**: First job loads model (~30s), subsequent jobs are faster
- **Memory**: Each model variant has different memory requirements
- **Concurrency**: Limited by GPU memory, typically 1-2 concurrent jobs
- **Duration**: Longer generations consume more memory and time

## Custom Presets

To add custom presets, edit `ghost-studio/presets.json`:

```json
{
  "your_preset": {
    "name": "Your Style",
    "description": "Custom music style",
    "prompt_base": "your custom prompt here",
    "suggested_bpm": 128,
    "suggested_duration": 10,
    "seed": 1234,
    "tags": ["custom", "unique"],
    "parameters": {
      "temperature": 1.1,
      "top_k": 200,
      "top_p": 0.85
    }
  }
}
```

Restart the API server to load new presets.

## Monitoring and Debugging

### Logs

Check API logs for generation progress:

```bash
# Development
tail -f logs/api.log

# Docker
docker logs son1k_api -f
```

### Job Queue

Monitor job queue file:

```bash
# View current jobs
cat ghost-studio/jobs.json | jq

# Watch for changes  
watch -n 2 'cat ghost-studio/jobs.json | jq ".[] | {id: .id, status: .status}"'
```

### Memory Usage

```bash
# Monitor GPU memory (NVIDIA)
watch nvidia-smi

# Monitor system memory
htop
```

---

**Ghost Studio** provides a powerful abstraction over MusicGen, making AI music generation accessible through simple API calls and preset configurations.