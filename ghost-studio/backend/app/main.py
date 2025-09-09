from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import jobs

app = FastAPI(title="Ghost Studio API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])

@app.get('/health')
def health():
    return {'status': 'ok'}
