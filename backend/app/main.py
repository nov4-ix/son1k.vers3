from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import music

app = FastAPI(
    title="Son1k Music Generation API",
    description="AI-powered music generation platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(music.router, tags=["music"])

@app.get("/")
async def root():
    return {"message": "Son1k Music Generation API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}