# 🎵 Son1k v3.0 - AI Music Generation Platform

Production-ready AI music generation platform using MusicGen with FastAPI backend and React frontend.

![Son1k](https://img.shields.io/badge/Son1k-v3.0-blue) ![Python](https://img.shields.io/badge/Python-3.11-green) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal) ![React](https://img.shields.io/badge/React-18-blue)

## ✨ Features

- 🎤 **Maqueta → Production**: Upload demos and get professional AI productions  
- 📊 **Advanced Audio Analysis**: Tempo, key, energy, vocal detection
- 🎛️ **Professional Postprocessing**: SSL EQ, Melodyne-like tuning, Neve saturation, mastering
- 🔄 **A/B Comparison**: Side-by-side demo vs production playback
- 🤖 **Ghost Studio**: Automated music generation with presets and job queue
- 🎛️ **Manual Generation**: Full control over MusicGen parameters
- 🎵 **High Quality Audio**: Professional postprocessing with normalization and limiting
- 🔐 **JWT Authentication**: Secure user management
- 🎨 **Modern UI**: React frontend with dark/light modes
- 📱 **Responsive Design**: Works on desktop and mobile
- 🐳 **Docker Support**: Easy deployment with containers
- 🧪 **Comprehensive Tests**: Smoke tests and integration testing

## 🚀 Quick Start

### Prerequisites

- **macOS** with Homebrew (tested, other OS should work)
- **Python 3.11**
- **Node.js 18+**
- **ffmpeg**: `brew install ffmpeg`
- **rubberband** (optional, for better pitch shifting): `brew install rubberband`

### Setup & Run

```bash
# Clone repository
git clone <your-repo>
cd son1k

# One-time setup (creates venv, installs deps, creates folders)
make setup

# Install additional tools
make install-deps

# Start development servers (API + Frontend)
make dev-all
```

That's it! 🎉

- **Frontend**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🎮 Usage

### Maqueta → Production (NEW! 🎤)

The revolutionary new workflow that transforms your demos into professional productions:

1. **Upload Demo**: Drop any audio file (WAV, MP3, FLAC, etc.)
2. **Describe Vision**: Tell the AI how you want it to sound ("make it a polished pop anthem", "give it jazz fusion vibes", etc.)
3. **AI Analysis**: System analyzes tempo, key, energy, vocal presence automatically
4. **AI Generation**: MusicGen creates new music using analysis + your vision
5. **Professional Processing**: Applies SSL EQ, pitch correction, Neve saturation, mastering chain
6. **A/B Compare**: Listen to original demo vs AI production side-by-side

**Processing Chain Applied:**
- **Analysis**: Tempo detection, key detection, energy/structure analysis, vocal detection
- **Generation**: MusicGen with context-aware prompts
- **SSL EQ**: 4-band parametric EQ (low shelf, 2 peak mids, high shelf, HPF)
- **Tuning**: Melodyne-like pitch correction (if vocals detected)
- **Neve Saturation**: Console-style harmonic enhancement with oversampling
- **Mastering**: LUFS normalization, brick-wall limiting, fade in/out

### Ghost Studio (Automated)

1. **Select Preset**: Choose from pre-configured music styles
2. **Add Instructions**: Optional customization prompts
3. **Create Job**: Automatic generation with job queue
4. **Monitor Progress**: Real-time job status updates
5. **Play Results**: Stream generated music

### Manual Generation

1. **Describe Music**: Write detailed text prompt
2. **Adjust Parameters**: Duration, temperature, top-k, top-p, seed
3. **Generate**: Direct MusicGen generation
4. **Download**: Access WAV files in `/output`

## 🏗️ Architecture

```
son1k/
├── src/                    # Backend (FastAPI)
│   ├── main.py            # FastAPI app with CORS & static files
│   ├── generate.py        # MusicGen integration
│   ├── audio_post.py      # Professional audio postprocessing (SSL, Neve, mastering)
│   ├── audio_analysis.py  # Audio analysis (tempo, key, energy, vocals)
│   ├── ghost_api.py       # Ghost Studio API (presets + maqueta workflow)
│   └── auth/              # JWT authentication
├── ghost-studio/          # Ghost Studio MVP
│   ├── presets.json       # Music style presets
│   ├── jobs.json          # Job queue storage
│   └── ghost_api.py       # Ghost Studio API
├── frontend/              # React + Vite frontend
│   └── src/App.jsx        # Main UI with tabs (Maqueta, Ghost, Manual)
├── uploads/               # Uploaded demo files
│   └── ghost/            # Maqueta sessions
├── output/                # Generated audio files
│   └── ghost/            # Production outputs
└── tests/                 # Smoke tests (including maqueta workflow)
```

## 🎵 Audio Pipeline

### Maqueta → Production Pipeline
1. **Upload**: Demo file (WAV/MP3/FLAC) → `uploads/ghost/{session_id}/`
2. **Analysis**: 
   - Tempo detection (librosa beat tracking)
   - Key detection (chroma + Krumhansl-Schmuckler algorithm)
   - Energy curve analysis & structure detection
   - Vocal presence detection (spectral + MFCC features)
3. **AI Prompt Construction**: Analysis context + user vision → final prompt
4. **Generation**: MusicGen (facebook/musicgen-small) with enhanced prompt
5. **Professional Postprocessing**:
   - **SSL EQ**: Low shelf (+1.5dB@80Hz), Mid cuts (-1dB@400Hz), High shelf (+1dB@8kHz), HPF@20Hz
   - **Pitch Correction**: Melodyne-like tuning using detected key (if vocals present)
   - **Neve Saturation**: Console modeling with 4x oversampling, harmonic enhancement
   - **Mastering**: LUFS normalization (-14dB), brick-wall limiter (-0.3dB), fade in/out
6. **Output**: Production WAV → `output/ghost/{session_id}/production.wav`

### Standard Generation Pipeline  
1. **Generation**: MusicGen (facebook/musicgen-small)
2. **Basic Postprocessing**:
   - RMS Normalization to -14 LUFS
   - Fade in (50ms) / Fade out (200ms)
   - Soft limiting to -0.3dBFS
3. **Output**: 32kHz mono WAV files
4. **Serving**: Static files via FastAPI

## 🎛️ Configuration

### Environment Variables

Copy `backend.env.example` to `src/.env`:

```bash
# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database  
DATABASE_URL=sqlite:///./son1k.db

# CORS
BACKEND_CORS_ORIGINS=http://localhost:5173

# App
APP_NAME=Son1k API
APP_DEBUG=true
```

### Frontend Environment

Create `frontend/.env`:

```bash
VITE_API_URL=http://localhost:8000
```

## 🧪 Testing

```bash
# All tests
make test

# Smoke tests only
make test-smoke

# Individual test
python -m pytest tests/test_api.py::test_health -v
```

### Test Coverage

- ✅ Health check endpoint
- ✅ Music generation (2s sample)
- ✅ Ghost Studio workflow (preset → job → completion)
- ✅ Static file serving
- ✅ Models endpoint
- ✅ Cache management

## 📊 Performance

### Model Memory Usage

| Model | Parameters | Memory | Generation Speed |
|-------|------------|--------|------------------|
| Small | 300M | ~2GB | Fastest (recommended) |
| Medium | 1.5B | ~8GB | Better quality |
| Large | 3.3B | ~16GB | Highest quality |

### Device Support

- **CUDA**: Best performance (if available)
- **MPS**: Apple Silicon support
- **CPU**: Fallback (slower)

### Generation Limits

- **Duration**: 1-30 seconds (MusicGen limitation)
- **Sample Rate**: 32kHz (for memory efficiency)
- **Channels**: Mono (configurable to stereo)

## 🐳 Docker Deployment

```bash
# Development
make docker-up

# Production build
docker compose -f docker-compose.prod.yml up --build
```

## 🔧 Development Commands

```bash
# Setup & Installation
make setup           # Initial setup
make install-deps    # Install additional tools

# Development
make dev-all         # Start API + Frontend (recommended)
make api            # API server only
make web            # Frontend only

# Testing & Maintenance
make test           # Run all tests  
make test-smoke     # Quick smoke tests
make clean          # Clean generated files
make build          # Production build

# Docker
make docker-up      # Start with Docker
make docker-down    # Stop containers

# Help
make help           # Show all commands
```

## 🎨 Ghost Studio Presets

Built-in music style presets:

- **🎸 Latin Rock**: Energetic guitars and percussion
- **🎤 Trap 808**: Heavy trap beats with dark atmosphere  
- **🎬 Ambient Cinematic**: Orchestral atmospheric soundscapes
- **🌆 Synthwave Retro**: 80s inspired electronic music
- **🎷 Jazz Fusion**: Complex jazz with electric instruments

## 🎤 Maqueta → Production Workflow

Son1k v3.0 introduces revolutionary **demo-to-production** capability that analyzes your rough recordings and generates professional-quality productions.

### How It Works

#### 1. Audio Analysis Engine
- **Tempo Detection**: Robust BPM estimation with onset-based fallback
- **Key Detection**: Krumhansl-Schmuckler key profiling with chromagram analysis  
- **Energy Structure**: RMS-based energy curve with verse/chorus section detection
- **Vocal Detection**: Spectral centroid + MFCC + zero-crossing rate analysis
- **Musical Characteristics**: Dynamic range, spectral rolloff, brightness analysis

#### 2. Context-Aware AI Generation  
Instead of generic prompts, the system builds intelligent prompts like:
```
"Using detected musical characteristics: tempo around 128 BPM, key of G major, 
moderate energy, includes vocal elements. Transform this musical foundation into: 
a polished pop anthem with modern production"
```

#### 3. Professional Audio Processing

**SSL-Style 4-Band EQ:**
- Low Shelf: +1.5dB @ 80Hz (warmth)
- Mid 1 Peak: -1.0dB @ 400Hz, Q=1.0 (clarity)  
- Mid 2 Peak: +1.5dB @ 3kHz, Q=0.8 (presence)
- High Shelf: +1.0dB @ 8kHz (air)
- High-pass: 20Hz (rumble removal)

**Melodyne-like Pitch Correction:**
- YIN algorithm fundamental frequency detection
- Scale-aware pitch correction to detected key  
- Formant preservation (with pyrubberband)
- Configurable correction strength

**Neve Console Saturation:**
- 4x oversampling for alias-free processing
- Asymmetric tanh + harmonic enhancement
- 2nd/3rd harmonic generation  
- Pre/post console EQ characteristics

**Mastering Chain:**
- LUFS loudness normalization (pyloudnorm)
- Brick-wall limiting with lookahead
- Professional fade curves

#### 4. A/B Comparison Interface
- Side-by-side demo vs production players
- Analysis metadata display
- Processing chain transparency  
- Session management with unique IDs

### API Usage

```bash
# Upload maqueta and generate production
curl -X POST http://localhost:8000/api/v1/ghost/maqueta \
  -F "file=@my_demo.wav" \
  -F "prompt=transform into uplifting electronic music" \
  -F "duration=15" \
  -F "tune_amount=0.7" \
  -F "lufs_target=-14"

# Response includes analysis + production URLs
{
  "ok": true,
  "demo": {
    "url": "/uploads/ghost/uuid/demo.wav",
    "analysis": {
      "tempo": {"bpm": 128.5},
      "key_guess": {"root": "G", "scale": "major"},
      "vocals": {"has_vocals": true, "vocal_probability": 0.85}
    }
  },
  "production": {
    "url": "/output/ghost/uuid/production.wav", 
    "post_metadata": {
      "processing_chain": ["tuning", "ssl_eq", "neve_saturation", "lufs_normalization", "limiter", "fades"],
      "lufs_gain_db": -2.3
    }
  },
  "prompt_final": "Using detected characteristics: tempo 128 BPM, G major, vocals. Transform into: uplifting electronic music"
}
```

### Supported Formats
- **Input**: WAV, MP3, FLAC, AIFF, M4A (up to 100MB)
- **Output**: 32kHz mono WAV (production ready)
- **Duration**: 5-30 seconds (MusicGen limitation)

### Performance Notes
- **Analysis**: ~2-5 seconds (depending on file length)
- **Generation**: ~15-45 seconds (first run downloads model)
- **Processing**: ~3-8 seconds (depending on chain complexity)
- **Total**: ~30-60 seconds end-to-end

## 🔍 Troubleshooting

### Model Download Issues

```bash
# Clear model cache
make clean
# Or via API
curl -X DELETE http://localhost:8000/api/v1/cache
```

### Memory Issues

- Use `musicgen-small` model (default)
- Reduce duration to 5-10 seconds
- Close other applications

### Audio Processing Issues

**"librosa not found"**: 
```bash
source .venv/bin/activate
pip install librosa
```

**"pyrubberband not available"**:
```bash
# Install rubberband system dependency
brew install rubberband
# Then install Python wrapper
pip install pyrubberband
```

**"pyloudnorm missing"**:
```bash
pip install pyloudnorm
```

### Maqueta Upload Issues

**"File too large"**: Max 100MB per upload
**"Unsupported format"**: Use WAV, MP3, FLAC, AIFF, or M4A
**"Analysis failed"**: File may be corrupted or very short (< 0.5s)

### Performance on Mac

- **M1/M2**: Enable MPS acceleration automatically
- **Intel**: Uses CPU (slower but works)
- Monitor with Activity Monitor

### Common Errors

**"No module named 'src'"**: Run from project root
**"CUDA not available"**: Normal on Mac, uses MPS/CPU
**"Port 8000 in use"**: Kill process: `lsof -ti:8000 | xargs kill -9`
**"Upload processing failed"**: Check file format and size

## 📚 API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Ghost Studio**: [docs/ghost_studio.md](docs/ghost_studio.md)

### Key Endpoints

```bash
# Health
GET /health

# Manual Generation
POST /api/v1/generate

# Maqueta → Production (NEW!)
POST /api/v1/ghost/maqueta     # Upload demo, generate production
GET  /api/v1/ghost/sessions/{id}  # Session status
DELETE /api/v1/ghost/sessions/{id}  # Delete session

# Ghost Studio  
GET  /api/v1/ghost/presets
POST /api/v1/ghost/job
GET  /api/v1/ghost/jobs
GET  /api/v1/ghost/stats       # Usage statistics

# Models & Cache
GET    /api/v1/models
DELETE /api/v1/cache
```

## 🚀 Production Deployment

1. **Environment**: Set production environment variables
2. **Database**: Use PostgreSQL instead of SQLite
3. **Reverse Proxy**: Nginx for static files and SSL
4. **Process Manager**: PM2 or systemd for API
5. **Monitoring**: Add logging and monitoring
6. **Storage**: External storage for generated files

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `make test`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push branch: `git push origin feature/amazing-feature`
6. Open Pull Request

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Credits

- **MusicGen**: Facebook Research
- **Transformers**: Hugging Face
- **FastAPI**: Sebastián Ramirez
- **React**: Meta

---

**Built with ❤️ by the Son1k Team**

For more details, check out the [Ghost Studio documentation](docs/ghost_studio.md) and explore the API at http://localhost:8000/docs