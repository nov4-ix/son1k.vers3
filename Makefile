SHELL := /bin/zsh

.PHONY: setup dev api web dev-all stop docker-up docker-down test clean install-deps

# Setup development environment
setup:
	@echo "🚀 Setting up Son1k development environment..."
	python3.11 -m venv .venv
	@echo "📦 Installing Python dependencies..."
	bash -lc 'source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt'
	@echo "📦 Installing Node.js dependencies..."
	cd frontend && npm ci
	@echo "📁 Creating directories..."
	mkdir -p output uploads/ghost output/ghost
	@echo "📄 Creating environment file..."
	@if [ ! -f src/.env ]; then cp backend.env.example src/.env && echo "Created src/.env from example"; fi
	@echo "✅ Setup complete! Run 'make dev-all' to start development servers."

# Install additional dependencies
install-deps:
	@echo "📦 Installing concurrently..."
	npm install -g concurrently
	@echo "🍺 Checking dependencies..."
	@echo "FFmpeg:" && (which ffmpeg || echo "⚠️  Please install: brew install ffmpeg")
	@echo "Rubberband (optional):" && (which rubberband || echo "💡 Optional for better pitch shifting: brew install rubberband")
	@echo "Python libraries status:"
	@bash -lc 'source .venv/bin/activate && python -c "
import sys
try:
    import librosa; print(\"✅ librosa available\")
except: print(\"❌ librosa missing\")
try:
    import pyloudnorm; print(\"✅ pyloudnorm available\")  
except: print(\"❌ pyloudnorm missing\")
try:
    import pyrubberband; print(\"✅ pyrubberband available\")
except: print(\"⚠️  pyrubberband missing (optional)\")
"'

# Run both API and frontend with concurrently
dev-all:
	@echo "🚀 Starting Son1k development servers..."
	npx concurrently -k -n "api,web" -c "blue,green" \
		"bash -lc 'source .venv/bin/activate && uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload'" \
		"bash -lc 'cd frontend && npm run dev'"

# Individual development servers
api:
	@echo "🔧 Starting API server..."
	bash -lc 'source .venv/bin/activate && uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload'

web:
	@echo "🌐 Starting frontend server..."
	cd frontend && npm run dev

# Legacy combined command (sequential)
dev: api web

# Stop development servers
stop:
	@echo "🛑 Use CTRL+C to stop dev-all servers"
	@echo "💡 This target is a placeholder - use CTRL+C in the dev-all terminal"

# Docker commands
docker-up:
	@echo "🐳 Starting Docker development environment..."
	docker compose up --build

docker-down:
	@echo "🐳 Stopping Docker containers..."
	docker compose down

# Run tests
test:
	@echo "🧪 Running API tests..."
	bash -lc 'source .venv/bin/activate && python -m pytest tests/ -v'

test-smoke:
	@echo "🔥 Running smoke tests..."
	bash -lc 'source .venv/bin/activate && python -m pytest tests/test_api.py::test_health -v'
	bash -lc 'source .venv/bin/activate && python -m pytest tests/test_api.py::test_generate_smoke -v'
	bash -lc 'source .venv/bin/activate && python -m pytest tests/test_api.py::test_ghost_studio_smoke -v'

# Clean up generated files and cache
clean:
	@echo "🧹 Cleaning up..."
	rm -rf output/*.wav
	rm -rf __pycache__ **/__pycache__ .pytest_cache
	bash -lc 'source .venv/bin/activate && python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"'
	@echo "✅ Cleanup complete"

# Production build
build:
	@echo "🏗️  Building frontend for production..."
	cd frontend && npm run build
	@echo "✅ Production build complete"

# Show help
help:
	@echo "Son1k Development Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup         - Initial setup (venv, deps, folders)"
	@echo "  make install-deps  - Install additional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make dev-all       - Start API + Frontend with concurrently"
	@echo "  make api           - Start API server only"
	@echo "  make web           - Start frontend only"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make test-smoke    - Run smoke tests"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     - Start with Docker"
	@echo "  make docker-down   - Stop Docker containers"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Clean generated files"
	@echo "  make build         - Build for production"
	@echo "  make help          - Show this help"

# Default target
.DEFAULT_GOAL := help