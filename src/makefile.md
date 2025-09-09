# Son1kVers3 - Makefile Final
.PHONY: install dev prod test clean setup frontend backend docs

# Variables
PYTHON := python3
PIP := pip3
UVICORN := uvicorn
PORT := 8000
HOST := 0.0.0.0

# Configuración por defecto
setup:
	@echo "🎵 Configurando Son1kVers3..."
	@mkdir -p storage/uploads storage/output storage/uploads/ghost storage/output/ghost
	@mkdir -p logs temp
	@echo "📁 Directorios creados"
	@echo "✅ Setup completado"

# Instalación de dependencias
install:
	@echo "📦 Instalando dependencias..."
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencias instaladas"

# Instalación completa (con dependencias opcionales)
install-full:
	@echo "📦 Instalación completa con dependencias opcionales..."
	$(PIP) install -r requirements.txt
	# Intentar instalar dependencias opcionales
	-$(PIP) install librosa scipy pyloudnorm pyrubberband
	@echo "⚠️  Algunas dependencias opcionales pueden fallar - el sistema funcionará con graceful fallback"
	@echo "✅ Instalación completa"

# Desarrollo - servidor con recarga automática
dev: setup
	@echo "🚀 Iniciando servidor de desarrollo..."
	@echo "📍 Servidor en http://$(HOST):$(PORT)"
	@echo "📖 Documentación en http://$(HOST):$(PORT)/docs"
	@echo "🔄 Recarga automática activada"
	$(UVICORN) src.main:app --host $(HOST) --port $(PORT) --reload --log-level info

# Producción - servidor optimizado
prod: setup
	@echo "🚀 Iniciando servidor de producción..."
	@echo "📍 Servidor en http://$(HOST):$(PORT)"
	$(UVICORN) src.main:app --host $(HOST) --port $(PORT) --workers 4 --log-level warning

# Frontend standalone (si quieres servir solo el HTML)
frontend:
	@echo "🎨 Sirviendo frontend..."
	@if [ -f "index.html" ]; then \
		echo "📍 Frontend en http://localhost:3000"; \
		$(PYTHON) -m http.server 3000; \
	else \
		echo "❌ No se encontró index.html"; \
		echo "💡 Usa 'make dev' para servidor completo"; \
	fi

# Backend solo API
backend: setup
	@echo "🔧 Iniciando solo backend API..."
	$(UVICORN) src.main:app --host $(HOST) --port $(PORT) --reload

# Test básico de conectividad
test:
	@echo "🧪 Probando conectividad..."
	@curl -f http://localhost:$(PORT)/health 2>/dev/null && echo "✅ API funcionando" || echo "❌ API no responde"
	@curl -f http://localhost:$(PORT)/ 2>/dev/null > /dev/null && echo "✅ Frontend accesible" || echo "❌ Frontend no accesible"

# Limpiar archivos temporales
clean:
	@echo "🧹 Limpiando archivos temporales..."
	@rm -rf __pycache__ .pytest_cache .coverage
	@rm -rf storage/uploads/* storage/output/* temp/* logs/*
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -name "*~" -delete
	@echo "✅ Limpieza completada"

# Documentación
docs:
	@echo "📖 Abriendo documentación..."
	@echo "🌐 Documentación interactiva en http://localhost:$(PORT)/docs"
	@echo "📚 Documentación alternativa en http://localhost:$(PORT)/redoc"
	@open http://localhost:$(PORT)/docs 2>/dev/null || xdg-open http://localhost:$(PORT)/docs 2>/dev/null || echo "🔗 Abre manualmente: http://localhost:$(PORT)/docs"

# Verificar estado del sistema
status:
	@echo "📊 Estado de Son1kVers3:"
	@echo "🐍 Python: $(shell $(PYTHON) --version)"
	@echo "📦 FastAPI: $(shell $(PIP) show fastapi 2>/dev/null | grep Version || echo 'No instalado')"
	@echo "🎵 Librosa: $(shell $(PIP) show librosa 2>/dev/null | grep Version || echo 'No instalado (opcional)')"
	@echo "📁 Directorios:"
	@ls -la storage/ 2>/dev/null || echo "  📁 storage/ no existe - ejecuta 'make setup'"
	@echo "🌐 Conectividad:"
	@curl -f http://localhost:$(PORT)/health 2>/dev/null > /dev/null && echo "  ✅ API activa" || echo "  ❌ API inactiva"

# Instalar y ejecutar en un comando
quickstart: install setup dev

# Ayuda
help:
	@echo "🎵 Son1kVers3 - Comandos disponibles:"
	@echo ""
	@echo "📦 INSTALACIÓN:"
	@echo "  make install      - Instalar dependencias básicas"
	@echo "  make install-full - Instalar dependencias completas (con opcionales)"
	@echo "  make setup        - Configurar directorios"
	@echo ""
	@echo "🚀 EJECUCIÓN:"
	@echo "  make dev          - Servidor desarrollo (recomendado)"
	@echo "  make prod         - Servidor producción"
	@echo "  make backend      - Solo API backend"
	@echo "  make frontend     - Solo frontend HTML"
	@echo ""
	@echo "🛠️  UTILIDADES:"
	@echo "  make test         - Probar conectividad"
	@echo "  make status       - Ver estado del sistema"
	@echo "  make docs         - Abrir documentación"
	@echo "  make clean        - Limpiar archivos temporales"
	@echo ""
	@echo "⚡ INICIO RÁPIDO:"
	@echo "  make quickstart   - Instalar + configurar + ejecutar"
	@echo ""
	@echo "📖 Documentación: http://localhost:8000/docs"
	@echo "🎯 Frontend: http://localhost:8000/"

# Default target
.DEFAULT_GOAL := help