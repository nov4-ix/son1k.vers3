#!/bin/bash
# Son1kVers3 - Script de instalación automática
# Prepara el repositorio son1k-fusion para ejecutar

set -e  # Salir si hay errores

echo "🎵 Son1kVers3 - Instalación Automática"
echo "======================================"

# Verificar que estamos en el directorio correcto
if [ ! -d "src" ] || [ ! -f "src/audio_analysis.py" ] || [ ! -f "src/audio_post.py" ]; then
    echo "❌ Error: No se encontraron los módulos de audio"
    echo "Asegúrate de estar en el directorio son1k-fusion con:"
    echo "  - src/audio_analysis.py"
    echo "  - src/audio_post.py"
    echo "  - src/main.py (se creará)"
    exit 1
fi

echo "✅ Módulos de audio encontrados"

# Verificar Python
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ -z "$python_version" ]; then
    echo "❌ Python 3 no encontrado"
    exit 1
fi

echo "✅ Python $python_version encontrado"

# Crear directorios
echo "📁 Creando directorios..."
mkdir -p storage/{uploads,output}/{,ghost}
mkdir -p logs temp
echo "✅ Directorios creados"

# Crear __init__.py si no existe
if [ ! -f "src/__init__.py" ]; then
    touch src/__init__.py
    echo "✅ src/__init__.py creado"
fi

# Instalar dependencias mínimas
echo "📦 Instalando dependencias básicas..."
pip3 install fastapi uvicorn numpy soundfile python-multipart pydantic

echo ""
echo "¿Instalar dependencias opcionales para funciones avanzadas? (y/n)"
read -r install_optional

if [ "$install_optional" = "y" ] || [ "$install_optional" = "Y" ]; then
    echo "📦 Instalando dependencias opcionales..."
    pip3 install librosa scipy pyloudnorm pyrubberband || echo "⚠️  Algunas dependencias opcionales fallaron - el sistema funcionará en modo básico"
fi

# Verificar instalación
echo ""
echo "🧪 Verificando instalación..."
python3 -c "import fastapi, uvicorn, numpy, soundfile; print('✅ Dependencias básicas OK')" || {
    echo "❌ Error en dependencias básicas"
    exit 1
}

# Mostrar estado
echo ""
echo "📊 Estado del sistema:"
echo "✅ Directorios configurados"
echo "✅ Dependencias instaladas"
echo "✅ Módulos de audio disponibles"

# Instrucciones finales
echo ""
echo "🚀 Instalación completada!"
echo ""
echo "Para ejecutar Son1kVers3:"
echo "  python3 run.py"
echo ""
echo "O usando Make:"
echo "  make dev"
echo ""
echo "Documentación estará en: http://localhost:8000/docs"
echo ""
echo "Disfruta creando música con Son1kVers3!"