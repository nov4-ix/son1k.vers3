#!/usr/bin/env python3
"""
Son1kVers3 - Script de inicio directo
Compatible con módulos audio_analysis.py y audio_post.py existentes
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Verificar versión de Python"""
    if sys.version_info < (3, 8):
        logger.error("Se requiere Python 3.8 o superior")
        sys.exit(1)
    logger.info(f"Python {sys.version_info.major}.{sys.version_info.minor} ✓")

def setup_directories():
    """Crear directorios necesarios"""
    directories = [
        "storage/uploads",
        "storage/output", 
        "storage/uploads/ghost",
        "storage/output/ghost",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    logger.info("Directorios configurados ✓")

def check_audio_modules():
    """Verificar módulos de audio personalizados"""
    audio_analysis_path = Path("src/audio_analysis.py")
    audio_post_path = Path("src/audio_post.py")
    
    if audio_analysis_path.exists() and audio_post_path.exists():
        logger.info("Módulos de audio profesional encontrados ✓")
        return True
    else:
        logger.warning("Módulos de audio no encontrados - usando modo básico")
        if not audio_analysis_path.exists():
            logger.warning(f"  Falta: {audio_analysis_path}")
        if not audio_post_path.exists():
            logger.warning(f"  Falta: {audio_post_path}")
        return False

def install_dependencies():
    """Instalar dependencias con opción de mínimas"""
    logger.info("Verificando dependencias...")
    
    # Verificar dependencias obligatorias
    required_modules = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn', 
        'numpy': 'NumPy',
        'soundfile': 'SoundFile',
        'pydantic': 'Pydantic'
    }
    
    missing_required = []
    for module, name in required_modules.items():
        try:
            __import__(module)
            logger.info(f"{name} ✓")
        except ImportError:
            missing_required.append(module)
            logger.warning(f"{name} ✗")
    
    # Verificar dependencias opcionales
    optional_modules = {
        'librosa': 'Librosa (análisis avanzado)',
        'scipy': 'SciPy (procesamiento de señales)',
        'pyloudnorm': 'PyLoudnorm (LUFS mastering)',
        'pyrubberband': 'PyRubberBand (pitch correction)'
    }
    
    missing_optional = []
    for module, name in optional_modules.items():
        try:
            __import__(module)
            logger.info(f"{name} ✓")
        except ImportError:
            missing_optional.append(module)
            logger.warning(f"{name} ✗ (opcional)")
    
    # Instalar si es necesario
    if missing_required:
        logger.error(f"Dependencias obligatorias faltantes: {', '.join(missing_required)}")
        
        response = input("¿Instalar dependencias automáticamente? (y/n): ")
        if response.lower() in ['y', 'yes', 'sí', 's']:
            try:
                # Instalar mínimas primero
                minimal_deps = ['fastapi', 'uvicorn[standard]', 'numpy', 'soundfile', 'python-multipart', 'pydantic']
                logger.info("Instalando dependencias mínimas...")
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + minimal_deps)
                
                # Preguntar por opcionales
                if missing_optional:
                    response_opt = input(f"¿Instalar también dependencias opcionales para funciones avanzadas? (y/n): ")
                    if response_opt.lower() in ['y', 'yes', 'sí', 's']:
                        try:
                            logger.info("Instalando dependencias opcionales...")
                            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_optional)
                        except subprocess.CalledProcessError:
                            logger.warning("Algunas dependencias opcionales fallaron - el sistema funcionará en modo básico")
                
                logger.info("Dependencias instaladas ✓")
                return True
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error instalando dependencias: {e}")
                return False
        else:
            logger.error("Dependencias requeridas no encontradas")
            return False
    
    elif missing_optional:
        logger.info("Dependencias obligatorias OK")
        logger.info(f"Dependencias opcionales faltantes: {len(missing_optional)}")
        logger.info("El sistema funcionará en modo básico")
        return True
    else:
        logger.info("Todas las dependencias encontradas ✓")
        return True

def validate_main_module():
    """Validar que src/main.py existe y es importable"""
    main_path = Path("src/main.py")
    if not main_path.exists():
        logger.error("src/main.py no encontrado")
        logger.error("Asegúrate de estar en el directorio son1k-fusion")
        return False
    
    # Verificar que el módulo src existe
    src_init = Path("src/__init__.py")
    if not src_init.exists():
        logger.info("Creando src/__init__.py...")
        src_init.touch()
    
    return True

def start_server(host="0.0.0.0", port=8000, reload=True):
    """Iniciar servidor"""
    logger.info("=" * 50)
    logger.info("🎵 Son1kVers3 - Resistencia Sonora")
    logger.info("=" * 50)
    logger.info(f"Servidor: http://{host}:{port}")
    logger.info(f"Documentación: http://{host}:{port}/docs")
    logger.info(f"Frontend: http://{host}:{port}/")
    logger.info("=" * 50)
    
    try:
        import uvicorn
        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
    except ImportError:
        logger.error("uvicorn no disponible - instala dependencias")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error iniciando servidor: {e}")
        sys.exit(1)

def show_status():
    """Mostrar estado del sistema"""
    print("🎵 Son1kVers3 - Estado del Sistema")
    print("=" * 40)
    
    # Python
    print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Archivos principales
    files_status = {
        "src/main.py": "API Principal",
        "src/audio_analysis.py": "Análisis de Audio",
        "src/audio_post.py": "Post-procesamiento",
        "requirements.txt": "Dependencias",
        "Makefile": "Comandos"
    }
    
    print("\n📁 Archivos:")
    for file_path, description in files_status.items():
        status = "✓" if Path(file_path).exists() else "✗"
        print(f"  {status} {description}: {file_path}")
    
    # Directorios
    print("\n📂 Directorios:")
    directories = ["storage/uploads", "storage/output", "src", "logs"]
    for directory in directories:
        status = "✓" if Path(directory).exists() else "✗"
        print(f"  {status} {directory}")
    
    # Dependencias
    print("\n📦 Dependencias:")
    modules = ["fastapi", "uvicorn", "numpy", "soundfile", "librosa", "scipy"]
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module}")

def main():
    """Función principal"""
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_status()
        return
    
    print("🎵 Son1kVers3 - Resistencia Sonora")
    print("=" * 40)
    
    # Verificaciones
    check_python_version()
    setup_directories()
    check_audio_modules()
    
    if not validate_main_module():
        sys.exit(1)
    
    if not install_dependencies():
        sys.exit(1)
    
    # Configuración desde variables de entorno
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    # Iniciar
    start_server(host, port, reload)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)