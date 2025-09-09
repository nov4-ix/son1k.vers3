# Son1kVers3 - Resistencia Sonora

**Plataforma de creación musical con IA que democratiza la producción musical para 400+ millones de hispanohablantes.**

> "Lo imperfecto también es sagrado" - Manifiesto Son1k

## Características Principales

- **Maqueta → Production**: Analiza demos y genera producciones profesionales
- **Generación Musical**: Crea música original desde prompts textuales
- **Análisis Profesional**: Detección de clave, tempo, estructura vocal
- **Post-procesamiento**: SSL EQ, Neve saturation, LUFS mastering
- **Interfaz Accesible**: Frontend HTML moderno con soporte completo ARIA
- **API REST**: FastAPI con documentación interactiva

## Arquitectura

```
son1k-fusion/
├── src/
│   ├── main.py              # API principal integrada
│   ├── audio_analysis.py    # Motor de análisis profesional
│   └── audio_post.py        # Post-procesamiento SSL/Neve
├── frontend/
│   └── index.html           # Interfaz HTML moderna
├── storage/
│   ├── uploads/             # Archivos subidos
│   └── output/              # Producciones generadas
├── requirements.txt         # Dependencias Python
├── Makefile                # Comandos de ejecución
└── README.md               # Esta documentación
```

## Instalación Rápida

### Opción 1: Inicio Rápido (Recomendado)
```bash
cd son1k-fusion/
make quickstart
```

### Opción 2: Paso a Paso
```bash
# 1. Configurar directorios
make setup

# 2. Instalar dependencias
make install

# 3. Ejecutar servidor
make dev
```

## Dependencias

### Obligatorias
- **FastAPI** - API REST moderna
- **NumPy** - Procesamiento numérico
- **SoundFile** - Lectura/escritura de audio

### Opcionales (con graceful fallback)
- **Librosa** - Análisis avanzado de audio
- **SciPy** - Procesamiento de señales
- **PyLoudnorm** - Normalización LUFS
- **PyRubberBand** - Corrección de pitch

## Uso

### 1. Servidor de Desarrollo
```bash
make dev
```
- Servidor en `http://localhost:8000`
- Documentación en `http://localhost:8000/docs`
- Recarga automática activada

### 2. Servidor de Producción
```bash
make prod
```

### 3. Endpoints Principales

#### Procesar Maqueta
```http
POST /api/v1/maqueta/process
Content-Type: multipart/form-data

audio_file: archivo.wav
style: "pop" | "rock" | "electronic" | "auto"
intensity: 0.0-100.0
```

#### Generar Música
```http
POST /api/v1/generate
Content-Type: multipart/form-data

prompt: "descripción musical"
duration: 12.0
tempo: 120
creativity: 70.0
variation: 50.0
```

#### Estado de Job
```http
GET /api/v1/jobs/{job_id}/status
```

#### Historial
```http
GET /api/v1/history
```

## Funcionalidades Técnicas

### Análisis de Audio
- **Detección de tempo** usando beat tracking y onset detection
- **Detección de clave** con correlación cromática Krumhansl-Schmuckler
- **Análisis espectral** (centroide, brillo, bandwidth)
- **Detección vocal** usando MFCCs y análisis armónico/percusivo
- **Estructura energética** con percentiles dinámicos

### Post-procesamiento Profesional
- **SSL EQ**: 4 bandas paramétricas + high-pass
- **Neve Saturation**: Saturación asimétrica con armónicos
- **LUFS Mastering**: Normalización según estándares broadcast
- **Limitador**: Brick-wall con lookahead
- **Fades**: Entrada/salida con curvas cuadráticas

### Generación Musical
- **Motor MusicGen**: Síntesis de audio realista
- **Prompts inteligentes**: Construcción automática basada en análisis
- **Post-procesamiento**: Cadena completa de masterización

## Comandos Make

```bash
# Instalación
make install        # Dependencias básicas
make install-full   # Con dependencias opcionales
make setup          # Configurar directorios

# Ejecución
make dev           # Desarrollo (recomendado)
make prod          # Producción
make backend       # Solo API
make frontend      # Solo HTML

# Utilidades
make test          # Probar conectividad
make status        # Estado del sistema
make docs          # Abrir documentación
make clean         # Limpiar temporales
make help          # Ver ayuda completa
```

## Estructura de Respuestas

### Maqueta Procesada
```json
{
  "ok": true,
  "session_id": "uuid",
  "production_url": "/output/production_uuid.wav",
  "analysis": {
    "key": "C",
    "tempo": 120.0,
    "duration": 30.5,
    "has_vocals": true,
    "energy": 0.7
  },
  "processing_chain": [
    "tuning", "ssl_eq", "neve_saturation", 
    "lufs_normalization", "limiter", "fades"
  ],
  "processing_time": 45.2
}
```

### Job Status
```json
{
  "job_id": "uuid",
  "status": "processing" | "completed" | "failed",
  "progress": 75,
  "message": "Aplicando procesamiento profesional...",
  "result": { /* resultado cuando completed */ }
}
```

## Frontend

### Interfaz Moderna
- **Tabs accesibles** con navegación por teclado
- **Knobs expresivos** con soporte touch y ARIA
- **Sistema de toasts** para notificaciones
- **Responsive design** para móvil/desktop
- **Soporte reduced motion** para accesibilidad

### Componentes
- **Maqueta → Production**: Upload y procesamiento
- **Generación Manual**: Prompts y controles expresivos
- **Ghost Studio**: Configuración del sistema
- **Historial**: Creaciones anteriores

## Desarrollo

### Estructura del Código
```python
# src/main.py - Punto de entrada
from .audio_analysis import AudioAnalyzer
from .audio_post import AudioPostProcessor

# Flujo principal
analyzer = AudioAnalyzer()
processor = AudioPostProcessor()

# Análisis → Generación → Post-procesamiento
analysis = analyzer.analyze_audio_file(file_path)
audio, metadata = musicgen_service.generate_music(prompt)
processed, post_meta = processor.process_master(audio)
```

### Logs
```bash
# Ver logs en tiempo real
tail -f logs/son1k.log

# Logs por nivel
grep "ERROR" logs/son1k.log
grep "INFO" logs/son1k.log
```

### Testing
```bash
# Test básico
make test

# Test manual via curl
curl -X GET http://localhost:8000/health
curl -X GET http://localhost:8000/api/v1/history
```

## Solución de Problemas

### Dependencias Faltantes
```bash
# Si falla librosa
pip install librosa

# Si falla scipy
pip install scipy

# Instalación completa
make install-full
```

### Puerto Ocupado
```bash
# Cambiar puerto
PORT=8001 make dev

# O editar Makefile
```

### Permisos de Archivos
```bash
# Crear directorios manualmente
mkdir -p storage/{uploads,output}/{,ghost}
chmod 755 storage -R
```

### Audio No Se Reproduce
- Verificar formato de archivo soportado
- Comprobar que el archivo no esté corrupto
- Revisar permisos de directorio `storage/output`

## Producción

### Variables de Entorno
```bash
export SON1K_ENV=production
export SON1K_LOG_LEVEL=warning
export SON1K_WORKERS=4
```

### Nginx (Opcional)
```nginx
server {
    listen 80;
    server_name son1kvers3.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /uploads/ {
        alias /path/to/son1k-fusion/storage/uploads/;
    }
    
    location /output/ {
        alias /path/to/son1k-fusion/storage/output/;
    }
}
```

## Contribuir

### Código
1. Fork del repositorio
2. Crear rama feature
3. Desarrollar con tests
4. Pull request

### Issues
- Reportar bugs en GitHub Issues
- Incluir logs relevantes
- Especificar sistema operativo

## Licencia

MIT License - Democratización musical para todos

## Contacto

- **Proyecto**: Son1kVers3
- **Filosofía**: "Resistencia Sonora"
- **Objetivo**: 400M+ hispanohablantes con acceso a producción musical

---

**¡Que la música fluya libre!** 🎵