"""
Son1k API Smoke Tests - Extended with Maqueta → Production
Tests basic functionality including the new Ghost Studio features
"""

import pytest
import requests
import time
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from io import BytesIO

# Test configuration
API_BASE = "http://localhost:8000"
OUTPUT_DIR = Path("output")
UPLOADS_DIR = Path("uploads")

def generate_test_audio(duration_s=2.0, sample_rate=44100, frequency=440.0):
    """Generate a simple sine wave for testing"""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), False)
    # Create a simple melody with some variation
    audio = 0.3 * (np.sin(2 * np.pi * frequency * t) + 
                   0.3 * np.sin(2 * np.pi * frequency * 1.5 * t))
    # Add some envelope to make it more realistic
    envelope = np.exp(-t * 0.5)
    audio = audio * envelope
    return audio, sample_rate

def test_health():
    """Test health check endpoint"""
    response = requests.get(f"{API_BASE}/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data
    print("✅ Health check passed")

def test_generate_smoke():
    """Test music generation with minimal parameters"""
    payload = {
        "prompt": "simple piano melody",
        "duration": 2.0,  # Minimal duration for speed
        "temperature": 1.0,
        "top_k": 250
    }
    
    print("🎵 Starting generation test...")
    response = requests.post(
        f"{API_BASE}/api/v1/generate",
        json=payload,
        timeout=120  # Allow time for model loading
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert data["ok"] is True
    assert "url" in data
    assert "filename" in data
    assert "device" in data
    assert "model" in data
    assert data["prompt"] == payload["prompt"]
    
    # Check file exists
    filename = data["filename"]
    output_path = OUTPUT_DIR / filename
    assert output_path.exists(), f"Generated file {filename} not found"
    assert output_path.stat().st_size > 0, f"Generated file {filename} is empty"
    
    print(f"✅ Generation test passed: {filename} ({data['duration']:.1f}s on {data['device']})")

def test_ghost_maqueta_flow():
    """Test complete Maqueta → Production workflow"""
    print("🎤 Testing Maqueta → Production workflow...")
    
    # Step 1: Generate test audio file
    test_audio, sr = generate_test_audio(duration_s=3.0, frequency=440.0)
    
    # Save to temporary WAV file
    temp_audio_path = Path("/tmp/test_demo.wav")
    sf.write(temp_audio_path, test_audio, sr)
    
    try:
        # Step 2: Upload maqueta
        with open(temp_audio_path, 'rb') as audio_file:
            files = {'file': ('test_demo.wav', audio_file, 'audio/wav')}
            data = {
                'prompt': 'transform into uplifting electronic music',
                'duration': '5.0',
                'tune_amount': '0.5',
                'lufs_target': '-16.0'
            }
            
            print("📤 Uploading maqueta and processing...")
            response = requests.post(
                f"{API_BASE}/api/v1/ghost/maqueta",
                files=files,
                data=data,
                timeout=180  # Allow more time for processing
            )
        
        assert response.status_code == 200, f"Maqueta processing failed: {response.status_code}"
        
        result = response.json()
        
        # Step 3: Validate response structure
        assert result["ok"] is True
        assert "demo" in result
        assert "production" in result
        assert "prompt_final" in result
        assert "session_id" in result
        
        # Validate demo analysis
        demo = result["demo"]
        assert "url" in demo
        assert "analysis" in demo
        assert "file_info" in demo["analysis"]
        assert "tempo" in demo["analysis"]
        assert "key_guess" in demo["analysis"]
        
        # Validate production
        production = result["production"]
        assert "url" in production
        assert "post_metadata" in production
        assert "processing_chain" in production["post_metadata"]
        
        # Step 4: Verify files exist and are accessible
        demo_response = requests.get(f"{API_BASE}{demo['url']}")
        assert demo_response.status_code == 200
        assert len(demo_response.content) > 0
        
        production_response = requests.get(f"{API_BASE}{production['url']}")
        assert production_response.status_code == 200
        assert len(production_response.content) > 0
        
        # Step 5: Validate audio files can be read
        # Check that we can read the production file
        session_id = result["session_id"]
        production_path = OUTPUT_DIR / "ghost" / session_id / production["filename"]
        assert production_path.exists(), f"Production file {production_path} not found"
        audio_info = sf.info(production_path)
        assert audio_info.samplerate > 0
        assert audio_info.frames > 0
        print(f"✅ Production file validated: {audio_info.frames} frames @ {audio_info.samplerate}Hz")
        
        print(f"✅ Maqueta workflow completed successfully!")
        print(f"   📊 Analysis: {demo['analysis']['tempo']['bpm']:.1f} BPM, {demo['analysis']['key_guess']['root']}{demo['analysis']['key_guess']['scale']}")
        print(f"   🎵 Processing chain: {' → '.join(production['post_metadata']['processing_chain'])}")
        print(f"   ⏱️  Total time: {result['processing_time_s']:.1f}s")
        
        return result
        
    finally:
        # Cleanup
        temp_audio_path.unlink(missing_ok=True)

def test_ghost_presets_endpoint():
    """Test Ghost Studio presets endpoint"""
    response = requests.get(f"{API_BASE}/api/v1/ghost/presets")
    assert response.status_code == 200
    
    data = response.json()
    assert "presets" in data
    assert "count" in data
    assert data["count"] > 0
    
    # Check preset structure
    for preset_name, preset_data in data["presets"].items():
        assert "name" in preset_data
        assert "description" in preset_data
        assert "prompt_base" in preset_data
        
    print(f"✅ Presets endpoint test passed: {data['count']} presets available")

def test_ghost_stats_endpoint():
    """Test Ghost Studio statistics endpoint"""
    response = requests.get(f"{API_BASE}/api/v1/ghost/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "sessions" in data
    assert "storage" in data
    assert "presets" in data
    
    print(f"✅ Ghost stats endpoint test passed")

def test_models_endpoint():
    """Test models listing endpoint"""
    response = requests.get(f"{API_BASE}/api/v1/models")
    assert response.status_code == 200
    
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0
    
    # Check first model structure
    model = data["models"][0]
    assert "name" in model
    assert "description" in model
    assert "parameters" in model
    
    print(f"✅ Models endpoint test passed: {len(data['models'])} models available")

def test_static_files():
    """Test that static files are served correctly"""
    # This test assumes at least one file exists from previous tests
    if not any(OUTPUT_DIR.glob("*.wav")):
        pytest.skip("No generated audio files to test static serving")
    
    # Get the first wav file
    wav_file = next(OUTPUT_DIR.glob("*.wav"))
    filename = wav_file.name
    
    # Test static file serving
    response = requests.get(f"{API_BASE}/output/{filename}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")
    assert len(response.content) > 0
    
    print(f"✅ Static files test passed: {filename}")

def test_cache_clear():
    """Test cache clearing endpoint"""
    response = requests.delete(f"{API_BASE}/api/v1/cache")
    assert response.status_code == 200
    
    data = response.json()
    assert data["ok"] is True
    assert "message" in data
    
    print("✅ Cache clear test passed")

def test_audio_analysis_functions():
    """Test audio analysis functions directly"""
    try:
        from src.audio_analysis import analyze_demo
        
        # Generate test audio file
        test_audio, sr = generate_test_audio(duration_s=5.0, frequency=261.6)  # C4 note
        temp_path = Path("/tmp/test_analysis.wav")
        sf.write(temp_path, test_audio, sr)
        
        try:
            # Run analysis
            analysis = analyze_demo(str(temp_path))
            
            # Validate analysis structure
            assert "file_info" in analysis
            assert "tempo" in analysis
            assert "key_guess" in analysis
            assert "energy_structure" in analysis
            assert "vocals" in analysis
            
            # Basic sanity checks
            assert analysis["file_info"]["samplerate"] == sr
            assert analysis["file_info"]["duration_s"] > 0
            assert 60 <= analysis["tempo"]["bpm"] <= 200  # Reasonable tempo range
            
            print(f"✅ Audio analysis test passed: {analysis['tempo']['bpm']:.1f}bpm, {analysis['key_guess']['root']}{analysis['key_guess']['scale']}")
            
        finally:
            temp_path.unlink(missing_ok=True)
            
    except ImportError:
        pytest.skip("Audio analysis module not available")

def test_audio_postprocessing():
    """Test audio postprocessing functions"""
    try:
        from src.audio_post import process_master
        
        # Generate test audio
        test_audio, sr = generate_test_audio(duration_s=2.0, frequency=440.0)
        
        # Apply processing
        processed_audio, metadata = process_master(
            test_audio, sr,
            eq_params={"low_gain_db": 2.0, "high_gain_db": 1.0},
            tune_params={"enabled": False},
            sat_params={"drive_db": 3.0, "mix": 0.2},
            master_params={"lufs_target": -16.0, "ceiling_db": -0.5}
        )
        
        # Validate results
        assert len(processed_audio) == len(test_audio)
        assert "processing_chain" in metadata
        assert "lufs_gain_db" in metadata
        assert len(metadata["processing_chain"]) > 0
        
        # Check that processing was applied (audio should be different)
        assert not np.array_equal(test_audio, processed_audio)
        
        print(f"✅ Audio postprocessing test passed: {len(metadata['processing_chain'])} stages applied")
        
    except ImportError:
        pytest.skip("Audio postprocessing module not available")

# Test fixtures and utilities
@pytest.fixture(autouse=True)
def ensure_directories():
    """Ensure required directories exist before tests"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    (UPLOADS_DIR / "ghost").mkdir(exist_ok=True)
    (OUTPUT_DIR / "ghost").mkdir(exist_ok=True)

def test_full_workflow():
    """Integration test of the complete workflow including new features"""
    print("🧪 Running full workflow test (extended)...")
    
    # 1. Health check
    test_health()
    
    # 2. Manual generation
    test_generate_smoke()
    
    # 3. Audio analysis (if available)
    test_audio_analysis_functions()
    
    # 4. Audio postprocessing (if available)
    test_audio_postprocessing()
    
    # 5. Ghost Studio presets
    test_ghost_presets_endpoint()
    
    # 6. Maqueta → Production workflow
    test_ghost_maqueta_flow()
    
    # 7. Static files
    test_static_files()
    
    # 8. Statistics
    test_ghost_stats_endpoint()
    
    print("✅ Full extended workflow test completed successfully")

if __name__ == "__main__":
    # Run individual tests if called directly
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if hasattr(sys.modules[__name__], test_name):
            getattr(sys.modules[__name__], test_name)()
        else:
            print(f"Test {test_name} not found")
    else:
        # Run all tests
        test_full_workflow()