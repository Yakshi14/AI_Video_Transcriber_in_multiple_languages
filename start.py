#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    required_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn", 
        "yt-dlp": "yt_dlp",
        "faster-whisper": "faster_whisper",
        "openai": "openai"
    }
    
    missing_packages = []
    for display_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(display_name)
    
    if missing_packages:
        print("❌ Missing the following dependency packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install dependencies using:")
        print("source venv/bin/activate && pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        print("🟩 FFmpeg is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found on this system.")
        print("Please install FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return False

def setup_environment():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it using: export OPENAI_API_KEY=your_api_key_here")
        return False
    
    print("✅ OPENAI_API_KEY is set")
    
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://oneapi.basevec.com/v1"
        print("✅ Default OpenAI Base URL applied")
    
    if not os.getenv("WHISPER_MODEL_SIZE"):
        os.environ["WHISPER_MODEL_SIZE"] = "base"
    
    print("🔑 OpenAI API configuration completed — summary features enabled")
    return True

def main():
    production_mode = "--prod" in sys.argv or os.getenv("PRODUCTION_MODE") == "true"
    
    print("🚀 AI Video Transcriber — Startup Check")
    if production_mode:
        print("🔒 Production Mode — Hot Reload Disabled")
    else:
        print("🔧 Development Mode — Hot Reload Enabled")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check FFmpeg
    if not check_ffmpeg():
        print("⚠️  FFmpeg is not installed. Some video formats may not be processed correctly.")
    
    # Setup environment variables
    setup_environment()
    
    print("\n🎉 Startup check completed!")
    print("=" * 50)
    
    # Start server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print("\n🌐 Starting server...")
    print(f"   URL: http://localhost:{port}")
    print(f"   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", host,
            "--port", str(port)
        ]
        
        # Enable hot reload only in development mode
        if not production_mode:
            cmd.append("--reload")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
