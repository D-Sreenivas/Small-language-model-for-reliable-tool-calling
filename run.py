"""Unified Launcher Script for Backend and Frontend."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_all():
    print("=" * 60)
    print("Launching Small Language Model for Reliable Tool Calling")
    print("=" * 60)
    
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    frontend_cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"]
    
    print("Starting FastAPI Backend at http://127.0.0.1:8000...")
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT))
    time.sleep(2)
    
    print("Starting Streamlit Frontend at http://127.0.0.1:8501...")
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(ROOT))
    
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    run_all()

