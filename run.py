import sys
import uvicorn
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import SAMPLES_DIR

def ensure_sample_videos():
    sample_files = list(SAMPLES_DIR.glob("*.mp4"))
    if not sample_files:
        print("[INFO] Generating synthetic CCTV surveillance test clips...")
        from app.scripts.generate_samples import generate_all_samples
        generate_all_samples()
        print("[INFO] Sample clips generated successfully.")

def main():
    print(r"""
========================================================================
    ___    ______ ____  ____ _____       _    __ _____ ____ _____ _   __
   /   |  / ____// __ \/  _// ___/      | |  / //  _// ___//  _/ / | / /
  / /| | / __/  / /_/ // /  \__ \ ______| | / / / /  \__ \ / /  /  |/ / 
 / ___ |/ /___ / _, _// /  ___/ //_____/| |/ /_/ /  ___/ // /  / /|  /  
/_/  |_/_____//_/ |_/___/ /____/        |___//___/ /____/___/ /_/ |_/   

  AI CRIME & SURVEILLANCE ANOMALY DETECTION PLATFORM
  Model: OPear/videomae-large-finetuned-UCF-Crime (323M Parameters)
========================================================================
[+] Server starting on: http://127.0.0.1:8000
[+] Web UI Dashboard:   http://127.0.0.1:8000
========================================================================
    """)
    ensure_sample_videos()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
