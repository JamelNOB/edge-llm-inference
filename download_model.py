"""
Edge LLM Inference Project - High-Speed Model Downloader
Prioritizes ModelScope CDN for max download speed
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_REPO_MODELSCOPE = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
TARGET_DIR = Path(__file__).parent / "models"
TARGET_FILE = TARGET_DIR / MODEL_FILENAME

def check_existing_model():
    if TARGET_FILE.exists():
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        if size_mb > 500:
            print(f"[+] Model already exists and complete: {TARGET_FILE} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"[!] Removing incomplete file ({size_mb:.2f} MB)...")
            TARGET_FILE.unlink()
    return False

def download_via_modelscope():
    print("[*] Connecting to ModelScope high-speed CDN...")
    try:
        from modelscope.hub.file_download import model_file_download
        downloaded_path = model_file_download(
            model_id=MODEL_REPO_MODELSCOPE,
            file_path=MODEL_FILENAME,
            local_dir=str(TARGET_DIR)
        )
        print(f"[+] Download successful: {downloaded_path}")
        return True
    except Exception as e:
        print(f"[-] ModelScope error: {e}")
        return False

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Edge LLM Model Downloader (ModelScope CDN)")
    print(f"Model: {MODEL_FILENAME} (~980 MB)")
    print(f"Directory: {TARGET_DIR.resolve()}")
    print("=" * 60)

    if check_existing_model():
        return

    success = download_via_modelscope()
    if success and TARGET_FILE.exists():
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 60)
        print(f"[+] Model downloaded successfully! Total size: {size_mb:.2f} MB")
        print("[+] Ready for inference. Next step: python app.py")
        print("=" * 60)
    else:
        print("\n[x] Download failed. Please check network connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()
