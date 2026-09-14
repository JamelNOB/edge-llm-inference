"""
Model Downloader & Asset Provisioning Tool
==========================================
Supports ModelScope high-speed CDN and Hugging Face mirror endpoints.
"""
import os
import sys
from pathlib import Path

# 将项目根目录添加进 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config import MODELS_DIR, DEFAULT_MODEL_FILENAME, DEFAULT_MODEL_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_REPO_MODELSCOPE = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"

def check_existing_model(target_file: Path) -> bool:
    if target_file.exists():
        size_mb = target_file.stat().st_size / (1024 * 1024)
        if size_mb > 500:
            print(f"[+] Model already exists and complete: {target_file} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"[!] Removing incomplete file ({size_mb:.2f} MB)...")
            target_file.unlink()
    return False

def download_via_modelscope(target_dir: Path, filename: str) -> bool:
    print("[*] Connecting to ModelScope high-speed CDN...")
    try:
        from modelscope.hub.file_download import model_file_download
        downloaded_path = model_file_download(
            model_id=MODEL_REPO_MODELSCOPE,
            file_path=filename,
            local_dir=str(target_dir)
        )
        print(f"[+] Download successful: {downloaded_path}")
        return True
    except Exception as e:
        print(f"[-] ModelScope download failed: {e}")
        return False

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_file = DEFAULT_MODEL_PATH

    print("=" * 65)
    print("Edge LLM Model Downloader (ModelScope CDN)")
    print(f"Target Model: {DEFAULT_MODEL_FILENAME} (~980 MB)")
    print(f"Target Directory: {MODELS_DIR.resolve()}")
    print("=" * 65)

    if check_existing_model(target_file):
        print("[+] Model ready for inference.")
        return

    success = download_via_modelscope(MODELS_DIR, DEFAULT_MODEL_FILENAME)
    if success and target_file.exists():
        size_mb = target_file.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 65)
        print(f"[+] Model downloaded successfully! Total size: {size_mb:.2f} MB")
        print("[+] Ready for inference. Next step: python main.py")
        print("=" * 65)
    else:
        print("\n[x] Download failed. Please check your network connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()
