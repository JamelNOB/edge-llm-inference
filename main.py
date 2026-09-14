"""
Edge LLM Inference Service - Main Entry Point
==============================================
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn
from src.core.config import SERVER_HOST, SERVER_PORT

def main():
    print("=" * 65)
    print("🚀 Edge LLM Inference Service - Starting Gateway...")
    print(f"📡 Serving on http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"📖 Interactive Docs: http://127.0.0.1:{SERVER_PORT}/docs")
    print("=" * 65)
    uvicorn.run("src.server.app:app", host=SERVER_HOST, port=SERVER_PORT, reload=False, workers=1)

if __name__ == "__main__":
    main()
