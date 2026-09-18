"""
Video Coach CLI Runner
======================
运行端侧视频流双频异步姿态质检流水线并生成带 HUD 标注的质检视频
"""
import sys
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.vision.synthetic_motion_generator import SyntheticSquatGenerator
from src.pipelines.video_coach import VideoPostureCoachPipeline


def main():
    parser = argparse.ArgumentParser(description="端侧视频流双频异步姿态质检流水线")
    parser.add_argument("--repeats", type=int, default=1, help="生成的合成深蹲动作循环次数")
    parser.add_argument("--output", type=str, default="output_coached_squat.mp4", help="导出的标注视频路径")
    parser.add_argument("--injury", type=str, default="右膝半月板轻微损伤 (TTL有效)", help="用户时效伤病设定")
    args = parser.parse_args()

    generator = SyntheticSquatGenerator()
    pipeline = VideoPostureCoachPipeline(user_id="demo_user", user_injury=args.injury)

    try:
        res = pipeline.process_stream(generator=generator, output_path=args.output)
        print(f"\n[SUCCESS] 演示视频成功渲染生成: {res['output_path']}")
        print(f"[*] 最终字幕: {res['latest_advice']}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
