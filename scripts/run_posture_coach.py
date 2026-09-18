"""
Posture Coach CLI Runner
========================
运行端侧深蹲姿态质检测试（覆盖达标、内扣代偿与伤病降级多场景）
"""
import sys
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

from src.pipelines.posture_coach import EdgePostureCoach


def main():
    coach = EdgePostureCoach()
    coach.setup_user_profile(
        user_id="demo_user",
        name="学员",
        injury="右膝半月板轻微损伤 (TTL时效保护)"
    )

    print("=" * 70)
    print("🏋️ [Edge-Motion-Coach] 端侧深蹲姿态质检微纠错实测")
    print("=" * 70)

    # 场景 1: 严重膝内扣代偿
    print("\n>>> [测试场景 1: 膝内扣代偿 + 既往半月板旧伤]")
    res1 = coach.analyze_and_coach(
        user_id="demo_user",
        hip=(200, 300),
        knee=(240, 400),  # 内扣偏移
        ankle=(200, 500),
        shoulder=(200, 150),
        action="squat"
    )
    print(f"[*] 屈曲角度: {res1['metrics']['knee_angle']}° | 躯干前倾: {res1['metrics']['torso_angle']}°")
    print(f"[*] 动作结论: {res1['verdict']}")
    print(f"[*] 核心缺陷: {res1['core_defect']}")
    print(f"[*] 纠错口令: >>> {res1['coach_cue']}")
    print(f"[*] 推理时延: {res1['llm_cost_ms']} ms | 端到端总时延: {res1['total_cost_ms']} ms")

    # 场景 2: 动作规范达标
    print("\n>>> [测试场景 2: 标准深蹲下蹲深度达标]")
    # 构造膝夹角 ~90度，无内扣
    res2 = coach.analyze_and_coach(
        user_id="demo_user",
        hip=(180, 300),
        knee=(200, 400),
        ankle=(200, 500),
        shoulder=(180, 150),
        action="squat"
    )
    print(f"[*] 屈曲角度: {res2['metrics']['knee_angle']}° | 躯干前倾: {res2['metrics']['torso_angle']}°")
    print(f"[*] 动作结论: {res2['verdict']}")
    print(f"[*] 核心缺陷: {res2['core_defect']}")
    print(f"[*] 纠错口令: >>> {res2['coach_cue']}")
    print(f"[*] 推理时延: {res2['llm_cost_ms']} ms | 端到端总时延: {res2['total_cost_ms']} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
