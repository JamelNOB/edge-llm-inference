"""
Lat Pulldown Edge Pipeline CLI Runner
=====================================
一键运行端侧高位下拉全链路质检评测（包含标准动作与代偿动作双场景验证）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.pipelines.lat_pulldown_pipeline import LatPulldownEdgePipeline


def main():
    pipeline = LatPulldownEdgePipeline()

    # 初始化测试学员档案（含 TTL 时效伤病记忆）
    pipeline.setup_user(
        user_id="demo_user",
        injury="腰肌轻微劳损 (L4-L5竖脊肌酸痛, TTL: 72h)",
        posture_baseline="轻度右侧耸肩习惯"
    )

    print("=" * 70)
    print("🏋️ [Edge-Motion-Coach] 高位下拉端侧全链路整合 Pipeline 实测")
    print("=" * 70)

    # 场景 1: 标准帧 (后仰 14°, 肩胛下沉充分 0.31, 触及上胸 0.12, 宽握 1.35)
    print("\n>>> [测试场景 1: 标准动作实测]")
    res1 = pipeline.process_motion_event(
        user_id="demo_user",
        torso_angle=14.0,
        scapula_ratio=0.31,
        pull_pos_ratio=0.12,
        grip_ratio=1.35,
        peak_frame=101,
        peak_time=3.37
    )
    print(f"[*] 极值定位: 帧 {res1['peak_frame']} ({res1['peak_time_s']}s) | 握法: {res1['grip_variant']}")
    print(f"[*] 用户记忆: {res1['user_status']}")
    print(f"[*] 规则裁决: {'[合格] 动作规范' if res1['is_compliant'] else '[代偿] 存在违规'}")
    print(f"[*] 契约化验单: {res1['payload']}")
    print(f"[*] 推理时延: {res1['llm_latency_ms']} ms | 全链路总时延: {res1['total_latency_ms']} ms")
    print(f"[*] 穿透短口令 (8~12字): >>> {res1['coach_cue']}")

    # 场景 2: 严重代偿超标 (后仰 32°, 严重耸肩 0.18, 触及剑突 0.38)
    print("\n>>> [测试场景 2: 违规后仰代偿 + 腰肌劳损禁忌触发]")
    res2 = pipeline.process_motion_event(
        user_id="demo_user",
        torso_angle=32.0,
        scapula_ratio=0.18,
        pull_pos_ratio=0.38,
        grip_ratio=1.35,
        peak_frame=264,
        peak_time=8.80
    )
    print(f"[*] 极值定位: 帧 {res2['peak_frame']} ({res2['peak_time_s']}s) | 握法: {res2['grip_variant']}")
    print(f"[*] 用户记忆: {res2['user_status']}")
    print(f"[*] 规则裁决: {'[合格] 动作规范' if res2['is_compliant'] else '[代偿] 存在违规'}")
    print(f"[!] 违规明细: {res2['rule_faults']}")
    print(f"[*] 契约化验单: {res2['payload']}")
    print(f"[*] 推理时延: {res2['llm_latency_ms']} ms | 全链路总时延: {res2['total_latency_ms']} ms")
    print(f"[*] 穿透短口令 (8~12字): >>> {res2['coach_cue']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
