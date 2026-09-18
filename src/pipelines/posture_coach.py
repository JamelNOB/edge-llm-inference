"""
Edge-Motion-Coach: Real-Time Posture Analysis & Coaching Pipeline
================================================================
结合：
1. 骨骼力学角度计算 (PoseAngleCalculator)
2. RAG 国家级运动解剖学力学标准 (biomechanics_rules.json)
3. SQLite 用户时效伤病记忆 (AgentMemoryEngine)
4. 统一端侧量化大模型底座 (EdgeLLMEngine via llama.cpp / Adaptive Mock)
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.core.config import DEFAULT_RULES_PATH, DEFAULT_MEMORY_DB
from src.vision.pose_angle_calculator import PoseAngleCalculator
from src.memory.agent_memory import AgentMemoryEngine
from src.engine.llm_engine import get_engine, EdgeLLMEngine


class EdgePostureCoach:
    """
    端侧运动姿态质检与微纠错总装控制器
    统一接入 EdgeLLMEngine，常驻内存零 I/O 极速推理
    """
    def __init__(
        self,
        engine: Optional[EdgeLLMEngine] = None,
        model_path: Optional[str] = None,
        rules_path: Optional[str] = None,
        memory_db: Optional[str] = None
    ):
        self.engine = engine or get_engine()
        self.rules_path = str(rules_path or DEFAULT_RULES_PATH)
        self.memory_engine = AgentMemoryEngine(db_path=str(memory_db or DEFAULT_MEMORY_DB))
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if os.path.exists(self.rules_path):
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def setup_user_profile(
        self,
        user_id: str = "demo_user",
        name: str = "运动学员",
        injury: Optional[str] = None,
        ttl_seconds: int = 72 * 3600
    ):
        """设置用户画像与临时生理伤病档案"""
        self.memory_engine.set_memory(user_id, "user_name", name, category="profile", importance=5)
        if injury:
            self.memory_engine.set_memory(
                user_id, "current_injury", injury, category="status", importance=5, ttl_seconds=ttl_seconds
            )

    def analyze_and_coach(
        self,
        user_id: str = "demo_user",
        hip: Optional[Tuple[float, float]] = None,
        knee: Optional[Tuple[float, float]] = None,
        ankle: Optional[Tuple[float, float]] = None,
        shoulder: Optional[Tuple[float, float]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        action: str = "squat"
    ) -> Dict[str, Any]:
        """
        单帧/动作极值点端到端质检流程：
        支持直接传入解算好的 metrics（避免上游视频流假数据或重复计算），
        亦支持传入四点原始骨架坐标。
        """
        start_t = time.perf_counter()

        # 1. 骨骼几何角度计算 (< 0.5ms)
        if metrics is None:
            if hip is None or knee is None or ankle is None or shoulder is None:
                raise ValueError("必须提供 metrics 字典或完整的 (hip, knee, ankle, shoulder) 关键点坐标。")
            metrics = PoseAngleCalculator.analyze_squat_posture(hip, knee, ankle, shoulder)

        # 2. 检索有效用户记忆 (包含 TTL 惰性淘汰与伤病提取)
        memories = self.memory_engine.get_effective_memories(user_id)
        user_injury = None
        for m in memories:
            if m["key"] in ("current_injury", "injury_status"):
                user_injury = m["content"]
                break

        # 3. 检索 RAG 力学规范约束
        squat_rule = self.rules.get(action, {})
        thresh = squat_rule.get("biomechanical_thresholds", {})

        # 4. 组装 ChatML 结构化 Prompt
        valgus_str = "膝关节严重内扣代偿" if metrics.get("has_valgus") else "膝关节对准脚尖同向"
        injury_str = f"检测到既往伤病档案: {user_injury}" if user_injury else "无伤病记录"

        system_prompt = (
            "你是一个国家级专业运动力学康复教练。根据运动员当前的骨骼几何角度，"
            "结合运动解剖学标准与用户生理状态，用中文给出极其精炼、干脆利落的质检诊断与纠错口令。\n"
            "【力学判定规范】: 下蹲深度要求膝夹角<=90度；膝内扣偏移>20px必须报警；有伤病史必须采取保护策略。\n"
            "请严格按照此格式输出:\n"
            "【动作结论】: 达标 / 不合格 / 伤病降级\n"
            "【核心缺陷】: (一句话分析力学问题)\n"
            "【纠错口令】: (简短有力的动作指导)"
        )

        user_content = (
            f"目标动作: 深蹲 (Squat)\n"
            f"运动状态: {metrics.get('phase', 'UNKNOWN')}\n"
            f"膝关节屈曲角: {metrics.get('knee_angle', 0.0)}°\n"
            f"膝关节对齐状态: {valgus_str} (偏移量: {metrics.get('valgus_offset_px', 0.0)}px)\n"
            f"躯干前倾角: {metrics.get('torso_angle', 0.0)}°\n"
            f"{injury_str}"
        )

        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 5. 调用统一常驻内存底座 (EdgeLLMEngine) 执行推理
        llm_start = time.perf_counter()
        raw_output = self.engine.generate_text(
            prompt=full_prompt,
            max_tokens=96,
            temperature=0.3,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        llm_cost_ms = (time.perf_counter() - llm_start) * 1000

        # 6. 解析结构化输出
        verdict = "达标"
        core_defect = "无明显缺陷"
        coach_cue = "保持动作规范！"

        m_verdict = re.search(r"【动作结论】[\s:：]+([^\n\r]+)", raw_output)
        if m_verdict:
            verdict = m_verdict.group(1).strip()

        m_defect = re.search(r"【核心缺陷】[\s:：]+([^\n\r]+)", raw_output)
        if m_defect:
            core_defect = m_defect.group(1).strip()

        m_cue = re.search(r"【纠错口令】[\s:：]+([^\n\r]+)", raw_output)
        if m_cue:
            coach_cue = m_cue.group(1).strip()
        else:
            # 兜底截取短句
            lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
            if lines:
                coach_cue = lines[-1]

        total_cost_ms = (time.perf_counter() - start_t) * 1000

        return {
            "action": action,
            "metrics": metrics,
            "user_injury": user_injury,
            "verdict": verdict,
            "core_defect": core_defect,
            "coach_cue": coach_cue,
            "advice": f"【{verdict}】{coach_cue}",
            "raw_output": raw_output,
            "llm_cost_ms": round(llm_cost_ms, 2),
            "total_cost_ms": round(total_cost_ms, 2)
        }
