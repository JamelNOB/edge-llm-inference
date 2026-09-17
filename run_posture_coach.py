"""
Edge-Motion-Coach: Real-Time Posture Analysis & Coaching Pipeline
================================================================
结合：
1. 骨骼力学角度计算 (PoseAngleCalculator)
2. RAG 国家级运动解剖学力学标准 (biomechanics_rules.json)
3. SQLite 用户时效伤病记忆 (AgentMemoryEngine)
4. 934MB 端侧量化大模型 (Qwen2.5 Q4_K_M via llama.cpp)
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# 导入关键模块
try:
    from pose_angle_calculator import PoseAngleCalculator
except ImportError:
    from src.vision.pose_angle_calculator import PoseAngleCalculator

try:
    from agent_memory import AgentMemoryEngine
except ImportError:
    from src.memory.agent_memory import AgentMemoryEngine


class EdgePostureCoach:
    def __init__(
        self,
        model_path: str,
        rules_path: str = "biomechanics_rules.json",
        memory_db: str = "posture_user_memory.db",
        llama_bin: Optional[str] = None
    ):
        self.model_path = model_path
        self.rules_path = rules_path
        self.memory_engine = AgentMemoryEngine(db_path=memory_db)
        self.llama_bin = llama_bin or self._find_llama_bin()
        self.rules = self._load_rules()

    def _find_llama_bin(self) -> str:
        candidates = [
            str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-simple"),
            str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli"),
            "llama-simple",
            "llama-cli"
        ]
        for c in candidates:
            if os.path.exists(c) and os.access(c, os.X_OK):
                return c
        return "llama-simple"

    def _load_rules(self) -> Dict[str, Any]:
        if os.path.exists(self.rules_path):
            with open(self.rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def setup_user_profile(self, user_id: str = "liubo", injury: Optional[str] = None, ttl_seconds: int = 72*3600):
        """设置用户画像与临时生理伤病档案"""
        self.memory_engine.set_memory(user_id, "user_name", "刘博", category="profile", importance=5)
        if injury:
            self.memory_engine.set_memory(user_id, "current_injury", injury, category="status", importance=5, ttl_seconds=ttl_seconds)

    def analyze_and_coach(
        self,
        user_id: str,
        hip: tuple,
        knee: tuple,
        ankle: tuple,
        shoulder: tuple,
        action: str = "squat"
    ) -> Dict[str, Any]:
        """
        单帧/动作关键点端到端质检流程
        """
        start_t = time.perf_counter()

        # 1. 骨骼几何角度计算 (< 1ms)
        metrics = PoseAngleCalculator.analyze_squat_posture(hip, knee, ankle, shoulder)

        # 2. 检索有效用户记忆 (包含 TTL 惰性淘汰与伤病提取)
        memories = self.memory_engine.get_effective_memories(user_id)
        user_injury = None
        for m in memories:
            if m["key"] == "current_injury":
                user_injury = m["content"]
                break

        # 3. 检索 RAG 力学规范约束
        squat_rule = self.rules.get(action, {})
        thresh = squat_rule.get("biomechanical_thresholds", {})
        contra = squat_rule.get("injury_contraindications", {})

        # 4. 组装 ChatML 结构化 Prompt
        valgus_str = "膝关节严重内扣代偿" if metrics["has_valgus"] else "膝关节对准脚尖同向"
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
            f"运动状态: {metrics['phase']}\n"
            f"膝关节屈曲角: {metrics['knee_angle']}°\n"
            f"膝关节对齐状态: {valgus_str} (偏移量: {metrics['valgus_offset_px']}px)\n"
            f"躯干前倾角: {metrics['torso_angle']}°\n"
            f"{injury_str}"
        )

        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 5. 调用端侧 934MB 模型 (llama-simple) 执行推理
        cmd = [
            self.llama_bin,
            "-m", self.model_path,
            "-n", "96",
            full_prompt
        ]

        llm_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        llm_cost_ms = (time.perf_counter() - llm_start) * 1000

        output_text = proc.stdout
        # 提取 assistant 后的文本
        if "<|im_start|>assistant" in output_text:
            output_text = output_text.split("<|im_start|>assistant")[-1]
        if "main: decoded" in output_text:
            output_text = output_text.split("main: decoded")[0]

        total_cost_ms = (time.perf_counter() - start_t) * 1000

        return {
            "metrics": metrics,
            "user_injury": user_injury,
            "advice": output_text.strip(),
            "llm_latency_ms": round(llm_cost_ms, 2),
            "total_latency_ms": round(total_cost_ms, 2)
        }


if __name__ == "__main__":
    # 本地演示脚本
    model = sys.argv[1] if len(sys.argv) > 1 else "/home/liubo/llm_practice/models/qwen_lora_merged.Q4_K_M.gguf"
    coach = EdgePostureCoach(model_path=model)
    coach.setup_user_profile("liubo", injury="右膝半月板轻微损伤 (TTL: 72小时有效)")

    print("======================================================================")
    print("🏋️ Edge-Motion-Coach 端侧实时运动姿态质检系统 (深蹲测试)")
    print("======================================================================")

    # 模拟场景 1: 严重半蹲不足 + 膝内扣
    hip = (200.0, 250.0)
    knee = (235.0, 370.0)  # 内扣
    ankle = (195.0, 480.0)
    shoulder = (210.0, 120.0)

    print("\n[帧 1: 下蹲底峰状态监测...]")
    res = coach.analyze_and_coach("liubo", hip, knee, ankle, shoulder)
    print(f"📊 骨骼几何角度: 膝夹角={res['metrics']['knee_angle']}°, 躯干角={res['metrics']['torso_angle']}°, 内扣={res['metrics']['has_valgus']}")
    print(f"🧠 用户伤病状态: {res['user_injury']}")
    print(f"⚡ 端侧 LLM 推理耗时: {res['llm_latency_ms']} ms")
    print(f"🎙️ 教练实时纠错建议:\n{res['advice']}")
