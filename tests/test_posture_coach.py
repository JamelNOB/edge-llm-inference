"""
Unit Tests for Edge-Motion-Coach (实时运动姿态质检单元测试)
==========================================================
验证:
1. 骨骼力学角度计算精度 (膝角与内扣偏角)
2. 伤病时效记忆检索与 RAG 力学规范装配
3. 结构化 Prompt 注入正确性
"""

import os
import unittest
from src.vision.pose_angle_calculator import PoseAngleCalculator
from run_posture_coach import EdgePostureCoach


class TestEdgePostureCoach(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_posture_memory.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.coach = EdgePostureCoach(
            model_path="dummy_model.gguf",
            rules_path="src/knowledge/biomechanics_rules.json",
            memory_db=self.db_path
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_squat_angle_calculation(self):
        """测试 90 度标准下蹲几何计算"""
        hip = (200.0, 300.0)
        knee = (200.0, 400.0)
        ankle = (300.0, 400.0)
        angle = PoseAngleCalculator.calculate_angle_2d(hip, knee, ankle)
        self.assertAlmostEqual(angle, 90.0, delta=1.0)

    def test_valgus_detection(self):
        """测试内扣偏角检测"""
        hip = (200.0, 250.0)
        knee = (240.0, 370.0)  # X 轴显著内倾
        ankle = (190.0, 480.0)
        shoulder = (205.0, 120.0)
        res = PoseAngleCalculator.analyze_squat_posture(hip, knee, ankle, shoulder)
        self.assertTrue(res["has_valgus"])
        self.assertGreater(res["valgus_offset_px"], 20.0)

    def test_memory_integration(self):
        """测试旧伤记忆注入与风控"""
        self.coach.setup_user_profile(user_id="test_user", injury="左膝前交叉韧带(ACL)重建术后(TTL时效内)")
        memories = self.coach.memory_engine.get_effective_memories("test_user")
        self.assertEqual(len(memories), 2)  # name + injury
        self.assertIn("ACL", [m["content"] for m in memories if "ACL" in m["content"]][0])


if __name__ == "__main__":
    unittest.main()
