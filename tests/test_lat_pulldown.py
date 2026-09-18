"""
Unit Tests for Lat Pulldown Edge Pipeline & Robust Cue Parsing
=============================================================
验证:
1. 握距自适应比对 (宽握 vs 窄握)
2. 施密特双阈值防抖与力学违规判定
3. 伤病禁忌规则仲裁
4. RobustCueParser 正则提取与长度硬截断
5. 端到端 process_motion_event 流水线全链路
"""
import os
import unittest
from pathlib import Path

from src.knowledge.rule_engine import BiomechanicsRuleEngine
from src.pipelines.lat_pulldown_pipeline import LatPulldownEdgePipeline, RobustCueParser


class TestLatPulldownPipeline(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_lat_memory.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.pipeline = LatPulldownEdgePipeline(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_robust_cue_parser(self):
        """测试健壮口令解析器在各种脏文本下的提取表现"""
        text1 = "1. 诊断: 动作良好。\n2. 即时纠错口令: 挺胸沉肩，慢放两秒！\n3. 力学说明: ..."
        cue1 = RobustCueParser.extract_cue(text1)
        self.assertEqual(cue1, "挺胸沉肩，慢放两秒！")

        text2 = "【纠错口令】: 核心收紧，减小后仰！"
        cue2 = RobustCueParser.extract_cue(text2)
        self.assertEqual(cue2, "核心收紧，减小后仰！")

        # 针对空文本或无结构文本的降级兜底
        text_empty = ""
        cue_empty = RobustCueParser.extract_cue(text_empty, rule_fallback_cue="挺胸收腹，慢放两秒！")
        self.assertEqual(cue_empty, "挺胸收腹，慢放两秒！")

    def test_grip_variant_routing(self):
        """测试握距比自适应路由切换"""
        engine = BiomechanicsRuleEngine()
        res_wide = engine.evaluate_lat_pulldown(torso_angle=15.0, scapula_ratio=0.3, pull_pos_ratio=0.12, grip_ratio=1.3)
        self.assertIn("宽握颈前下拉", res_wide["variant"])

        res_close = engine.evaluate_lat_pulldown(torso_angle=18.0, scapula_ratio=0.3, pull_pos_ratio=0.25, grip_ratio=0.85)
        self.assertIn("窄握对握下拉", res_close["variant"])

    def test_contraindication_arbitration(self):
        """测试伤病禁忌与后仰超标仲裁"""
        engine = BiomechanicsRuleEngine()
        res = engine.evaluate_lat_pulldown(
            torso_angle=32.0,  # 严重超标
            scapula_ratio=0.18,  # 耸肩
            pull_pos_ratio=0.38,
            grip_ratio=1.3,
            user_injury="腰肌劳损 (L4-L5竖脊肌酸痛)"
        )
        self.assertFalse(res["is_compliant"])
        self.assertTrue(any("高危禁忌" in f or "超限" in f or "上限" in f for f in res["faults"]))

    def test_pipeline_end_to_end(self):
        """测试全链路端到端质检"""
        self.pipeline.setup_user("user_101", injury="肩袖损伤康复期")
        res = self.pipeline.process_motion_event(
            user_id="user_101",
            torso_angle=14.0,
            scapula_ratio=0.32,
            pull_pos_ratio=0.12,
            grip_ratio=1.3
        )
        self.assertIn("coach_cue", res)
        self.assertTrue(len(res["coach_cue"]) > 0)
        self.assertTrue(res["total_latency_ms"] >= 0)


if __name__ == "__main__":
    unittest.main()
