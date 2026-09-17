"""
Unit Tests for Video Posture Pipeline
======================================
测试:
1. 有限状态机 (SquatStateMachine) 相位迁移与拐点检测
2. 合成动作发生器 (SyntheticSquatGenerator) 数据流
3. HUD 渲染器与视频生成管道端到端闭环
"""

import os
import unittest
import numpy as np

from src.vision.motion_analyzer import SquatStateMachine, MotionPhase
from src.vision.synthetic_motion_generator import SyntheticSquatGenerator
from src.vision.video_annotator import VideoPostureAnnotator
from run_video_coach import VideoPostureCoachPipeline


class TestVideoPosturePipeline(unittest.TestCase):

    def test_state_machine_cycle(self):
        """测试深蹲下蹲、触底、回弹上升与计数完整闭环"""
        sm = SquatStateMachine()
        self.assertEqual(sm.current_phase, MotionPhase.STANDING)

        # 模拟下蹲过程
        angles = [160.0, 140.0, 110.0, 90.0, 80.0, 85.0, 120.0, 155.0]
        phases = []
        triggered = []

        for a in angles:
            phase, trig, reason = sm.update({
                "knee_angle": a,
                "has_valgus": False
            })
            phases.append(phase)
            if trig:
                triggered.append(reason)

        # 验证触底拐点是否被精准捕捉并触发 LLM
        self.assertIn("bottom_peak", triggered)
        self.assertEqual(sm.rep_count, 1)
        self.assertEqual(sm.current_phase, MotionPhase.STANDING)

    def test_valgus_debounce_trigger(self):
        """测试膝内扣连续 4 帧防抖触发"""
        sm = SquatStateMachine(valgus_debounce_frames=3)
        trig_reasons = []

        for _ in range(4):
            _, trig, reason = sm.update({
                "knee_angle": 130.0,
                "has_valgus": True
            })
            if trig:
                trig_reasons.append(reason)

        self.assertIn("severe_valgus_warning", trig_reasons)

    def test_synthetic_generator(self):
        """测试合成运动流生成规格"""
        gen = SyntheticSquatGenerator(width=320, height=320, fps=10)
        # 只跑前 5 帧验证数据契约
        for idx, frame, lm in gen.generate_stream():
            self.assertEqual(frame.shape, (320, 320, 3))
            self.assertIn("knee", lm)
            self.assertIn("hip", lm)
            if idx >= 5:
                break

    def test_annotator_rendering(self):
        """测试 HUD 渲染无报错"""
        annotator = VideoPostureAnnotator()
        dummy_frame = np.zeros((480, 480, 3), dtype=np.uint8)

        # 骨骼渲染
        dummy_frame = annotator.draw_skeleton(
            dummy_frame,
            shoulder=(200, 100),
            hip=(200, 250),
            knee=(210, 380),
            ankle=(200, 470),
            has_valgus=False
        )
        self.assertEqual(dummy_frame.shape, (480, 480, 3))

        # 仪表盘渲染
        dummy_frame = annotator.draw_hud(
            dummy_frame,
            metrics={"knee_angle": 88.0, "torso_angle": 42.0, "valgus_offset_px": 5.0, "has_valgus": False},
            phase_str="触底拐点",
            rep_count=1
        )
        self.assertEqual(dummy_frame.shape, (480, 480, 3))


if __name__ == "__main__":
    unittest.main()
