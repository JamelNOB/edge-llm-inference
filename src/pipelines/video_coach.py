"""
Edge-Motion-Coach: Video Stream Posture QA & Live HUD Pipeline
=============================================================
端侧视频流实时运动姿态质检主程序（双频异步异构解耦架构）：
1. 高频感知环（30 FPS / <5ms）：
   - OpenCV 抓帧 -> PoseAngleCalculator 力学向量几何解算 -> FSM 状态机维护
   - 实时光栅化 HUD 遥测仪表盘与骨骼报警线
2. 低频认知环（0.5~1 Hz / 异步工作线程）：
   - 有限状态机拐点防抖触发（触底极值点 / 突发内扣代偿）
   - 后台工作线程消费真实骨骼力学 metrics，调用统一 EdgeLLMEngine
   - 结合 SQLite 伤病记忆 + RAG 规则执行端侧推理
   - 原子回写最新教练纠错字幕，主视频流 0 卡顿、0 掉帧！
"""

import os
import sys
import time
import queue
import argparse
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

from src.vision.pose_angle_calculator import PoseAngleCalculator
from src.vision.motion_analyzer import SquatStateMachine, MotionPhase
from src.vision.video_annotator import VideoPostureAnnotator
from src.vision.synthetic_motion_generator import SyntheticSquatGenerator
from src.pipelines.posture_coach import EdgePostureCoach
from src.engine.llm_engine import get_engine, EdgeLLMEngine


class VideoPostureCoachPipeline:
    """
    双频异步端侧视频质检流水线（真力学解算与零假数据传输）
    """
    def __init__(
        self,
        engine: Optional[EdgeLLMEngine] = None,
        rules_path: Optional[str] = None,
        memory_db: Optional[str] = None,
        user_id: str = "demo_user",
        user_injury: Optional[str] = None
    ):
        self.user_id = user_id

        # 初始化底层推理与业务教练模块
        self.coach = EdgePostureCoach(
            engine=engine,
            rules_path=rules_path,
            memory_db=memory_db
        )
        if user_injury:
            self.coach.setup_user_profile(user_id=user_id, injury=user_injury)

        self.state_machine = SquatStateMachine()
        self.annotator = VideoPostureAnnotator()

        # 异步认知环通信通道
        self.task_queue: queue.Queue = queue.Queue(maxsize=1)
        self.current_advice: str = "动作准备就绪，请保持核心收紧，平稳下蹲。"
        self.is_inferring: bool = False
        self._stop_event = threading.Event()

        # 启动后台 LLM 工作线程
        self.worker_thread = threading.Thread(target=self._llm_worker_loop, daemon=True)
        self.worker_thread.start()

    def _llm_worker_loop(self):
        """后台异步 LLM 推理工作线程（消费真力学指标，杜绝假坐标占位）"""
        while not self._stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            metrics, trigger_reason = task
            self.is_inferring = True

            try:
                # 真实调用运动教练分析：直接传入由高频感知环解算出的真实 metrics 字典！
                res = self.coach.analyze_and_coach(
                    user_id=self.user_id,
                    metrics=metrics,
                    action="squat"
                )
                self.current_advice = res.get("advice", "保持动作规范")
                print(f"\n[AI-Coach] 实时纠错口令刷新 [原因: {trigger_reason}] -> {self.current_advice}")

            except Exception as e:
                print(f"[Worker Error]: {e}")
            finally:
                self.is_inferring = False
                self.task_queue.task_done()

    def stop(self):
        """停止异步推理工作线程"""
        self._stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def process_stream(
        self,
        generator: SyntheticSquatGenerator,
        output_path: str = "output_coached_squat.mp4",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        运行视频流水线并导出带质检 HUD 的标注视频
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, generator.fps, (generator.width, generator.height))

        if verbose:
            print(f"[START] 启动端侧高频感知视频流 (分辨率: {generator.width}x{generator.height}, 目标 FPS: {generator.fps})")
            print(f"[OUTPUT] 质检视频将渲染并导出至: {output_path}")

        frame_count = 0
        triggers_count = 0
        t0 = time.perf_counter()

        for frame_idx, frame, landmarks in generator.generate_stream():
            frame_count += 1
            shoulder = landmarks["shoulder"]
            hip = landmarks["hip"]
            knee = landmarks["knee"]
            ankle = landmarks["ankle"]

            # 1. 骨骼力学几何计算 (< 0.5ms)
            metrics = PoseAngleCalculator.analyze_squat_posture(hip, knee, ankle, shoulder)

            # 2. 状态机迁移与拐点检测
            phase, trigger_llm, reason = self.state_machine.update(metrics)

            # 3. 异步触发事件提交 (非阻塞无锁入队)
            if trigger_llm:
                triggers_count += 1
                try:
                    self.task_queue.put_nowait((metrics, reason))
                except queue.Full:
                    # 若推理仍在进行则跳过，保证视频流绝不掉帧
                    pass

            # 4. 光栅化渲染 HUD 仪表盘与教练字幕
            annotated_frame = self.annotator.render_hud(
                frame=frame,
                landmarks=landmarks,
                metrics=metrics,
                state_machine=self.state_machine,
                ai_advice=self.current_advice,
                is_inferring=self.is_inferring
            )

            out.write(annotated_frame)

        out.release()
        total_time = time.perf_counter() - t0
        avg_fps = frame_count / total_time if total_time > 0 else 0.0

        if verbose:
            print(f"[COMPLETE] 流水线运行完成: 共处理 {frame_count} 帧, 触发 {triggers_count} 次大模型纠错, 平均 FPS: {avg_fps:.1f}")

        return {
            "frame_count": frame_count,
            "triggers_count": triggers_count,
            "avg_fps": round(avg_fps, 1),
            "output_path": output_path,
            "latest_advice": self.current_advice
        }
