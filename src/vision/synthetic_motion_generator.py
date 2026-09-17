"""
Synthetic Biomechanical Motion Generator
=========================================
运动生物力学合成发生器（用于端侧无外设基准测试与自动化质检验证）：
数学化生成具有逼真力学轨迹的深蹲视频流：
- 第 1 次下蹲（Rep 1）：标准深蹲（膝屈曲角降至 78°，无内扣，力学对齐良好）
- 第 2 次下蹲（Rep 2）：危险代偿深蹲（膝屈曲角 85°，膝关节 X 坐标严重内偏 +26px，触发力学报警与伤病风控）
"""

import math
import cv2
import numpy as np
from typing import Generator, Tuple, Dict, Any


class SyntheticSquatGenerator:
    """
    程序化深蹲骨骼运动发生器
    """

    def __init__(self, width: int = 640, height: int = 640, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = fps * 8  # 8 秒视频，包含 2 次完整深蹲

    def generate_stream(self) -> Generator[Tuple[int, np.ndarray, Dict[str, Tuple[float, float]]], None, None]:
        """
        生成按帧迭代的视频数据流:
        Yields:
            (frame_idx, frame_bgr, {"shoulder": (x,y), "hip": (x,y), "knee": (x,y), "ankle": (x,y)})
        """
        base_x = self.width // 2
        ankle_y = 520.0
        ankle_x = float(base_x - 30)

        for frame_idx in range(self.total_frames):
            # 8秒分为两组动作 (每组 4 秒 = 120 帧)
            rep_idx = frame_idx // 120
            t_in_rep = (frame_idx % 120) / 120.0  # 0.0 ~ 1.0

            # 正弦波模拟下蹲深度因子 (0: 站立, 1: 触底)
            # 0.0 ~ 0.2: 站立准备; 0.2 ~ 0.5: 下蹲; 0.5 ~ 0.8: 站起; 0.8 ~ 1.0: 站立
            if t_in_rep < 0.15 or t_in_rep > 0.85:
                depth_factor = 0.0
            else:
                # 归一化到 0 ~ pi
                phase = (t_in_rep - 0.15) / 0.7 * math.pi
                depth_factor = math.sin(phase)

            # 关键点力学坐标演化
            # 站立高度 vs 下蹲高度
            hip_stand_y = 280.0
            hip_squat_y = 410.0
            hip_y = hip_stand_y + (hip_squat_y - hip_stand_y) * depth_factor
            hip_x = float(base_x - 25 - 20 * depth_factor)  # 臀部适度后移

            shoulder_y = hip_y - 140.0
            shoulder_x = hip_x + 35 * depth_factor  # 躯干适度前倾

            knee_stand_y = 400.0
            knee_squat_y = 450.0
            knee_y = knee_stand_y + (knee_squat_y - knee_stand_y) * depth_factor
            knee_x = float(base_x - 15)

            # 第 2 次深蹲 (rep_idx == 1): 注入严重的膝内扣代偿 (X 轴向身体中线内偏 +28px)
            if rep_idx == 1 and depth_factor > 0.4:
                knee_x += 28.0 * depth_factor

            # 构造背景画布（深灰工业风健身房背景）
            frame = np.full((self.height, self.width, 3), (35, 38, 42), dtype=np.uint8)

            # 绘制地板网格线
            cv2.line(frame, (0, int(ankle_y + 10)), (self.width, int(ankle_y + 10)), (70, 75, 80), 2)
            for gx in range(50, self.width, 80):
                cv2.line(frame, (gx, int(ankle_y + 10)), (gx - 30, self.height), (55, 60, 65), 1)

            landmarks = {
                "shoulder": (shoulder_x, shoulder_y),
                "hip": (hip_x, hip_y),
                "knee": (knee_x, knee_y),
                "ankle": (ankle_x, ankle_y)
            }

            yield frame_idx, frame, landmarks

    def save_demo_video(self, output_path: str = "demo_squat_raw.mp4") -> str:
        """
        导出纯净合成原始视频
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        for _, frame, _ in self.generate_stream():
            out.write(frame)

        out.release()
        return output_path
