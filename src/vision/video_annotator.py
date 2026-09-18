"""
Video Posture Visualizer & HUD Overlay Renderer
================================================
负责在视频帧上渲染工业级运动质检抬头显示（HUD）：
1. 骨骼拓扑线渲染（正常呈现明绿/青蓝，力学代偿或内扣呈现警示鲜红）
2. 实时力学遥测仪表盘（半透明磨砂黑底，展示膝角、躯干角、内扣偏移量与动作阶段）
3. 底部 AI 教练纠错口令条（支持中文字符集渲染，与端侧大模型异步输出对齐）
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


class VideoPostureAnnotator:
    """
    OpenCV + PIL 混合高精度运动姿态 HUD 渲染器
    """

    def __init__(self):
        self.font_path = self._find_chinese_font()

    def _find_chinese_font(self) -> Optional[str]:
        candidates = [
            # Windows 常见中文字体
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            # Linux 常见中文字体
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def draw_text_cn(
        self,
        img_bgr: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
        font_size: int = 18
    ) -> np.ndarray:
        """
        在 OpenCV 图像上渲染高质量中文文本
        """
        if not self.font_path:
            # 降级使用 OpenCV 原生英文渲染
            cv2.putText(img_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, text_color, 1, cv2.LINE_AA)
            return img_bgr

        try:
            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(self.font_path, font_size)
            draw.text(pos, text, font=font, fill=(text_color[2], text_color[1], text_color[0]))
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(img_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, text_color, 1, cv2.LINE_AA)
            return img_bgr

    def draw_skeleton(
        self,
        frame: np.ndarray,
        shoulder: Tuple[float, float],
        hip: Tuple[float, float],
        knee: Tuple[float, float],
        ankle: Tuple[float, float],
        has_valgus: bool = False
    ) -> np.ndarray:
        """
        绘制运动链关键骨骼段与关节关节点
        """
        color_trunk = (0, 230, 255)       # 躯干：明黄
        color_thigh = (0, 255, 120)       # 大腿：正常绿
        color_shank = (0, 255, 120)       # 小腿：正常绿

        # 若内扣严重，小腿与膝关节变红警示
        if has_valgus:
            color_shank = (0, 50, 255)
            color_thigh = (0, 100, 255)

        pts = {
            "shoulder": (int(shoulder[0]), int(shoulder[1])),
            "hip": (int(hip[0]), int(hip[1])),
            "knee": (int(knee[0]), int(knee[1])),
            "ankle": (int(ankle[0]), int(ankle[1]))
        }

        # 骨骼连线
        cv2.line(frame, pts["shoulder"], pts["hip"], color_trunk, 4, cv2.LINE_AA)
        cv2.line(frame, pts["hip"], pts["knee"], color_thigh, 4, cv2.LINE_AA)
        cv2.line(frame, pts["knee"], pts["ankle"], color_shank, 4, cv2.LINE_AA)

        # 关节节点光晕
        for name, p in pts.items():
            cv2.circle(frame, p, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 9, (0, 0, 0), 2, cv2.LINE_AA)

        # 膝关节特殊警示标记
        if has_valgus:
            cv2.circle(frame, pts["knee"], 15, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        metrics: Dict[str, Any],
        phase_str: str,
        rep_count: int,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        绘制左上角半透明遥测仪表盘
        """
        h, w = frame.shape[:2]
        panel_w, panel_h = 320, 160
        x1, y1 = 20, 20
        x2, y2 = x1 + panel_w, y1 + panel_h

        # 半透明蒙层
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 180, 255), 1, cv2.LINE_AA)

        # 遥测文本
        knee_deg = metrics.get("knee_angle", 0.0)
        torso_deg = metrics.get("torso_angle", 0.0)
        valgus_px = metrics.get("valgus_offset_px", 0.0)
        has_valgus = metrics.get("has_valgus", False)

        valgus_tag = "【内扣警告!】" if has_valgus else "【力线对齐】"
        valgus_color = (0, 50, 255) if has_valgus else (0, 255, 120)

        frame = self.draw_text_cn(frame, f"⚙️ 端侧姿态力学感知 HUD ({fps:.1f} FPS)", (x1 + 10, y1 + 10), (0, 220, 255), 14)
        frame = self.draw_text_cn(frame, f"运动相位: {phase_str}", (x1 + 10, y1 + 35), (255, 255, 255), 15)
        frame = self.draw_text_cn(frame, f"完成次数: 第 {rep_count} 次", (x1 + 10, y1 + 60), (255, 215, 0), 15)
        frame = self.draw_text_cn(frame, f"膝关节屈角: {knee_deg:.1f}° (标准 <= 90°)", (x1 + 10, y1 + 85), (220, 220, 220), 14)
        frame = self.draw_text_cn(frame, f"躯干倾角: {torso_deg:.1f}° | 膝偏量: {valgus_px:+.1f}px", (x1 + 10, y1 + 110), (200, 200, 200), 13)
        frame = self.draw_text_cn(frame, f"力学质检: {valgus_tag}", (x1 + 10, y1 + 132), valgus_color, 14)

        return frame

    def draw_coach_banner(
        self,
        frame: np.ndarray,
        coach_advice: str,
        is_inferring: bool = False
    ) -> np.ndarray:
        """
        绘制底部 AI 教练口令条
        """
        h, w = frame.shape[:2]
        banner_h = 75
        y1 = h - banner_h - 15
        y2 = h - 15
        x1, x2 = 20, w - 20

        # 半透明磨砂底板
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # 呼吸边框
        border_color = (0, 165, 255) if is_inferring else (0, 255, 120)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)

        # 标题与口令
        status_tag = "🎙️ AI 动作质检教练 (Qwen2.5-1.5B Q4_K_M · 934MB):"
        if is_inferring:
            status_tag += " [🧠 正在研判姿态力学...]"

        frame = self.draw_text_cn(frame, status_tag, (x1 + 15, y1 + 8), (0, 200, 255), 14)

        display_text = coach_advice if coach_advice else "动作进行中，请保持核心收紧、呼吸均匀。"
        # 防止超长折行截断
        if len(display_text) > 46:
            line1 = display_text[:46]
            line2 = display_text[46:90]
            frame = self.draw_text_cn(frame, line1, (x1 + 15, y1 + 30), (255, 255, 255), 15)
            frame = self.draw_text_cn(frame, line2, (x1 + 15, y1 + 52), (255, 255, 255), 14)
        else:
            frame = self.draw_text_cn(frame, display_text, (x1 + 15, y1 + 36), (255, 255, 255), 16)

        return frame

    def render_hud(
        self,
        frame: np.ndarray,
        landmarks: Dict[str, Tuple[float, float]],
        metrics: Dict[str, Any],
        state_machine: Any,
        ai_advice: str,
        is_inferring: bool = False
    ) -> np.ndarray:
        """
        高阶端到端渲染：骨骼拓扑连线 + 左上角遥测仪表盘 + 底部 AI 教练口令条
        """
        if "shoulder" in landmarks and "hip" in landmarks and "knee" in landmarks and "ankle" in landmarks:
            frame = self.draw_skeleton(
                frame=frame,
                shoulder=landmarks["shoulder"],
                hip=landmarks["hip"],
                knee=landmarks["knee"],
                ankle=landmarks["ankle"],
                has_valgus=metrics.get("has_valgus", False)
            )

        phase_str = state_machine.get_phase_desc() if hasattr(state_machine, "get_phase_desc") else str(getattr(state_machine, "current_phase", "STAND"))
        rep_count = getattr(state_machine, "rep_count", 0)
        frame = self.draw_hud(
            frame=frame,
            metrics=metrics,
            phase_str=phase_str,
            rep_count=rep_count
        )

        frame = self.draw_coach_banner(
            frame=frame,
            coach_advice=ai_advice,
            is_inferring=is_inferring
        )
        return frame
