"""
Motion State Machine & Biomechanical Trajectory Analyzer
========================================================
负责运动时序轨迹的高频时序分析与有限状态机（FSM）管理：
1. 平滑滤波：抑制单帧关键点跳变噪声（滑动平均 / EMA）
2. 动作相位识别：站立(STANDING) -> 下蹲(DESCENDING) -> 底峰拐点(BOTTOM_PEAK) -> 蹬伸(ASCENDING)
3. 事件触发器（Event Trigger）：
   - 动作触底极值点触发（评估下蹲深度与对称性）
   - 危险力学代偿防抖触发（膝关节连续内扣超过防抖阈值）
4. 计数器：精准记录完成次数（Repetition Counter）
"""

from enum import Enum
from typing import Dict, Any, Tuple, Optional, List


class MotionPhase(str, Enum):
    STANDING = "站立位 (Standing)"
    DESCENDING = "下蹲中 (Descending)"
    BOTTOM_PEAK = "触底拐点 (Bottom Peak)"
    ASCENDING = "蹬伸上升 (Ascending)"


class SquatStateMachine:
    """
    深蹲有限状态机与极值拐点检测器
    """

    def __init__(
        self,
        knee_standing_threshold: float = 150.0,
        knee_bottom_threshold: float = 100.0,
        valgus_debounce_frames: int = 4
    ):
        self.standing_thresh = knee_standing_threshold
        self.bottom_thresh = knee_bottom_threshold
        self.valgus_debounce = valgus_debounce_frames

        self.current_phase = MotionPhase.STANDING
        self.rep_count = 0
        self.min_knee_angle_in_rep = 180.0
        self.rep_has_valgus = False

        # 历史窗口
        self.angle_history: List[float] = []
        self.valgus_streak = 0
        self._llm_triggered_in_current_rep = False

    def update(self, metrics: Dict[str, Any]) -> Tuple[MotionPhase, bool, str]:
        """
        输入单帧骨骼力学特征，更新状态机并判定是否触发端侧 LLM 认知事件。

        Returns:
            (当前运动相位, 是否触发大模型评估, 触发原因)
        """
        knee_angle = metrics["knee_angle"]
        has_valgus = metrics["has_valgus"]

        self.angle_history.append(knee_angle)
        if len(self.angle_history) > 10:
            self.angle_history.pop(0)

        # 1. 膝内扣防抖统计
        if has_valgus:
            self.valgus_streak += 1
            self.rep_has_valgus = True
        else:
            self.valgus_streak = max(0, self.valgus_streak - 1)

        trigger_llm = False
        trigger_reason = ""

        # 2. 状态机迁移逻辑
        if self.current_phase == MotionPhase.STANDING:
            if knee_angle < self.standing_thresh - 5.0:
                self.current_phase = MotionPhase.DESCENDING
                self.min_knee_angle_in_rep = knee_angle
                self._llm_triggered_in_current_rep = False
                self.rep_has_valgus = False

        elif self.current_phase == MotionPhase.DESCENDING:
            if knee_angle < self.min_knee_angle_in_rep:
                self.min_knee_angle_in_rep = knee_angle

            # 拐点检测：若当前角度比本次动作最低点回弹反弹至少 4 度，说明已触底并开始反弹上升
            if knee_angle >= self.min_knee_angle_in_rep + 4.0:
                self.current_phase = MotionPhase.BOTTOM_PEAK
                if not self._llm_triggered_in_current_rep:
                    trigger_llm = True
                    trigger_reason = "bottom_peak" if self.min_knee_angle_in_rep <= self.bottom_thresh else "shallow_squat_warning"
                    self._llm_triggered_in_current_rep = True

        elif self.current_phase == MotionPhase.BOTTOM_PEAK:
            if knee_angle > self.min_knee_angle_in_rep + 8.0:
                self.current_phase = MotionPhase.ASCENDING

        elif self.current_phase == MotionPhase.ASCENDING:
            if knee_angle >= self.standing_thresh - 5.0:
                self.current_phase = MotionPhase.STANDING
                self.rep_count += 1
                self.min_knee_angle_in_rep = 180.0

        # 3. 紧急代偿触发检查：即使未触底，严重且持续内扣必须紧急介入
        if self.valgus_streak >= self.valgus_debounce and not self._llm_triggered_in_current_rep:
            trigger_llm = True
            trigger_reason = "severe_valgus_warning"
            self._llm_triggered_in_current_rep = True

        return self.current_phase, trigger_llm, trigger_reason

    def reset(self):
        """重置状态机"""
        self.current_phase = MotionPhase.STANDING
        self.rep_count = 0
        self.min_knee_angle_in_rep = 180.0
        self.rep_has_valgus = False
        self.angle_history.clear()
        self.valgus_streak = 0
        self._llm_triggered_in_current_rep = False
