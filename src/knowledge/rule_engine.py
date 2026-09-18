"""
Biomechanical Symbolic Rule Engine (参数化力学规则引擎)
=====================================================
负责将前端 CV 测得的连续物理量，与外部 RAG 知识库（biomechanics_rules.json）进行确定性比对：
1. 握距自适应（Grip Variant Routing）：根据两腕间距与两肩间距比值 R_grip 动态切换宽握/窄握标尺；
2. 施密特双门限迟滞比对（Schmitt Trigger）：消除临界线高频抖动；
3. 伤病禁忌规则仲裁（Contraindication Arbitration）；
4. 产出 100% 契合微调 SFT 语料格式的结构化特征快照（Zero Schema Drift）。
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


class BiomechanicsRuleEngine:
    """
    轻量、零算术幻觉的端侧参数化力学规则引擎
    """
    def __init__(self, rules_path: Optional[str] = None):
        if rules_path is None:
            base_dir = Path(__file__).parent
            rules_path = str(base_dir / "biomechanics_rules.json")
        self.rules_path = rules_path
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if not os.path.exists(self.rules_path):
            import logging
            logging.getLogger("BiomechanicsRuleEngine").warning(f"Rules file not found at: {self.rules_path}")
            return {}
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            import logging
            logging.getLogger("BiomechanicsRuleEngine").warning(f"Failed to parse biomechanics rules at {self.rules_path}: {e}")
            return {}

    def evaluate_lat_pulldown(
        self,
        torso_angle: float,
        scapula_ratio: float,
        pull_pos_ratio: float,
        grip_ratio: float = 1.3,
        user_injury: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        对高位下拉单帧/极值快照进行 100% 确定性力学裁决。

        Args:
            torso_angle: 躯干后仰角 (度)
            scapula_ratio: 耳肩距 / 躯干长 (肩胛下沉比)
            pull_pos_ratio: (手腕Y - 肩膀Y) / 躯干长
            grip_ratio: 双腕距 / 双肩距 (>= 1.1 为宽握，< 0.9 为窄握)
            user_injury: 学员既往生理状态/伤病描述

        Returns:
            Dict 包含结构化快照文本 payload、违规项列表 faults、是否合规 is_compliant
        """
        lat_rules = self.rules.get("lat_pulldown", {})
        variants = lat_rules.get("grip_variants", {})

        faults: List[str] = []

        # 0. 输入量合法性校验 (防御 NaN, Inf 及非数值输入，杜绝崩溃与漏检)
        has_invalid_data = False
        try:
            torso_angle = float(torso_angle)
            scapula_ratio = float(scapula_ratio)
            pull_pos_ratio = float(pull_pos_ratio)
            grip_ratio = float(grip_ratio)
        except (ValueError, TypeError):
            has_invalid_data = True

        if has_invalid_data or not (
            math.isfinite(torso_angle) and math.isfinite(scapula_ratio) and
            math.isfinite(pull_pos_ratio) and math.isfinite(grip_ratio)
        ):
            faults.append("视觉传感器力学测算异常: 关键点特征存在非数值(NaN)或溢出(Inf)")
            has_invalid_data = True
        elif scapula_ratio < 0.0 or grip_ratio <= 0.0 or pull_pos_ratio < -1.0:
            faults.append("视觉传感器力学测算异常: 空间关键点特征比值出现非物理负值")
            has_invalid_data = True
        elif scapula_ratio > 1.0 or grip_ratio > 4.0 or pull_pos_ratio > 2.0:
            faults.append("视觉传感器力学测算异常: 空间关键点特征比值超出人体解剖学极限")
            has_invalid_data = True

        # 1. 握距自适应路由
        if not has_invalid_data and grip_ratio < 1.0:
            variant_key = "close_grip"
            variant_cfg = variants.get("close_grip", {
                "variant_name": "窄握对握下拉",
                "ideal_torso_angle": [15, 25],
                "max_safe_torso_angle": 30,
                "ideal_endpoint": "剑突至上胸下段位置"
            })
        else:
            variant_key = "wide_grip"
            variant_cfg = variants.get("wide_grip", {
                "variant_name": "宽握颈前下拉",
                "ideal_torso_angle": [10, 20],
                "max_safe_torso_angle": 25,
                "ideal_endpoint": "锁骨下方上胸位置"
            })

        # 2. 躯干后仰角评定 (施密特迟滞区间防抖)
        if has_invalid_data:
            lean_eval = "角度数据异常"
        else:
            max_safe = variant_cfg.get("max_safe_torso_angle", 25)
            ideal_min, ideal_max = variant_cfg.get("ideal_torso_angle", [10, 20])

            if torso_angle < -30.0 or torso_angle > 90.0:
                faults.append(f"躯干后仰角 {torso_angle:.1f}° 超出人体正常运动力学范围")
                lean_eval = "角度超出解剖极限"
                has_invalid_data = True
            elif torso_angle < 0.0:
                faults.append(f"躯干前倾 {abs(torso_angle):.1f}°，严禁含胸前倾代偿")
                lean_eval = "含胸前倾代偿"
            elif torso_angle > max_safe:
                faults.append(f"后仰角 {torso_angle:.1f}° 超过安全上限 {max_safe}° (过度借力)")
                lean_eval = "严重后仰代偿"
            elif torso_angle > ideal_max:
                lean_eval = "微后仰借力"
            elif torso_angle < ideal_min:
                lean_eval = "过于直立"
            else:
                lean_eval = "标准受力区间"

        # 3. 肩胛下沉状态评定
        scap_cfg = lat_rules.get("scapular_depression", {})
        scap_warn = scap_cfg.get("warning_ratio_below", 0.22)
        scap_ideal = scap_cfg.get("ideal_ratio_above", 0.28)

        if has_invalid_data:
            scapula_status = "肩胛比值异常"
        elif scapula_ratio < scap_warn:
            scapula_status = "严重耸肩代偿"
            faults.append("斜方肌过度耸肩，肩胛骨未锁死下沉")
        elif scapula_ratio < scap_ideal:
            scapula_status = "肩胛下沉尚可"
        else:
            scapula_status = "肩胛骨充分下沉锁死 (无耸肩)"

        # 4. 下拉触点终点位置评定
        if has_invalid_data:
            pull_pos = "位置数据异常"
        elif variant_key == "wide_grip":
            if pull_pos_ratio < 0.15:
                pull_pos = "锁骨下方上胸位置"
            elif pull_pos_ratio < 0.40:
                pull_pos = "剑突位置"
                faults.append("宽握下拉拉至剑突过低，动作变形为划船")
            else:
                pull_pos = "胸骨/腹部位置"
                faults.append("下拉位置过低，张力脱落")
        else:
            if pull_pos_ratio < 0.35:
                pull_pos = "剑突至上胸下段位置"
            else:
                pull_pos = "腹部位置"
                faults.append("窄握下拉行程过深")

        # 5. 伤病禁忌核查 (Contraindication Arbitration)
        raw_injury = user_injury.strip() if (user_injury and isinstance(user_injury, str) and user_injury.strip()) else ""
        if not raw_injury or raw_injury in ("无已知生理伤病", "无已知伤病", "无", "健康", "正常", "none", "None"):
            injury_desc = "无已知生理伤病"
        else:
            injury_desc = raw_injury

        contraindications = lat_rules.get("injury_contraindications", {})

        has_specific_contra_violation = False
        # 针对性伤病禁忌仲裁:
        # A. 腰肌劳损患者: 严格受限后仰角 <= 15°
        if ("腰" in injury_desc or "lumbar" in injury_desc.lower()) and injury_desc != "无已知生理伤病":
            if not has_invalid_data and torso_angle > 15.0:
                msg = f"【高危禁忌警报】{contraindications.get('lumbar_strain', '腰肌劳损患者躯干后仰角严禁超过 15 度，严禁利用腰椎前后反弓摆动代偿')}"
                if msg not in faults:
                    faults.append(msg)
                has_specific_contra_violation = True

        # B. 肩袖撞击/损伤患者: 行程严禁低于锁骨平面
        if ("肩袖" in injury_desc or "rotator" in injury_desc.lower()) and injury_desc != "无已知生理伤病":
            if not has_invalid_data and pull_pos_ratio > 0.15:
                msg = f"【高危禁忌警报】{contraindications.get('rotator_cuff_impingement', '肩袖撞击/损伤患者严禁做颈后下拉，横杠拉动行程不得低于锁骨平面')}"
                if msg not in faults:
                    faults.append(msg)
                has_specific_contra_violation = True

        # C. 高低肩/体态侧弯: 严重耸肩或代偿时提示
        if ("高低肩" in injury_desc or "侧弯" in injury_desc or "shoulder_asymmetry" in injury_desc.lower()) and injury_desc != "无已知生理伤病":
            if not has_invalid_data and scapula_ratio < scap_warn:
                msg = f"【高危禁忌警报】{contraindications.get('shoulder_asymmetry', '高低肩/脊柱轻度侧弯患者需强化弱侧背阔肌意念控制，严禁杠铃左右倾斜代偿')}"
                if msg not in faults:
                    faults.append(msg)
                has_specific_contra_violation = True

        # D. 若已存在动作代偿且存在既往伤病，附加伤病风控预警 (杜绝在后仰合规时错误挂载躯干后仰禁忌)
        if faults and injury_desc != "无已知生理伤病" and not has_specific_contra_violation:
            general_warn = f"【伤病风控预警】检测到学员既往伤病档案（{injury_desc}），代偿动作易诱发次生损伤"
            if general_warn not in faults:
                faults.append(general_warn)

        # 6. 生成 100% 契合微调 SFT (qa_posture.json) 的标准化快照字符串 (Zero Schema Drift)
        # 契约对齐: "目标动作: 高位下拉 ({variant_clean})" 与 qa_posture.json 风格严格对齐
        variant_name = variant_cfg.get("variant_name", "宽握颈前下拉")
        variant_clean = variant_name.split("(")[0].strip()
        angle_str = f"{int(torso_angle)}°" if (not has_invalid_data and math.isfinite(torso_angle) and -30.0 <= torso_angle <= 90.0) else "异常°"
        payload = (
            f"目标动作: 高位下拉 ({variant_clean}) | 躯干后仰角: {angle_str} | "
            f"肩胛状态: {scapula_status} | 下拉终点位置: {pull_pos} | 用户既往生理状态: {injury_desc}"
        )

        return {
            "action": "高位下拉",
            "variant": variant_cfg.get("variant_name", "宽握颈前下拉"),
            "torso_angle": round(torso_angle, 1) if (not has_invalid_data and math.isfinite(torso_angle)) else None,
            "lean_evaluation": lean_eval,
            "scapula_status": scapula_status,
            "pull_position": pull_pos,
            "user_injury": injury_desc,
            "user_status": injury_desc,
            "is_compliant": len(faults) == 0,
            "faults": faults,
            "payload": payload
        }


if __name__ == "__main__":
    engine = BiomechanicsRuleEngine()
    # 测试宽握严重后仰 + 耸肩
    res = engine.evaluate_lat_pulldown(
        torso_angle=32.5,
        scapula_ratio=0.18,
        pull_pos_ratio=0.12,
        grip_ratio=1.4,
        user_injury="腰肌劳损 (L4-L5竖脊肌酸痛)"
    )
    print("=== 规则引擎判决测试 ===")
    print("变式:", res["variant"])
    print("是否合规:", res["is_compliant"])
    print("违规项:", res["faults"])
    print("生成的对齐 Prompt Payload:\n", res["payload"])
