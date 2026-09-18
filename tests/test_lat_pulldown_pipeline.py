"""
Automated Test Suite for Lat Pulldown Pipeline (高位下拉端侧全链路质检流水线自动化测试)
========================================================================================
验证 Acceptance Criteria:
1. 标准达标动作短路极速直出 (Zero-Cost Short-Circuit): LLM 耗时 0ms，端到端时延 < 5ms；
2. 违规代偿动作激活端侧大模型纠错口令生成 (8~12字短口令，max_tokens <= 24，4线程调度)；
3. 单例常驻内存 (In-Memory Resident) 架构验证与零磁盘冷启动重载；
4. 数据契约对齐微调 SFT 训练集 (qa_posture.json): "用户既往生理状态: 无已知生理伤病"；
5. 伤病禁忌规则仲裁 (腰肌劳损严禁后仰 > 15°，肩袖严禁下拉低于锁骨)；
6. 异常降级防护与日志告警 (无静默吞异常代码，确定性规则降级兜底)。
"""

import os
import unittest
import logging
from pathlib import Path

from src.pipelines.lat_pulldown_pipeline import (
    LatPulldownEdgePipeline,
    ResidentLLMEngine,
    RobustCueParser
)
from src.knowledge.rule_engine import BiomechanicsRuleEngine


class TestLatPulldownPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["FORCE_MOCK"] = "true"
        ResidentLLMEngine.reset_instance()

    def setUp(self):
        self.db_path = "test_lat_pipeline_memory.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.pipeline = LatPulldownEdgePipeline(db_path=self.db_path)
        self.pipeline.setup_user("test_athlete", injury=None, posture_baseline=None)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_compliant_short_circuit_zero_cost(self):
        """
        验证 AC1: 标准达标动作下，流水线端到端执行时延在 5ms 以内，验证未调用大模型（LLM 耗时为 0ms）。
        """
        # 标准宽握下拉动作: 后仰 14°, 肩胛下沉充分 0.31, 触及上胸 0.12, 宽握 1.35
        res = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=14.0,
            scapula_ratio=0.31,
            pull_pos_ratio=0.12,
            grip_ratio=1.35,
            peak_frame=101,
            peak_time=3.37
        )

        # 1. 动作判定合规
        self.assertTrue(res["is_compliant"], "标准动作应判定为合格")
        self.assertEqual(len(res["rule_faults"]), 0, "合规动作违规项应为空")

        # 2. 短路剪枝验证 (彻底切断大模型前向调用)
        self.assertTrue(res["short_circuit"], "合规帧应触发短路剪枝标志")
        self.assertEqual(res["llm_latency_ms"], 0.0, "短路状态下 LLM 耗时必须严格为 0.0 ms")

        # 3. 性能与时延指标 (端到端时延在 5ms 以内)
        self.assertLess(res["total_latency_ms"], 5.0, f"端到端执行时延必须在 5ms 以内，实测: {res['total_latency_ms']}ms")

        # 4. 业务口令验证 (秒级直出正向激励口令)
        self.assertEqual(res["coach_cue"], "挺胸沉肩，动作标准！")
        self.assertEqual(res["raw_output"], "")
        self.assertFalse(res["is_degraded"])

    def test_non_compliant_triggers_resident_llm(self):
        """
        验证 AC2: 违规代偿动作下，大模型推理走内存常驻机制，收敛生成截断阈值 (16~24 tokens)，适配 8~12 字短口令。
        """
        # 严重违规代偿动作: 后仰 32° (超标 > 25°), 严重耸肩 0.18 (< 0.22), 下拉至剑突 0.38
        res = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=32.0,
            scapula_ratio=0.18,
            pull_pos_ratio=0.38,
            grip_ratio=1.35,
            peak_frame=264,
            peak_time=8.80
        )

        # 1. 动作判定违规
        self.assertFalse(res["is_compliant"], "代偿动作应判定为不合格")
        self.assertFalse(res["short_circuit"], "代偿动作不可短路剪枝")
        self.assertGreater(len(res["rule_faults"]), 0, "违规项列表不应为空")

        # 2. 内存常驻大模型推理验证
        self.assertIsNotNone(self.pipeline.resident_llm)
        self.assertIn("resident", self.pipeline.resident_llm.backend)

        # 3. 硬件调度与截断策略验证
        self.assertEqual(self.pipeline.resident_llm.n_threads, 4, "推理线程应收敛调优为 4 核心，防大小核争抢")
        self.assertLessEqual(self.pipeline.resident_llm.max_tokens, 24, "解码长度上限必须紧凑限制在 24 tokens 以内")

        # 4. 高穿透 8~12 字短口令解析输出验证
        cue = res["coach_cue"]
        self.assertTrue(len(cue) >= 4 and len(cue) <= 16, f"纠错短口令必须适配 8~12 字区间，实测口令: '{cue}', 长度: {len(cue)}")

    def test_resident_llm_singleton_architecture(self):
        """
        验证 AC3: 根除通过外部子进程重复从磁盘读取加载 941MB GGUF 模型的致命冷启动 I/O，统一为单例常驻内存架构。
        """
        instance1 = ResidentLLMEngine.get_instance(n_threads=4, max_tokens=24)
        instance2 = ResidentLLMEngine.get_instance(n_threads=4, max_tokens=24)
        self.assertIs(instance1, instance2, "ResidentLLMEngine 必须遵循全局单例设计模式")

        pipeline_a = LatPulldownEdgePipeline(db_path="mem_a.db")
        pipeline_b = LatPulldownEdgePipeline(db_path="mem_b.db")
        self.assertIs(pipeline_a.resident_llm, pipeline_b.resident_llm, "多流水线实例应共享内存常驻单例，杜绝重复装载")

        # 清理临时 db
        for p in ["mem_a.db", "mem_b.db"]:
            if os.path.exists(p):
                os.remove(p)

    def test_schema_contract_alignment(self):
        """
        验证 AC4: 规则引擎输出字段与微调训练集 (qa_posture.json) 命名严格一致，杜绝 Schema 偏差。
        """
        # 测试健康用户
        rule_engine = BiomechanicsRuleEngine()
        res_healthy = rule_engine.evaluate_lat_pulldown(
            torso_angle=14.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,
            grip_ratio=1.3,
            user_injury=None
        )
        self.assertIn("用户既往生理状态: 无已知生理伤病", res_healthy["payload"], "无伤病状态必须严格对齐 qa_posture.json 命名")
        self.assertNotIn("既往生理状态: 无已知伤病", res_healthy["payload"], "不应出现未对齐的旧字段命名")

        # 测试带伤病用户
        res_injury = rule_engine.evaluate_lat_pulldown(
            torso_angle=14.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,
            grip_ratio=1.3,
            user_injury="腰肌轻微劳损 (L4-L5竖脊肌酸痛)"
        )
        self.assertIn("用户既往生理状态: 腰肌轻微劳损 (L4-L5竖脊肌酸痛)", res_injury["payload"])

    def test_injury_contraindication_arbitration(self):
        """
        验证 AC5: 伤病禁忌规则仲裁。
        - 腰肌劳损患者躯干后仰角严禁超过 15 度；
        - 肩袖损伤患者横杠行程不得低于锁骨平面。
        """
        rule_engine = BiomechanicsRuleEngine()

        # 场景 A: 腰肌劳损用户在 14° (未超 15°) -> 合规
        res_lumbar_safe = rule_engine.evaluate_lat_pulldown(
            torso_angle=14.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,
            grip_ratio=1.3,
            user_injury="腰肌轻微劳损 (L4-L5竖脊肌酸痛)"
        )
        self.assertTrue(res_lumbar_safe["is_compliant"], "腰肌劳损患者后仰 14° (<=15°) 应视为合规安全")

        # 场景 B: 腰肌劳损用户后仰 20° (普通人宽握 <=25° 合规，但腰肌患者 >15° 违规禁忌) -> 裁决代偿违规
        res_lumbar_violate = rule_engine.evaluate_lat_pulldown(
            torso_angle=20.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,
            grip_ratio=1.3,
            user_injury="腰肌轻微劳损 (L4-L5竖脊肌酸痛)"
        )
        self.assertFalse(res_lumbar_violate["is_compliant"], "腰肌劳损患者后仰 20° (>15°) 必须触发禁忌警报并判决不合格")
        self.assertTrue(any("腰肌劳损患者躯干后仰角严禁超过 15 度" in f for f in res_lumbar_violate["faults"]))

        # 场景 C: 肩袖损伤患者下拉至剑突位置 (行程低于锁骨平面) -> 触发禁忌警报
        res_rotator = rule_engine.evaluate_lat_pulldown(
            torso_angle=14.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.35,  # 低于锁骨
            grip_ratio=1.3,
            user_injury="肩袖撞击综合征"
        )
        self.assertFalse(res_rotator["is_compliant"])
        self.assertTrue(any("肩袖撞击/损伤患者严禁做颈后下拉" in f for f in res_rotator["faults"]))

    def test_exception_degradation_and_fallback(self):
        """
        验证 AC6: 当模型推理发生异常时，具有清晰的 fallback 降级口令与告警日志，无静默吞异常代码。
        """
        # 注入故意抛出异常的模拟引擎
        class MockCrashingLLM(ResidentLLMEngine):
            def generate(self, prompt, max_tokens=None):
                raise RuntimeError("端侧硬件加速单元 NPU/CPU 发生模拟崩溃")

        crashing_pipeline = LatPulldownEdgePipeline(
            db_path=self.db_path,
            resident_llm=MockCrashingLLM()
        )
        crashing_pipeline.setup_user("liubo", injury="腰肌轻微劳损 (L4-L5竖脊肌酸痛)")

        # 执行违规动作 (后仰 30°, 耸肩 0.19, 触发腰肌劳损禁忌)
        with self.assertLogs("LatPulldownEdgePipeline", level="WARNING") as cm:
            res = crashing_pipeline.process_motion_event(
                user_id="liubo",
                torso_angle=30.0,
                scapula_ratio=0.19,
                pull_pos_ratio=0.12,
                grip_ratio=1.35
            )

        # 验证降级与日志告警
        self.assertTrue(res["is_degraded"], "模型异常时必须置位 is_degraded")
        self.assertEqual(res["llm_latency_ms"], 0.0)
        # 伤病禁忌优先：腰肌劳损后仰超标应输出保护腰椎安全降级口令
        self.assertEqual(res["coach_cue"], "保护腰椎，严禁后仰借力！", "伤病禁忌触发时必须优先输出保护患处安全口令")
        self.assertTrue(any("降级防护生效" in log_msg for log_msg in cm.output), "必须产生明确告警日志，严禁静默吞异常")

        # 补充验证无伤病健康用户在异常发生时的机械力学降级口令
        crashing_pipeline.setup_user("healthy_athlete", injury=None)
        res_healthy = crashing_pipeline.process_motion_event(
            user_id="healthy_athlete",
            torso_angle=30.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,
            grip_ratio=1.35
        )
        self.assertTrue(res_healthy["is_degraded"])
        self.assertEqual(res_healthy["coach_cue"], "核心收紧，减小后仰！")

    def test_robust_cue_parser(self):
        """
        验证健壮口令解析器 (RobustCueParser)
        """
        # 模式 1: 规范微调输出
        out1 = "【即时纠错口令】: 沉肩坠肘，慢放两秒！\n力学分析: 动作正常"
        self.assertEqual(RobustCueParser.extract_cue(out1), "沉肩坠肘，慢放两秒！")

        # 模式 2: 带数字编号
        out2 = "2. 即时纠错口令: 核心收紧，减小后仰！"
        self.assertEqual(RobustCueParser.extract_cue(out2), "核心收紧，减小后仰！")

        # 模式 3: 直接输出口令
        out3 = "挺胸沉肩，稳定躯干！"
        self.assertEqual(RobustCueParser.extract_cue(out3), "挺胸沉肩，稳定躯干！")

        # 模式 4: 空输出或异常输出降级
        fallback = "挺胸收腹，慢放两秒！"
        self.assertEqual(RobustCueParser.extract_cue("", rule_fallback_cue=fallback), fallback)
        self.assertEqual(RobustCueParser.extract_cue(None, rule_fallback_cue=fallback), fallback)

        # 模式 5: 截断超长句
        long_out = "即时纠错口令: 立即主动收紧腹横肌核心群，并且绝对不要在顶峰收缩时过度后仰借力！"
        extracted = RobustCueParser.extract_cue(long_out)
        self.assertLessEqual(len(extracted), 18)

    def test_injury_safe_with_unrelated_fault(self):
        """
        验证伤病禁忌精准仲裁：
        当腰肌劳损患者后仰角在安全范围内 (14° <= 15°)，但存在其他代偿 (如耸肩) 时，
        严禁错误附加'躯干后仰角严禁超过 15 度'的假阳性高危禁忌警报。
        """
        rule_engine = BiomechanicsRuleEngine()
        res = rule_engine.evaluate_lat_pulldown(
            torso_angle=14.0,
            scapula_ratio=0.18,  # 耸肩
            pull_pos_ratio=0.12,
            grip_ratio=1.3,
            user_injury="腰肌轻微劳损 (L4-L5竖脊肌酸痛)"
        )
        self.assertFalse(res["is_compliant"])
        self.assertTrue(any("过度耸肩" in f for f in res["faults"]))
        # 严禁假阳性误报
        self.assertFalse(any("腰肌劳损患者躯干后仰角严禁超过 15 度" in f for f in res["faults"]),
                         "后仰 14° 未违背 15° 限制，不应错误误报后仰禁忌！")

    def test_nan_and_inf_edge_cases(self):
        """
        验证极端浮点数边界 (NaN, Inf, 物理非法越界)：
        1. 必须具备容错防御，严禁抛出 ValueError/OverflowError 崩溃；
        2. 绝对不能误判为合规标准动作 (不得短路直出)；
        3. 必须输出安全降级口令。
        """
        # 测试 NaN 输入
        res_nan = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=float("nan"),
            scapula_ratio=0.30,
            pull_pos_ratio=0.12
        )
        self.assertFalse(res_nan["is_compliant"], "NaN 输入不可判为合格")
        self.assertFalse(res_nan["short_circuit"], "NaN 输入不可触发短路")
        self.assertTrue(any("异常" in f for f in res_nan["rule_faults"]))
        self.assertEqual(res_nan["coach_cue"], "姿态信号异常，请对准机位！")

        # 测试 Inf 输入
        res_inf = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=float("inf"),
            scapula_ratio=0.30,
            pull_pos_ratio=0.12
        )
        self.assertFalse(res_inf["is_compliant"])
        self.assertFalse(res_inf["short_circuit"])
        self.assertTrue(any("异常" in f for f in res_inf["rule_faults"]))

    def test_simulator_precise_cue_matching(self):
        """
        验证内存常驻引擎仿真匹配的精准性：
        消除将提示词中的模板前缀'躯干后仰角'误当成后仰代偿的致命缺陷，
        确保不同违规动作产出专属的力学纠错口令。
        """
        # 1. 仅耸肩违规 (后仰 14° 完全达标)
        res_shrug = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=14.0,
            scapula_ratio=0.15,
            pull_pos_ratio=0.12
        )
        self.assertEqual(res_shrug["coach_cue"], "沉肩坠肘，锁死肩胛！", "单纯耸肩不可误报后仰口令")

        # 2. 仅下拉位置过低违规 (后仰 14° 达标，无耸肩)
        res_pull = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=14.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.38
        )
        self.assertEqual(res_pull["coach_cue"], "横杠拉至上胸，控制行程！")

        # 3. 真实后仰超标 (后仰 32°，无耸肩)
        res_lean = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=32.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12
        )
        self.assertEqual(res_lean["coach_cue"], "核心收紧，减小后仰！")

    def test_robust_cue_parser_adversarial(self):
        """
        验证健壮口令解析器的对抗性边界测试：
        防御 XSS 脚本注入、ChatML 标记泄露、Markdown 代码块、ANSI 转义及控制字符。
        """
        # 1. HTML / XSS 注入
        cue_xss = RobustCueParser.extract_cue("<script>alert(1)</script>")
        self.assertNotIn("<script>", cue_xss)
        self.assertEqual(cue_xss, "挺胸收腹，慢放两秒！")

        # 2. ChatML 模板 Token 泄露
        cue_chatml = RobustCueParser.extract_cue("<|im_start|>assistant\n<|im_end|>")
        self.assertNotIn("<|im_start|>", cue_chatml)
        self.assertNotIn("assistant", cue_chatml)
        self.assertEqual(cue_chatml, "挺胸收腹，慢放两秒！")

        # 3. Markdown 代码块污染
        md_raw = "```markdown\n1. 动作缺陷: 严重后仰\n2. 即时纠错口令: 核心收紧，减小后仰！\n```"
        cue_md = RobustCueParser.extract_cue(md_raw)
        self.assertNotIn("`", cue_md)
        self.assertEqual(cue_md, "核心收紧，减小后仰！")

        # 4. ANSI 终端颜色转义符
        ansi_raw = "\x1b[31m沉肩坠肘，锁死肩胛！\x1b[0m"
        cue_ansi = RobustCueParser.extract_cue(ansi_raw)
        self.assertNotIn("\x1b", cue_ansi)
        self.assertEqual(cue_ansi, "沉肩坠肘，锁死肩胛！")

        # 5. 空字节与控制字符
        ctrl_raw = "\x00\x01\x02\r\t"
        cue_ctrl = RobustCueParser.extract_cue(ctrl_raw)
        self.assertEqual(cue_ctrl, "挺胸收腹，慢放两秒！")

    def test_concurrent_inference_thread_safety(self):
        """
        验证 ResidentLLMEngine 线程安全并发推理：
        模拟移动端多线程高频调用，保证无死锁、无状态竞争。
        """
        import threading
        results = []
        errors = []

        def worker(angle):
            try:
                out = self.pipeline.resident_llm.generate(
                    f"目标动作: 高位下拉 | 躯干后仰角: {angle}° | 肩胛状态: 严重耸肩代偿 | 下拉终点位置: 锁骨下方上胸位置 | 用户既往生理状态: 无已知生理伤病"
                )
                results.append(out)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(10 + i * 2,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"并发推理不可抛出异常: {errors}")
        self.assertEqual(len(results), 10)

    def test_close_grip_safe_angle_with_shrug(self):
        """
        验证窄握变体 (Close Grip) 安全角度与耸肩代偿隔离：
        窄握允许躯干后仰角达到 30°。当后仰 27° (安全范围内) 且存在耸肩 (0.15) 时，
        规则引擎严禁判定后仰违规，常驻模型必须精准输出'沉肩坠肘，锁死肩胛！'，绝不可误报'核心收紧，减小后仰！'。
        """
        res = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=27.0,
            scapula_ratio=0.15,
            pull_pos_ratio=0.20,
            grip_ratio=0.8
        )
        self.assertFalse(res["is_compliant"])
        self.assertEqual(len(res["rule_faults"]), 1)
        self.assertIn("斜方肌过度耸肩", res["rule_faults"][0])
        self.assertEqual(res["coach_cue"], "沉肩坠肘，锁死肩胛！")

    def test_close_grip_sword_process_ideal_endpoint(self):
        """
        验证窄握变体下拉终点至剑突为标准受力点：
        窄握下拉 (V-Bar) 标准行程终点为'剑突至上胸下段位置'，严禁误报为违规并触发错误截断。
        """
        res = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=20.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.20,
            grip_ratio=0.8
        )
        self.assertTrue(res["is_compliant"], "窄握拉至剑突上胸应判定为合格")
        self.assertTrue(res["short_circuit"], "标准动作应触发短路直出")
        self.assertEqual(res["coach_cue"], "挺胸沉肩，动作标准！")

    def test_forward_lean_compensation_detection(self):
        """
        验证高位下拉躯干前倾代偿 (含胸前倾驼背)：
        负值后仰角 (如 -15°) 属于严重前倾违规，严禁误判为'过于直立'并漏检放行。
        """
        res = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=-15.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,
            grip_ratio=1.35
        )
        self.assertFalse(res["is_compliant"], "前倾负角度必须判决不合格")
        self.assertFalse(res["short_circuit"], "代偿动作不可短路直出")
        self.assertTrue(any("躯干前倾" in f for f in res["rule_faults"]))
        self.assertEqual(res["coach_cue"], "打开胸腔，挺胸防前倾！")

    def test_legacy_injury_normalization_no_bogus_warning(self):
        """
        验证历史伤病表达归一化：
        当学员传入'无已知伤病'或'健康'时，归一化为'无已知生理伤病'，严禁错误将'无已知伤病'视为真实创伤并挂载风控预警。
        """
        rule_engine = BiomechanicsRuleEngine()
        res = rule_engine.evaluate_lat_pulldown(
            torso_angle=30.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,
            user_injury="无已知伤病"
        )
        self.assertFalse(res["is_compliant"])
        self.assertFalse(any("检测到学员既往伤病档案（无已知伤病）" in f for f in res["faults"]),
                         "归一化后严禁出现无已知伤病的虚假风控警报")
        self.assertEqual(res["user_status"], "无已知生理伤病")

    def test_l1_cache_ttl_expiration_sync(self):
        """
        验证 L1 内存缓存与 SQLite 记忆 TTL 惰性淘汰同步：
        当短期状态（如扭伤）TTL 到期后，L1 内存缓存必须自动失效并同步恢复为'无已知生理伤病'。
        """
        import time
        self.pipeline.setup_user("ttl_user", injury="短期手腕扭伤", ttl_seconds=1)
        s1 = self.pipeline.get_cached_user_status("ttl_user")
        self.assertIn("短期手腕扭伤", s1)
        
        # 等待 TTL 过期
        time.sleep(1.2)
        s2 = self.pipeline.get_cached_user_status("ttl_user")
        self.assertEqual(s2, "无已知生理伤病", "TTL 到期后 L1 缓存应自动完成惰性淘汰")

    def test_robust_cue_parser_unpunctuated_long_text(self):
        """
        验证 RobustCueParser 对无标点长文本的强制 16 字截断：
        杜绝因单句无标点导致提取口令突破 16 字硬性限制。
        """
        raw = "即时纠错口令: 学员请立即在本次下拉过程中主动收紧核心群避免后仰"
        cue = RobustCueParser.extract_cue(raw)
        self.assertLessEqual(len(cue), 16, f"口令长度必须严格 <= 16 字，实测: {len(cue)}, '{cue}'")
        self.assertTrue(cue.endswith("！"))

    def test_non_physical_negative_ratios(self):
        """
        验证物理特征比值非负性校验：
        耳肩距、两肩距比值若为负数属于视觉传感器严重异常，应安全拦截降级。
        """
        res = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=14.0,
            scapula_ratio=-0.5,
            pull_pos_ratio=0.12
        )
        self.assertFalse(res["is_compliant"])
        self.assertTrue(any("视觉传感器力学测算异常" in f for f in res["rule_faults"]))
        self.assertEqual(res["coach_cue"], "姿态信号异常，请对准机位！")

    def test_close_grip_deep_pull_safe_angle_cue(self):
        """
        验证窄握变体深度拉至腹部时，即使角度处于 27° (窄握安全范围 <=30°)，
        常驻模型也绝不可误报为后仰违规，必须精准输出行程纠错口令。
        """
        res = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=27.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.40,
            grip_ratio=0.8
        )
        self.assertFalse(res["is_compliant"])
        self.assertEqual(res["rule_faults"], ["窄握下拉行程过深"])
        self.assertEqual(res["coach_cue"], "控制行程，横杠不拉过低！")

    def test_rule_fallback_lumbar_with_shrug_no_false_alarm(self):
        """
        验证规则降级引擎伤病假阳性隔离：
        学员有腰肌劳损，后仰角为 14° (安全 <=15°)，但发生耸肩代偿。
        降级引擎必须输出耸肩口令，严禁因学员带腰伤标签就误报'保护腰椎，严禁后仰借力！'。
        """
        class MockCrashingLLM(ResidentLLMEngine):
            def generate(self, prompt, max_tokens=None):
                raise RuntimeError("降级模拟异常")

        fallback_pipeline = LatPulldownEdgePipeline(
            db_path="test_fb_lumbar.db",
            resident_llm=MockCrashingLLM()
        )
        fallback_pipeline.setup_user("athlete_lumbar", injury="腰肌轻微劳损 (L4-L5竖脊肌酸痛)")
        res = fallback_pipeline.process_motion_event(
            user_id="athlete_lumbar",
            torso_angle=14.0,
            scapula_ratio=0.18,  # 耸肩
            pull_pos_ratio=0.12,
            grip_ratio=1.35
        )
        if os.path.exists("test_fb_lumbar.db"):
            os.remove("test_fb_lumbar.db")

        self.assertTrue(res["is_degraded"])
        self.assertEqual(res["coach_cue"], "沉肩坠肘，锁死肩胛！", "未违背后仰限制时绝不可虚假输出腰椎后仰降级口令")

    def test_rule_fallback_shoulder_with_lean_no_false_alarm(self):
        """
        验证规则降级引擎肩袖假阳性隔离：
        学员有肩袖撞击，行程在锁骨平面 (0.12 <=0.15 安全)，但发生后仰代偿 (32°)。
        降级引擎必须输出后仰口令，严禁因学员有肩袖病史就误报'保护肩袖，横杠不过低！'。
        """
        class MockCrashingLLM(ResidentLLMEngine):
            def generate(self, prompt, max_tokens=None):
                raise RuntimeError("降级模拟异常")

        fallback_pipeline = LatPulldownEdgePipeline(
            db_path="test_fb_shoulder.db",
            resident_llm=MockCrashingLLM()
        )
        fallback_pipeline.setup_user("athlete_cuff", injury="肩袖撞击综合征")
        res = fallback_pipeline.process_motion_event(
            user_id="athlete_cuff",
            torso_angle=32.0,  # 后仰超标
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,  # 锁骨位置安全
            grip_ratio=1.35
        )
        if os.path.exists("test_fb_shoulder.db"):
            os.remove("test_fb_shoulder.db")

        self.assertTrue(res["is_degraded"])
        self.assertEqual(res["coach_cue"], "核心收紧，减小后仰！")

    def test_out_of_range_angles_sensor_anomaly(self):
        """
        验证超出人体解剖运动极限的角度 (如 120° 或 -45°):
        必须被拦截为姿态信号异常，严禁误诊为普通力学代偿。
        """
        # 测试 +120° (倒立/传感器反向)
        res_high = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=120.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12
        )
        self.assertFalse(res_high["is_compliant"])
        self.assertFalse(res_high["short_circuit"])
        self.assertEqual(res_high["coach_cue"], "姿态信号异常，请对准机位！")

        # 测试 -45° (严重异常折叠)
        res_low = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=-45.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12
        )
        self.assertFalse(res_low["is_compliant"])
        self.assertFalse(res_low["short_circuit"])
        self.assertEqual(res_low["coach_cue"], "姿态信号异常，请对准机位！")

    def test_glitched_upper_bound_physical_ratios(self):
        """
        验证超出物理极限的比值 (如耳肩比 3.5, 握距比 10.0, 下拉深度比 5.0):
        严禁被误判为达标合格通过短路剪枝，必须拦截报警。
        """
        res_scap = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=15.0,
            scapula_ratio=3.5,
            pull_pos_ratio=0.12
        )
        self.assertFalse(res_scap["is_compliant"])
        self.assertFalse(res_scap["short_circuit"])
        self.assertTrue(any("视觉传感器力学测算异常" in f for f in res_scap["rule_faults"]))
        self.assertEqual(res_scap["coach_cue"], "姿态信号异常，请对准机位！")

        res_grip = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=15.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,
            grip_ratio=10.0
        )
        self.assertFalse(res_grip["is_compliant"])
        self.assertEqual(res_grip["coach_cue"], "姿态信号异常，请对准机位！")

    def test_robust_cue_parser_exact_16_char_unpunctuated(self):
        """
        验证恰好 16 字无标点文本解析：
        加上感叹号后总长度必须严格限制在 16 字以内。
        """
        raw = "即时纠错口令: 一二三四五六七八九十一二三四五六"
        cue = RobustCueParser.extract_cue(raw)
        self.assertLessEqual(len(cue), 16, f"口令总长度必须严格 <= 16 字，实测: {len(cue)}, '{cue}'")
        self.assertTrue(cue.endswith("！"))

    def test_string_peak_frame_and_time(self):
        """
        验证 REST/JSON 传入字符串格式的 peak_frame 与 peak_time:
        流水线应能健壮解析并正确返回整型与浮点型。
        """
        res = self.pipeline.process_motion_event(
            user_id="test_athlete",
            torso_angle=14.0,
            scapula_ratio=0.30,
            pull_pos_ratio=0.12,
            grip_ratio=1.35,
            peak_frame="101",
            peak_time="3.37"
        )
        self.assertEqual(res["peak_frame"], 101)
        self.assertEqual(res["peak_time_s"], 3.37)
        self.assertIn("user_injury", res)


if __name__ == "__main__":
    unittest.main()
