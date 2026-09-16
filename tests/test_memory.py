"""
Unit Tests for Agent Memory Engine
Verifies:
1. 3-Tier Categorization (Profile, Preference, Status)
2. TTL Lazy Eviction (Short-term status expiration)
3. Slot UPSERT Conflict Resolution (Overwriting conflicting facts)
"""

import time
import os
import unittest
from src.memory.agent_memory import AgentMemoryEngine


class TestAgentMemory(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_agent_memory.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.engine = AgentMemoryEngine(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_upsert_conflict_resolution(self):
        """测试相同的 key 会被原子覆写，消解矛盾事实"""
        # 1. 昨天用户喜好
        self.engine.set_memory("liubo", "diet_pref", "超级喜欢吃重辣川菜", category="preference", importance=4)
        memories = self.engine.get_effective_memories("liubo")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["content"], "超级喜欢吃重辣川菜")

        # 2. 今天突发医嘱禁辣，使用相同 key 覆盖
        self.engine.set_memory("liubo", "diet_pref", "医生诊断胃溃疡，一个月内绝对禁辣！", category="preference", importance=5)
        memories = self.engine.get_effective_memories("liubo")
        self.assertEqual(len(memories), 1, "应该只有一条最新记录，旧冲突条目必须被清除")
        self.assertEqual(memories[0]["content"], "医生诊断胃溃疡，一个月内绝对禁辣！")

    def test_ttl_lazy_eviction(self):
        """测试 TTL 存活期过后自动惰性淘汰"""
        # 设置一条仅存活 1 秒的临时状态记忆
        self.engine.set_memory("liubo", "acute_fatigue", "右膝扭伤，需要卧床休息", category="status", importance=5, ttl_seconds=1)
        
        # 立即读取，应该存在
        memories = self.engine.get_effective_memories("liubo")
        self.assertEqual(len(memories), 1)
        
        # 等待 1.2 秒后读取，应该被惰性淘汰
        time.sleep(1.2)
        memories_after = self.engine.get_effective_memories("liubo")
        self.assertEqual(len(memories_after), 0, "过期记忆应该被自动清除，防止污染上下文")

    def test_prompt_formatting(self):
        """测试生成 System Prompt 上下文格式"""
        self.engine.set_memory("liubo", "name", "刘博", category="profile", importance=5)
        self.engine.set_memory("liubo", "device", "RTX 3090 + Intel Xeon", category="preference", importance=3)
        
        prompt = self.engine.format_memory_prompt("liubo")
        self.assertIn("【用户长期与当前生理记忆上下文】", prompt)
        self.assertIn("刘博", prompt)
        self.assertIn("RTX 3090", prompt)


if __name__ == "__main__":
    unittest.main()
