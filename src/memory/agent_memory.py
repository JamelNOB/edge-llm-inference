"""
Agent Memory Governance Engine (SQLite + 3-Tier Hierarchy + TTL + Conflict UPSERT)
[简历核心支撑点: 分层记忆治理与时效状态淘汰]

设计动机:
大模型本身为无状态黑盒。若将所有历史对话无脑拼接注入 Context，会导致:
1. 显存与计算开销呈平方级暴涨 (KV Cache 溢出);
2. 历史短期状态 (如生理不适、临时行程) 永久污染长期用户画像;
3. 相互矛盾的事实 (如喜好变更) 同时出现导致模型人格分裂。

本模块提供:
1. 三层记忆划分: Profile(永久长期)、Preference(中期偏好)、Status(短时生理/时效状态);
2. 惰性 TTL 淘汰 (Lazy Eviction): 读取时自动清理已失效条目，无需后台轮询线程;
3. Slot UPSERT 冲突消解: 强制 UNIQUE(user_id, key) 并在写入时原子覆写，杜绝矛盾事实并存。
"""

import time
import sqlite3
from typing import List, Dict, Any, Optional


class AgentMemoryEngine:
    """
    工业级轻量 Agent 记忆治理引擎
    """
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL DEFAULT 'profile',
                        importance INTEGER NOT NULL DEFAULT 3,
                        created_at REAL NOT NULL,
                        expires_at REAL,
                        UNIQUE(user_id, key) ON CONFLICT REPLACE
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_expires 
                    ON memories (user_id, expires_at)
                """)
        finally:
            conn.close()

    def set_memory(
        self,
        user_id: str,
        key: str,
        content: str,
        category: str = "profile",
        importance: int = 3,
        ttl_seconds: int = 0
    ) -> None:
        """
        写入或覆盖记忆条目 (UPSERT 原子操作)
        
        Args:
            user_id: 用户唯一标识
            key: 记忆槽位键 (如 'diet_preference', 'current_injury')
            content: 事实内容
            category: 类别 ('profile' 长期, 'preference' 偏好, 'status' 短期状态)
            importance: 重要度权重 (1-5 级，用于注入时排序截断)
            ttl_seconds: 存活秒数 (0 代表永久有效)
        """
        now = time.time()
        expires_at = (now + ttl_seconds) if ttl_seconds > 0 else None
        
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (user_id, key, content, category, importance, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, key, content, category, importance, now, expires_at))
        finally:
            conn.close()

    def get_effective_memories(
        self,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取当前有效且未过期的记忆列表 (附带惰性淘汰)
        """
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.cursor()
                # 1. 惰性淘汰 (Lazy Eviction): 顺手清理已过期的垃圾记录
                cursor.execute("""
                    DELETE FROM memories 
                    WHERE expires_at IS NOT NULL AND expires_at <= ?
                """, (now,))
                
                # 2. 查询有效记录
                if category:
                    query = """
                        SELECT key, content, category, importance, created_at, expires_at
                        FROM memories 
                        WHERE user_id = ? AND category = ?
                        ORDER BY importance DESC, created_at DESC
                        LIMIT ?
                    """
                    params = (user_id, category, limit)
                else:
                    query = """
                        SELECT key, content, category, importance, created_at, expires_at
                        FROM memories 
                        WHERE user_id = ?
                        ORDER BY importance DESC, created_at DESC
                        LIMIT ?
                    """
                    params = (user_id, limit)
                    
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                results = []
                for r in rows:
                    results.append({
                        "key": r[0],
                        "content": r[1],
                        "category": r[2],
                        "importance": r[3],
                        "created_at": r[4],
                        "expires_at": r[5]
                    })
                return results
        finally:
            conn.close()

    def format_memory_prompt(self, user_id: str, max_tokens: int = 256) -> str:
        """
        将有效记忆格式化为大模型 System Prompt 提示词片段
        """
        memories = self.get_effective_memories(user_id)
        if not memories:
            return ""
            
        lines = ["【用户长期与当前生理记忆上下文】:"]
        for m in memories:
            lines.append(f"- [{m['category'].upper()}] {m['key']}: {m['content']}")
        return "\n".join(lines)
