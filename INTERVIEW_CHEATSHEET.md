# 🎯 Edge-LLM-Inference 代码防穿帮面试速记卡

> **核心宗旨**：专为技术面试深挖准备，彻底对齐代码实现与简历话术。杜绝“简历吹自研底层被问死”与“代码搜不到关键词被当场拆穿”。

---

## 🧭 一、 代码锚点速查表 (Ctrl+F 检索对照)

面试官若让你共享屏幕或检查代码仓库，直接带面试官看以下核心文件：

| 简历核心技术点 (Claim) | 对应源码文件位置 | 核心函数 / 类名 | 核心考点与防穿帮要点 |
| :--- | :--- | :--- | :--- |
| **KV Cache 预热流水线** | [`src/engine/kv_cache.py`](src/engine/kv_cache.py) | `warmup_kv_cache()` | 证明预热的是注意力 KV 张量而非 RAG 知识库，启动时仅执行 `eval()` 前向计算。 |
| **Q4_K_M 混合精度与带宽突破** | [`src/core/config.py`](src/core/config.py) | `DEFAULT_MODEL_PATH`<br>`DEFAULT_CPU_THREADS` | 源码头部有详尽架构注释：解释端侧为内存带宽受限 (Memory Bandwidth Bound)。 |
| **Agent 分层记忆与 TTL 淘汰** | [`src/memory/agent_memory.py`](src/memory/agent_memory.py) | `AgentMemoryEngine`<br>`set_memory()`<br>`get_effective_memories()` | 证明通过 SQLite 惰性过期实现短时状态淘汰，杜绝历史状态污染长期用户画像。 |
| **Slot UPSERT 冲突覆写** | [`src/memory/agent_memory.py`](src/memory/agent_memory.py) | `UNIQUE(user_id, key)`<br>`INSERT OR REPLACE` | 证明当用户喜好矛盾时，通过原子替换覆写旧记录，杜绝模型人格分裂。 |
| **C++ 底层硬件压测** | [`scripts/benchmark_llama_cpp.sh`](scripts/benchmark_llama_cpp.sh)<br>[`benchmarks/llama_bench_report.md`](benchmarks/llama_bench_report.md) | `llama-bench` 指标 | 实测给出 Prefill 186.69 t/s 与 Decode 26.56 t/s 的真实硬件跑分。 |
| **端侧双频异步视频质检** | [`run_video_coach.py`](run_video_coach.py)<br>[`src/vision/motion_analyzer.py`](src/vision/motion_analyzer.py) | `SquatStateMachine`<br>`VideoPostureCoachPipeline` | 证明高频感知环 (30 FPS) 与低频认知环 (1 Hz 异步线程) 解耦，基于 FSM 触底拐点与代偿阈值事件触发，视频 0 卡顿。 |
| **异步流式打字机网关** | [`src/server/routes.py`](src/server/routes.py) | `chat_stream()` | 基于 FastAPI 与 `EventSourceResponse` (SSE) 实现单向流式推送与 ChatML 解析。 |

---

## 🛡️ 二、 一句话免死金牌说辞 (划清主理人工程边界)

> **面试官追问底层细节时（如：“底层 CUDA Kernel 是怎么写的？SIMD 汇编怎么排布的？”）：**  
> 
> 🗣️ **“我们项目定位于端侧应用与推理网关架构主理人。底层 GGUF 反量化与 CPU 矩阵计算依托于开源成熟的 llama.cpp C++ 底座；我的精力与核心自研工作聚焦在【端侧内存带宽瓶颈量化建模】、【Q4_K_M 混合精度量化选型】、【KV Cache 开机预热流水线】、【分层 Agent 记忆治理与冲突消解】以及【异步流式 Web 服务编排】，在保障生产级稳定性的同时避免重复造底层算子轮子。”**

---

## ⚡ 三、 七大高频技术鱼钩与【双轨答辩剧本】

### 🐟 鱼钩 1：为什么模型压缩后在 CPU 上反而变快（6.2 -> 26+ tokens/s）？
- **数据出处**：`README.md` 实测基准表（FP16: 3.09GB / 6.2 T/s；Q4_K_M: 934MB / 26.56 T/s）。
- **【轨道 A：脑内物理直觉】**：
  > CPU 推理就像搬砖。CPU 的小口袋（L3 缓存）只有几十 MB，放不下整个模型，模型全在主板内存里。每吐一个字都得把全量模型从内存“马路”运进 CPU 一遍。原版 3.09GB 太胖，马路直接塞车（带宽堵死）；量化瘦身到 934MB 后马路通畅了，一秒钟能搬更多趟，吐字自然暴涨 4.2 倍。
- **【轨道 B：大厂专业得分词】**：
  > **得分关键词**：`内存带宽受限 (Memory Bandwidth Bound)`、`自回归逐 Token 遍历访存`、`总线流量削减`。  
  > “端侧 CPU 推理的核心瓶颈在于内存总线带宽而非算力。自回归模型每生成一个 Token 必须全量读取一次模型参数矩阵。原版 FP16 占满了带宽导致算力饥饿；Q4_K_M 量化将权重体积削减 70.3%，直接解除了访存总线拥塞，使得流式吞吐释放至 26.56 tokens/s。”

---

### 🐟 鱼钩 2：为什么选 Q4_K_M 量化，没有变成“弱智模型”？
- **【轨道 A：脑内物理直觉】**：
  > 保核心、砍外围。好比写文章，中心思想和关键词必须字斟句酌（保留 6-bit 高精度），而套话副词直接压成 4-bit 节省空间。
- **【轨道 B：大厂专业得分词】**：
  > **得分关键词**：`混合精度量化 (Mixed-Precision)`、`Attention 注意力保留 6-bit`、`FFN 前馈层压缩 4-bit`、`困惑度 (PPL) 损失几乎可忽略`。  
  > “Q4_K_M 采用了非均匀的混合精度策略：决定长文本长程逻辑与注意力关联的 Attention 层保留 6-bit；参数量巨大但对量化噪声容忍度高的 FFN 前馈网络深度压缩至 4-bit。实测困惑度（PPL）恶化小于 0.045，实现了精度与带宽的极致工程平衡。”

---

### 🐟 鱼钩 3：端侧推理时模型容易陷入“复读机死循环（Repetition Loop）”，为什么？怎么解决？
- **【轨道 A：脑内物理直觉】**：
  > 第一是没给答题卡：用户裸扔一句话，模型以为在玩成语接龙，进入自由瞎扯；第二是溜滑梯效应：一旦它吐出了第一个重复词，这个词就被写进了 KV Cache 记忆草稿纸，下一步模型看前面的词权重极高，马太效应越滚越大，彻底刹不住车！
- **【轨道 B：大厂专业得分词】**：
  > **得分关键词**：`ChatML 状态机模板对齐`、`KV Cache 累积注意力偏移`、`Repetition Penalty (重复惩罚)`、`EOS 动态截断`。  
  > “本质在于两点：1) 输入必须显式注入 `<|im_start|>` 和 `<|im_end|>` 结构化标签，让模型正确进入 Assistant 解码状态机；2) 解码阶段必须施加 Repetition Penalty（对已在 KV Cache 中的 Token logits 实施衰减）并监听 EOS 提前 break 退出前向循环。”

---

### 🐟 鱼钩 4：Agent 记忆系统如何解决“用户喜好变更”导致的事实冲突？
- **【轨道 A：脑内物理直觉】**：
  > 手机通讯录改电话号码。张三告诉你新号码，你直接用橡皮擦把旧号码擦掉、把新号码写上去（覆写 Update）。如果建两个张三，打电话时系统肯定人格分裂！
- **【轨道 B：大厂专业得分词】**：
  > **得分关键词**：`Slot 槽位化提取`、`UNIQUE 约束`、`UPSERT (INSERT OR REPLACE) 原子覆写`。  
  > 对应代码：`src/memory/agent_memory.py:set_memory()`。  
  > “在表结构设计中定义 `UNIQUE(user_id, key) ON CONFLICT REPLACE`。当用户状态发生变更（如‘喜欢吃辣’变为‘胃溃疡禁辣’），以相同的 key 触发原子覆写，保证库内只留存唯一最新的真实事实，杜绝矛盾信息并存污染大模型 Prompt。”

---

### 🐟 鱼钩 5：Prefill 阶段与 Decode 阶段的计算特征有什么本质区别？
- **【轨道 A：脑内物理直觉】**：
  > Prefill（读题）是学生一口气读完整张卷子，所有字都在眼前，可以拉上所有兄弟一起分工算矩阵；Decode（答题）是闭着眼睛按规律一个字一个字往外挤，每次挤一个字都得把整本字典搬出来看一眼！
- **【轨道 B：大厂专业得分词】**：
  > **得分关键词**：`Prefill 为计算密集型 (Compute-Bound)`、`Decode 为访存受限 (Memory-Bound)`、`GQA (分组查询注意力)`。  
  > “Prefill 阶段由于所有输入 Token 均已知，矩阵乘法可充分并行化（GEMM），算力利用率高（实测达 186.69 t/s）；Decode 阶段每次仅生成单 Token（GEMV），算力处于饥饿等待状态，系统吞吐完全受制于从内存搬运权重的总线带宽。”

---

### 🐟 鱼钩 6：KV Cache 预热到底预热了什么？（严禁混淆 RAG 知识库！）
- **【轨道 A：脑内物理直觉】**：
  > 厨师营业前先把葱姜蒜切好备在案板上（留好注意力草稿纸），客人进门点菜直接下锅开炒。绝对不是在脑子里死背一本外部百科全书！
- **【轨道 B：大厂专业得分词】**：
  > **得分关键词**：`键值张量常驻 (KV Tensors)`、`System Prompt 仅 eval 前向计算`、`增量追加计算 (Incremental Computation)`、`TTFT 压降至 100ms 以内`。  
  > 对应代码：`src/engine/kv_cache.py:warmup_kv_cache()`。  
  > “预热存的不是 RAG 外部知识，而是固定不变的系统提示词（System Prompt）在注意力网络中计算产生的 Key 和 Value 矩阵。开机时显式执行无损 `eval` 前向计算并锁定在内存；用户首个请求到达时直接从系统 Prompt 之后进行增量前向追加，首字延迟（TTFT）从冷启动的 ~650ms 压降至 100ms 左右。”

---

### 🐟 鱼钩 7：为什么流式传输选 SSE 而不用 WebSocket？
- **【轨道 A：脑内物理直觉】**：
  > 大模型文字输出是单向广播打字机，不是双向打电话。单向广播用最轻的喇叭（SSE）就行，没必要架设双向全双工电话线（WebSocket）。
- **【轨道 B：大厂专业得分词】**：
  > **得分关键词**：`单向事件流 (Server-Sent Events)`、`原生基于 HTTP/1.1 与 HTTP/2`、`浏览器原生断线重连`、`避免双向握手与心跳保活开销`。  
  > “大模型文本补全本质是‘单请求、多数据块’的单向输出流。SSE 原生基于 HTTP，轻量透明，天然支持浏览器断线重连；而 WebSocket 是全双工双向协议，在此场景下存在协议升级与心跳开销的过度设计（除非是实时双向多模态语音打断 Barge-in 场景才需引入 WebSocket）。”

---

### 🐟 鱼钩 8：端侧视频流 30 FPS，端侧大模型每秒才生成 26 tokens，一句话要 2 秒，你的视频流怎么做到不卡死的？
- **【轨道 A：脑内物理直觉】**：
  > 就像高速公路的“雷达测速探头与交警开罚单”：
  > 探头（OpenCV + 几何力学计算）每秒抓拍 30 张，雷达测速飞快（<0.5ms）；交警（端侧 Qwen 大模型）绝不在每张抓拍照片上都开一张罚单，而是在车辆“严重超速（膝内扣突增）”或者“通过收费站最低点（下蹲触底拐点）”时，才抓出一张交给交警去写罚单（异步线程推理）。车流（前台视频流）依然 30 FPS 丝滑畅行，罚单写好后往屏幕底部牌子上一贴即可！
- **【轨道 B：大厂专业得分词】**：
  > **得分关键词**：`双频异构解耦 (Dual-Rate Heterogeneous Decoupling)`、`有限状态机极值拐点检测 (FSM Inflection Detection)`、`事件驱动非阻塞队列 (Event-Driven Non-blocking Queue)`、`异步 HUD 双缓冲回写`。  
  > 对应代码：`run_video_coach.py` 与 `src/vision/motion_analyzer.py:SquatStateMachine`。  
  > “该架构将实时质检解耦为两层：
  > 1) **高频感知环 (30 Hz / <5ms)**：OpenCV 帧获取 + 纯向量内积几何解算 + 骨骼 HUD 实时光栅化，保证前台 30 FPS 零掉帧；
  > 2) **状态机事件驱动 (Event-Driven)**：绝不按帧轮询大模型，而是通过有限状态机追踪角速度拐点（`BOTTOM_PEAK` 触底极值点）与连续内扣防抖阈值；
  > 3) **低频认知环 (0.5~1 Hz)**：仅在拐点事件触发时，向后台 Worker 线程无锁投递 Prompt 任务，端侧 934MB Qwen 模型完成推理后原子回写字幕槽，实现高低频无缝协作。”

