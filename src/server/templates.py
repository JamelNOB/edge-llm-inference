"""
HTML Web Console Template
"""
HTML_CONSOLE_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Edge LLM High-Performance Streaming Console</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; height: 100vh; }
    header { background: #1e293b; padding: 14px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; }
    .title-area h1 { font-size: 16px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
    .badge { background: #0369a1; color: #e0f2fe; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
    #chat-container { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px; }
    .msg { max-width: 80%; padding: 12px 16px; border-radius: 10px; font-size: 14px; line-height: 1.6; word-break: break-word; white-space: pre-wrap; }
    .msg.user { align-self: flex-end; background: #2563eb; color: #fff; }
    .msg.bot { align-self: flex-start; background: #1e293b; border: 1px solid #334155; }
    .metrics { font-size: 11px; color: #94a3b8; margin-top: 6px; font-family: monospace; border-top: 1px dashed #334155; padding-top: 4px; }
    footer { background: #1e293b; padding: 16px 24px; border-top: 1px solid #334155; }
    .input-box { display: flex; gap: 12px; }
    input[type="text"] { flex: 1; background: #0f172a; border: 1px solid #475569; border-radius: 8px; padding: 12px 16px; color: #fff; font-size: 14px; outline: none; }
    input[type="text"]:focus { border-color: #38bdf8; box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2); }
    button { background: #0284c7; color: #fff; border: 0; border-radius: 8px; padding: 0 20px; font-weight: 600; cursor: pointer; transition: background .15s; }
    button:hover { background: #0369a1; }
  </style>
</head>
<body>
  <header>
    <div class="title-area">
      <h1>Edge LLM Streaming Console <span class="badge">Qwen2.5-1.5B Q4_K_M</span></h1>
    </div>
    <div style="font-size:12px; color:#94a3b8;">FastAPI + llama.cpp C++ Engine (SSE)</div>
  </header>
  <div id="chat-container">
    <div class="msg bot">你好！我是运行在你本地端侧的轻量化大语言模型 (Qwen2.5-1.5B INT4)。支持毫秒级流式打字机响应与 KV Cache 预热，请输入你的问题体验端侧推理！</div>
  </div>
  <footer>
    <div class="input-box">
      <input type="text" id="userInput" placeholder="输入问题测试流式响应与 TTFT / 吞吐性能..." onkeydown="if(event.key==='Enter') sendMsg()" />
      <button onclick="sendMsg()">发送 (Send)</button>
    </div>
  </footer>
  <script>
    async function sendMsg() {
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';

      const container = document.getElementById('chat-container');
      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.textContent = text;
      container.appendChild(userDiv);

      const botDiv = document.createElement('div');
      botDiv.className = 'msg bot';
      const textSpan = document.createElement('span');
      botDiv.appendChild(textSpan);
      container.appendChild(botDiv);
      container.scrollTop = container.scrollHeight;

      const t0 = performance.now();
      let ttft = null;
      let tokenCount = 0;

      const resp = await fetch('/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: text }],
          max_tokens: 512,
          stream: true
        })
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const raw = line.substring(6).trim();
            if (raw === '[DONE]') break;
            try {
              const data = JSON.parse(raw);
              const delta = data.choices[0]?.delta?.content || '';
              if (delta) {
                if (ttft === null) ttft = performance.now() - t0;
                tokenCount++;
                textSpan.textContent += delta;
                container.scrollTop = container.scrollHeight;
              }
            } catch(e) {}
          }
        }
      }

      const totalTimeSec = (performance.now() - t0) / 1000.0;
      const speed = tokenCount > 0 ? (tokenCount / totalTimeSec).toFixed(1) : 0;
      const metricsDiv = document.createElement('div');
      metricsDiv.className = 'metrics';
      metricsDiv.textContent = `⚡ TTFT: ${ttft ? ttft.toFixed(0) : 0}ms | Tokens: ${tokenCount} | Speed: ${speed} tokens/s`;
      botDiv.appendChild(metricsDiv);
    }
  </script>
</body>
</html>
"""
