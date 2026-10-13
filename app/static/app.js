// 旅行助手前端：调用 /api/v1/chat/stream（SSE），流式展示回答（含轻量 Markdown 渲染）。
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const streamToggle = document.getElementById("streamToggle");
const newSessionBtn = document.getElementById("newSession");
const clearMemoryBtn = document.getElementById("clearMemory");
const tokenInput = document.getElementById("tokenInput");

// 访问口令：可用 ?token=xxx 打开一次（会自动记住并清理地址栏）
const urlToken = new URLSearchParams(location.search).get("token");
if (urlToken) {
  localStorage.setItem("travel_token", urlToken);
  history.replaceState(null, "", location.pathname);
}
tokenInput.value = localStorage.getItem("travel_token") || "";
tokenInput.addEventListener("change", () => {
  localStorage.setItem("travel_token", tokenInput.value.trim());
});

function authHeaders() {
  const token = (tokenInput.value || "").trim();
  return token ? { "X-Access-Token": token } : {};
}

// 会话 id：同一页面持续复用以获得多轮记忆
let sessionId = newId();

// 用户 id：跨会话长期记忆（存 localStorage，刷新页面、开新会话都还在）
let userId = localStorage.getItem("travel_user_id");
if (!userId) {
  userId = newId();
  localStorage.setItem("travel_user_id", userId);
}

function newId() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : "sess-" + Math.random().toString(36).slice(2) + Date.now();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

function addUser(text) {
  const msg = el("div", "msg user");
  msg.appendChild(el("div", "bubble", text));
  chat.appendChild(msg);
  scrollToBottom();
}

function addAssistantBubble(placeholder) {
  const msg = el("div", "msg assistant");
  const bubble = el("div", "bubble", placeholder || "");
  const meta = el("div", "meta");
  msg.appendChild(bubble);
  msg.appendChild(meta);
  chat.appendChild(msg);
  scrollToBottom();
  return { bubble, meta };
}

// ---------- 轻量 Markdown 渲染（先转义再拼，避免 XSS） ----------
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderInline(text) {
  return text
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );
}

function renderMarkdown(md) {
  const lines = escapeHtml(md).split("\n");
  let html = "";
  let list = null; // "ul" | "ol" | null
  const closeList = () => {
    if (list) {
      html += `</${list}>`;
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    let m;

    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
      closeList();
      const level = Math.min(m[1].length + 2, 6);
      html += `<h${level}>${renderInline(m[2])}</h${level}>`;
      continue;
    }
    if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) {
      if (list !== "ul") {
        closeList();
        html += "<ul>";
        list = "ul";
      }
      html += `<li>${renderInline(m[1])}</li>`;
      continue;
    }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      if (list !== "ol") {
        closeList();
        html += "<ol>";
        list = "ol";
      }
      html += `<li>${renderInline(m[1])}</li>`;
      continue;
    }
    if (!line.trim()) {
      closeList();
      continue;
    }
    closeList();
    html += `<p>${renderInline(line)}</p>`;
  }
  closeList();
  return html;
}

// ---------- 元信息：意图标签 + 引用来源 ----------
function renderMeta(metaEl, data) {
  metaEl.innerHTML = "";
  const chips = el("div", "chips");
  if (data.intent) chips.appendChild(el("span", "chip", "意图: " + data.intent));
  if (typeof data.confidence === "number") {
    chips.appendChild(el("span", "chip", "置信度: " + data.confidence));
  }
  if (data.source) chips.appendChild(el("span", "chip", "来源: " + data.source));
  if (chips.childElementCount) metaEl.appendChild(chips);

  const citations = data.citations || [];
  if (citations.length) {
    const box = el("div", "citations");
    box.appendChild(el("div", "citations-title", "参考资料"));
    citations.forEach((c) => {
      box.appendChild(el("div", "citation", `• ${c.title}（得分 ${c.score}）`));
    });
    metaEl.appendChild(box);
  }
}

// 非流式：一次拿完整回答
async function sendOnce(text, bubble, meta) {
  const res = await fetch("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ message: text, session_id: sessionId, user_id: userId }),
  });
  if (res.status === 401) throw new Error("口令错误或未填写（请在顶部「口令」里填写）");
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "HTTP " + res.status);
  }
  const data = await res.json();
  bubble.innerHTML = renderMarkdown(data.reply || "(空回复)");
  renderMeta(meta, data);
}

// 流式：SSE 逐字返回
async function sendStream(text, bubble, meta) {
  const res = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ message: text, session_id: sessionId, user_id: userId }),
  });
  if (res.status === 401) throw new Error("口令错误或未填写（请在顶部「口令」里填写）");
  if (!res.ok || !res.body) throw new Error("HTTP " + res.status);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let started = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, sep).trim();
      buffer = buffer.slice(sep + 2);
      if (!raw.startsWith("data:")) continue;
      let event;
      try {
        event = JSON.parse(raw.slice(5).trim());
      } catch {
        continue;
      }

      if (event.type === "status") {
        if (!started) bubble.textContent = "…" + event.text;
      } else if (event.type === "delta") {
        started = true;
        answer += event.text;
        bubble.innerHTML = renderMarkdown(answer);
        scrollToBottom();
      } else if (event.type === "done") {
        renderMeta(meta, event);
      } else if (event.type === "error") {
        answer += "\n\n[错误] " + (event.error || "");
        bubble.innerHTML = renderMarkdown(answer);
      }
    }
  }
  if (!started) bubble.textContent = "(空回复)";
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  autosize();
  addUser(text);

  const { bubble, meta } = addAssistantBubble("连接中…");
  sendBtn.disabled = true;
  try {
    if (streamToggle.checked) {
      await sendStream(text, bubble, meta);
    } else {
      await sendOnce(text, bubble, meta);
    }
  } catch (e) {
    bubble.textContent = "请求失败：" + e.message;
  } finally {
    sendBtn.disabled = false;
    scrollToBottom();
  }
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
input.addEventListener("input", autosize);

newSessionBtn.addEventListener("click", () => {
  sessionId = newId();
  chat.innerHTML = "";
  const msg = el("div", "msg assistant");
  const bubble = el("div", "bubble");
  bubble.innerHTML =
    "已开启新会话。可以帮你规划行程、估算花费、查签证与出行攻略。<br />" +
    "试试问：<b>去日本旅游要准备什么签证材料</b>";
  msg.appendChild(bubble);
  chat.appendChild(msg);
});

clearMemoryBtn.addEventListener("click", async () => {
  chat.innerHTML = "";
  const msg = el("div", "msg assistant");
  const bubble = el("div", "bubble");
  try {
    await fetch("/api/v1/profile/" + encodeURIComponent(userId), {
      method: "DELETE",
      headers: authHeaders(),
    });
    bubble.textContent = "已清除长期记忆，之后我不会再记得你之前告诉我的偏好。";
  } catch (e) {
    bubble.textContent = "清除失败：" + e.message;
  }
  msg.appendChild(bubble);
  chat.appendChild(msg);
});
