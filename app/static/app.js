// 旅行助手前端：调用 /api/v1/chat/stream（SSE），逐字展示回答，并显示意图与引用来源。
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

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

// 渲染意图标签与引用来源
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

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  autosize();
  addUser(text);

  const { bubble, meta } = addAssistantBubble("连接中…");
  sendBtn.disabled = true;
  let started = false;
  try {
    const res = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok || !res.body) throw new Error("HTTP " + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

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
          if (!started) {
            bubble.textContent = "";
            started = true;
          }
          bubble.textContent += event.text;
          scrollToBottom();
        } else if (event.type === "done") {
          renderMeta(meta, event);
        } else if (event.type === "error") {
          bubble.textContent += "\n[错误] " + (event.error || "");
        }
      }
    }
    if (!started && !bubble.textContent) bubble.textContent = "(空回复)";
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
