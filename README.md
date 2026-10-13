# 旅行助手（Travel Assistant）

个人旅行 AI Agent：对话式帮你**规划行程、估算预算、查询签证与出行攻略**。

核心是一条「**意图路由**」链路：先判断你在问什么，再决定是去**本地知识库检索**、还是**调用旅行工具**，最后有依据地生成回答——而不是所有问题都丢给大模型自由发挥。

## 功能

- **意图识别**：规则快车道（正则，零成本低延迟）+ LLM 慢车道（拿不准时才调用），两路融合。
- **本地检索（RAG）**：关键词通道 + 意图通道并行召回、合并去重；攻略/签证类问题基于检索结果作答，并回传**参考资料**。
- **旅行工具**：行程草稿（逐日骨架 + 费用）、预算估算（总额/人均/构成），参数用 Pydantic 校验；信息不全时**主动追问**。
- **健壮性**：模型偶发返回空内容会重试；调用失败有指数退避。
- **Web 界面**：原生 HTML/CSS/JS 聊天界面，展示回答、意图标签与参考资料；支持**流式开关**与新建会话。
- **多轮会话**：传入相同 `session_id` 即带上历史对话，支持「那改成 5 天」这类追问。

## 架构

```
Browser UI (app/static)
      │  POST /api/v1/chat
      ▼
  FastAPI (app/main.py)
      │
      ▼
  TravelAgent (app/agent.py)  ← 编排
      ├─ IntentRecognizer (app/intents.py)   规则 + LLM 快慢车道
      │
      ├─ 检索分支 → KnowledgeBase (app/rag.py)  关键词 / 意图两路召回
      │            → LLM 依资料生成 + 返回引用
      │
      └─ 工具分支 → tools (app/tools.py)       行程 / 预算
                   → 抽取参数 → 调工具 → 润色成自然语言
      ▼
  LLMClient (app/llm.py)  OpenAI 兼容接口 + 重试
```

## 目录结构

```
travel-assistant/
├── app/
│   ├── main.py          # FastAPI 入口、路由
│   ├── config.py        # 配置（.env / 环境变量）
│   ├── llm.py           # LLM 客户端（OpenAI 兼容 + 重试）
│   ├── intents.py       # 意图识别（规则 + LLM 慢车道）
│   ├── rag.py           # 本地知识库检索（关键词 + 意图）
│   ├── tools.py         # 旅行工具（行程 / 预算）
│   ├── agent.py         # 编排：意图 → 检索 / 工具 / 直接回答
│   ├── json_utils.py    # 稳健解析模型返回的 JSON
│   └── static/          # 前端（index.html / styles.css / app.js）
├── data/knowledge.json  # 知识库语料
├── tests/               # pytest 测试
├── requirements.txt
└── .env.example
```

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # 填入 OPENAI_API_KEY 等
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/> 即可对话；接口文档在 <http://127.0.0.1:8000/docs>。

## 配置（.env）

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | 兼容 OpenAI 接口的密钥 |
| `OPENAI_BASE_URL` | 接口地址（如 `https://api.deepseek.com`） |
| `OPENAI_MODEL` | 模型名 |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT` | 生成参数 |

## API

### `POST /api/v1/chat`

```json
{ "message": "去日本旅游要准备什么签证材料" }
```

响应：

```json
{
  "reply": "……",
  "model": "deepseek-flash",
  "intent": "guide",
  "confidence": 0.9,
  "source": "rule",
  "citations": [{ "doc_id": "kb-visa", "title": "签证与证件准备", "score": 0.219 }],
  "tool_result": null
}
```

- `intent`：`trip_planning` / `budget` / `guide` / `transport` / `stay` / `general`
- `source`：`rule`（规则命中）/ `llm`（慢车道）/ `hybrid`（融合）
- `citations`：RAG 命中的参考资料
- `tool_result`：调用工具时的结构化结果

传入相同 `session_id` 可获得**多轮记忆**（进程内存，最多保留最近 6 轮）。

### `GET /api/v1/health`

返回 `status`、`model`、`kb_size`。

### `POST /api/v1/chat/stream`

流式对话，返回 `text/event-stream`，每行 `data: {JSON}`，事件类型：

- `status`：当前阶段（如「检索资料」「整理结果」）
- `delta`：增量正文
- `done`：结束，附带 `intent` / `confidence` / `source` / `citations` / `tool_result`
- `error`：出错信息

前端默认走这个接口，回答**逐字出现**。

## 测试

```bash
pytest
```

测试使用假 LLM，不会真实调用 API。

## 说明

个人学习与实践项目。行程与预算为粗略估算，签证等时效性信息请以官方最新要求为准。
