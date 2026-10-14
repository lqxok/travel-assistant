# 旅行助手（Travel Assistant）

个人旅行 AI Agent：对话式帮你**规划行程、估算预算、查询签证与出行攻略**。

核心是一条「**意图路由**」链路：先判断你在问什么，再决定是去**本地知识库检索**、还是**调用旅行工具**，最后有依据地生成回答——而不是所有问题都丢给大模型自由发挥。

## 功能

- **意图识别**：规则快车道（正则，零成本低延迟）+ LLM 慢车道（拿不准时才调用），两路融合。
- **本地检索（RAG）**：**词面（中文二元组 + IDF 加权）+ 意图 + 语义**三路并行召回，用 **RRF 融合**，再用 **cross-encoder 重排**；攻略/签证类问题基于检索结果作答，并回传**参考资料**与命中通道。语义/重排用本地 ONNX 模型（`bge-small-zh` / `bge-reranker-base`），模型不可用时自动降级为词面检索。
- **旅行工具**：行程草稿（逐日骨架 + 费用）、预算估算（总额/人均/构成），参数用 Pydantic 校验；信息不全时**主动追问**。
- **健壮性**：模型偶发返回空内容会重试；调用失败有指数退避。
- **Web 界面**：原生 HTML/CSS/JS 聊天界面，展示回答、意图标签与参考资料；支持**流式开关**与新建会话。
- **多轮会话**：传入相同 `session_id` 即带上历史对话，支持「那改成 5 天」这类追问。
- **用户长期记忆**：按 `user_id` 记住稳定偏好/事实（如常驻城市、饮食忌口），**跨会话生效**并落盘保存。
- **生产化**：每个请求分配**追踪 ID**（响应头 `X-Request-ID`，贯穿全部日志）；支持**结构化 JSON 日志**（`LOG_FORMAT=json`）；**接口限流**（滑动窗口，超限 429，健康检查豁免）。
- **访问控制**：支持**访问口令**（`X-Access-Token`）保护接口；也可开启**演示模式**免口令对外开放，并用「每 IP 每日配额」把模型成本限死。

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
      ├─ 检索分支 → KnowledgeBase (app/rag.py)  词面 / 意图 / 语义 三路召回
      │            → RRF 融合 → cross-encoder 重排（app/embedder.py）
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
│   ├── rag.py           # 本地知识库检索（词面 + 意图 + 语义 → RRF 融合 → 重排）
│   ├── embedder.py      # 本地向量/重排模型封装（fastembed，懒加载 + 降级）
│   ├── tools.py         # 旅行工具（行程 / 预算）
│   ├── agent.py         # 编排：意图 → 检索 / 工具 / 直接回答
│   ├── observability.py # 请求追踪 ID / 结构化日志 / 限流
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
| `ACCESS_TOKEN` | 访问口令；填了之后调用接口需带请求头 `X-Access-Token`（留空=不校验） |
| `DEMO_MODE` | 演示模式：开启后**免口令**对外开放；会把 demo 状态写进 `/api/v1/health` |
| `DEMO_DAILY_LIMIT` | 演示模式下每个 IP 每天的聊天次数上限，默认 20；设 `0` 不限 |
| `EMBED_MODEL` | 语义检索模型，默认 `BAAI/bge-small-zh-v1.5`；填 `none` 关闭语义通道 |
| `RERANK_MODEL` | 重排模型，默认 `BAAI/bge-reranker-base`；填 `none` 关闭重排（只用 RRF 融合） |
| `HF_ENDPOINT` | HuggingFace 镜像地址，国内建议 `https://hf-mirror.com` |
| `RATE_LIMIT_PER_MINUTE` | 每客户端每分钟限流次数，默认 60；设 `0` 关闭 |
| `LOG_FORMAT` | 日志格式：`text`（可读）/ `json`（结构化） |

> 对外部署时建议设置 `ACCESS_TOKEN`，否则任何人拿到地址都能免费用你的模型额度。
> 前端顶部有「口令」输入框；也可用 `http://<host>/?token=xxx` 打开一次（会自动记住）。

**想放公网 demo（别人点开就能用）？** 开启演示模式即可：

```bash
DEMO_MODE=1
DEMO_DAILY_LIMIT=20   # 每个 IP 每天最多 20 次聊天
```

演示模式下**不再校验口令**，但聊天接口按 **IP + 24 小时滚动窗口** 计数，超出即返回 `429`；
`/api/v1/health` 与首页不计入配额（容器探针与页面访问不受影响）。
这样既能让人直接体验，又能把 API key 的成本**限死在每日上限内**。

**自己人不想被限？** 演示模式下 `ACCESS_TOKEN` 仍然有效：请求带对
`X-Access-Token`（或 `?token=`）的**不计每日配额**——陌生人受限，管理员不限。
前端顶部的口令框**默认留空**，填入正确口令即切换为不受限模式。

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
  "citations": [{ "doc_id": "kb-visa", "title": "签证与证件准备", "score": 0.9, "channel": "rerank" }],
  "tool_result": null
}
```

- `intent`：`trip_planning` / `budget` / `guide` / `transport` / `stay` / `general`
- `source`：`rule`（规则命中）/ `llm`（慢车道）/ `hybrid`（融合）
- `citations`：RAG 命中的参考资料（`channel` 为命中通道：`keyword` / `intent` / `vector` / `rerank`）
- `tool_result`：调用工具时的结构化结果

传入相同 `session_id` 可获得**多轮记忆**（进程内存，最多保留最近 6 轮）。
传入 `user_id` 可获得**跨会话长期记忆**（落盘于 `data/profiles.json`，重启不丢）。

### `DELETE /api/v1/profile/{user_id}`

清除某个用户的长期记忆。

### `GET /api/v1/health`

返回 `status`、`model`、`kb_size`、`vector` / `rerank`（两路是否启用）、`auth_required`、`rate_limit_per_minute`。

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

## 评测

`evals/` 下有一套评测用例与指标：

```bash
python -m evals.run_evals --offline            # 离线指标（不调用模型）
python -m evals.run_evals --offline --semantic # 离线 + 语义通道 / 重排（需已下载模型）
python -m evals.run_evals                      # 离线 + 在线指标（需要 API Key）
python -m evals.run_evals --limit 3            # 在线部分每组只跑 3 条（省钱）
```

| 指标 | 是否需要模型 | 说明 |
|------|:---:|------|
| 规则意图覆盖 / 覆盖内准确率 | 否 | 快车道能覆盖多少、覆盖得对不对 |
| 检索 Recall@1 / Recall@3 | 否 | 期望资料是否被召回 |
| 整体意图准确率 | 是 | 含慢车道 |
| 工具调用准确率 | 是 | 是否调用了正确的工具 |
| 回答忠实度 | 是 | LLM 判官打分：是否只依据参考资料、不编造 |
| 延迟 mean / p50 / p95 | 是 | 端到端耗时 |

结果写入 `evals/report.json`。CI 会跑**离线评测**；`Evals` 工作流可手动触发**完整评测**（需配置 Secrets）。

## 生产化（可观测性与限流）

- **请求追踪**：每个请求分配 `request_id`（也接受上游传入的 `X-Request-ID`），
  放进 `ContextVar` 并写回响应头，于是**同一次请求的所有日志**都能通过它串起来。
- **结构化日志**：`LOG_FORMAT=json` 时每条日志输出一行 JSON（`ts/level/logger/request_id/message`），
  便于被 ELK、云日志直接采集；默认 `text` 便于本地调试。
- **限流**：单实例内存中的**滑动窗口**计数器，按客户端 IP 限制每分钟请求数，超限返回 `429`；
  `/api/v1/health` 豁免（供容器健康检查）。多实例部署时可换成 Redis 计数器。

```bash
LOG_FORMAT=json RATE_LIMIT_PER_MINUTE=120 uvicorn app.main:app --port 8000
```

## 部署（Docker）

```bash
# 1) 服务器安装 Docker（Ubuntu 示例）
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker

# 2) 拉代码
git clone https://github.com/lqxok/travel-assistant.git
cd travel-assistant

# 3) 配置密钥
cp .env.example .env
vim .env            # 填 OPENAI_API_KEY / OPENAI_MODEL 等

# 4) 启动（后台常驻）
docker compose up -d --build
docker compose logs -f

# 5) 更新版本时
git pull && docker compose up -d --build
```

- 访问 `http://<服务器公网IP>:8000/`；记得在云控制台**安全组放行 TCP 8000**。
- 首次构建会**预下载本地语义/重排模型**（约 1.1 GB，走 `HF_ENDPOINT` 镜像）；模型层在 `COPY app` 之前，
  之后改代码重建可命中缓存，不会重复下载。内存吃紧（如 4G 机器同时跑别的服务）可在 `.env` 里设
  `RERANK_MODEL=none` 只保留语义通道或 `EMBED_MODEL=none` 退回纯词面检索。
- `./data` 以卷挂载，用户长期记忆 `profiles.json` **持久化**，容器重建不丢。
- 镜像以**非 root 用户**运行，并内置 `/api/v1/health` 健康检查与 `restart: unless-stopped`。
- 生产建议前置 Nginx + HTTPS，密钥通过环境变量注入（不要写进镜像）。

## 说明

个人学习与实践项目。行程与预算为粗略估算，签证等时效性信息请以官方最新要求为准。
