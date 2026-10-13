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
- **运营数据看板**：每次对话落库（MySQL，本地回退 SQLite），统计调用量、意图/工具/命中通道分布、延迟分位与近 7 天趋势；Vue 3 管理后台（`/admin`）可视化，统计结果经 Redis 缓存。

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
      ├─ 工具分支 → tools (app/tools.py)       行程 / 预算
      │            → 抽取参数 → 调工具 → 润色成自然语言
      │
      └─ 每次调用落库 → store (app/store.py, MySQL)  ←→  admin (app/admin.py) + Vue 看板
                                        └─ Redis 缓存 (app/cache.py)
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
│   ├── store.py         # 对话日志存储与统计（SQLAlchemy，MySQL/SQLite）
│   ├── cache.py         # Redis 统计缓存（可选，未配置自动跳过）
│   ├── admin.py         # 运营看板接口
│   ├── auth.py          # 访问口令守卫
│   ├── json_utils.py    # 稳健解析模型返回的 JSON
│   └── static/          # 前端（聊天页 + admin.html 看板）
├── data/knowledge.json  # 知识库语料
├── scripts/
│   ├── crawler.py       # Playwright 爬虫：抓公开城市信息并回填数据库
│   └── requirements-crawler.txt
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
| `EMBED_MODEL` | 语义检索模型，默认 `BAAI/bge-small-zh-v1.5`；填 `none` 关闭语义通道 |
| `RERANK_MODEL` | 重排模型，默认 `BAAI/bge-reranker-base`；填 `none` 关闭重排（只用 RRF 融合） |
| `HF_ENDPOINT` | HuggingFace 镜像地址，国内建议 `https://hf-mirror.com` |
| `DATABASE_URL` | 对话日志库；留空回退 SQLite（本地零依赖）。compose 里默认指向内置 MySQL |
| `REDIS_URL` | 统计缓存；留空则不启用 |

> 对外部署时建议设置 `ACCESS_TOKEN`，否则任何人拿到地址都能免费用你的模型额度。
> 前端顶部有「口令」输入框；也可用 `http://<host>/?token=xxx` 打开一次（会自动记住）。

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

返回 `status`、`model`、`kb_size`、`vector` / `rerank`（两路是否启用）、`database` / `cache`、`auth_required`。

### `GET /api/v1/admin/stats`（需口令）

运营看板主数据：`total_calls` / `today_calls` / `sessions` / `users` / `intent_distribution` /
`tool_distribution` / `channel_distribution` / `latency_ms`（mean / p50 / p95 / max）/ `trend`（近 N 天，`?days=`）。
命中 Redis 缓存时 `cached` 为 `true`。

### `GET /api/v1/admin/logs`（需口令）

最近调用记录（倒序，`?limit=`），字段含意图、来源、工具、命中通道、延迟、会话。

> 管理后台页面：`http://<host>:8000/admin`（Vue 3，同样需要口令）。

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

## 自动化爬虫（RPA）

`scripts/crawler.py` 用 **Playwright** 抓取公开的城市旅行信息并**自动回填数据库**，
演示浏览器自动化 + 反爬应对 + 数据入库的完整闭环（依赖单独放置，不装进主镜像）：

```bash
pip install -r scripts/requirements-crawler.txt
playwright install chromium
python -m scripts.crawler --cities 京都 清迈 巴厘岛        # 抓取并入库
python -m scripts.crawler --cities 京都 --notify          # 抓完推送飞书
```

- **反爬应对**：伪装 UA / 语言头、请求间随机延迟、失败重试、遵守 `robots.txt`；
- **结构化抽取**：浏览器渲染后取正文，压缩空白；
- **自动回填**：经 `app.store` 写入 `crawl_items` 表（MySQL / SQLite），按 `城市+标题` 去重；
- **飞书通知**：配置 `FEISHU_WEBHOOK` 后推送本轮结果。

> 目标站点在脚本内可配置；请遵守目标站点条款与 robots.txt，仅用于学习。

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
- `docker compose up -d` 会一并启动 **mysql / redis** 两个容器（`mysql_data` / `redis_data` 命名卷持久化）。
  若 4G 内存与模型同跑吃紧，可删掉 compose 里的 `mysql` / `redis` 两个 service——app 会自动回退 SQLite、跳过缓存。
- 运营数据看板：`http://<公网IP>:8000/admin`（用同一口令）。
- 镜像以**非 root 用户**运行，并内置 `/api/v1/health` 健康检查与 `restart: unless-stopped`。
- 生产建议前置 Nginx + HTTPS，密钥通过环境变量注入（不要写进镜像）。

## 说明

个人学习与实践项目。行程与预算为粗略估算，签证等时效性信息请以官方最新要求为准。
