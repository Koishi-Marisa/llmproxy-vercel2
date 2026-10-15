# 任务：让反代的 `/gemini` 支持 Gemini 原生「Grounding with Google Search」

## 背景（你看不到我们的对话，这里是全部上下文）

- 调用方是一台树莓派上的 Koishi 机器人（Docker 容器 `koishi`，宿主机 `/home/pi/qqbot/koishi-data`）。
- 机器人里有个本地转发插件 `koishi-plugin-llm-fallback`，对外暴露 `http://127.0.0.1:5140/llm-fallback/v1`，模型名统一是 `xiaoyu`，内部按 **gemini → groq → openrouter → zhipu** 顺序做上游回退。
- gemini 这一档现在指向你的反代：`https://ai.xiaowan.dpdns.org/gemini/chat/completions`，模型 `gemini-3.5-flash-lite`，鉴权是 `Authorization: Bearer <AQ. 开头的 key>`（key 在部署方手里，不要外泄、不要写进日志或错误响应）。
- **目标**：让 gemini 这一档能用到 **Grounding with Google Search**——搜索由 Google 服务端完成，模型直接带着检索结果和引用回答。这样调用方就不需要本地跑 web_search 工具，也不会有 `tool_calls` 往返（现在这条链路因为 `thought_signature` 问题根本走不通工具回合）。

## 现状（我实测的，2026-10-16，从调用方这台机器发起）

| 请求 | 结果 |
|---|---|
| `POST /gemini/chat/completions` + `"tools":[{"google_search":{}}]` | **HTTP 200，但工具被静默忽略**：问「今天有什么科技新闻」，回的是 2024 年的旧事（"OpenAI 刚发布 GPT-4o…"），`usage.prompt_tokens=14`，响应里没有任何引用/接地字段 |
| `POST /gemini/chat/completions` + `"tools":[{"type":"google_search","google_search":{}}]` | **HTTP 500** |
| `POST /gemini/v1beta/models/gemini-3.5-flash-lite:generateContent` | **HTTP 404** `{"detail":"Not Found"}` |
| `GET /gemini/v1beta/models` | **HTTP 404** |
| `GET /gemini/models` | `Platform 'gemini' not supported. Supported: cerebras, groq, mistral, nvidia, openai, sambanova`（说明 gemini 是特例路由，不是一等平台） |

结论：这个反代目前只有 OpenAI 兼容层，**不支持搜索接地**；传了接地工具还会假装成功——这是最糟糕的失败模式。

## 请实现（方案 A 或 B，二选一即可，都做更好）

### 方案 A（推荐，最省事）：原生 API 透传
- 新增 `POST /gemini/v1beta/models/{model}:generateContent`（可选再加 `:streamGenerateContent`），把请求体**原样**转发到 `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`，**必须保留 `tools:[{"google_search":{}}]`**。
- 鉴权：接受 `x-goog-api-key: <反代key>` 或 `Authorization: Bearer <反代key>`，服务端替换成真正的 Google key。
- 响应：原样返回 Google 的 JSON，**必须包含 `candidates[0].groundingMetadata`**（`webSearchQueries` / `groundingChunks[].web.{uri,title}` / `groundingSupports`）。

### 方案 B：让 OpenAI 兼容层支持接地
- `POST /gemini/chat/completions` 里的 `"tools":[{"google_search":{}}]` 翻译成原生 `google_search` 工具，由 Google 服务端执行。
- 引用怎么返回：来源以 markdown 链接形式内联/追加进 `choices[0].message.content`，并**额外**挂一个非标准字段 `choices[0].message.grounding_metadata`（原始 `webSearchQueries` + `groundingChunks`）。两种形式至少给一种，且在交付物里写清是哪种。

### 两种方案都必须遵守
**不支持时要明确报错**（4xx + 可读的 error message）。绝不能像现在这样 200 + 静默忽略：调用方会以为搜索生效了，最后拿着模型编的答案去回复用户。

## 不能破坏的现有调用方

- `/gemini/chat/completions` 非流式与流式都要继续可用（ChatLuna 侧配置 `nonStreaming: true`，但别把流式弄坏）。
- **视觉**：`messages[].content` 里 `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}` 必须继续工作——群里识图全靠它。
- 响应结构保持 OpenAI 形状：`choices[0].message.content`、`finish_reason`、`usage`。
- 现有字段不要改名、不要动语义。

## 自测与验收（请把原始输出贴回来，key 打码）

```bash
KEY='<反代里的 gemini key>'
BASE='https://ai.xiaowan.dpdns.org/gemini'

# 1) 有搜索：必须带 groundingMetadata，且内容与「今天」对得上
curl -s "$BASE/v1beta/models/gemini-3.5-flash-lite:generateContent" \
  -H "x-goog-api-key: $KEY" -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"今天有什么重要的科技新闻？两句话，必须是最新的"}]}],"tools":[{"google_search":{}}]}'

# 2) 对照组：同样问题但不带 tools —— 应当没有 groundingMetadata
curl -s "$BASE/v1beta/models/gemini-3.5-flash-lite:generateContent" \
  -H "x-goog-api-key: $KEY" -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"今天有什么重要的科技新闻？两句话"}]}]}'

# 3) 老接口没坏
curl -s "$BASE/chat/completions" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.5-flash-lite","messages":[{"role":"user","content":"说一句你好"}],"stream":false}'
```

验收标准：
1. 第 1 条 **HTTP 200 且 `candidates[0].groundingMetadata.webSearchQueries` 非空**——这是「原生搜索真的生效」的唯一硬证据；
2. 第 2 条没有 `groundingMetadata`（证明差异确实来自搜索）；
3. 第 3 条依然 200；
4. 附一句说明：**哪些模型名支持接地**、免费额度/RPM/RPD 限制、超额时返回什么（必须是明确的 429/4xx，不能静默降级成不搜）。

参考文档：
- Grounding with Google Search（原生）：https://ai.google.dev/gemini-api/docs/google-search
- Gemini 的 OpenAI 兼容层：https://ai.google.dev/gemini-api/docs/openai
- 注意官方说明：**Grounding with Google Search 只在 Gemini 3 及更新的模型上可用**（我们用的 `gemini-3.5-flash-lite` 属 Gemini 3 系，没问题）。

## 坑（我踩过的，别重复）

- **Node ≥20 的 Happy Eyeballs**：这台树莓派没有 IPv6 路由，Node 默认 `autoSelectFamily=true` 会让访问 Cloudflare 系域名 `ETIMEDOUT`（同一时刻 curl 正常）。凡是跑在这台机器上的 Node 进程/测试脚本，都要在启动时 `require('net').setDefaultAutoSelectFamily(false)`。
- 这台机器在国内网络，**直连 `generativelanguage.googleapis.com` 不一定通**——所以接地必须做在**反代这一侧**（你那边已有通到 Google 的线路），不要指望调用方直连 Google。
- 免费额度优先：这是个人自用机器人，**只用免费档**；额度用尽要报错，不要偷偷变成「不搜」。
- 顺带排除一个备选：Groq 的 compound / compound-mini（内置 web search）已于 2026-08-24 宣布弃用、2026-09-21 下线，不要再把它当替代方案。
- 别把 key 打进日志、错误体或返回体。

## 交付物

1. 最终可用的调用方式：完整 URL + 请求体示例 + 一段真实的 `groundingMetadata` 返回原文（key 打码）。
2. 一句结论：**调用方该用哪个路径、怎么表达「我要接地」**（例如「native 路径 + `tools:[{"google_search":{}}]`，其它参数都不用传」）。
3. 支持的模型名与额度限制说明。
4. 没做/做不了的部分及原因。
