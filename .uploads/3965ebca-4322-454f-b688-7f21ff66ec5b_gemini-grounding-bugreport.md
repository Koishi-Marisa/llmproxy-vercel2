# Bug 报告：Gemini 原生搜索接地，线上验收不通过

你上一轮的改动我已经在**线上**复测了，5 个点里 **没一个生效**。下面是可复现的证据，请按「先确认部署」→「再修代码」的顺序处理。

## 0. 最重要的一条：线上跑的不是你改的那份代码

`GET https://ai.xiaowan.dpdns.org/openapi.json` 返回的路由表**和你改之前完全一样**，没有任何 `v1beta` / `generateContent` 路由：

```json
["/hello/", "/gemini/chat/completions", "/{platform}/v1/chat/completions",
 "/{platform}/chat/completions", "/chat/completions", "/v1/chat/completions",
 "/{platform}/models", "/"]
```

而且 `POST /gemini/v1beta/models/gemini-3.5-flash-lite:generateContent` 连打 3 次都是：

```
HTTP 404
{"detail":"Not Found"}
```

这个 404 是 **FastAPI 自己的 JSON 格式**（不是 nginx/Cloudflare 的 HTML 404），说明请求**已经进到了应用**，只是应用里没有这条路由 —— 也就是线上那个进程加载的代码里没有你的新端点。

不过有一个迹象表明**确实动过部署**：现在只带 `x-goog-api-key` 或完全不带鉴权时会返回一条新文案

```
HTTP 401 {"detail":"missing credential. Send 'Authorization: Bearer <your gemini api key>'."}
```

这条文案以前是没有的。所以请你确认并回答：

1. 线上那个服务是**哪个进程/容器/目录**，加载的文件是不是你改的 `/workspace/api/servers/gemini.py`？
2. 有没有**重新部署 + 重启**？如果重启过，是不是有第二份代码（比如打的镜像、另一份副本、或另一个 worker）覆盖了？
3. 你上一轮报告里说的 5 个点，哪些是**已经部署上线**的，哪些只是改了代码？请给出版本/提交标识。

## 1. 方案 A（原生透传）不生效

见上：404 ×3。

## 2. 方案 B（OpenAI 兼容层接地）不生效：200 但没有接地

```bash
KEY='<线上 gemini key>'
curl -s https://ai.xiaowan.dpdns.org/gemini/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.5-flash-lite","messages":[{"role":"user","content":"今天2026年10月16日有什么科技新闻？一句话，要最新的"}],"tools":[{"google_search":{}}],"stream":false}'
```

实测返回：

```
HTTP 200
顶层字段: ["id","object","created","model","usage","choices"]
message 字段: ["role","content"]          ← 没有 grounding_metadata
回答: 由于我的知识库更新截止到2026年1月，且无法获取此时此刻（2026年10月16日）的实时联网资讯，
      我无法为您提供今天的最新科技新闻。…
```

对照组（**不带** tools）回答：

```
抱歉，我无法提供今天的最新新闻，因为我的知识库截至2025年1月，无法获取2026年10月16日的实时信息。
```

换 `gemini-3.5-flash` + tools 也一样，回答「2026年10月16日是未来的日期，目前无法获取该日期的实时科技新闻。」

**两组回答都明确说自己没有实时联网能力，且 message 里没有任何 `grounding_metadata`、没有 markdown 来源列表** —— 也就是 `google_search` 依然没被真正交给 Google 执行（虽然现在是 200，不再像以前那样用 2024 年的旧闻假装成功）。

## 3. 一条很可能是 bug 的线索：带 `google_search` 时短暂 401

就在你们那次部署的前后，我抓到过这个（**同一路径、同一 key、无工具的请求同时是 200**）：

```
POST /gemini/chat/completions   + {"tools":[{"google_search":{}}]}
HTTP 401 {"error":{"message":"Invalid API Key","type":"invalid_request_error","code":"invalid_api_key"}}
```

同一个 body 去掉 `tools` 立刻 200；带上 tools 连续两次都是 401。我的怀疑：**走接地分支时，把「调用方传进来的反代 key」当成上游 Google 的 key 用了**（或者从 `x-goog-api-key` 取上游 key，而调用方只发了 Authorization）。请重点查这条分支的凭据来源，别用调用方的 key 去请求 Google。

## 4. 鉴权行为与约定不符

约定是「`x-goog-api-key` **或** `Authorization: Bearer` 都接受，服务端再换成真正的 Google key」。实测（普通请求）：

| 请求头 | 结果 |
|---|---|
| `Authorization: Bearer <key>` | 200 ✅ |
| `x-goog-api-key: <key>` | 401 `missing credential. Send 'Authorization: Bearer <your gemini api key>'` ❌ |
| 两个都带 | 200 ✅ |
| 都不带 | 401 ✅（符合预期） |

我们的调用方（ChatLuna 的 OpenAI-like 适配器）用的是 `Authorization: Bearer`，所以现在能用；但请让实现和文档一致，或者明确只支持 Bearer 并同步更新说明。

## 5. 顺手一起修：上游抖动很严重

同一个模型、同一个请求连打多次，结果随机：

| 模型 | 结果 |
|---|---|
| gemini-3.5-flash-lite（不带工具） | 500 / 200 / 500 |
| gemini-3.5-flash-lite（带工具） | 500 / 500 / 200 |
| gemini-3.5-flash | 200 |
| gemini-3.8-flash | 503 high demand |
| gemini-3.6-flash、gemini-3.1-flash-lite、gemini-flash-lite-latest | 500 |
| gemini-3.8-flash-lite、gemini-3.8-pro、gemini-3-flash-lite、gemini-2.5-flash | 404 |

500 跟有没有带 tools **无关**（不带工具也 500），所以不是工具参数造成的，是上游本身在抖。这个对我们影响很大：gemini 是我们多上游链的第一档，它一抖就掉到后面的小模型上，回答质量直接下降。

## 6. 请按这个顺序验收，并把**原始输出**贴回来

```bash
KEY='<线上 gemini key>'
BASE='https://ai.xiaowan.dpdns.org/gemini'

# ① 路由表里必须出现新端点
curl -s https://ai.xiaowan.dpdns.org/openapi.json | python3 -c 'import sys,json;print(list(json.load(sys.stdin)["paths"]))'

# ② 有搜索：必须有非空 groundingMetadata
curl -s "$BASE/v1beta/models/gemini-3.5-flash-lite:generateContent" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"今天2026年10月16日有什么科技新闻？一句话，要最新的"}]}],"tools":[{"google_search":{}}]}'

# ③ 对照：不带 tools，应当**没有** groundingMetadata
curl -s "$BASE/v1beta/models/gemini-3.5-flash-lite:generateContent" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"今天2026年10月16日有什么科技新闻？一句话"}]}]}'

# ④ OpenAI 兼容层：message 里应带引用
curl -s "$BASE/chat/completions" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.5-flash-lite","messages":[{"role":"user","content":"今天2026年10月16日有什么科技新闻？一句话"}],"tools":[{"google_search":{}}],"stream":false}'
```

判定标准（和上次一样，只有这几条算通过）：

1. ① 的路由表里出现 `/gemini/v1beta/models/{model}:generateContent`；
2. ② HTTP 200 且 `candidates[0].groundingMetadata.webSearchQueries` **非空**，同时回答内容和「今天」对得上（这是接地生效的唯一硬证据）；
3. ③ 没有 `groundingMetadata`；
4. ④ `choices[0].message` 里有 `grounding_metadata`（或 content 末尾有 markdown 来源列表）；
5. 不支持的工具（如 `{"type":"web_search"}`）返回 **400 + 明确文案**，不再静默忽略；已知不支持接地的模型（1.5/2.x）请求接地时也应 400。

请连**部署方式**一起回答，并附上 ② 的完整 JSON（key 打码）。如果 ② 仍然拿不到 `groundingMetadata`，请说明是 Google 侧不支持、凭据问题，还是模型不支持——不要用「测试通过」结案，这次要的是线上原始返回。
