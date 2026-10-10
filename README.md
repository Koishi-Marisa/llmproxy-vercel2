<!-- todo:
1. change icon
2. one-click deploy
3. support cerebras
-->

<p align="center">
 <img width="780px" src="public/flow.png" align="center" alt="Deploy on Vercel" />
 <h2 align="center"> LLM API 反向代理 </h2>

<p align="center">
  <a href="https://github.com/ultrasev/llmproxy-vercel/issues">
    <img alt="Issues" src="https://img.shields.io/github/issues/ultrasev/llmproxy-vercel?style=flat&color=336791" />
  </a>
  <a href="https://github.com/ultrasev/llmproxy-vercel/pulls">
    <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/ultrasev/llmproxy-vercel?style=flat&color=336791" />
  </a>
  <br />
</p>

本项目旨在提供一个反向代理服务，解决在部分国家或地区无法直接访问 Google, Groq, Cerebras（Amazon cloudfront）等平台 API 的问题。

# 功能

通过 Vercel 边缘网络，反向代理 OpenAI、Groq、Google、Cerebras 等平台的 API 请求。

- 支持供应商：Groq、Google、OpenAI、Cerebras、NVIDIA、Mistral、Sambanova
- 支持流式输出
- 兼容 OpenAI API 规范

注：大陆不可直接访问 vercel.app 域名。如想直接访问，可参考之前作者的另一个项目[llmproxy](https://github.com/ultrasev/llmproxy)，通过 cloudflare worker 部署 LLM API 反向代理。

# 使用
部署到 Vercel 后，可使用自己的 API 地址为：https://your-project-name.vercel.app/。
测试 API 地址：
- https://llmproxy-vercel.vercel.app/ : 即 vercel 提供的 API 地址，局域网内不能直接访问，需科学上网。
- https://llm.cufo.cc/ : 作者通过 cloudflare + 美区 VPS 搭建的反向代理，大陆等地区可直接访问，调用路径为 local -> CF(HK) -> VPS(US) -> OpenAI。

两个地址使用时均不需要携带 `/v1` 后缀，即 `base_url="https://llm.cufo.cc/openai"` 或者 `chat_url="https://llm.cufo.cc/openai/chat/completions"`。

## 示例 1： OpenAI

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-proj-...",
    base_url="https://llmproxy-vercel.vercel.app/openai", # 没有 /v1 后缀
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world!"}],
)

print(response.choices[0].message.content)
```

## 示例 2： Google Gemini

```python
from openai import OpenAI

client = OpenAI(
    api_key="...",
    base_url="https://llmproxy-vercel.vercel.app/gemini",
)

response = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=[{"role": "user", "content": "Hello world!"}],
)

print(response.choices[0].message.content)
```

## 示例 3： Cerebras

```bash
curl --location 'https://llmproxy-vercel.vercel.app/cerebras/chat/completions' \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${CEREBRAS_API_KEY}" \
  --data '{
    "model": "llama3.1-8b",
    "stream": false,
    "messages": [{"content": "why is fast inference important?", "role": "user"}],
    "temperature": 0,
    "max_tokens": 1024,
    "seed": 0,
    "top_p": 1
}'
```

# Vercel 一键部署

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fultrasev%2Fllmproxy-vercel)

# 本地开发测试

```bash
pip3 install -r requirements.txt
pip3 install uvicorn
uvicorn main:app --host 0.0.0.0 --port 3000 --reload
```

# License

Copyright © 2024 [ultrasev](https://github.com/ultrasev).<br />
This project is [MIT](LICENSE) licensed.

# Support me

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/ultrasev)

## Contacts

- [![Twitter Follow](https://img.shields.io/twitter/follow/ultrasev?style=social)](https://twitter.com/slippertopia)
- [![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UCt0Op8mQvqwjp18B8vNPjzg?style=social)](https://www.youtube.com/channel/UCt0Op8mQvqwjp18B8vNPjzg)
