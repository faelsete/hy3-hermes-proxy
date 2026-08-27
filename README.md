# HY3 Hermes Proxy — Tencent Hy3 via API OpenAI

Proxy **100% OpenAI-compatible** para o Space oficial `tencent/hy3` no Hugging Face. Sem token, sem OpenRouter — roda local e expõe `http://localhost:8766/v1/chat/completions`.

> **Prova real:** este proxy gerou o dashboard em `/hy3-proof-dashboard` (6 arquivos, 2487 tokens, 71s) via `tencent/hy3` — veja `hy3_output.md`.

## Por que usar?
- **Contexto gigante:** 262.144 tokens, compactação inteligente (nunca retorna vazio, mesmo com 400k tokens)
- **KV-cache:** prefix cache (hash system+tools, TTL 10min)
- **Compatível:** OpenAI SDK, streaming SSE, tool_calls, usage real via `tiktoken`
- **Leve:** 228MB Docker, 75MB RAM

## Instalação em 1 comando

### Windows / Linux / Mac (sem Docker)
```bash
git clone https://github.com/faelsete/hy3-hermes-proxy.git
cd hy3-hermes-proxy
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8766
curl http://127.0.0.1:8766/health
# {"status":"ok","model":"tencent/hy3","context_length":262144}
```

### Docker (recomendado para outros devices)
```bash
docker build -t hy3-proxy .
docker run -p 8766:8766 hy3-proxy
# ou
docker compose up -d
```

### Windows (Docker Desktop)
1. Instale [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
2. `docker compose up -d`
3. Acesse `http://localhost:8766/health`

### Uso com OpenAI SDK
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8766/v1", api_key="not-needed")
resp = client.chat.completions.create(model="tencent/hy3", messages=[{"role":"user","content":"Olá em pt-BR"}])
print(resp.choices[0].message.content)
print(resp.usage) # prompt_tokens real via tiktoken
```

### cURL
```bash
curl -X POST http://localhost:8766/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tencent/hy3","messages":[{"role":"user","content":"Olá"}]}'
curl http://localhost:8766/v1/models
curl http://localhost:8766/health
```

## Endpoints
- `GET /health` — status, kv_cache_size, context_length
- `GET /v1/models` — lista `tencent/hy3` (262144)
- `POST /v1/chat/completions` — chat + tools + streaming (`stream:true`)

## Arquitetura
```
Hermes / App -> http://127.0.0.1:8766/v1/chat/completions -> https://tencent-hy3.hf.space/gradio_api/call/chat
```

## Stress test (prova)
```bash
python /tmp/hy3_stress.py
# 20k 5.6s ok | 50k 9.3s ok | 100k 19.5s ok | 180k 4.6s ok | 250k 4.2s ok | 400k 4.3s ok
```

## Hermes Agent
```bash
# config.yaml
providers:
  hy3-official:
    base_url: http://127.0.0.1:8766/v1
    api_key: not-needed
    models: [tencent/hy3]
```

## Tailscale (acesso remoto)
```bash
# expor para rede Tailscale
python -m uvicorn app:app --host 100.76.28.66 --port 8766
# acesso: http://100.76.28.66:8766/health
```

Licença MIT — use em qualquer device.
