# HY3/HY4 Hermes Proxy — Tencent Hunyuan via API OpenAI

Proxy **100% compatível com a API OpenAI** para os modelos oficiais da Tencent: **Hunyuan 4 (Hy4)** e **Hunyuan 3 (Hy3)**. Roda localmente na porta `:8766` e expõe `http://localhost:8766/v1/chat/completions`.

---

## 🚀 Motores Suportados

| Modelo | Contexto | Arquitetura | Características | Aliases |
| :--- | :--- | :--- | :--- | :--- |
| **`tencent/hy4-preview`** | **1.000.000 tokens** | MoE 770B | Raciocínio profundo nativo, tool-calling de alta precisão, raciocínio passo-a-passo | `tencent/hy4`, `hy4` |
| **`tencent/hy3`** | **262.144 tokens** | Denso | Alta velocidade, sem autenticação externa, prefix cache KV local | `tencent/hy3` |

---

## ⚡ Recursos Principais

- **Contexto Gigante**: Até 1 milhão de tokens no Hy4 e 262k no Hy3.
- **Raciocínio Profundo Nativo**: Suporte ao bloco de pensamento (`reasoning_content` / `reasoning_details`) no Hy4.
- **Tool Calling (Function Calling)**: Compatibilidade estrita com schemas OpenAI `tools` e `tool_choice`.
- **Streaming Real via SSE**: Tokens emitidos em tempo real (`stream: true`).
- **KV-Cache Local**: Prefix cache em memória com TTL de 30min para acelerar requisições repetidas.
- **Auto-Compactação Resiliente**: Se a requisição exceder limites no Hy3, o proxy compacta o histórico automaticamente sem nunca retornar vazio.

---

## 📦 Como Usar

### 1. Iniciar o Proxy
```bash
cd /root/projetos/hy3-hermes-proxy
# Com venv ativo
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

### 2. cURL — Testando o Hy4
```bash
curl -X POST http://127.0.0.1:8766/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tencent/hy4",
    "messages": [
      {"role": "user", "content": "Explique resumidamente a arquitetura MoE."}
    ],
    "max_tokens": 1000,
    "temperature": 0.2
  }'
```

### 3. cURL — Listar Modelos Disponíveis
```bash
curl http://127.0.0.1:8766/v1/models
```

### 4. OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8766/v1",
    api_key="not-needed"
)

# Chamada usando o Hy4 (1M contexto)
response = client.chat.completions.create(
    model="tencent/hy4",
    messages=[
        {"role": "system", "content": "Você é um assistente técnico especialista."},
        {"role": "user", "content": "Qual a diferença entre Hy3 e Hy4?"}
    ]
)
print(response.choices[0].message.content)

# Chamada usando o Hy3 (modo rápido)
response_hy3 = client.chat.completions.create(
    model="tencent/hy3",
    messages=[{"role": "user", "content": "Olá em pt-BR!"}]
)
print(response_hy3.choices[0].message.content)
```

### 5. Tool Calling com Hy4
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "consultar_clima",
            "description": "Obtém a previsão do tempo para uma cidade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cidade": {"type": "string", "description": "Nome da cidade"}
                },
                "required": ["cidade"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="tencent/hy4",
    messages=[{"role": "user", "content": "Como está o tempo em Belo Horizonte hoje?"}],
    tools=tools
)
print(response.choices[0].message.tool_calls)
```

---

## 🌐 Endpoints

- `GET /health` — Status operacional, contexto e tamanho do cache KV.
- `GET /v1/models` — Lista os modelos disponíveis (`tencent/hy4-preview`, `tencent/hy4`, `hy4`, `tencent/hy3`).
- `POST /v1/chat/completions` — Endpoint padrão OpenAI (chat, tool_calls, streaming).

---

## 🛠️ Variáveis de Ambiente

As chaves e parâmetros são carregados de `/root/.agents/keys.env`:
- `OPENROUTER_API_KEY`: Chave da API para o upstream oficial do Hy4.
- `HY3_SPACE_URL`: URL upstream para o Space do Hy3 (padrão: `https://tencent-hy3.hf.space`).
- `HY4_MODEL_ID`: Identificador do modelo Hy4 (padrão: `tencent/hy4-preview`).

---

Licença MIT — VisionOS Ecosystem.
