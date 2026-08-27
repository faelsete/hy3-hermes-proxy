import json

import pytest

from app import (
    build_upstream_payload,
    build_nonstream_response,
    extract_conversation,
    normalize_tool_calls,
    parse_gradio_data,
    payload_size_chars,
    reasoning_level,
)


def test_extract_conversation_separates_system_and_last_user():
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "developer", "content": "Use tools carefully."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]

    system_prompt, history, message = extract_conversation(messages)

    assert "You are Hermes.\n\nUse tools carefully." in system_prompt
    assert "DIRETIVA DE IDIOMA" in system_prompt
    assert history == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    assert message == "Second question"


def test_extract_conversation_continues_after_tool_result():
    messages = [
        {"role": "user", "content": "Check the time"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "clock", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
    ]

    _, history, message = extract_conversation(messages)

    assert history[-1]["role"] == "tool"
    assert history[-1]["content"] == "12:00"
    assert "Continue" in message


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, "high"),
        ({"reasoning_effort": "none"}, "no_think"),
        ({"reasoning_effort": "minimal"}, "no_think"),
        ({"reasoning_effort": "low"}, "low"),
        ({"reasoning_effort": "medium"}, "low"),
        ({"reasoning_effort": "high"}, "high"),
        ({"reasoning": {"enabled": False}}, "no_think"),
        ({"reasoning": {"enabled": True}}, "high"),
    ],
)
def test_reasoning_level_mapping(payload, expected):
    assert reasoning_level(payload) == expected


def test_build_upstream_payload_passes_tools_and_sampling():
    request = {
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo text",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "temperature": 0.4,
        "top_p": 0.8,
        "max_tokens": 321,
        "reasoning_effort": "low",
    }

    payload = build_upstream_payload(request)

    assert payload["data"][0] == "Use a tool"
    assert payload["data"][3] == "low"
    assert payload["data"][4] == 0.4
    assert payload["data"][5] == 321
    assert payload["data"][6] == 0.8
    assert json.loads(payload["data"][8]) == request["tools"]


def test_parse_gradio_data_extracts_latest_snapshot():
    raw = json.dumps(
        [[
            "Final answer",
            "Reasoning text",
            [{"id": "call_1", "function": {"name": "echo", "arguments": {"text": "x"}}}],
            [{"role": "user", "content": "hello"}],
        ]]
    )

    snapshot = parse_gradio_data(raw)

    assert snapshot["content"] == "Final answer"
    assert snapshot["reasoning"] == "Reasoning text"
    assert snapshot["history"][0]["role"] == "user"


def test_normalize_tool_calls_outputs_openai_shape():
    calls = normalize_tool_calls(
        [
            {
                "id": "call_1",
                "name": "echo",
                "arguments": {"text": "hello"},
            }
        ]
    )

    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "echo"
    assert json.loads(calls[0]["function"]["arguments"]) == {"text": "hello"}


def test_build_nonstream_response_includes_reasoning_and_tools():
    response = build_nonstream_response(
        model="tencent/hy3",
        content="",
        reasoning="Need a tool",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "echo", "arguments": "{}"},
            }
        ],
    )

    assert response["object"] == "chat.completion"
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning_content"] == "Need a tool"
    assert choice["message"]["tool_calls"][0]["id"] == "call_1"


def test_build_upstream_payload_trims_old_turns_to_context_budget():
    messages = [{"role": "system", "content": "System rules"}]
    for index in range(20):
        messages.extend(
            [
                {"role": "user", "content": f"old-question-{index} " + ("x" * 180)},
                {"role": "assistant", "content": f"old-answer-{index} " + ("y" * 180)},
            ]
        )
    messages.append({"role": "user", "content": "olá"})

    payload = build_upstream_payload(
        {"messages": messages, "max_tokens": 256},
        max_input_chars=1_200,
    )

    history = payload["data"][2]
    assert payload["data"][0] == "olá"
    assert payload_size_chars(payload["data"]) <= 1_200
    assert 0 < len(history) < 40
    assert history[0]["role"] == "user"
    assert history[-1]["content"].startswith("old-answer-19")
    assert "Histórico anterior compactado" in payload["data"][1]


def test_build_upstream_payload_does_not_trim_small_context():
    request = {
        "messages": [
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ]
    }

    payload = build_upstream_payload(request, max_input_chars=10_000)

    assert "System rules" in payload["data"][1]
    assert "DIRETIVA DE IDIOMA" in payload["data"][1]
    assert payload["data"][5] == 16_384
    assert payload["data"][2] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]


def test_extract_tool_calls_from_xml_and_json():
    from app import extract_tool_calls_from_text

    raw = """Vou verificar o sistema agora.
<tool_call>
<computer_use>
  <action>bash</action>
  <args>
    <command>systemctl is-active hermes-jr</command>
  </args>
</computer_use>
</tool_call>"""

    clean, tools = extract_tool_calls_from_text(raw)
    assert clean == "Vou verificar o sistema agora."
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "computer_use"
    args = json.loads(tools[0]["function"]["arguments"])
    assert args["action"] == "bash"
    assert args["command"] == "systemctl is-active hermes-jr"
