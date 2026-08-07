import pytest

from anima.train.common import (
    ChatTemplateError,
    build_sft_chat_rows,
    record_to_messages,
    render_messages_with_chat_template,
    render_sft_completion,
    render_sft_completion_content,
)


class FakeTokenizer:
    chat_template = "{% for message in messages %}{{ message['role'] }}{% endfor %}"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "<CHAT_TEMPLATE_RENDERED>"


class NoTemplateTokenizer:
    chat_template = None


def _record():
    return {
        "id": "chat_template_test",
        "character": "李白",
        "source_work": "唐诗",
        "profile": "唐代诗人，豪放洒脱。",
        "conversations": [{"role": "user", "content": "今日为何饮酒？"}],
        "gold_focus": ["SHOULD_NOT_LEAK"],
        "gold_focus_attr": "SECRET_ATTR",
        "reference_answer": "SECRET_REFERENCE",
    }


def test_chat_template_renderer_calls_tokenizer_apply_chat_template():
    tokenizer = FakeTokenizer()
    messages = record_to_messages(_record())

    rendered = render_messages_with_chat_template(messages, tokenizer, mode="formal")

    assert rendered == "<CHAT_TEMPLATE_RENDERED>"
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0]["tokenize"] is False
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[0]["messages"] == messages


def test_record_to_messages_keeps_gold_fields_out_of_prompt_messages():
    messages = record_to_messages(_record())
    joined = "\n".join(message["content"] for message in messages)

    assert "李白" in joined
    assert "今日为何饮酒？" in joined
    assert "SHOULD_NOT_LEAK" not in joined
    assert "SECRET_ATTR" not in joined
    assert "SECRET_REFERENCE" not in joined


def test_record_to_messages_maps_character_turns_to_assistant():
    record = dict(_record())
    record["conversations"] = [
        {"role": "user", "content": "你是谁？"},
        {"role": "李白", "content": "吾乃青莲居士。"},
        {"role": "user", "content": "今日为何饮酒？"},
    ]

    messages = record_to_messages(record)

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "吾乃青莲居士。"


def test_formal_renderer_fails_closed_without_chat_template():
    with pytest.raises(ChatTemplateError, match="chat-template|apply_chat_template|formal"):
        render_messages_with_chat_template(
            record_to_messages(_record()),
            NoTemplateTokenizer(),
            mode="formal",
        )


def test_legacy_renderer_is_local_only():
    rendered = render_messages_with_chat_template(
        record_to_messages(_record()),
        None,
        mode="dry_run",
    )

    assert rendered.startswith("system:")
    assert "今日为何饮酒？" in rendered


def test_build_sft_chat_rows_uses_conversational_prompt_completion(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        __import__("json").dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8"
    )

    prepared = build_sft_chat_rows([path], require_gold_format=True)

    assert prepared.skipped == 0
    assert len(prepared.rows) == 1
    row = prepared.rows[0]
    assert [message["role"] for message in row["prompt"]] == ["system", "user"]
    prompt_text = "\n".join(message["content"] for message in row["prompt"])
    assert "SECRET_ATTR" not in prompt_text
    assert "SECRET_REFERENCE" not in prompt_text
    assert row["completion"][0]["role"] == "assistant"
    completion = row["completion"][0]["content"]
    assert completion.startswith("<think>")
    assert "<focus_attr>SECRET_ATTR</focus_attr>" in completion
    assert "\\boxed{SECRET_REFERENCE}" in completion
    assert "<|im_end|>" not in completion


def test_chat_template_completion_content_leaves_turn_boundary_to_tokenizer():
    record = _record()

    content = render_sft_completion_content(record, answer="举杯。")
    legacy = render_sft_completion(record, answer="举杯。")

    assert content.endswith("\\boxed{举杯。}")
    assert "<|im_end|>" not in content
    assert legacy.endswith("\\boxed{举杯。}<|im_end|>")
