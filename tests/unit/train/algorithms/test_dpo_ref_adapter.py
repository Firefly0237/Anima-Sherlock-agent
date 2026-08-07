import sys
import types

import pytest

from anima.train.algorithms import dpo
from anima.train.common import load_config, load_dpo_policy_with_ref_adapter
from tests.support.environment.paths import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"


class CapturingDPOConfig:
    """Accepts everything, so _build_dpo_config passes the full requested dict."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MinimalDPOConfig:
    """Mimics an old TRL without the dual-adapter DPOConfig fields."""

    def __init__(self, beta=None, loss_type=None, max_length=None, max_prompt_length=None):
        self.beta = beta
        self.loss_type = loss_type
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length


def test_build_dpo_config_injects_dual_adapter_names_with_sft_adapter(tmp_path):
    args = dpo._build_dpo_config(CapturingDPOConfig, {}, tmp_path, use_ref_adapter=True)

    assert args.kwargs["model_adapter_name"] == "default"
    assert args.kwargs["ref_adapter_name"] == "reference"


def test_build_dpo_config_omits_adapter_names_without_sft_adapter(tmp_path):
    args = dpo._build_dpo_config(CapturingDPOConfig, {}, tmp_path, use_ref_adapter=False)

    assert "model_adapter_name" not in args.kwargs
    assert "ref_adapter_name" not in args.kwargs


def test_build_dpo_config_defaults_use_lora_scale_lr(tmp_path):
    args = dpo._build_dpo_config(CapturingDPOConfig, {}, tmp_path)

    assert args.kwargs["learning_rate"] == pytest.approx(5e-6)
    assert args.kwargs["warmup_ratio"] == pytest.approx(0.1)


def test_missing_dual_adapter_support_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="model_adapter_name.*ref_adapter_name|critical keys"):
        dpo._build_dpo_config(MinimalDPOConfig, {}, tmp_path, use_ref_adapter=True)


def test_missing_dual_adapter_support_is_fine_without_adapter(tmp_path, capsys):
    args = dpo._build_dpo_config(MinimalDPOConfig, {}, tmp_path, use_ref_adapter=False)

    assert isinstance(args, MinimalDPOConfig)


def test_persona_v4_strict_semantics_rejects_missing_completion_limit(tmp_path):
    with pytest.raises(RuntimeError, match="completion truncation"):
        dpo._build_dpo_config(
            MinimalDPOConfig,
            {"strict_config_semantics": True},
            tmp_path,
            use_ref_adapter=False,
        )


def test_reference_policy_semantics_values():
    assert (
        dpo.reference_policy_semantics("/models/sft_rolebench")
        == "sft_adapter_via_ref_adapter_name"
    )
    assert dpo.reference_policy_semantics(None) == "base_model_adapter_disabled"
    assert "SFT" in dpo.reference_policy_note("/models/sft_rolebench")
    assert "base" in dpo.reference_policy_note(None)


def test_dpo_configs_use_lora_scale_lr():
    config = load_config(CONFIG_DIR / "dpo.yaml")
    assert config["learning_rate"] == pytest.approx(5e-6)
    assert config["warmup_ratio"] == pytest.approx(0.1)
    assert config["beta"] == pytest.approx(0.1)
    assert config["precompute_ref_log_probs"] is False


def test_load_dpo_policy_with_ref_adapter_mounts_same_adapter_twice(monkeypatch):
    calls = {}

    class FakeTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<|im_end|>"
            self.padding_side = None

    class FakeModelConfig:
        use_cache = True

    class FakeBaseModel:
        config = FakeModelConfig()

    class FakePeftModel:
        def __init__(self, base):
            self.base = base
            self.load_adapter_calls = []
            self.active_adapter = None

        @classmethod
        def from_pretrained(cls, model, path, **kwargs):
            calls["peft_from_pretrained"] = {"path": path, **kwargs}
            return cls(model)

        def load_adapter(self, path, adapter_name, *, is_trainable):
            self.load_adapter_calls.append((path, adapter_name, is_trainable))

        def set_adapter(self, name):
            self.active_adapter = name

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["base_from_pretrained"] = {"name": name, **kwargs}
            return FakeBaseModel()

    def fake_bnb_config(**kwargs):
        return {"bnb": kwargs}

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    fake_peft.prepare_model_for_kbit_training = lambda model: model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.BitsAndBytesConfig = fake_bnb_config

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model, tokenizer, peft_config = load_dpo_policy_with_ref_adapter(
        {"model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"},
        padding_side="right",
        adapter_path="/models/sft_rolebench",
    )

    assert peft_config is None
    assert calls["peft_from_pretrained"]["path"] == "/models/sft_rolebench"
    assert calls["peft_from_pretrained"]["is_trainable"] is True
    assert calls["peft_from_pretrained"]["adapter_name"] == "default"
    assert model.load_adapter_calls == [("/models/sft_rolebench", "reference", False)]
    assert model.active_adapter == "default"
    assert tokenizer.padding_side == "right"
    assert tokenizer.pad_token == tokenizer.eos_token
