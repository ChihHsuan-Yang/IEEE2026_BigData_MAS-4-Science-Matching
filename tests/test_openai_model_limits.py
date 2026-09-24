"""Context-window and model-alias behaviour for the OpenAI-compatible client.

These tests exercise the gateway-specific hooks without naming any site: the
gateway is identified purely by the OPENAI_GATEWAY_* substrings the caller sets.
"""
import importlib


GATEWAY_HOST = "gateway.example.invalid"
ALIAS_PATH = "/deployment-a/"
GATEWAY_URL = f"https://{GATEWAY_HOST}{ALIAS_PATH}v1/"
PLAIN_URL = f"https://{GATEWAY_HOST}/deployment-b/v1/"


def _reload_with_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_GATEWAY_HOST_MATCH", GATEWAY_HOST)
    monkeypatch.setenv("OPENAI_GATEWAY_ALIAS_MATCH", ALIAS_PATH)
    import agentverse.llms.openai as mod

    return importlib.reload(mod)


def test_alias_deployment_keeps_full_context_limit(monkeypatch) -> None:
    mod = _reload_with_gateway(monkeypatch)
    model = mod._normalize_gateway_model_name(
        "google/gemma-4-31B-it", base_url=GATEWAY_URL
    )
    assert model == "gemma-4-31B-it"
    assert mod.OpenAIChat.send_token_limit(model, base_url=GATEWAY_URL) == 131072


def test_non_alias_deployment_keeps_provider_prefixed_name(monkeypatch) -> None:
    mod = _reload_with_gateway(monkeypatch)
    model = mod._normalize_gateway_model_name(
        "google/gemma-4-31B-it", base_url=PLAIN_URL
    )
    assert model == "google/gemma-4-31B-it"
    assert mod.OpenAIChat.send_token_limit(model, base_url=PLAIN_URL) == 131072


def test_llama31_window_is_capped_on_configured_gateway(monkeypatch) -> None:
    mod = _reload_with_gateway(monkeypatch)
    limit = mod.OpenAIChat.send_token_limit(
        "meta-llama/Meta-Llama-3.1-70B-Instruct", base_url=GATEWAY_URL
    )
    assert limit == 16384


def test_llama31_window_is_uncapped_off_gateway(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_GATEWAY_HOST_MATCH", raising=False)
    monkeypatch.delenv("OPENAI_GATEWAY_ALIAS_MATCH", raising=False)
    import agentverse.llms.openai as mod

    mod = importlib.reload(mod)
    limit = mod.OpenAIChat.send_token_limit(
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        base_url="https://api.example.invalid/v1/",
    )
    assert limit == 32768
