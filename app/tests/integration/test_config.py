import pytest
from pydantic import ValidationError

from neorag.config import Settings, EmbedderConfig, LLMConfig


class TestSettingsDefaults:
    def test_default_embedder(self):
        config = Settings()
        assert config.embedder.type == "bge-m3"
        assert config.embedder.model_name == "BAAI/bge-m3"
        assert config.embedder.device == "cpu"

    def test_default_qdrant(self):
        config = Settings()
        assert config.qdrant.host == "qdrant"
        assert config.qdrant.port == 6333
        assert config.qdrant.collection_name == "documents"

    def test_default_llm(self):
        config = Settings()
        assert config.llm.type == "ollama"
        assert config.llm.model == "llama3.2:8b"
        assert config.llm.temperature == 0.1

    def test_default_retrieval(self):
        config = Settings()
        assert config.retrieval.alpha == 0.5
        assert config.retrieval.top_k_retrieve == 20
        assert config.retrieval.min_score == 0.6

    def test_default_session(self):
        config = Settings()
        assert config.session.storage == "sqlite"
        assert config.session.db_path == "data/sessions.db"


class TestSettingsFromEnv:
    def test_embedder_type_override(self, monkeypatch):
        monkeypatch.setenv("EMBEDDER__TYPE", "sentence-transformer")
        config = Settings()
        assert config.embedder.type == "sentence-transformer"

    def test_qdrant_host_override(self, monkeypatch):
        monkeypatch.setenv("QDRANT__HOST", "localhost")
        config = Settings()
        assert config.qdrant.host == "localhost"

    def test_llm_temperature_override(self, monkeypatch):
        monkeypatch.setenv("LLM__TEMPERATURE", "0.5")
        config = Settings()
        assert config.llm.temperature == 0.5

    def test_multiple_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM__MODEL", "mistral:7b")
        monkeypatch.setenv("LLM__TEMPERATURE", "0.0")
        config = Settings()
        assert config.llm.model == "mistral:7b"
        assert config.llm.temperature == 0.0

    def test_retrieval_alpha_override(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL__ALPHA", "0.8")
        config = Settings()
        assert config.retrieval.alpha == 0.8


class TestSettingsValidation:
    def test_invalid_embedder_type(self):
        with pytest.raises(ValidationError):
            EmbedderConfig(type="invalid-model")

    def test_invalid_llm_type(self):
        with pytest.raises(ValidationError):
            LLMConfig(type="invalid-llm")

    def test_negative_temperature(self):
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-1.0)

    def test_negative_top_k(self):
        with pytest.raises(ValidationError):
            from neorag.config import RetrievalConfig
            RetrievalConfig(top_k_retrieve=-1)


class TestSettingsExport:
    def test_model_dump(self):
        config = Settings()
        dump = config.model_dump()
        assert "embedder" in dump
        assert "qdrant" in dump
        assert "llm" in dump
        assert dump["embedder"]["type"] == "bge-m3"

    def test_model_dump_json(self):
        config = Settings()
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "bge-m3" in json_str
