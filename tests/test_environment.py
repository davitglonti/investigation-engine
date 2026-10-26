"""Test environment setup and core configurations."""
import pytest
from src.core.config import settings


def test_settings_load():
    assert settings.PROJECT_NAME == "AI Investigation Engine"
    assert settings.CHUNK_SIZE > 0
    assert settings.CHUNK_OVERLAP >= 0
    assert settings.TOP_K_RETRIEVAL > 0
    assert settings.RRF_K > 0
