import os
from pathlib import Path
import tomllib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    data_dir: Path = Path('data')
    ai_config: Path = Path('env.toml')
    target_roles_file: Path = Path('target_roles.md')
    generation_model: str = 'generation'
    decision_model: str = 'decision'
    kb_url: str = 'http://127.0.0.1:8000/mcp'
    kb_headers: dict[str, str] = Field(default_factory=dict)
    redactor_url: str = Field(default='http://127.0.0.1:8509', min_length=1)
    redactor_headers: dict[str, str] = Field(default_factory=dict)
    redactor_detection: Literal['ner', 'llm', 'both'] = 'both'
    redactor_mask_level: Literal['certain', 'strong', 'medium', 'weak', 'faint'] = 'strong'
    allowed_hosts: list[str] = Field(default_factory=lambda: ['127.0.0.1', 'localhost', 'testserver'])
    scope_paths: list[str] = Field(default_factory=list)
    exclude_paths: list[str] = Field(default_factory=list)
    search_mode: Literal['hybrid', 'semantic', 'keyword'] = 'hybrid'
    candidate_limit: int = Field(default=5, ge=1, le=50)
    necessity_threshold: float = Field(default=0.5, ge=0, le=1)
    timeout_seconds: float = Field(default=180, gt=0)
    retry_count: int = Field(default=1, ge=0, le=5)
    concurrency: int = Field(default=2, ge=1, le=10)
    max_files: int = Field(default=10, ge=1, le=50)
    max_file_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_expanded_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_spreadsheet_cells: int = Field(default=200000, gt=0)
    max_input_chars: int = Field(default=120000, gt=0)
    max_document_chars: int = Field(default=120000, gt=0)
    max_input_tokens: int = Field(default=50000, gt=0)


def load_settings() -> Settings:
    path = Path(os.environ.get('AADD_CONFIG', 'config.toml')).resolve()
    values = tomllib.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    settings = Settings.model_validate(values)
    settings.data_dir = (path.parent / settings.data_dir).resolve()
    settings.ai_config = (path.parent / settings.ai_config).resolve()
    settings.target_roles_file = (path.parent / settings.target_roles_file).resolve()
    return settings
