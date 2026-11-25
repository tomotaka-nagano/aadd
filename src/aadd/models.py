from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue, model_validator


def identifier() -> str:
    return uuid4().hex


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Change(BaseModel):
    id: str = Field(default_factory=identifier, min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=2000)
    before: str | None = Field(default=None, max_length=12000)
    after: str | None = Field(default=None, max_length=12000)
    sources: list[str] = Field(default_factory=list, max_length=100)
    unknowns: list[str] = Field(default_factory=list, max_length=100)


class Extraction(BaseModel):
    changes: list[Change] = Field(max_length=100)


class SpecificationSearch(BaseModel):
    query: str = Field(min_length=1, max_length=160)
    question: str = Field(min_length=1, max_length=500)


class ExtractionPlan(Extraction):
    searches: list[SpecificationSearch] = Field(default_factory=list, max_length=12)


class RefinedChange(Change):
    evidence_ids: list[str] = Field(default_factory=list, max_length=6)


class Refinement(ExtractionPlan):
    changes: list[RefinedChange] = Field(max_length=100)
    reviewed_evidence_ids: list[str] = Field(default_factory=list, max_length=6)


class MaskPreview(BaseModel):
    message: str
    system_prompt: str
    response_schema: dict[str, JsonValue]


class ChangeUpdate(Extraction):
    changes: list[Change] = Field(min_length=1, max_length=100)
    revision: int = Field(ge=0)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({item.id for item in self.changes}) != len(self.changes):
            raise ValueError('変更IDが重複しています。')
        return self


class RunRequest(BaseModel):
    revision: int = Field(ge=0)


class Attachment(BaseModel):
    id: str = Field(default_factory=identifier)
    filename: str
    text: str
    warnings: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    document_id: str
    file_path: str
    title: str
    max_score: float = Field(allow_inf_nan=False)


class Snapshot(BaseModel):
    document_id: str
    file_path: str
    text: str
    sha256: str


class Assessment(BaseModel):
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    model: str


class Generated(BaseModel):
    modified: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    change_ids: list[str] = Field(min_length=1)


class Proposal(Generated):
    id: str = Field(default_factory=identifier)
    document_id: str
    file_path: str
    original: str
    diff: str


RunStatus = Literal['running', 'completed', 'review', 'failed', 'interrupted']


class Run(BaseModel):
    id: str = Field(default_factory=identifier)
    kind: Literal['extract', 'generate']
    revision: int
    status: RunStatus = 'running'
    stage: str = 'queued'
    created_at: str = Field(default_factory=timestamp)
    changes: list[Change] = Field(default_factory=list)
    candidates: dict[str, list[Candidate]] = Field(default_factory=dict)
    documents: dict[str, Snapshot] = Field(default_factory=dict)
    assessments: dict[str, dict[str, Assessment]] = Field(default_factory=dict)
    selections: dict[str, str] = Field(default_factory=dict)
    proposals: dict[str, Proposal] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    attempt: int = Field(default=1, ge=1)


class Project(BaseModel):
    id: str = Field(default_factory=identifier)
    title: str
    request: str
    created_at: str = Field(default_factory=timestamp)
    revision: int = 0
    attachments: list[Attachment] = Field(default_factory=list)
    changes: list[Change] = Field(default_factory=list)
    runs: list[Run] = Field(default_factory=list)
