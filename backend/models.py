from __future__ import annotations

from pydantic import BaseModel, Field


class ComplianceResult(BaseModel):
    is_supported: bool
    company_name: str | None = None
    company_name_ok: bool = False
    has_signature_or_seal: bool = False
    notes: list[str] = Field(default_factory=list)


class PageSummary(BaseModel):
    page_number: int
    summary: str


class DocumentIngestion(BaseModel):
    filename: str
    page_summaries: list[PageSummary]
    compliance: ComplianceResult


class QuestionRequest(BaseModel):
    session_id: str
    questions: list[str]
    depth: int = Field(default=1, ge=1, le=5)


class AnswerSource(BaseModel):
    filename: str
    page_number: int
    summary: str | None = None


class AnswerItem(BaseModel):
    question: str
    answer: str
    extracted_answer: str | None = None
    sources: list[AnswerSource]
    summary: str | None = None


class QuestionResponse(BaseModel):
    session_id: str
    answers: list[AnswerItem]


class QuestionnaireItem(BaseModel):
    key: str
    type: str
    question: str
    options: list[str] | None = None


class QuestionnaireResponse(BaseModel):
    items: list[QuestionnaireItem]


class ReadmeRequest(BaseModel):
    content: str
