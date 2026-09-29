"""API for generating EagleEye-compatible decision-tree drafts."""

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()


class AIProvider(str, Enum):
    AI_PLATFORM = "ai_platform"
    SOCLAAS = "soclaas"


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    model: str
    headers: dict[str, str]
    reasoning_effort: str | None = None


def get_provider_config() -> ProviderConfig:
    provider_name = os.getenv("AI_PROVIDER", AIProvider.AI_PLATFORM.value).lower()
    try:
        provider = AIProvider(provider_name)
    except ValueError as error:
        supported = ", ".join(item.value for item in AIProvider)
        raise HTTPException(
            status_code=503,
            detail=f"Unsupported AI_PROVIDER '{provider_name}'. Choose one of: {supported}.",
        ) from error

    if provider is AIProvider.SOCLAAS:
        api_key = os.getenv("SOCLAAS_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="SOCLAAS_API_KEY is not configured on the server.",
            )
        return ProviderConfig(
            base_url=os.getenv(
                "SOCLAAS_BASE_URL", "https://soclaas-api.comp.nus.edu.sg/v1"
            ).rstrip("/"),
            model=os.getenv("SOCLAAS_MODEL", "qwen3.5:9b"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            reasoning_effort=os.getenv("SOCLAAS_REASONING_EFFORT", "none"),
        )

    api_key = os.getenv("AI_PLATFORM_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI_PLATFORM_API_KEY is not configured on the server.",
        )
    return ProviderConfig(
        base_url=os.getenv(
            "AI_PLATFORM_BASE_URL", "https://api.ai.tech.gov.sg/platform/models"
        ).rstrip("/"),
        model=os.getenv("AI_PLATFORM_MODEL", "gpt-4o-mini"),
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        reasoning_effort=os.getenv("AI_PLATFORM_REASONING_EFFORT"),
    )


app = FastAPI(title="EagleEye Decision Tree API", version="0.1.0")
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        ",".join(
            [
                "http://localhost:3000",
                "http://localhost:5050",
                "http://localhost:5500",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5050",
                "http://127.0.0.1:5500",
            ]
        ),
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class GenerationRequest(BaseModel):
    question: Annotated[str, Field(min_length=3, max_length=8_000)]
    answer: Annotated[str, Field(min_length=3, max_length=12_000)]


class Topic(BaseModel):
    name: str
    description: str = ""
    example_phrases: str = ""


class Question(BaseModel):
    name: str
    description: str = ""
    example_phrases: str = ""


class LeadingQuestion(BaseModel):
    name: str
    question_to_ask: str
    required: bool = True
    allowed_values: str = ""
    example_values: str = ""


class Answer(BaseModel):
    final_decision: str
    answer_text: str
    next_steps: str = ""
    escalation_note: str = ""


class Rule(BaseModel):
    conditions: dict[str, str] = Field(default_factory=dict)
    answer_index: int = Field(ge=0)
    priority: int = Field(ge=1)


class DecisionTree(BaseModel):
    topic: Topic
    question: Question
    leading_questions: list[LeadingQuestion] = Field(default_factory=list)
    answers: list[Answer] = Field(min_length=1)
    rules: list[Rule] = Field(min_length=1)


class GenerationResponse(BaseModel):
    decision_tree: DecisionTree


TREE_SCHEMA = DecisionTree.model_json_schema()
SYSTEM_PROMPT = """You create concise, safe decision-tree drafts for EagleEye, a
knowledge-base authoring tool. Your response will be validated and then converted into
five related workbook entities:

1. A topic is a broad category of enquiry. It can contain several questions.
2. A question is one specific user need within that topic. This draft creates exactly
   one topic and one question.
3. Leading questions collect only the facts needed to choose an outcome. A leading
   question has a human-readable name, the question to ask, whether it is required,
   allowed_values, examples, and display order.
4. Answers are the approved outcome texts. A rule chooses one answer by its zero-based
   answer_index. final_decision should normally be Answered, Clarify, or Escalate.
5. Rules define when an answer applies. Their conditions are an AND combination: every
   condition must match. A rule can intentionally mention only the leading questions
   relevant to that outcome; this lets EagleEye stop collecting information as soon as a
   conclusive rule matches. If more than one rule matches, lower priority numbers win
   (1 is highest priority).

Follow these schema rules exactly:
- Return only a JSON object conforming to the supplied JSON schema; do not use Markdown.
- Every conditions key must exactly equal a leading_questions[].name, including casing.
- Every conditions value must be one of that leading question's allowed_values. Provide
  allowed_values as a clear comma- or semicolon-separated list whenever the question is
  used by a rule. Do not use free-text conditions that are absent from allowed_values.
- Each rule must have a distinct condition combination and refer to an answer_index that
  exists in answers. Create enough rules to cover the meaningful combinations, including
  a targeted Clarify or Escalate fallback only when supported by the supplied guidance.
- An empty conditions object is an unconditional fallback; never create more than one.
  Give it a lower precedence than specific rules so it cannot override them.
- Do not add leading questions merely to make the tree look complete. Required means the
  information is normally needed, but a specific matching rule may still be conclusive
  without it. Mark a question optional when it should not block an outcome.
- Keep topic and question names short and searchable. Put alternate user wording in
  example_phrases. Keep answer_text as user-facing guidance; put actions in next_steps
  and only human hand-off criteria in escalation_note.

Use only information supplied by the user. Do not invent policy terms, eligibility,
limits, procedures, or legal conclusions. If the supplied guidance cannot safely support
a definitive outcome, use an Escalate answer that states what a human needs to review."""


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ai/status")
async def ai_status() -> dict[str, bool | str]:
    """Expose availability without exposing provider credentials."""
    try:
        provider = get_provider_config()
    except HTTPException:
        return {"available": False}
    return {"available": True, "provider": os.getenv("AI_PROVIDER", "ai_platform")}


@app.post("/decision-trees/generate", response_model=GenerationResponse)
async def generate_decision_tree(request: GenerationRequest) -> GenerationResponse:
    provider = get_provider_config()

    payload = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User question:\n{request.question}\n\nApproved answer or guidance:\n{request.answer}",
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "eagleeye_decision_tree", "strict": True, "schema": TREE_SCHEMA},
        },
        "temperature": 0.2,
    }
    if provider.reasoning_effort:
        payload["reasoning_effort"] = provider.reasoning_effort

    try:
        async with httpx.AsyncClient(timeout=480) as client:
            response = await client.post(
                f"{provider.base_url}/chat/completions",
                headers=provider.headers,
                json=payload,
            )
            response.raise_for_status()
    except httpx.TimeoutException as error:
        raise HTTPException(status_code=504, detail="The AI platform timed out.") from error
    except httpx.HTTPStatusError as error:
        detail = error.response.text[:500] or "The AI platform rejected the request."
        raise HTTPException(status_code=502, detail=detail) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="Could not reach the AI platform.") from error

    try:
        content = response.json()["choices"][0]["message"]["content"]
        tree = DecisionTree.model_validate(json.loads(content))
        validate_tree_references(tree)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise HTTPException(status_code=502, detail="The AI platform returned an invalid decision-tree draft.") from error

    return GenerationResponse(decision_tree=tree)


def validate_tree_references(tree: DecisionTree) -> None:
    leading_question_names = {item.name for item in tree.leading_questions}
    rule_combinations: set[tuple[tuple[str, str], ...]] = set()
    for rule in tree.rules:
        unknown_names = set(rule.conditions) - leading_question_names
        if unknown_names:
            raise ValueError("A rule references an unknown leading question.")
        if rule.answer_index >= len(tree.answers):
            raise ValueError("A rule references an unknown answer.")
        combination = tuple(sorted(rule.conditions.items()))
        if combination in rule_combinations:
            raise ValueError("Two rules use the same leading-question combination.")
        rule_combinations.add(combination)
