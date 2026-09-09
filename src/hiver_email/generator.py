"""Retrieval-grounded suggested-reply generation."""

from __future__ import annotations

from .llm import LLMClient, LLMError
from .retrieval import DEFAULT_TOP_K, TfidfRetriever
from .schemas import GenerationResult, RetrievedExample

SYSTEM_PROMPT = """You draft suggested email replies for a busy professional.

You are given the new incoming email plus up to three similar historical
emails and the replies real people actually sent to them.

Rules:
- Use the retrieved examples as guidance for style, structure and the kind of
  information a good reply contains.
- Answer the ACTUAL new email. Do not copy an example reply.
- Do NOT fabricate commitments, dates, prices, numbers, approvals, attachments,
  transactions or completed actions that the incoming email and the retrieved
  examples do not support.
- Preserve a tone consistent with the retrieved correspondence: professional,
  direct and concise.
- If something needed to answer properly is unknown, acknowledge the
  uncertainty or say you will follow up. Never invent the missing detail.
- Return ONLY the suggested reply body. No subject line, no preamble, no
  commentary, no markdown fences."""

MAX_EXAMPLE_CHARS = 1200
MAX_QUERY_CHARS = 3000


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def build_prompt(incoming_email: str, retrieved: list[RetrievedExample]) -> str:
    blocks: list[str] = []
    if retrieved:
        for i, ex in enumerate(retrieved, start=1):
            blocks.append(
                f"### Historical example {i} (similarity {ex.similarity:.3f})\n"
                f"Incoming email:\n{_truncate(ex.incoming_email, MAX_EXAMPLE_CHARS)}\n\n"
                f"Reply that was actually sent:\n{_truncate(ex.human_reply, MAX_EXAMPLE_CHARS)}"
            )
        grounding = "\n\n".join(blocks)
    else:
        grounding = "(No sufficiently similar historical email was found.)"

    return (
        "RETRIEVED HISTORICAL EXAMPLES\n"
        f"{grounding}\n\n"
        "=====================\n"
        "NEW INCOMING EMAIL\n"
        f"{_truncate(incoming_email, MAX_QUERY_CHARS)}\n\n"
        "=====================\n"
        "Write the suggested reply to the NEW INCOMING EMAIL."
    )


class SuggestedReplyGenerator:
    def __init__(
        self,
        retriever: TfidfRetriever,
        client: LLMClient,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.2,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.top_k = top_k
        self.temperature = temperature

    def generate(self, incoming_email: str) -> GenerationResult:
        retrieved = self.retriever.retrieve(incoming_email, top_k=self.top_k)
        prompt = build_prompt(incoming_email, retrieved)
        try:
            reply = self.client.complete(
                system=SYSTEM_PROMPT,
                user=prompt,
                model=self.client.gen_model,
                temperature=self.temperature,
                max_tokens=1200,
            )
        except LLMError as exc:
            return GenerationResult(
                reply="", retrieved=retrieved, model=self.client.gen_model, error=str(exc)
            )
        return GenerationResult(
            reply=_strip_fences(reply), retrieved=retrieved, model=self.client.gen_model
        )


THINKING_MARKERS = (
    "here's a thinking process",
    "here is a thinking process",
    "**analyze the",
    "let me think through",
)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
    return _strip_thinking(t.strip())


def _strip_thinking(text: str) -> str:
    """Drop leaked chain-of-thought.

    Some reasoning models prepend their scratchpad to the answer. We keep what
    follows an explicit </think> tag, and otherwise refuse to pass on a reply
    that is visibly a scratchpad: an empty reply is recorded as a failure,
    which is honest, whereas shipping the scratchpad would quietly pollute the
    demo output and the scores.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    head = text[:120].lower()
    if any(marker in head for marker in THINKING_MARKERS):
        return ""
    return text.strip()
