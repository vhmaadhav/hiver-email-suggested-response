"""LLM-as-a-judge rubric scoring.

The judge is told explicitly that the historical human reply is *one* valid
answer, not a wording target. Without that instruction the judge collapses
into a paraphrase detector and we lose the whole point of the rubric.
"""

from __future__ import annotations

from .llm import LLMClient, LLMError, parse_json_object
from .schemas import JudgeVerdict, RetrievedExample, coerce_verdict

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of suggested email replies.

You see an incoming email, the reply a real human actually sent, a candidate
reply produced by a system, and the historical examples the system was given.

CRITICAL FRAMING: the historical human response is ONE valid response, not
wording that must be copied. A candidate that solves the sender's problem in
completely different words is fully correct. Judge substance, not overlap.

Score each dimension from 1 to 5 (1 = very poor, 5 = excellent):
- task_fulfillment: does the candidate address what the incoming email
  actually requires or asks for?
- action_alignment: does it preserve the important commitments and actions the
  reference reply represents, WITHOUT introducing consequential actions,
  approvals or promises that nothing supports?
- completeness: does it cover the information needed for an effective reply?
- tone: is it clear, concise, professional and appropriate to the exchange?

Set critical_error = true if the candidate does any of:
- contradicts the incoming email
- invents consequential facts, numbers, dates, prices or commitments
- falsely claims an attachment was included or an action was completed
- fundamentally misses the action the sender requested

Set acceptable = true only if a competent professional could send the reply
after at most trivial edits.

Return ONLY a JSON object with exactly these keys:
{"task_fulfillment": int, "action_alignment": int, "completeness": int,
 "tone": int, "critical_error": bool, "acceptable": bool, "reason": string}
"reason" must be one or two concise sentences."""

MAX_CHARS = 1500


def _clip(text: str, limit: int = MAX_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def build_judge_prompt(
    incoming_email: str,
    human_reply: str,
    candidate_reply: str,
    retrieved: list[RetrievedExample],
) -> str:
    grounding = (
        "\n\n".join(
            f"Example {i} (similarity {ex.similarity:.3f})\n"
            f"Incoming: {_clip(ex.incoming_email, 600)}\n"
            f"Human reply: {_clip(ex.human_reply, 600)}"
            for i, ex in enumerate(retrieved, start=1)
        )
        or "(none retrieved)"
    )
    return (
        f"## INCOMING EMAIL\n{_clip(incoming_email)}\n\n"
        f"## REFERENCE HUMAN REPLY (one valid answer, not a wording target)\n"
        f"{_clip(human_reply)}\n\n"
        f"## CANDIDATE REPLY TO EVALUATE\n{_clip(candidate_reply)}\n\n"
        f"## GROUNDING EXAMPLES THE SYSTEM SAW\n{grounding}\n\n"
        "Return the JSON object now."
    )


class RubricJudge:
    def __init__(self, client: LLMClient, temperature: float = 0.0) -> None:
        self.client = client
        self.temperature = temperature

    def judge(
        self,
        incoming_email: str,
        human_reply: str,
        candidate_reply: str,
        retrieved: list[RetrievedExample] | None = None,
    ) -> JudgeVerdict:
        """Score one candidate. Raises on transport failure or malformed JSON.

        Failing loudly matters: a silently defaulted verdict would quietly
        inflate or deflate the headline number.
        """
        if not candidate_reply or not candidate_reply.strip():
            # An empty candidate is a definite failure, not an LLM question.
            return JudgeVerdict(
                task_fulfillment=1,
                action_alignment=1,
                completeness=1,
                tone=1,
                critical_error=True,
                acceptable=False,
                reason="Empty candidate reply; nothing was produced.",
            )
        prompt = build_judge_prompt(
            incoming_email, human_reply, candidate_reply, retrieved or []
        )
        raw = self.client.complete(
            system=JUDGE_SYSTEM_PROMPT,
            user=prompt,
            model=self.client.judge_model,
            temperature=self.temperature,
            max_tokens=1200,
            json_mode=True,
        )
        payload = parse_json_object(raw)  # ValueError on malformed output
        return coerce_verdict(payload)  # pydantic ValidationError on bad ranges


__all__ = ["RubricJudge", "build_judge_prompt", "JUDGE_SYSTEM_PROMPT", "LLMError"]
