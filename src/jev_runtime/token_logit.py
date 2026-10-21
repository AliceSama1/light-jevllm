"""Token-logit Decision adapter and answer decoding."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .decision_ir import Candidate, CompiledQuestion, ReadoutResult
from .heads.readouts import normalized_entropy, stable_softmax


class TokenLogitError(ValueError):
    """The prompt or backend output cannot support strict candidate readout."""


@dataclass(frozen=True)
class TokenLogitQuestion:
    question: CompiledQuestion
    prompt: str
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.id for candidate in self.question.candidates)


class TokenLogitAdapter:
    """Compile labels to exactly one token and decode candidate logits."""

    def __init__(self, encode: Callable[[str], Sequence[int]]):
        self._encode = encode

    def compile(self, question_id: str, kind: str, prompt: str,
                candidates: Sequence[Candidate | str], *,
                instructions: str | None = None) -> TokenLogitQuestion:
        normalized = tuple(candidate if isinstance(candidate, Candidate) else Candidate(str(candidate)) for candidate in candidates)
        if len(normalized) < 2:
            raise TokenLogitError("a token-logit question needs at least two candidates")
        if len({candidate.id for candidate in normalized}) != len(normalized):
            raise TokenLogitError("candidate IDs must be unique")
        if kind not in ("noul", "choice", "score"):
            raise TokenLogitError(f"unsupported question kind: {kind}")
        if kind == "noul" and {candidate.id for candidate in normalized} != {"true", "false"}:
            raise TokenLogitError("noul candidates must be exactly 'true' and 'false'")
        token_ids: list[int] = []
        token_texts: list[str] = []
        for candidate in normalized:
            text = candidate.label if candidate.label is not None else candidate.id
            encoded = tuple(self._encode(text))
            if len(encoded) != 1 or isinstance(encoded[0], bool) or not isinstance(encoded[0], int):
                raise TokenLogitError(f"candidate {candidate.id!r} must encode to exactly one integer token")
            token_ids.append(encoded[0])
            token_texts.append(text)
        if len(set(token_ids)) != len(token_ids):
            raise TokenLogitError("candidate labels must map to distinct token IDs")
        compiled = CompiledQuestion(question_id, kind, normalized, instructions)  # type: ignore[arg-type]
        return TokenLogitQuestion(compiled, prompt, tuple(token_ids), tuple(token_texts))

    def decode(self, item: TokenLogitQuestion, logits: Sequence[float]) -> dict[str, Any]:
        values = tuple(float(value) for value in logits)
        if len(values) != len(item.candidate_ids):
            raise TokenLogitError("backend did not return every candidate logit")
        if any(not math.isfinite(value) for value in values):
            raise TokenLogitError("backend returned a non-finite candidate logit")
        probabilities = stable_softmax(values)
        probability_map = dict(zip(item.candidate_ids, probabilities, strict=True))
        confidence = normalized_entropy(probabilities)
        kind = item.question.kind
        common = {"probabilities": probability_map, "confidence": confidence,
                  "confidence_type": "normalized_entropy", "calibration_version": None}
        if kind == "noul":
            return {"type": "noul", "noul": float(probability_map["true"]), **common}
        if kind == "score":
            try:
                score = sum(float(key) * probability_map[key] for key in item.candidate_ids)
            except ValueError as exc:
                raise TokenLogitError("score candidate IDs must be numeric") from exc
            if not math.isfinite(score):
                raise TokenLogitError("score must be finite")
            return {"type": "score", "score": score, **common,
                    "legend": {candidate.id: str(candidate.description)
                               for candidate in item.question.candidates
                               if candidate.description is not None}}
        winner = item.candidate_ids[max(range(len(values)), key=values.__getitem__)]
        return {"type": "choice", "choice": winner, **common}

    @staticmethod
    def result_logits(item: TokenLogitQuestion, result: ReadoutResult) -> tuple[float, ...]:
        if result.question_id != item.question.id:
            raise TokenLogitError("backend result question ID does not match the request")
        if result.candidate_ids != item.candidate_ids:
            raise TokenLogitError("backend candidate order does not match the request")
        return result.logits


__all__ = ["TokenLogitAdapter", "TokenLogitError", "TokenLogitQuestion"]
