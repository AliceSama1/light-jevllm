"""Strict vLLM System One response validation and local mapping."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .vllm_systemone import VLLMSystemOneError


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise VLLMSystemOneError(f"invalid {field}", kind="response")
    return float(value)


def _probabilities(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise VLLMSystemOneError("probabilities must be a non-empty object", kind="response")
    result = {str(key): _number(item, "probability") for key, item in value.items()}
    if any(item < 0 or item > 1 for item in result.values()) or not math.isclose(sum(result.values()), 1.0, abs_tol=1e-5):
        raise VLLMSystemOneError("probabilities must be normalized", kind="response")
    return result


def _answer(answer: Any) -> dict[str, Any]:
    if not isinstance(answer, Mapping):
        raise VLLMSystemOneError("answer must be an object", kind="response")
    allowed = {"type", "choice", "noul", "score", "probabilities", "confidence", "legend", "error", "message"}
    if set(answer) - allowed:
        raise VLLMSystemOneError("answer contains unsupported fields", kind="response")
    kind = answer.get("type")
    if kind not in ("choice", "noul", "score"):
        raise VLLMSystemOneError("answer type is incompatible", kind="response")
    if "error" in answer:
        if not isinstance(answer["error"], str) or not answer["error"]:
            raise VLLMSystemOneError("answer error is invalid", kind="response")
        return dict(answer)
    result = dict(answer)
    probabilities = _probabilities(answer.get("probabilities"))
    result["probabilities"] = probabilities
    confidence = _number(answer.get("confidence"), "confidence")
    if not 0 <= confidence <= 1:
        raise VLLMSystemOneError("confidence must be between zero and one", kind="response")
    if kind == "choice" and not isinstance(answer.get("choice"), str):
        raise VLLMSystemOneError("choice answer is missing choice", kind="response")
    if kind == "noul":
        result["noul"] = _number(answer.get("noul"), "noul")
        if not 0 <= result["noul"] <= 1:
            raise VLLMSystemOneError("noul must be between zero and one", kind="response")
    if kind == "score":
        result["score"] = _number(answer.get("score"), "score")
        if "legend" not in answer or not isinstance(answer["legend"], Mapping):
            raise VLLMSystemOneError("score answer is missing legend", kind="response")
    return result


def map_systemone_response(body: Mapping[str, Any], expected_question_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Validate native response without inventing confidence metadata.

    In particular, ``confidence`` and its numeric meaning are copied verbatim;
    this adapter never derives normalized entropy or adds a calibration label.
    """
    if not isinstance(body, Mapping):
        raise VLLMSystemOneError("System One response must be an object", kind="response")
    if set(body) - {"model", "answers", "usage", "meta"}:
        raise VLLMSystemOneError("System One response contains unsupported fields", kind="response")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise VLLMSystemOneError("System One response model is invalid", kind="response")
    answers = body.get("answers")
    if not isinstance(answers, Mapping) or not answers:
        raise VLLMSystemOneError("System One answers must be a non-empty object", kind="response")
    if expected_question_ids is not None and list(answers) != list(expected_question_ids):
        raise VLLMSystemOneError("System One answer order does not match request", kind="response")
    result = dict(body)
    result["answers"] = {str(question_id): _answer(answer) for question_id, answer in answers.items()}
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        raise VLLMSystemOneError("System One usage is invalid", kind="response")
    return result


__all__ = ["map_systemone_response"]
