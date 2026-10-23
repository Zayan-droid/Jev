"""A dependency-free client for ``decider-serve``.

from decider.client import DeciderClient, choice, noul, score

client = DeciderClient("http://127.0.0.1:8000")
result = client.decide(
    {"ticket": "Order 4417 arrived broken"},
    refund=noul("Does `ticket` ask for a refund?"),
    team=choice("Which team should handle `ticket`?",
                billing="payments and refunds", returns="damaged or wrong items"),
    urgency=score("How urgent is `ticket`?", ["can wait", "today", "immediately"]),
)
result.answers["team"].choice
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from decider.schema import (
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulQuestion,
    Question,
    ScoreQuestion,
    State,
)


def noul(instructions: str) -> NoulQuestion:
    return NoulQuestion(instructions=instructions)


def choice(
    instructions: str, criteria: dict[str, str] | None = None, **options: str
) -> ChoiceQuestion:
    return ChoiceQuestion(instructions=instructions, criteria={**(criteria or {}), **options})


def score(instructions: str, levels: list[str]) -> ScoreQuestion:
    return ScoreQuestion(instructions=instructions, levels=levels)


class DeciderError(RuntimeError):
    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class DeciderClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"content-type": "application/json"}
        if request_id is not None:
            headers["x-request-id"] = request_id
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise DeciderError(exc.code, detail) from None

    def decide(self, state: State, **questions: Question) -> DecisionResponse:
        return self.send(DecisionRequest(state=state, questions=questions))

    def send(self, request: DecisionRequest, request_id: str | None = None) -> DecisionResponse:
        """``request_id`` is echoed back and keys the server's logs (default: a random id)."""
        body = request.model_dump(mode="json")
        payload = self._request("POST", "/v1/decide", body, request_id)
        return DecisionResponse.model_validate(payload)

    def model(self) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", "/v1/model")
        return result

    def ready(self) -> bool:
        try:
            self._request("GET", "/readyz")
        except (DeciderError, OSError):
            return False
        return True
