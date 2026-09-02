# SPDX-FileCopyrightText: 2026 Yusef Syed
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import Any

import pytest

from rai_toolkit.assessment import Assessor
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.scorers.base import BaseScorer, ScorerResult


class StubModel(BaseModel):
    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> ModelResponse:
        return ModelResponse(output="stub")


class StubScorer(BaseScorer):
    def score(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        return ScorerResult(
            score=1.0,
            passed=True,
            category="custom",
            explanation="stub",
        )


def _install_fake_weave_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adapter: Any,
    captured: dict[str, Any],
    expected_result: object,
) -> None:
    weave_package = ModuleType("integrations.weave_integration")
    weave_package.__path__ = []  # type: ignore[attr-defined]

    evaluation_module = ModuleType("integrations.weave_integration.evaluation")

    class FakeWeaveEvaluationRunner:
        def __init__(self, engine: object) -> None:
            self.engine = engine

        async def get_detailed_evaluation(
            self,
            model: object,
            profile: object,
            dataset: object,
            **kwargs: Any,
        ) -> tuple[object, dict[str, Any]]:
            captured.update(kwargs)
            return object(), {}

    evaluation_module.WeaveEvaluationRunner = FakeWeaveEvaluationRunner  # type: ignore[attr-defined]

    scorers_module = ModuleType("integrations.weave_integration.scorers")
    scorers_module.make_weave_rai_scorer = adapter  # type: ignore[attr-defined]

    models_module = ModuleType("integrations.weave_integration.models")
    models_module.WeaveModel = lambda **kwargs: object()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "integrations.weave_integration", weave_package)
    monkeypatch.setitem(
        sys.modules,
        "integrations.weave_integration.evaluation",
        evaluation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "integrations.weave_integration.scorers",
        scorers_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "integrations.weave_integration.models",
        models_module,
    )

    from rai_toolkit.evaluation import weave_adapter

    monkeypatch.setattr(
        weave_adapter,
        "weave_eval_results_to_evaluation_results",
        lambda *args, **kwargs: expected_result,
    )


@pytest.mark.asyncio
async def test_weave_evaluation_receives_adapted_additional_scorers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = StubScorer()
    adapted_scorer = object()
    captured: dict[str, Any] = {}
    expected_result = object()
    _install_fake_weave_modules(
        monkeypatch,
        adapter=lambda value: adapted_scorer if value is scorer else None,
        captured=captured,
        expected_result=expected_result,
    )

    assessor = Assessor(
        model=StubModel(),
        preset="general",
        datasets=["unused"],
        additional_scorers=[scorer],
    )
    monkeypatch.setattr(assessor, "_should_use_weave_evaluation", lambda: True)
    profile = assessor.engine.create_profile_from_preset("general")

    result = await assessor._run_evaluation(profile, [{"input": "hello"}])

    assert result is expected_result
    assert captured["additional_scorers"] == [adapted_scorer]


@pytest.mark.asyncio
async def test_weave_evaluation_warns_and_keeps_adaptable_scorers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adaptable = StubScorer(name="adaptable")
    unsupported = StubScorer(name="unsupported")
    adapted_scorer = object()
    captured: dict[str, Any] = {}
    expected_result = object()

    def adapt(scorer: StubScorer) -> object:
        if scorer is unsupported:
            raise TypeError("unsupported scorer")
        return adapted_scorer

    _install_fake_weave_modules(
        monkeypatch,
        adapter=adapt,
        captured=captured,
        expected_result=expected_result,
    )
    assessor = Assessor(
        model=StubModel(),
        preset="general",
        datasets=["unused"],
        additional_scorers=[adaptable, unsupported],
    )
    monkeypatch.setattr(assessor, "_should_use_weave_evaluation", lambda: True)
    profile = assessor.engine.create_profile_from_preset("general")

    with caplog.at_level(logging.WARNING):
        result = await assessor._run_evaluation(profile, [{"input": "hello"}])

    assert result is expected_result
    assert captured["additional_scorers"] == [adapted_scorer]
    assert "Could not adapt additional scorer StubScorer for Weave" in caplog.text
