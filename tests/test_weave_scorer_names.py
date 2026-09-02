# SPDX-FileCopyrightText: 2026 Yusef Syed
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Regression coverage for configured RAI scorer names in Weave evaluations."""

import pytest

try:
    import weave
except ModuleNotFoundError:
    weave = None

from rai_toolkit.scorers.base import BaseScorer, ScorerResult


class _ConfiguredScoreScorer(BaseScorer):
    def score(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: object,
    ) -> ScorerResult:
        score = 1.0 if self.name == "alpha" else 0.0
        return ScorerResult(
            score=score,
            passed=bool(score),
            category="",
            explanation=self.name,
        )


@pytest.mark.asyncio
@pytest.mark.skipif(weave is None, reason="requires the weave extra")
async def test_weave_evaluation_keeps_configured_names_for_same_scorer_class() -> None:
    from integrations.weave_integration.scorers import make_weave_rai_scorer

    class StaticWeaveModel(weave.Model):  # type: ignore[union-attr,misc]
        @weave.op()  # type: ignore[union-attr]
        async def predict(self, input_text: str) -> dict[str, str]:
            return {"output": "response"}

    evaluation = weave.Evaluation(
        dataset=[{"input_text": "prompt"}],
        scorers=[
            make_weave_rai_scorer(_ConfiguredScoreScorer(name="alpha")),
            make_weave_rai_scorer(_ConfiguredScoreScorer(name="beta")),
        ],
    )

    results = await evaluation.get_eval_results(StaticWeaveModel())
    row = next(iter(results.rows))

    assert set(row["scores"]) == {"alpha", "beta"}
    assert row["scores"]["alpha"]["score"] == 1.0
    assert row["scores"]["beta"]["score"] == 0.0

    summary = await evaluation.summarize(results)

    assert summary["alpha"]["score"]["mean"] == 1.0
    assert summary["beta"]["score"]["mean"] == 0.0
