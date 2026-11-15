# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MMLU answer extraction: format tolerance and false-positive guards."""

import pytest

from nemo_evaluator.benchmarks.mmlu import _extract_letter, mmlu_scorer
from nemo_evaluator.scoring import ScorerInput


def score(response: str, target: str) -> dict:
    return mmlu_scorer(ScorerInput(response=response, target=target))


class TestExtractLetter:
    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("Answer: C", "C"),
            ("Answer: c", "C"),
            ("The answer is C", "C"),
            ("The answer is: C", "C"),
            ("The correct answer is C", "C"),
            ("The final answer is C", "C"),
            ("Answer: (C)", "C"),
            ("Answer: **C**", "C"),
            ("Answer: C.", "C"),
            ("Ans: C", "C"),
            ("\\boxed{C}", "C"),
            ("\\boxed{C}\n", "C"),
            ("答案是 C", "C"),
            ("正确答案是 C", "C"),
            ("答案：C", "C"),
            ("(C)", "C"),
            ("C) The capital is Paris", "C"),
            ("C", "C"),
            ("...\nC", "C"),
            ("...\nC.", "C"),
        ],
    )
    def test_formats(self, response, expected):
        assert _extract_letter(response) == expected

    def test_last_declaration_wins(self):
        response = "Answer: A\n\nWait, reconsidering. Answer: B"
        assert _extract_letter(response) == "B"

    def test_declaration_overrides_trailing_paren(self):
        response = "The answer is A, although option (D) also sounds plausible."
        assert _extract_letter(response) == "A"

    def test_no_letter_in_word(self):
        assert _extract_letter("The word 'both' contains a B.") is None

    def test_ambiguous_both_is_ignored(self):
        assert _extract_letter("Both A and B could work.") is None

    def test_empty_response(self):
        assert _extract_letter("") is None
        assert _extract_letter("   ") is None


class TestScorer:
    def test_correct(self):
        assert score("Answer: C", "C")["correct"] is True

    def test_correct_case_insensitive(self):
        assert score("Answer: c", "C")["correct"] is True

    def test_wrong(self):
        assert score("Answer: D", "C")["correct"] is False

    def test_no_answer_found(self):
        result = score("I'm not sure about this question.", "C")
        assert result["correct"] is False
        assert result["extracted"] is None

    def test_restated_answer(self):
        assert score("Answer: A ... Actually, the answer is C.", "C")["correct"] is True

    def test_score_shape(self):
        result = score("Answer: C", "C")
        assert set(result) == {"correct", "extracted"}
