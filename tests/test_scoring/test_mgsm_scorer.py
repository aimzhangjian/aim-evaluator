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
"""MGSM answer extraction: multilingual markers, formats, and false-positive guards."""

import pytest

from nemo_evaluator.benchmarks.mgsm import _extract_answer, mgsm_scorer
from nemo_evaluator.scoring import ScorerInput


def score(response: str, target: str) -> dict:
    return mgsm_scorer(ScorerInput(response=response, target=target))


class TestMultilingualMarkers:
    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("...\nAnswer: 42", "42"),
            ("...\nAntwort: 42", "42"),
            ("...\nAntwort ist 42", "42"),
            ("...\nRéponse: 42", "42"),
            ("...\nReponse est 42", "42"),
            ("...\nRespuesta: 42", "42"),
            ("...\nRespuesta es 42", "42"),
            ("...\nОтвет: 42", "42"),
            ("...\nОтвет: 42", "42"),
            ("...\nJibu: 42", "42"),
            ("...\n答案是 42", "42"),
            ("...\n答案：42", "42"),
            ("...\n答えは42", "42"),
            ("...\nคำตอบคือ 42", "42"),
            ("...\nజవాబు: 42", "42"),
            ("...\nউত্তর: 42", "42"),
        ],
    )
    def test_native_markers(self, response, expected):
        assert _extract_answer(response)[0] == expected

    def test_last_statement_wins_across_languages(self):
        response = "Respuesta: 5.\n\nNo, wait. Respuesta: 7."
        assert _extract_answer(response)[0] == "7"


class TestFormats:
    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("...\nAnswer: 42 dollars.", "42"),
            ("...\nAnswer: $42.", "42"),
            ("...\nAnswer: 42 apples", "42"),
            ("...\nAnswer: 1,234", "1234"),
            ("...\nAnswer: 3.50", "3.5"),
            ("...\nAnswer: 3/4", "0.75"),
            ("...\nAnswer: 1 1/2", "1.5"),
            ("...\nAnswer: -3", "-3"),
            ("...\nAnswer: **42**", "42"),
            ("...\nAnswer: \\boxed{42}", "42"),
            ("...\n#### 42", "42"),
            ("Answer: 6*7 = 42", "42"),
            ("...\nAnswer: 42%", "42"),
        ],
    )
    def test_robust_formats(self, response, expected):
        assert _extract_answer(response)[0] == expected

    def test_tail_without_marker(self):
        response = "6 boxes of 7 apples.\n6 * 7 = 42\nSo there are 42 apples in total."
        assert _extract_answer(response)[0] == "42"

    def test_tail_pure_number_line(self):
        assert _extract_answer("First double 21.\nThen subtract nothing.\n42")[0] == "42"

    def test_empty_response(self):
        assert _extract_answer("") == ("", "empty")


class TestScorer:
    def test_correct_basic(self):
        assert score("Answer: 15", "15")["correct"] is True

    def test_correct_ignores_distractor_number_in_reasoning(self):
        response = "I checked twice, step 3 gives 4, then add to get 15.\nAnswer: 15"
        assert score(response, "15")["correct"] is True

    def test_distractor_in_tail_is_overridden_by_marker(self):
        response = "The total is 15, but I need 2 more boxes.\nAnswer: 15"
        assert score(response, "15")["correct"] is True

    def test_wrong_answer(self):
        assert score("Answer: 16", "15")["correct"] is False

    def test_fraction_equivalence(self):
        assert score("Answer: 3/4", "0.75")["correct"] is True
        assert score("Answer: 0.75", "3/4")["correct"] is True

    def test_thousands_separator(self):
        assert score("Answer: 1,234", "1234")["correct"] is True

    def test_no_number_in_response(self):
        result = score("I cannot solve this problem.", "15")
        assert result["correct"] is False
        assert result["extracted"] == "I cannot solve this problem."

    def test_score_shape(self):
        result = score("Answer: 42", "42")
        assert set(result) == {"correct", "extracted"}
        assert isinstance(result["extracted"], str)
