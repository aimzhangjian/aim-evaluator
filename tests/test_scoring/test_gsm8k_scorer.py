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
"""GSM8K answer extraction: format robustness and false-positive guards."""

import pytest

from nemo_evaluator.benchmarks.gsm8k import _extract_answer, _prepare, gsm8k_scorer
from nemo_evaluator.scoring import ScorerInput


def score(response: str, target: str) -> dict:
    return gsm8k_scorer(ScorerInput(response=response, target=target))


class TestExtractAnswer:
    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("...\nThe answer is 42", "42"),
            ("...\nThe answer is 42.", "42"),
            ("...\nAnswer: 42", "42"),
            ("...\nAnswer: 42 apples", "42"),
            ("...\nThe answer is $42.", "42"),
            ("...\nThe answer is 42 dollars.", "42"),
            ("...\nThe final answer is: 42", "42"),
            ("...\nThe answer is approximately 42", "42"),
            ("...\nThe answer is **42**", "42"),
            ("...\nThe answer is \\boxed{42}", "42"),
            ("...\n\\boxed{42}", "42"),
            ("...\n#### 42", "42"),
            ("...\n答案是 42", "42"),
            ("...\n答案：42个", "42"),
            ("...\nThe answer is -3", "-3"),
            ("...\nThe answer is 1,234", "1234"),
            ("...\nThe answer is 3.50", "3.5"),
            ("...\nThe answer is 3/4", "0.75"),
            ("...\nThe answer is 1 1/2", "1.5"),
            ("The answer is 6*7 = 42", "42"),
            ("42", "42"),
        ],
    )
    def test_formats(self, response, expected):
        assert _extract_answer(response)[0] == expected

    def test_last_statement_wins(self):
        response = "The answer is 5.\n\nWait, I forgot the last step.\nThe answer is 7."
        assert _extract_answer(response)[0] == "7"

    def test_ignores_echoed_instruction(self):
        response = (
            "Put your final numerical answer after 'The answer is'.\n\n"
            "There are 6 boxes with 7 apples each, so 6*7=42.\n"
            "The answer is 42"
        )
        assert _extract_answer(response)[0] == "42"

    def test_tail_without_marker(self):
        response = "6 boxes of 7 apples.\n6 * 7 = 42\nSo there are 42 apples in total."
        assert _extract_answer(response)[0] == "42"

    def test_tail_without_marker_or_equals(self):
        response = "First double 21 to get 42.\nThen subtract nothing.\n42"
        assert _extract_answer(response)[0] == "42"

    def test_empty_response(self):
        assert _extract_answer("") == ("", "empty")
        assert _extract_answer("   \n ") == ("", "empty")


class TestScorer:
    def test_correct_with_units(self):
        assert score("The answer is 42 dollars.", "42")["correct"] is True
        assert score("The answer is 42 dollars.", "42")["extracted"] == "42"

    def test_correct_thousands_separator(self):
        assert score("The answer is 1,234.", "1234")["correct"] is True
        assert score("The answer is 1234.", "1,234")["correct"] is True

    def test_correct_fraction_target(self):
        assert score("The answer is 3/4", "0.75")["correct"] is True
        assert score("The answer is 0.75", "3/4")["correct"] is True

    def test_wrong_answer(self):
        assert score("The answer is 43", "42")["correct"] is False

    def test_wrong_answer_mentioned_in_reasoning(self):
        result = score("42 is close, but 43 is my conclusion.\nThe answer is 43", "42")
        assert result["correct"] is False

    def test_no_number_in_response(self):
        result = score("I cannot solve this problem.", "42")
        assert result["correct"] is False
        assert result["extracted"] == "I cannot solve this problem."

    def test_case_insensitive_fallback_for_non_numeric_target(self):
        assert score("The answer is Unknown", "unknown")["correct"] is True

    def test_score_is_json_serializable(self):
        result = score("The answer is 42", "42")
        assert set(result) == {"correct", "extracted"}
        assert isinstance(result["extracted"], str)


class TestPrepare:
    def test_extracts_after_hash_marker(self):
        row = {"question": "q", "answer": "Natalia sold 48 clips.\n#### 48"}
        assert _prepare(row, 0, None)["answer"] == "48"

    def test_strips_commas_and_dollar(self):
        assert _prepare({"answer": "#### $1,234"}, 0, None)["answer"] == "1234"

    def test_strips_trailing_period(self):
        assert _prepare({"answer": "#### 42."}, 0, None)["answer"] == "42"

    def test_keeps_raw_when_no_marker(self):
        assert _prepare({"answer": "42"}, 0, None)["answer"] == "42"


class TestExpressionVsAnswer:
    def test_expression_after_marker(self):
        assert _extract_answer("Answer: 6*7 = 42")[0] == "42"

    def test_leading_answer_then_restatement(self):
        assert _extract_answer("The answer is 42. Check: 6*7 = 42.")[0] == "42"

    def test_leading_answer_then_unrelated_product(self):
        assert _extract_answer("The answer is 7, although 7*6 = 42")[0] == "7"

    def test_fraction_is_not_an_expression(self):
        assert _extract_answer("The answer is 3/4")[0] == "0.75"

    def test_division_expression_after_equals(self):
        assert _extract_answer("Answer: 84/2 = 42")[0] == "42"
