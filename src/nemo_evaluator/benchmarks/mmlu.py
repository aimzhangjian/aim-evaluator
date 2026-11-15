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
"""MMLU -- Massive Multitask Language Understanding (14K 4-choice QA)."""

import re

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput

_PROMPT = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)

# High-priority declarations: an explicit "the answer is B" / "\boxed{B}" / "答案：B"
# overrides any bare letter found elsewhere in the reasoning.
_HIGH_PRIORITY = re.compile(
    r"\\boxed\s*\{\s*(?P<boxed>[A-Da-d])\s*\}"
    r"|(?:"
    r"(?:(?:final\s+|correct\s+)?answer\s*(?:is\s*:?|=|:)\s*"
    r"|(?:正确答案|答案)\s*(?:是|为|:|：|=)\s*"
    r"|ans\s*:?\s*)"
    r"\(?\s*(?P<declared>[A-Da-d])\s*\)?(?![A-Za-z])"
    r")",
    re.IGNORECASE,
)

# Lower-priority positional forms: "(B)", "B) ...", or a bare "B" on its own line.
_LOW_PRIORITY = re.compile(
    r"\((?P<paren>[A-Da-d])\)"
    r"|^[\s>*-]*(?P<line>[A-Da-d])[\.\)]\s*(?=[A-Za-z]|$)"
    r"|^[\s>*-]*(?P<bare>[A-Da-d])\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_letter(response: str) -> str | None:
    """Extract the model's ABCD choice, tolerant of common answer formats."""
    text = re.sub(r"[*`]", "", (response or "").strip())
    if not text:
        return None
    for pattern in (_HIGH_PRIORITY, _LOW_PRIORITY):
        matches = list(pattern.finditer(text))
        if matches:
            match = matches[-1]
            return next(g for g in match.groups() if g is not None).upper()
    return None


def _prepare(row, idx, rng):
    c = row["choices"]
    return {
        **row,
        "Question": row["question"],
        "A": c[0],
        "B": c[1],
        "C": c[2],
        "D": c[3],
        "answer": "ABCD"[row["answer"]],
        "category": row.get("subject", ""),
    }


@benchmark(
    name="mmlu",
    dataset="hf://cais/mmlu?config=all&split=test",
    prompt=_PROMPT,
    target_field="answer",
    prepare_row=_prepare,
)
@scorer
def mmlu_scorer(sample: ScorerInput) -> dict:
    extracted = _extract_letter(sample.response)
    return {
        "correct": extracted == str(sample.target).upper(),
        "extracted": extracted,
    }
