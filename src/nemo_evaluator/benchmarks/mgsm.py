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
"""MGSM -- Multilingual Grade School Math (11 languages)."""

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput

_LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "th", "te", "bn", "sw"]

_INSTRUCTIONS = {
    "en": "Solve this math problem. Give reasoning steps then the final answer on the last line as 'Answer:'.\n\n{input}",
    "de": "Lose dieses Problem. Gib Schritte an, dann die Antwort als 'Antwort:'.\n\n{input}",
    "fr": "Resolvez ce probleme. Donnez les etapes puis la reponse comme 'Reponse:'.\n\n{input}",
    "es": "Resuelve este problema. Da los pasos y la respuesta como 'Respuesta:'.\n\n{input}",
    "ru": "Reshite etu zadachu. Privedite shagi, zatem otvet kak 'Otvet:'.\n\n{input}",
    "zh": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "ja": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "th": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "te": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "bn": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "sw": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
}

# Answer declarations, most authoritative first. The markers are the native word for
# "answer" in each MGSM language, so an in-language reply is still captured. Models
# frequently restate the answer at the end, so for every marker the *last* parseable
# match wins.
_MARKERS = (
    r"####\s*(?P<answer>[^\n]+)",
    r"\\boxed\s*\{(?P<answer>[^{}]*)\}",
    r"(?i)\bfinal\s+answer\s*(?:is|:|=)\s*(?P<answer>[^\n]+)",
    r"(?i)\b(?:the\s+)?answer\s*(?:is|:|=)\s*(?P<answer>[^\n]+)",
    r"(?i)\bantwort\s*(?:ist|:|=)\s*(?P<answer>[^\n]+)",
    r"(?i)\br[ée]ponse\s*(?:est|:|=)\s*(?P<answer>[^\n]+)",
    r"(?i)\brespuesta\s*(?:es|:|=)\s*(?P<answer>[^\n]+)",
    r"(?i)\bответ\s*(?::|=|—|-)?\s*(?P<answer>[^\n]+)",
    r"(?i)\b(?:jibu|jibu\s+ni)\s*(?::|=)\s*(?P<answer>[^\n]+)",
    r"(?:最终)?答案\s*(?:是|为|:|=)\s*(?P<answer>[^\n]+)",
    r"答え\s*(?:は|:|=)\s*(?P<answer>[^\n]+)",
    r"คำตอบ\s*(?:คือ|:|=)\s*(?P<answer>[^\n]+)",
    r"జవాబు\s*(?::|=)\s*(?P<answer>[^\n]+)",
    r"উত্তর\s*(?::|=)\s*(?P<answer>[^\n]+)",
)

_TRANSLATE = str.maketrans({"−": "-", "–": "-", "—": "-", "﹣": "-", "\u00a0": " "})

# Fractions/mixed numbers come first so "3/4" is not read as two separate integers.
_NUMBER_RE = re.compile(
    r"(?P<mixed>-?\d+\s+\d+\s*/\s*\d+)"
    r"|(?P<frac>-?\d+\s*/\s*\d+)"
    r"|(?P<num>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
)

_MAX_EXTRACT_CHARS = 120

# "6*7 = 42" starts with an expression rather than the answer; "3/4" does not.
_LEADING_EXPR = re.compile(r"^\s*[^\d]{0,8}-?\d[\d,]*(?:\.\d+)?\s*(?:[+*×÷]|-(?!\d)|/(?!\d))")


def _clean(text: str) -> str:
    """Strip the LaTeX/markdown decoration that usually wraps a numeric answer."""
    text = text.translate(_TRANSLATE)
    for _ in range(2):
        text = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\(?:boxed|text|mathrm|mathbf|textbf|operatorname)\s*", "", text)
    text = re.sub(r"\\(?:left|right|quad|qquad|,|;|!)\s*", "", text)
    text = re.sub(r"\\(?:times|cdot)\s*", "*", text)
    text = re.sub(r"[\${}`]", "", text)
    text = text.replace("**", "")
    return re.sub(r"[ \t\u00a0]+", " ", text)


def _to_fraction(raw: str) -> Fraction | None:
    s = raw.strip().replace(" ", "").replace(",", "")
    if "/" in s:
        parts = s.split("/")
        if len(parts) != 2:
            return None
        try:
            return Fraction(Decimal(parts[0])) / Fraction(Decimal(parts[1]))
        except (InvalidOperation, ZeroDivisionError):
            return None
    try:
        return Fraction(Decimal(s))
    except InvalidOperation:
        return None


def _numbers(text: str) -> list[Fraction]:
    out: list[Fraction] = []
    for m in _NUMBER_RE.finditer(text):
        token = m.group(0).strip()
        if m.lastgroup == "mixed":
            whole, rest = token.split(None, 1)
            frac = _to_fraction(rest)
            if frac is None:
                continue
            out.append((1 if int(whole) >= 0 else -1) * (abs(int(whole)) + abs(frac)))
            continue
        value = _to_fraction(token)
        if value is not None:
            out.append(value)
    return out


def _format(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return format((Decimal(value.numerator) / Decimal(value.denominator)).normalize(), "f")


def _pick_from_candidate(candidate: str) -> Fraction | None:
    """Pick the answer from a declared-answer fragment, RHS of any trailing '=' wins."""
    if _LEADING_EXPR.match(candidate):
        rhs = _numbers(candidate.rsplit("=", 1)[-1]) if "=" in candidate else []
        if rhs:
            return rhs[-1]
        values = _numbers(candidate)
        return values[-1] if values else None
    values = _numbers(candidate)
    return values[0] if values else None


def _extract_answer(response: str) -> tuple[str, str]:
    """Return (canonical answer, extraction source). The source aids debugging."""
    text = _clean(response or "")
    if not text.strip():
        return "", "empty"

    fallback_text = ""
    for marker in _MARKERS:
        candidates = [_clean(m.group("answer")) for m in re.finditer(marker, text)]
        usable = [c for c in candidates if _pick_from_candidate(c)]
        if usable:
            value = _pick_from_candidate(usable[-1])
            return _format(value), "marker"
        if candidates and not fallback_text:
            fallback_text = candidates[-1].strip()

    # No usable marker: fall back to the closing lines, where a solver normally leaves
    # its result ("... so 6*3 = 18").
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines[-3:]):
        values = _numbers(line.rsplit("=", 1)[-1]) or _numbers(line)
        if values:
            return _format(values[-1]), "tail"

    values = _numbers(text)
    if values:
        return _format(values[-1]), "fallback"

    return (fallback_text or text.strip())[:_MAX_EXTRACT_CHARS], "none"


def _load_mgsm():
    from datasets import load_dataset

    rows = []
    for lang in _LANGS:
        ds = load_dataset("juletxara/mgsm", lang, split="test", trust_remote_code=True)
        for row in ds:
            answer = str(row["answer_number"]).rstrip("0").rstrip(".")
            rows.append({"question": row["question"], "answer": answer, "language": lang})
    return rows


def _seed_mgsm(row, idx):
    lang = row["language"]
    template = _INSTRUCTIONS.get(lang, _INSTRUCTIONS["en"])
    prompt = template.format(input=row["question"])
    return SeedResult(
        prompt=prompt,
        expected_answer=row["answer"],
        messages=[{"role": "user", "content": prompt}],
        metadata={"category": lang},
    )


@benchmark(name="mgsm", dataset=_load_mgsm, target_field="answer", prompt="", seed_fn=_seed_mgsm)
@scorer
def mgsm_scorer(sample: ScorerInput) -> dict:
    extracted, _source = _extract_answer(sample.response)
    target = _numbers(_clean(str(sample.target)))
    got = _to_fraction(extracted) if extracted else None

    if got is not None and target:
        correct = got == target[0]
    else:
        correct = extracted.strip().lower() == str(sample.target).strip().lower()

    return {"correct": correct, "extracted": extracted}
