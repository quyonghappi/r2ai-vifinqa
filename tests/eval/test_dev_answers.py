import json
from pathlib import Path


def test_dev_v1_answers_cover_exactly_the_same_questions():
    root = Path(__file__).resolve().parents[2] / "eval" / "dev_questions"
    questions = [json.loads(x) for x in (root / "dev_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    answers = [json.loads(x) for x in (root / "dev_v1_answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {x["id"] for x in questions} == {x["id"] for x in answers}
    assert len(answers) == 18  # 14 original + 4 retrieval-representation-sprint additions (627, 591, 959, 961)
    assert all(isinstance(x["expected_answer"], (int, float)) and x["derivation"] for x in answers)
