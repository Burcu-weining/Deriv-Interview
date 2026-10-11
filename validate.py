
import json
from pathlib import Path

ARTIFACTS = Path("artifacts")

REQUIRED_FILES = [
    "documents.json",
    "chunks.json",
    "retrieval_results.json",
    "answers.json",
    "citation_validation.json",
    "final_answers.json",
]


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate():
    issues = []

    for filename in REQUIRED_FILES:
        if not (ARTIFACTS / filename).is_file():
            issues.append(f"Missing artifact: {filename}")

    if issues:
        return issues

    try:
        data = {
            filename: load_json(ARTIFACTS / filename)
            for filename in REQUIRED_FILES
        }
        questions = load_json(Path("questions.json"))
    except (ValueError, OSError) as error:
        return [f"Could not load JSON: {error}"]

    answers = data["answers.json"]
    retrieval = data["retrieval_results.json"]
    validations = data["citation_validation.json"]
    finals = data["final_answers.json"]

    question_ids = [q["id"] for q in questions]
    answer_ids = [a["id"] for a in answers]

    if len(answer_ids) != len(set(answer_ids)):
        issues.append("Duplicate answer IDs")

    if len(answers) != len(questions):
        issues.append("Incorrect number of answers")

    if set(question_ids) != set(answer_ids):
        issues.append("Question and answer IDs do not match")

    retrieval_ids = [r["id"] for r in retrieval]
    if len(retrieval_ids) != len(set(retrieval_ids)):
        issues.append("Duplicate retrieval IDs")
    if len(retrieval) != len(questions) or set(retrieval_ids) != set(question_ids):
        issues.append("Missing or extra retrieval results")

    retrieval_by_id = {r["id"]: r for r in retrieval}

    for result in retrieval:
        if len(result["retrieved_chunks"]) > 3:
            issues.append(f'Too many chunks for {result["id"]}')

    for answer in answers:
        question_id = answer["id"]
        retrieved = retrieval_by_id.get(question_id)

        if retrieved is None:
            continue

        valid_chunk_ids = {
            c["chunk_id"]
            for c in retrieved["retrieved_chunks"]
        }

        citations = answer["citations"]

        if any(c not in valid_chunk_ids for c in citations):
            issues.append(f"Invalid citation for {question_id}")

        if answer["supported"] and not citations:
            issues.append(f"Missing citation for {question_id}")

        if not answer["supported"] and citations:
            issues.append(f"Unsupported answer cites sources: {question_id}")

    validation_ids = [v["id"] for v in validations]
    final_ids = [f["id"] for f in finals]

    if (len(validation_ids) != len(question_ids)
            or set(validation_ids) != set(question_ids)):
        issues.append("Incorrect citation validation records")

    if len(final_ids) != len(question_ids) or set(final_ids) != set(question_ids):
        issues.append("Incorrect final answer records")

    for record in validations:
        if not record["passed"]:
            issues.append(f'Grounding validation failed: {record["id"]}')

    return issues


if __name__ == "__main__":
    problems = validate()

    if problems:
        print("Validation FAILED:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("Validation PASSED: all required checks successful.")
