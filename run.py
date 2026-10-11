import os
import re
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path("docs")
ARTIFACTS_DIR = Path("artifacts")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def save_json(filename, data):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(ARTIFACTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# STAGE 1: DOCUMENT INGESTION
def load_documents():
    documents = []

    for path in sorted(DOCS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".md", ".txt"):
            continue

        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(DOCS_DIR).as_posix()

        doc_id = hashlib.sha256(relative_path.encode()).hexdigest()[:12]

        documents.append({
            "doc_id": doc_id,
            "path": str(path),
            "text": text
        })

    save_json("documents.json", documents)
    return documents


# STAGE 2: CHUNKING
def create_chunks(documents):
    chunks = []

    for doc in documents:
        text = doc["text"]
        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))

            chunk_id = "chunk_" + hashlib.sha256(
                f'{doc["doc_id"]}:{start}:{end}'.encode()
            ).hexdigest()[:12]

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "text": text[start:end],
                "start_char": start,
                "end_char": end
            })

            if end == len(text):
                break

            start = end - CHUNK_OVERLAP

    save_json("chunks.json", chunks)
    return chunks


# STAGE 3: BUILD SEARCH INDEX
def build_index(chunks):
    if not chunks:
        raise ValueError("No chunks available. Check the docs/ folder.")

    vectorizer = TfidfVectorizer(stop_words="english")
    texts = [chunk["text"] for chunk in chunks]
    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


# STAGE 4: LOAD QUESTIONS
def load_questions():
    with open("questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    return questions


# STAGE 5: RETRIEVE TOP 3 CHUNKS
def retrieve_chunks(questions, chunks, vectorizer, matrix):
    results = []

    for question in questions:
        query_vector = vectorizer.transform([question["question"]])
        scores = cosine_similarity(query_vector, matrix)[0]

        top_indices = scores.argsort()[::-1][:3]

        retrieved = []

        for index in top_indices:
            # Ignore chunks with no matching terms
            if scores[index] <= 0:
                continue

            chunk = chunks[index]

            retrieved.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "score": round(float(scores[index]), 4),
                "text": chunk["text"]
            })

        results.append({
            "id": question["id"],
            "question": question["question"],
            "retrieved_chunks": retrieved
        })

    save_json("retrieval_results.json", results)
    return results


# STAGE 6: ANSWER GENERATION

UNSUPPORTED = (
    "The knowledge base does not provide enough "
    "information to answer this."
)


def local_answer(question, retrieved_chunks):
    """
    Conservative fallback when no API key is available.
    Uses a matching source sentence, not invented facts.
    """
    question_words = set(re.findall(
        r"\b[a-z0-9]+\b", question.lower()
    ))

    stop_words = {
        "what", "which", "where", "when", "does",
        "the", "a", "an", "is", "are", "can", "i",
        "you", "do", "for", "to", "of", "in", "how",
        "long", "there", "have", "has", "it", "if"
    }
    question_words -= stop_words

    best_sentence = None
    best_chunk = None
    best_overlap = 0

    for chunk in retrieved_chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])

        for sentence in sentences:
            sentence_words = set(re.findall(
                r"\b[a-z0-9]+\b", sentence.lower()
            ))
            overlap = len(question_words & sentence_words)

            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()
                best_chunk = chunk["chunk_id"]

    # Conservative threshold: weak matches are unsupported.
    if best_overlap < 2:
        return UNSUPPORTED, False, []

    return best_sentence, True, [best_chunk]


def generate_answers(retrieval_results):
    answers = []

    api_key = os.getenv("OPENAI_API_KEY")
    client = None

    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    log_path = ARTIFACTS_DIR / "llm_calls.jsonl"
    log_path.write_text("", encoding="utf-8")

    for result in retrieval_results:
        question = result["question"]
        chunks = result["retrieved_chunks"]

        if not chunks:
            answer, supported, citations = UNSUPPORTED, False, []

        elif client is None:
            answer, supported, citations = local_answer(
                question, chunks
            )

        else:
            context = "\n\n".join(
                f'Chunk ID: {c["chunk_id"]}\n{c["text"]}'
                for c in chunks
            )

            prompt = f"""
You are a knowledge-base assistant.

Rules:
- Answer ONLY using the provided context.
- If the context does not support an answer, say:
  "{UNSUPPORTED}"
- Never invent product features, policies, or facts.
- Return ONLY valid JSON with keys:
  answer, supported, citations.
- citations must be an array of exact chunk IDs.
- Cite only chunks that directly support the answer.
- If supported is false, citations must be [].

Question:
{question}

Context:
{context}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )

            raw = json.loads(response.choices[0].message.content)

            answer = str(raw.get("answer", UNSUPPORTED))
            supported = raw.get("supported") is True
            citations = raw.get("citations", [])

            if not isinstance(citations, list):
                citations = []

            if not supported:
                answer = UNSUPPORTED
                citations = []

            log = {
                "stage": "answer_generation",
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "provider": "openai",
                "model": "gpt-4o-mini",
                "question_id": result["id"],
                "prompt_hash": hashlib.sha256(
                    prompt.encode()
                ).hexdigest(),
                "input_artifacts": [
                    "artifacts/retrieval_results.json"
                ],
                "output_artifact": "artifacts/answers.json"
            }

            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")

        answers.append({
            "id": result["id"],
            "question": question,
            "answer": answer,
            "supported": supported,
            "citations": citations
        })

    save_json("answers.json", answers)
    return answers



# STAGE 7: CITATION VALIDATION
def validate_citations(answers, retrieval_results):
    retrieval_by_id = {
        result["id"]: result
        for result in retrieval_results
    }

    validations = []

    for answer in answers:
        issues = []
        result = retrieval_by_id[answer["id"]]

        retrieved = {
            chunk["chunk_id"]: chunk
            for chunk in result["retrieved_chunks"]
        }

        citations = answer["citations"]

        for citation in citations:
            if citation not in retrieved:
                issues.append(f"Invalid citation: {citation}")

        if answer["supported"] and not citations:
            issues.append("Supported answer has no citations")

        if not answer["supported"] and citations:
            issues.append("Unsupported answer has citations")

        # Grounding check: numbers in the answer
        # must appear in the cited source text.
        if answer["supported"]:
            cited_text = " ".join(
                retrieved[c]["text"]
                for c in citations
                if c in retrieved
            )

            answer_numbers = set(
                re.findall(r"\b\d+(?:\.\d+)?\b", answer["answer"])
            )
            source_numbers = set(
                re.findall(r"\b\d+(?:\.\d+)?\b", cited_text)
            )

            for number in answer_numbers - source_numbers:
                issues.append(
                    f"Unsupported numeric claim: {number}"
                )

        validations.append({
            "id": answer["id"],
            "passed": len(issues) == 0,
            "issues": issues
        })

    save_json("citation_validation.json", validations)
    return validations


# STAGE 8: FINAL EXPORT
def export_results(answers, retrieval_results, validations):
    retrieval_by_id = {
        r["id"]: r for r in retrieval_results
    }

    validation_by_id = {
        v["id"]: v for v in validations
    }

    final_results = []

    for answer in answers:
        question_id = answer["id"]

        final_results.append({
            **answer,
            "retrieved_chunk_ids": [
                c["chunk_id"]
                for c in retrieval_by_id[question_id]["retrieved_chunks"]
            ],
            "validation": validation_by_id[question_id]
        })

    save_json("final_answers.json", final_results)
    return final_results



if __name__ == "__main__":
    documents = load_documents()
    chunks = create_chunks(documents)

    vectorizer, matrix = build_index(chunks)
    questions = load_questions()

    retrieval_results = retrieve_chunks(
        questions, chunks, vectorizer, matrix
    )

    answers = generate_answers(retrieval_results)

    validations = validate_citations(
        answers, retrieval_results
    )

    final_results = export_results(
        answers, retrieval_results, validations
    )

    failures = sum(not v["passed"] for v in validations)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Questions loaded: {len(questions)}")
    print(f"Supported answers: {sum(a['supported'] for a in answers)}")
    print(f"Validation failures: {failures}")
    print(f"Final answers exported: {len(final_results)}")
    print("Pipeline complete!")

