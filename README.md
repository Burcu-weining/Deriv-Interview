## Base RAG Assistant

## What does it do?
- this program answers questions using a local markdown and ahs different steps of text documentations and chunk citations. 

## Features
- Loads .md and .txt documents dynamically from docs/

- Creates overlapping text chunks with stable IDs

- Retrieves the top 3 relevant chunks using TF-IDF and cosine similarity

- Generates answers using OpenAI when configured, otherwise uses a local sentence-based fallback

- Identifies unsupported questions

- Validates citations and numeric claims

- Exports machine-readable JSON artifacts

## Locla Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Use Python xxx.py to run the cde

## Open API Key
Set up an open API key environment to enable LLM based answers. 


## Limitations
TF-IDF relies on word overlap rather than semantic understanding. Citation validation verifies evidence references and numeric claims, but cannot guarantee that every answer is fully supported.

## Output Files Generated
documents.json

chunks.json

retrieval_results.json

answers.json

citation_validation.json

final_answers.json