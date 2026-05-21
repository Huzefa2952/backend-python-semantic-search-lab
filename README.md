# Assignment: Lab — Build a Semantic Search Workflow

## Overview

You will be **implementing a Python semantic search tool** within a **backend documentation search scenario** in connection to **Identify → Assemble → Execute → Verify** with **document preparation, embedding workflow design, cosine similarity, top-k ranking, source traceability, and retrieval-quality checks** to a standard where your code passes the provided pytest test suite and returns ranked results that match the meaning of a user query.

In this lab, you will build the retrieval foundation that later AI-powered applications use before generating answers. A semantic retrieval workflow helps a backend system compare a user’s natural-language question to stored documents, rank the closest matches, and return source-grounded results that could later support a Flask API, React interface, vector database, LangChain retriever, or RAG system. The module outcome expects you to implement a semantic search system that embeds documents and a user query, computes similarity scores, and returns ranked results based on relevance.

### You’ll get

**Support:**

- Starter code in `lib/semantic_search.py`
- A provided document set
- A required function structure
- A deterministic pytest suite
- A fake embedding model used by the tests
- Optional `OllamaEmbeddingModel` support for local experimentation

**Rules:**

- You may use the starter code, course lessons, Python documentation, and your notes.
- You may run pytest as many times as needed.
- You should not modify the test files for grading.
- You should not hard-code answers for specific test queries or document IDs.
- You should not use Chroma, LangChain, Flask, React, or a live RAG system for this lab.
- You do not need internet access or Ollama to pass the tests.
- Ollama is optional for local experimentation after your tests pass.
- Your grade is based on pytest results only.

### You will be able to

- **Prepare** document records using metadata and searchable text.
- **Generate and use** document and query embeddings through a provided embedding model interface.
- **Compute** cosine similarity using basic Python.
- **Rank** documents by similarity score.
- **Return** top-k results with source metadata.
- **Verify** that semantic retrieval returns meaning-based matches instead of relying only on exact keyword overlap.

### You’ll show it by

Creating and submitting a completed `lib/semantic_search.py` file that implements:

- `build_search_text()`
- `prepare_documents()`
- `cosine_similarity()`
- `embed_documents()`
- `rank_documents()`
- `semantic_search()`

Your submission will be tested with `pytest`.

### How you’ll work

This lab uses the technical process **Identify → Assemble → Execute → Verify**. That process fits semantic retrieval because you need to clarify the retrieval goal, assemble documents and model inputs, execute the embedding and ranking workflow, and verify that results are relevant and traceable. The module uses this same process for technical and methodical learning tasks.

You will work independently, but the starter code and tests provide scaffolding. The tests guide you toward the expected behavior without requiring a manually graded reflection or written explanation.

### To meet the standard, your work must

- Pass the provided pytest suite.
- Prepare documents without mutating the original input.
- Validate required document fields.
- Create searchable text from title, category, summary, and tags.
- Embed each prepared document.
- Embed the user query.
- Compute cosine similarity correctly.
- Rank results from highest score to lowest score.
- Respect the `top_k` value.
- Return source-traceable results that include document metadata.
- Retrieve the best meaning-based match for multiple query intents.

---

## Scenario

You are a junior backend developer on a platform team. The team maintains internal developer documentation for account access, API authentication, billing, dashboard performance, and profile settings. Developers often search with everyday language, but the documentation uses more formal titles and summaries.

For example, a developer might search:

> “The mobile client says its login key is stale.”

The most useful article may be titled:

> “Fixing Invalid API Authentication Tokens”

A keyword-only search might miss or under-rank that article because the user did not type the exact words “API authentication token.” A semantic search workflow can compare the meaning of the query with the meaning of each document and return the most relevant documentation first.

Your task is to complete a small backend retrieval workflow that prepares documents, embeds documents, embeds the query, compares similarity scores, ranks the results, and returns the top matches with source information.

---

## Tools and Resources

You will need:

- Python 3.10 or newer
- A code editor such as Visual Studio Code
- Terminal or integrated terminal
- `pipenv`
- `pytest`
- Starter repository files
- Course lessons on:
  - Keyword vs semantic search
  - Embeddings
  - Semantic retrieval workflow
  - Cosine similarity
  - Top-k ranking
  - Source traceability

Optional local experimentation tools:

- Ollama installed locally
- An approved embedding model such as `embeddinggemma`
- Python package: `ollama`

The pytest suite uses a deterministic fake embedding model, so your grade does not depend on Ollama, internet access, or a specific live model output.

### Project structure

Your repository should include:

```text
backend-python-semantic-search-lab/
├── Pipfile
├── lib/
│   ├── __init__.py
│   └── semantic_search.py
└── tests/
    └── test_semantic_search.py
```
## Instructions

Follow the process: Identify → Assemble → Execute → Verify.

### Setup

From the root of the repository, install dependencies:

```bash
pipenv install
```

Enter the virtual environment:

```bash
pipenv shell
```

Run the tests:

```bash
pytest
```

You can also run:

```bash
pytest -x
```

The -x flag stops after the first failing test, which can make debugging easier.

### Step 1: Identify the retrieval goal and output contract

Before writing code, inspect the starter file and test file.

Your goal is to complete a backend semantic search workflow that can:

* receive a user query,
* prepare searchable document text,
* embed documents,
* embed the query,
* compare query and document embeddings,
* rank documents by similarity,
* return the top-k matches,
* and preserve source metadata.

Your output should be a list of result dictionaries. Each result should include:

```json
{
    "id": "DOC-102",
    "title": "Fixing Invalid API Authentication Tokens",
    "category": "api",
    "summary": "...",
    "source": "platform-docs/api/authentication-tokens",
    "score": 0.87
}
```

Do not return only text or only scores. A retrieval result needs metadata so a user, frontend, or future RAG system can trace where the result came from.

### Step 2: Assemble the starter code, inputs, and constraints

Open `lib/semantic_search.py`.

Review these provided parts:

* REQUIRED_DOCUMENT_FIELDS
* DEFAULT_DOCUMENTS
* EmbeddingModel
* OllamaEmbeddingModel
* Function docstrings
* Required return formats

Then open `tests/test_semantic_search.py`.

Use the tests as your checklist. They define the expected behavior for each function.

Pay close attention to these constraints:

* Do not mutate the original input documents.
* Use the same embedding model interface for document embeddings and query embeddings.
* Do not use NumPy for cosine similarity.
* Do not hard-code test results.
* Do not remove required metadata.
* Do not assume top_k will always be 3.
* Do not assume the document list will always contain exactly five documents.

### Step 3: Implement build_search_text()

Complete:

```python
def build_search_text(document: dict[str, Any]) -> str:
```

This function should create one clean searchable string for a document.

Your searchable text must include:

* title
* category
* summary
* tags, when available

This matters because embedding only a title may not give the model enough meaning. Including title, category, summary, and tags helps the embedding represent the document more clearly.

Expected behavior:

* Return a non-empty string.
* Include meaningful document fields.
* Handle tags when they are present.
* Avoid returning raw dictionaries or lists.

### Step 4: Implement prepare_documents()

Complete:

```python
def prepare_documents(raw_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
```

This function should validate and prepare the document collection.

Your function must:

* accept a list of document dictionaries,
* reject an empty list,
* validate that each document includes:
    * id
    * title
    * category
    * summary
    * source
* raise ValueError if a required field is missing,
* return a new list,
* preserve metadata,
* add a text field using build_search_text(),
* avoid mutating the original input documents.

This step supports the Assemble part of the workflow because you are preparing the searchable data before embedding it.

### Step 5: Implement cosine_similarity()

Complete:

```python
def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
```

This function should compare two vectors and return a similarity score.

Your function must:

* return a float,
* return 1.0 for identical non-zero vectors,
* return 0.0 for orthogonal vectors,
* return 0.0 if either vector has zero magnitude,
* raise ValueError if the vectors have different dimensions,
* use basic Python math instead of NumPy.

Cosine similarity helps estimate how close two embeddings are based on their direction. This is one of the core comparison methods used in semantic retrieval workflows. The module identifies cosine similarity, ranking, query embeddings, and document embeddings as key retrieval concepts.

### Step 6: Implement embed_documents()

Complete:

```python
def embed_documents(
    prepared_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
) -> list[dict[str, Any]]:
```

This function should embed every prepared document.

Your function must:

* require each document to have a text field,
* call embedding_model.embed(document["text"]) once per document,
* return a new list,
* add an embedding field to each returned document,
* preserve source metadata,
* avoid mutating the input documents.

This step is part of Execute because the backend is converting prepared text into vectors.

### Step 7: Implement rank_documents()

Complete:

```python
def rank_documents(
    query: str,
    embedded_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:
```

This function should rank embedded documents against a user query.

Your function must:

* reject an empty query,
* reject invalid `top_k` values,
* embed the query,
* compare the query embedding to each document embedding,
* compute a score for each result,
* sort results from highest score to lowest score,
* return only the top-k results,
* preserve:
    * id
    * title
    * category
    * summary
    * source
    * score

Do not return embeddings in the final search results. The final results should be useful to a user, frontend, or future RAG workflow, not just to the internal ranking algorithm.

### Step 8: Implement semantic_search()

Complete:

```python
def semantic_search(
    query: str,
    raw_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:
```

This function should orchestrate the full workflow.

The required sequence is:

1. Prepare documents.
2. Embed documents.
3. Embed the query.
4. Compute similarity scores.
5. Rank results.
6. Return top-k results with source metadata.

A strong solution uses the helper functions instead of duplicating all logic inside `semantic_search()`.

The module describes this same retrieval sequence as documents → document embeddings → user query → query embedding → similarity comparison → ranked results → top-k output.

### Step 9: Evaluate with pytest

Run:

```bash
pytest
```

Use failing tests as feedback.

If a test fails, read:

* the test name,
* the assertion,
* the function being tested,
* and the expected behavior in the function docstring.

Common issues to check:

* Missing text field: Confirm `prepare_documents()` calls `build_search_text()`
* Mutated input documents: Return copied dictionaries instead of editing originals
* Incorrect score order: Sort by score in descending order
* Missing metadata: Include `id`, `title`, `category`, `summary`, and `source` in results
* Invalid cosine similarity: Check dot product, magnitudes, zero vectors, and dimension mismatch
* Wrong number of results: Respect `top_k`, but do not return more documents than exist
* Query not embedded: Confirm `rank_documents()` embeds the query before scoring

Your work is complete when all tests pass.

### Step 10: Ungraded Reflection

This step is not submitted and is not manually graded.

After your tests pass, ask yourself:

* Does my code follow the retrieval sequence clearly?
* Can I explain why source metadata matters?
* Can my code handle a different document set?
* Can my code handle a different query intent?
* Did I avoid hard-coding test-specific answers?
* Would these results be traceable enough to support a future RAG workflow?

The module emphasizes that verification is not only about whether code runs. Strong retrieval workflows check consistency, traceability, similarity behavior, relevance, and readiness for future retrieval or RAG use.


## Notes for Local Experimentation

The tests use a deterministic fake embedding model. That model is designed for reliable grading.

After passing the test suite, you may optionally try the workflow with Ollama using the provided `OllamaEmbeddingModel`.

Example:

```python
from lib.semantic_search import (
    DEFAULT_DOCUMENTS,
    OllamaEmbeddingModel,
    semantic_search,
)

embedder = OllamaEmbeddingModel(model_name="embeddinggemma")

results = semantic_search(
    query="Why does the mobile app say my token is expired?",
    raw_documents=DEFAULT_DOCUMENTS,
    embedding_model=embedder,
    top_k=3,
)

for result in results:
    print(result["title"], result["score"], result["source"])
```

Your exact scores may vary when using a real embedding model. That is expected. What matters is that the workflow prepares documents, embeds consistently, compares similarity, ranks results, and preserves source information.