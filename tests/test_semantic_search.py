import pytest

from lib.semantic_search import (
    REQUIRED_DOCUMENT_FIELDS,
    build_search_text,
    prepare_documents,
    cosine_similarity,
    embed_documents,
    rank_documents,
    semantic_search,
)


class DeterministicEmbeddingModel:
    """
    A deterministic fake embedding model for autograding.

    This model does not call Ollama and does not need internet access.
    It maps text into a small topic vector so the tests can check whether
    students implemented the retrieval workflow correctly.

    Vector dimensions:
    0. api/authentication/token meaning
    1. billing/plan/invoice meaning
    2. dashboard/performance meaning
    3. profile/settings meaning
    4. password/account recovery meaning
    """

    TOPIC_TERMS = [
        {
            "api",
            "authentication",
            "auth",
            "token",
            "tokens",
            "credential",
            "credentials",
            "bearer",
            "key",
            "stale",
            "expired",
            "malformed",
            "request",
            "requests",
            "call",
            "calls",
            "blocked",
        },
        {
            "billing",
            "bill",
            "invoice",
            "invoices",
            "plan",
            "plans",
            "monthly",
            "usage",
            "limit",
            "limits",
            "upgrade",
            "upgrades",
            "payment",
        },
        {
            "dashboard",
            "performance",
            "slow",
            "loading",
            "latency",
            "rendering",
            "delay",
            "delays",
            "page",
            "client-side",
        },
        {
            "profile",
            "settings",
            "email",
            "name",
            "notification",
            "notifications",
            "preferences",
            "details",
        },
        {
            "password",
            "forgotten",
            "reset",
            "change",
            "account",
            "verification",
            "recover",
            "recovery",
        },
    ]

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)

        normalized = text.lower().replace(".", " ").replace(",", " ")
        tokens = set(normalized.split())

        vector = []
        for topic_terms in self.TOPIC_TERMS:
            score = sum(1 for term in topic_terms if term in tokens)
            vector.append(float(score))

        return vector


class FixedEmbeddingModel:
    """Returns one fixed query vector for rank_documents unit tests."""

    def __init__(self, query_vector):
        self.query_vector = query_vector
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return self.query_vector


@pytest.fixture
def sample_documents():
    return [
        {
            "id": "DOC-101",
            "title": "Resetting a Forgotten Password",
            "category": "account",
            "summary": (
                "Explains how users can change or reset their account password "
                "after identity verification."
            ),
            "source": "platform-docs/account/password-reset",
            "tags": ["password", "account", "verification"],
        },
        {
            "id": "DOC-102",
            "title": "Fixing Invalid API Authentication Tokens",
            "category": "api",
            "summary": (
                "Explains how to resolve API calls blocked by expired, missing, "
                "or malformed bearer credentials."
            ),
            "source": "platform-docs/api/authentication-tokens",
            "tags": ["api", "authentication", "credentials"],
        },
        {
            "id": "DOC-103",
            "title": "Understanding Monthly Billing Limits",
            "category": "billing",
            "summary": (
                "Explains how usage limits affect invoices, monthly plans, "
                "and account upgrades."
            ),
            "source": "platform-docs/billing/monthly-limits",
            "tags": ["billing", "invoice", "plan"],
        },
        {
            "id": "DOC-104",
            "title": "Troubleshooting Slow Dashboard Loading",
            "category": "performance",
            "summary": (
                "Explains common causes of slow page loads, dashboard latency, "
                "and client-side rendering delays."
            ),
            "source": "platform-docs/performance/dashboard-loading",
            "tags": ["dashboard", "performance", "latency"],
        },
        {
            "id": "DOC-105",
            "title": "Updating User Profile Settings",
            "category": "account",
            "summary": (
                "Explains how users can update email, display name, "
                "notification preferences, and profile details."
            ),
            "source": "platform-docs/account/profile-settings",
            "tags": ["profile", "settings", "email"],
        },
    ]


def test_build_search_text_combines_title_category_summary_and_tags():
    document = {
        "id": "DOC-200",
        "title": "Configuring Two-Factor Login",
        "category": "security",
        "summary": "Explains how to require a second verification step during sign-in.",
        "source": "platform-docs/security/two-factor-login",
        "tags": ["mfa", "login", "verification"],
    }

    search_text = build_search_text(document)

    assert isinstance(search_text, str)
    assert search_text.strip() != ""

    lowered = search_text.lower()

    assert "configuring two-factor login" in lowered
    assert "security" in lowered
    assert "second verification step" in lowered
    assert "mfa" in lowered
    assert "login" in lowered
    assert "verification" in lowered


def test_prepare_documents_adds_text_and_preserves_metadata_without_mutating_input(
    sample_documents,
):
    original_first_document = sample_documents[0].copy()

    prepared = prepare_documents(sample_documents)

    assert len(prepared) == len(sample_documents)
    assert "text" in prepared[0]
    assert prepared[0]["text"].strip() != ""

    for field in REQUIRED_DOCUMENT_FIELDS:
        assert prepared[0][field] == original_first_document[field]

    assert "text" not in sample_documents[0]


def test_prepare_documents_rejects_empty_document_list():
    with pytest.raises(ValueError):
        prepare_documents([])


def test_prepare_documents_rejects_missing_required_fields():
    incomplete_documents = [
        {
            "id": "DOC-999",
            "title": "Missing Source Example",
            "category": "api",
            "summary": "This document is missing the source field.",
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        prepare_documents(incomplete_documents)

    assert "source" in str(exc_info.value).lower()


def test_cosine_similarity_scores_identical_orthogonal_and_zero_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0])


def test_embed_documents_calls_model_once_per_document_and_preserves_metadata(
    sample_documents,
):
    prepared = prepare_documents(sample_documents)
    model = DeterministicEmbeddingModel()

    embedded = embed_documents(prepared, model)

    assert len(embedded) == len(prepared)
    assert len(model.calls) == len(prepared)

    for document in embedded:
        assert "embedding" in document
        assert isinstance(document["embedding"], list)

        for field in REQUIRED_DOCUMENT_FIELDS:
            assert field in document

    assert "embedding" not in prepared[0]


def test_rank_documents_returns_top_k_sorted_by_score_with_source_metadata():
    embedded_documents = [
        {
            "id": "DOC-A",
            "title": "Strong Match",
            "category": "api",
            "summary": "Strong API authentication match.",
            "source": "docs/a",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "id": "DOC-B",
            "title": "Partial Match",
            "category": "api",
            "summary": "Partial API authentication match.",
            "source": "docs/b",
            "embedding": [0.6, 0.4, 0.0],
        },
        {
            "id": "DOC-C",
            "title": "Unrelated Match",
            "category": "billing",
            "summary": "Billing information.",
            "source": "docs/c",
            "embedding": [0.0, 1.0, 0.0],
        },
    ]

    model = FixedEmbeddingModel(query_vector=[1.0, 0.0, 0.0])

    results = rank_documents(
        query="api authentication issue",
        embedded_documents=embedded_documents,
        embedding_model=model,
        top_k=2,
    )

    assert len(results) == 2
    assert [result["id"] for result in results] == ["DOC-A", "DOC-B"]

    assert results[0]["score"] >= results[1]["score"]

    for result in results:
        assert "score" in result
        assert isinstance(result["score"], float)

        assert result["id"]
        assert result["title"]
        assert result["category"]
        assert result["summary"]
        assert result["source"]


def test_rank_documents_rejects_empty_query(sample_documents):
    prepared = prepare_documents(sample_documents)
    model = DeterministicEmbeddingModel()
    embedded = embed_documents(prepared, model)

    with pytest.raises(ValueError):
        rank_documents("", embedded, model, top_k=3)


def test_rank_documents_rejects_invalid_top_k(sample_documents):
    prepared = prepare_documents(sample_documents)
    model = DeterministicEmbeddingModel()
    embedded = embed_documents(prepared, model)

    with pytest.raises(ValueError):
        rank_documents("api authentication issue", embedded, model, top_k=0)


def test_semantic_search_retrieves_best_meaning_match_without_exact_title_words(
    sample_documents,
):
    model = DeterministicEmbeddingModel()

    results = semantic_search(
        query="The mobile client says its login key is stale.",
        raw_documents=sample_documents,
        embedding_model=model,
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["id"] == "DOC-102"
    assert results[0]["title"] == "Fixing Invalid API Authentication Tokens"

    assert results[0]["score"] >= results[1]["score"]
    assert results[1]["score"] >= results[2]["score"]


def test_semantic_search_handles_different_query_intents(sample_documents):
    model = DeterministicEmbeddingModel()

    billing_results = semantic_search(
        query="Why did my invoice change after upgrading my monthly plan?",
        raw_documents=sample_documents,
        embedding_model=model,
        top_k=2,
    )

    performance_results = semantic_search(
        query="The dashboard page is loading slowly and has high latency.",
        raw_documents=sample_documents,
        embedding_model=model,
        top_k=2,
    )

    profile_results = semantic_search(
        query="How do I change my email notification preferences?",
        raw_documents=sample_documents,
        embedding_model=model,
        top_k=2,
    )

    assert billing_results[0]["id"] == "DOC-103"
    assert performance_results[0]["id"] == "DOC-104"
    assert profile_results[0]["id"] == "DOC-105"


def test_semantic_search_returns_all_documents_when_top_k_exceeds_corpus_size(
    sample_documents,
):
    model = DeterministicEmbeddingModel()

    results = semantic_search(
        query="API credential issue",
        raw_documents=sample_documents[:2],
        embedding_model=model,
        top_k=10,
    )

    assert len(results) == 2


def test_semantic_search_does_not_mutate_raw_documents(sample_documents):
    original_documents = [document.copy() for document in sample_documents]
    model = DeterministicEmbeddingModel()

    semantic_search(
        query="API credential issue",
        raw_documents=sample_documents,
        embedding_model=model,
        top_k=3,
    )

    assert sample_documents == original_documents