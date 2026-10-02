from langchain_chroma import Chroma

from .config import settings
from .models import get_embeddings

# Cosine distances are only comparable because every collection shares one
# embedding model, so a single score threshold works across all five domains.
# Below this the match is treated as "no confident domain" and the caller
# decides whether to fall back or abstain.
ROUTER_MAX_DISTANCE = 0.55

# When the top two domains are this close in cosine distance the router cannot
# honestly claim to know which one owns the question, so it defers to the LLM
# instead of guessing. 0.05 is a fixed "effectively tied" cutoff on cosine
# distance, chosen once rather than tuned per question; clear cases clear it by
# a wide margin and only genuine ties fall through.
ROUTER_MARGIN = 0.05


def _store(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def retrieve_from_chroma(query: str, collection_name: str, k: int | None = None):
    docs = _store(collection_name).similarity_search(query, k=k or settings.retrieval_k)
    if not docs:
        # ASCII only: a non-encodable character here crashes the run on a
        # cp1252 console (Windows) before the caller can handle empty context.
        print(f"[warn] no docs retrieved for query: {query}", flush=True)
    return docs


def route_by_similarity(query: str, margin: float = ROUTER_MARGIN):
    """Pick the domain whose documents are closest to the query.

    Returns ``(intent, distance)`` for the nearest domain when it beats the
    runner-up by ``margin``. Returns ``"general"`` when nothing is close enough
    to be a NovaTech policy question, and ``(None, distance)`` when two domains
    are too close to separate so the caller can defer to the LLM.

    Embedding similarity generalises to unseen phrasings, which a
    question-specific prompt rule cannot do.
    """
    scored = []
    for intent, collection_name in settings.collections.items():
        result = _store(collection_name).similarity_search_with_relevance_scores(
            query, k=1
        )
        if not result:
            continue
        # Chroma returns relevance in [0, 1]; convert to a distance so that a
        # smaller number is a better match and the threshold reads naturally.
        scored.append((intent, 1.0 - float(result[0][1])))

    if not scored:
        return None, float("inf")

    scored.sort(key=lambda pair: pair[1])
    best_intent, best_distance = scored[0]
    runner_up = scored[1][1] if len(scored) > 1 else float("inf")

    if best_distance > ROUTER_MAX_DISTANCE:
        # Nothing in the knowledge base is close to this question. That is the
        # definition of out of scope here: capitals, stocks, poems and sports all
        # sit far from every domain centroid, while genuine policy questions sit
        # well inside the threshold. Answer "general" directly instead of asking
        # a model to re-derive a distinction the distances already show.
        return "general", best_distance
    if runner_up - best_distance < margin:
        # Two domains are effectively tied; the caller defers to the LLM.
        return None, best_distance
    return best_intent, best_distance


def format_context(documents):
    return "\n\n".join(f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in documents)