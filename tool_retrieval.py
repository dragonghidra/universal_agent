"""Vector-based tool retrieval utilities for dynamic tool binding."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

try:  # pragma: no cover - optional performance dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - graceful fallback
    faiss = None


@dataclass
class ToolRecord:
    """Lightweight metadata describing a callable tool."""

    name: str
    server: str
    description: str
    input_schema: Dict[str, object]
    examples: List[str]
    tags: List[str]
    risk: str  # "low"|"medium"|"high"
    aliases: Optional[List[str]] = None


@dataclass
class ScoredTool:
    """Tool plus similarity score and textual reasons."""

    tool: ToolRecord
    score: float
    reasons: List[str]


def flatten_schema(schema: Dict[str, object]) -> str:
    """Flatten a JSON schema into a readable whitespace-delimited string."""

    pieces: List[str] = []

    def walk(node: Dict[str, object], path: str = ""):
        node = node or {}
        schema_type = node.get("type")
        enum = node.get("enum")
        if schema_type or enum:
            head = f"{path or 'root'}:{schema_type or 'any'}"
            if enum:
                head += f" enums={','.join(map(str, enum))}"
            pieces.append(head)
        properties = node.get("properties") or {}
        if isinstance(properties, dict):
            for key, value in properties.items():
                child_path = f"{path}.{key}" if path else key
                if isinstance(value, dict):
                    walk(value, child_path)
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, f"{path}[]" if path else "items[]")

    try:
        walk(schema or {})
    except Exception:  # pragma: no cover - defensive
        return ""
    return " ".join(pieces[:500])


def make_views(record: ToolRecord) -> Dict[str, str]:
    """Create text views (doc, schema, examples) used for embeddings."""

    doc_view = "\n".join(
        filter(
            None,
            [
                record.name,
                record.description,
                f"Server:{record.server}",
                "Tags:" + ",".join(record.tags or []),
                "Aliases:" + ",".join(record.aliases or []) if record.aliases else "",
            ],
        )
    )
    schema_view = flatten_schema(record.input_schema or {})
    examples_view = "\n".join((record.examples or [])[:3])
    return {"doc": doc_view, "schema": schema_view, "examples": examples_view}


def default_intent_string(task_text: str) -> str:
    """Very small heuristic classifier for obvious intents."""

    lowered = task_text.lower()
    if re.search(r"\b(weather|forecast)\b", lowered):
        match = re.search(r"\b(in|for)\s+([A-Z][a-zA-Z]+)\b", task_text)
        city = match.group(2) if match else "unknown"
        return f"retrieve:weather city={city} units=metric"
    if re.search(r"\b(csv|table|delimiter|spreadsheet)\b", lowered):
        return "transform:csv parse"
    if "browser" in lowered or "screenshot" in lowered:
        return "automation:browser"
    return "generic:intent"


class Embedder:
    """Wrapper around the embedding backend used for tool retrieval."""

    def __init__(self, *, dim: int = 1024, model_name: Optional[str] = None):
        self.dim = dim
        self.model_name = model_name or os.environ.get("AGENT_TOOL_EMBED_MODEL")
        self._client = None
        if self.model_name:
            self._client = self._init_openai_embeddings(self.model_name)

    @staticmethod
    def _init_openai_embeddings(model_name: str):
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception:  # pragma: no cover - optional dependency
            return None
        try:
            return OpenAIEmbeddings(model=model_name)
        except Exception:
            return None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return a 2D numpy array of embeddings for the provided texts."""

        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        if self._client is not None:
            try:
                vectors = self._client.embed_documents(texts)
                arr = np.asarray(vectors, dtype="float32")
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.ndim == 2 and arr.shape[1] > 0:
                    self.dim = arr.shape[1]
                return arr
            except Exception:
                # Fall back to deterministic pseudo-embeddings if the API call fails.
                pass

        rng = np.random.default_rng(7)
        rows: List[np.ndarray] = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            seeded_rng = np.random.default_rng(seed)
            vec = seeded_rng.normal(size=self.dim).astype("float32")
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            rows.append(vec)
        return np.vstack(rows)


class VectorIndex:
    """ANN wrapper that prefers FAISS but falls back to brute-force search."""

    def __init__(self, dim: int):
        self.dim = dim
        self._backend = "faiss" if faiss is not None else "brute"
        if self._backend == "faiss":
            self._index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
        else:
            self._index = None
        self._keys: List[str] = []
        self._vectors = np.zeros((0, dim), dtype="float32")

    def add(self, vecs: np.ndarray, keys: List[str]) -> None:
        if not len(vecs):
            return
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        normalized = vecs / norms
        if self._backend == "faiss":
            self._index.add(normalized)  # type: ignore[union-attr]
        else:
            if not len(self._vectors):
                self._vectors = normalized
            else:
                self._vectors = np.vstack([self._vectors, normalized])
        self._keys.extend(keys)

    def search(self, query: np.ndarray, *, topk: int = 200) -> List[Tuple[str, float]]:
        if not self._keys:
            return []
        norms = np.linalg.norm(query, axis=1, keepdims=True) + 1e-9
        normalized_query = query / norms
        if self._backend == "faiss":
            size = min(topk, len(self._keys))
            distances, indices = self._index.search(normalized_query, size)  # type: ignore[union-attr]
            results: List[Tuple[str, float]] = []
            for idx, score in zip(indices[0], distances[0]):
                if 0 <= idx < len(self._keys):
                    results.append((self._keys[idx], float(score)))
            return results
        scores = self._vectors @ normalized_query.T
        scores = scores.reshape(-1)
        order = np.argsort(-scores)[:topk]
        return [(self._keys[i], float(scores[i])) for i in order if scores[i] > 0]


def mmr(
    candidates: List[Tuple[str, float, np.ndarray]],
    *,
    lambda_mult: float = 0.65,
    topk: int = 12,
) -> List[Tuple[str, float]]:
    """Maximal Marginal Relevance w/ cosine similarity for diversity."""

    selected: List[Tuple[str, float]] = []
    selected_vecs: List[np.ndarray] = []
    pool = candidates[:]
    while pool and len(selected) < topk:
        if not selected_vecs:
            best = max(pool, key=lambda item: item[1])
            selected.append((best[0], best[1]))
            selected_vecs.append(best[2])
            pool.remove(best)
            continue
        mmr_scores = []
        for key, score, vec in pool:
            sim = max(float(np.dot(vec, existing)) for existing in selected_vecs)
            diversified = lambda_mult * score - (1 - lambda_mult) * sim
            mmr_scores.append((key, diversified, score, vec))
        best = max(mmr_scores, key=lambda item: item[1])
        selected.append((best[0], best[2]))
        selected_vecs.append(best[3])
        pool = [candidate for candidate in pool if candidate[0] != best[0]]
    return selected


class ToolRetriever:
    """Hybrid lexical + ANN retriever designed for small tool catalogs."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        include_defaults: Iterable[str] = (),
        risky_opt_in: bool = True,
        popularity_prior: Optional[Dict[str, float]] = None,
    ):
        self.embedder = embedder
        self.include_defaults = list(include_defaults)
        self.risky_opt_in = risky_opt_in
        self.popularity_prior = popularity_prior or {}
        self.catalog: Dict[str, ToolRecord] = {}
        self.views: Dict[str, Dict[str, str]] = {}
        self.view_index_names = ["doc", "schema", "examples"]
        self.ann: Dict[str, Optional[VectorIndex]] = {name: None for name in self.view_index_names}
        self.view_vecs: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in self.view_index_names}
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_keys: List[str] = []

    def _key(self, record: ToolRecord) -> str:
        return f"{record.server}:{record.name}"

    # ---------- building ----------

    def build_catalog(self, records: Iterable[ToolRecord]) -> None:
        """Store tool metadata and derived text views."""

        self.catalog.clear()
        self.views.clear()
        for record in records:
            key = self._key(record)
            self.catalog[key] = record
            self.views[key] = make_views(record)

    def build_indexes(self) -> None:
        """Create ANN + BM25 indexes from the current catalog."""

        self.view_vecs = {name: {} for name in self.view_index_names}
        for name in self.view_index_names:
            texts: List[str] = []
            keys: List[str] = []
            for key, view in self.views.items():
                text = (view.get(name) or "").strip()
                if text:
                    texts.append(text)
                    keys.append(key)
            if not texts:
                self.ann[name] = None
                continue
            vectors = self.embedder.encode(texts)
            dim = vectors.shape[1] if vectors.ndim == 2 and vectors.shape[1] else self.embedder.dim
            index = VectorIndex(dim)
            index.add(vectors, keys)
            self.ann[name] = index
            for key, vector in zip(keys, vectors):
                norm = np.linalg.norm(vector) + 1e-9
                self.view_vecs[name][key] = vector / norm

        bm25_corpus: List[str] = []
        bm25_keys: List[str] = []
        for key, record in self.catalog.items():
            haystack = " ".join(
                filter(
                    None,
                    [
                        record.name,
                        record.description,
                        " ".join(record.tags or []),
                        " ".join(record.aliases or []),
                        record.server,
                    ],
                )
            )
            bm25_corpus.append(haystack.lower())
            bm25_keys.append(key)
        if bm25_corpus:
            tokens = [text.split() for text in bm25_corpus]
            self.bm25 = BM25Okapi(tokens)
            self.bm25_keys = bm25_keys
        else:
            self.bm25 = None
            self.bm25_keys = []

    # ---------- selection ----------

    def select(
        self,
        user_message: str,
        recent_context: str = "",
        plan_step: str = "",
        *,
        K_dynamic: int = 12,
        allow_risky: bool = False,
    ) -> List[ScoredTool]:
        """Return an ordered shortlist of tools for the current turn."""

        if not self.catalog:
            return []

        task_text = self._task_repr(user_message, recent_context, plan_step)
        intent = default_intent_string(task_text)

        lexical_scores: Dict[str, float] = {}
        if self.bm25:
            query_tokens = (user_message + " " + intent).lower().split()
            scores = self.bm25.get_scores(query_tokens)
            for key, score in zip(self.bm25_keys, scores):
                if score > 0:
                    lexical_scores[key] = float(score)

        qvec = self.embedder.encode([task_text])
        ann_scores: Dict[str, float] = {}
        ann_candidates: List[Tuple[str, float, np.ndarray]] = []
        for name in self.view_index_names:
            index = self.ann.get(name)
            if not index:
                continue
            hits = index.search(qvec, topk=200)
            for key, score in hits:
                ann_scores[key] = max(ann_scores.get(key, -1.0), score)
        for key, score in ann_scores.items():
            vec = self._pick_vec(key)
            if vec is not None:
                ann_candidates.append((key, score, vec))

        fused_scores: Dict[str, float] = {}
        keys = set(list(lexical_scores.keys()) + list(ann_scores.keys()))
        for key in keys:
            embed_score = ann_scores.get(key, 0.0)
            lex_score = lexical_scores.get(key, 0.0)
            prior = self.popularity_prior.get(key, 0.0)
            fused_scores[key] = 0.70 * embed_score + 0.25 * _log1p(lex_score) + 0.15 * prior

        ranked = sorted(
            [(key, fused_scores.get(key, 0.0), self._pick_vec(key)) for key in keys if self._pick_vec(key) is not None],
            key=lambda item: item[1],
            reverse=True,
        )
        diversified = mmr(ranked, lambda_mult=0.65, topk=max(6, K_dynamic)) if ranked else []
        shortlist_keys = [key for key, _ in diversified]
        for default_name in self.include_defaults:
            default_key = self._find_by_name(default_name)
            if default_key and default_key not in shortlist_keys:
                shortlist_keys.append(default_key)

        gated: List[str] = []
        for key in shortlist_keys:
            record = self.catalog.get(key)
            if not record:
                continue
            if self.risky_opt_in and not allow_risky and (record.risk or "").lower() == "high":
                continue
            gated.append(key)

        output: List[ScoredTool] = []
        for key in gated[: K_dynamic or 1]:
            record = self.catalog.get(key)
            if not record:
                continue
            output.append(
                ScoredTool(
                    tool=record,
                    score=fused_scores.get(key, 0.0),
                    reasons=self._reasons(task_text, record),
                )
            )
        return output

    def expand_and_retry(
        self,
        user_message: str,
        recent_context: str,
        plan_step: str,
        prev_shortlist: List[ScoredTool],
        *,
        K_expand: int = 24,
    ) -> List[ScoredTool]:
        """Broader search used when the agent asks for additional tools."""

        expanded = self.select(
            user_message,
            recent_context,
            plan_step,
            K_dynamic=K_expand,
            allow_risky=False,
        )
        prev_names = {entry.tool.name for entry in prev_shortlist}
        expanded_names = {entry.tool.name for entry in expanded}
        if expanded_names.issubset(prev_names):
            expanded = self.select(
                user_message,
                recent_context,
                plan_step,
                K_dynamic=K_expand,
                allow_risky=True,
            )
        return expanded

    # ---------- helpers ----------

    def _find_by_name(self, name: str) -> Optional[str]:
        lname = name.lower()
        for key, record in self.catalog.items():
            if record.name.lower() == lname:
                return key
        return None

    def _pick_vec(self, key: str) -> Optional[np.ndarray]:
        for name in self.view_index_names:
            vector = self.view_vecs.get(name, {}).get(key)
            if vector is not None:
                return vector
        return None

    @staticmethod
    def _task_repr(user_msg: str, ctx: str, step: str) -> str:
        intent = default_intent_string(f"{user_msg} {ctx} {step}")
        return f"user_goal: {user_msg}\ncontext: {ctx}\nplan_step: {step}\nintent: {intent}"

    @staticmethod
    def _reasons(task_text: str, record: ToolRecord) -> List[str]:
        lowered = task_text.lower()
        haystack = (record.description + " " + " ".join(record.tags or [])).lower()
        reasons: List[str] = []
        for token in ("csv", "path", "delimiter", "url", "table", "weather", "image", "browser", "shell"):
            if token in lowered and token in haystack:
                reasons.append(f"matches '{token}'")
        if (record.risk or "").lower() == "high":
            reasons.append("high-risk (opt-in)")
        return reasons


def _log1p(value: float) -> float:
    try:
        return math.log1p(value)
    except Exception:  # pragma: no cover - defensive
        return 0.0


__all__ = [
    "Embedder",
    "ScoredTool",
    "ToolRecord",
    "ToolRetriever",
    "default_intent_string",
    "flatten_schema",
]
