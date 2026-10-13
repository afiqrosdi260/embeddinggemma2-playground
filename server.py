"""Local multimodal retrieval playground for google/embeddinggemma-2.

No external LLM is involved: the /search endpoint returns source passages,
not generated answers. Data and embeddings are in memory, uploads on local disk.
"""
from __future__ import annotations

import asyncio
import io
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / ".uploads"
UPLOADS.mkdir(exist_ok=True)
MAX_BYTES = 15 * 1024 * 1024  # Prevent accidental large media uploads in the local prototype
ALLOWED = {
    ".txt": "text", ".md": "text", ".pdf": "text",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".wav": "audio", ".flac": "audio", ".mp3": "audio",
    ".mp4": "video", ".mov": "video", ".webm": "video",
}
DIMS = (128, 256, 512, 768)

SEEDS = [
    ("What are embeddings?", "An embedding is a numerical representation of the meaning of an input. Similar content often maps to nearby vectors. Embeddings support semantic search, recommendation, clustering, and retrieval for RAG systems."),
    ("How retrieval-augmented generation works", "RAG retrieves relevant passages from a document collection, passes the evidence to a generative language model, and asks it to answer with citations. The embedding model performs retrieval, not the generation of the answer."),
    ("Vector search versus keyword search", "Keyword search matches exact terms. Vector search compares the semantic meaning of a question and indexed content. A query about automobile maintenance may retrieve a passage about taking care of a car even when the wording differs."),
    ("EmbeddingGemma 2 model", "EmbeddingGemma 2 is a 740-million-parameter multimodal embedding model from Google DeepMind. It maps text, code, images, audio, and video into a shared 768-dimensional vector space. Smaller Matryoshka dimensions include 512, 256, and 128."),
    ("Cross-modal retrieval", "You can embed a photograph of a drilling rig and retrieve it with a text search such as offshore drilling equipment. Images and text are compared in the same embedding space when indexed by the same multimodal model."),
    ("API capabilities and semantic discovery", "An API capability registry describes verified resources, endpoints, authentication, field schemas, and supported operations. Embedding search can help locate relevant capability descriptions, but it must not replace deterministic API validation."),
    ("Geospatial well data", "An oil well dataset can include latitude, longitude, field name, depth, operator and status. Spatial analysis requires actual coordinates and GIS functions. Semantic embeddings alone cannot calculate geographic distance or perform geospatial joins."),
    ("Bahasa Melayu and multilingual search", "Carian semantik boleh mencari dokumen berdasarkan maksud ayat dan bukan hanya padanan perkataan. Model embedding pelbagai bahasa boleh membantu pencarian silang bahasa Melayu dan bahasa Inggeris."),
    ("Precision and dimension trade-offs", "EmbeddingGemma 2 supports full 768-dimensional vectors and truncated 512, 256, and 128-dimensional vectors. After truncation, vectors must be L2-normalized again. Smaller vectors save memory and storage but may reduce retrieval quality."),
]


def chunks(text: str, size: int = 950, overlap: int = 130) -> list[str]:
    """Deterministic character chunks, respecting words where possible."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    result = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary != -1:
                end = boundary
        part = text[start:end].strip()
        if part:
            result.append(part)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return result


def unit(vector: np.ndarray, dimension: int) -> np.ndarray:
    """Truncate a 768-d Matryoshka embedding and re-normalize before dot product."""
    if dimension not in DIMS:
        raise ValueError("Unsupported embedding dimension")
    v = np.asarray(vector, dtype=np.float32).reshape(-1)[:dimension]
    if len(v) != dimension or not np.all(np.isfinite(v)):
        raise ValueError("Non-finite or unexpectedly-sized embedding returned")
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        raise ValueError("Zero-length embedding returned")
    return v / norm


@dataclass
class Document:
    id: str
    title: str
    kind: str
    content: str = ""
    path: str | None = None
    segments: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "kind": self.kind,
                "preview": (self.content[:175] if self.content else f"{self.kind.title()} file"),
                "chunks": len(self.segments) if self.kind == "text" else 1,
                "media_url": f"/api/media/{self.id}" if self.kind == "image" else None}


class Store:
    def __init__(self):
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.model = None
        self.state = "unloaded"
        self.error: str | None = None
        self.docs: list[Document] = []
        self.index: list[tuple[str, int, np.ndarray]] = []
        for title, content in SEEDS:
            self.add_text(title, content)

    def add_text(self, title: str, content: str) -> Document:
        doc = Document(uuid.uuid4().hex, title[:140], "text", content=content, segments=chunks(content))
        with self.lock:
            self.docs.append(doc)
        return doc

    def add_media(self, title: str, kind: str, path: Path) -> Document:
        doc = Document(uuid.uuid4().hex, title[:140], kind, path=str(path))
        with self.lock:
            self.docs.append(doc)
        return doc

    def _encode(self, doc: Document) -> list[np.ndarray]:
        assert self.model is not None
        if doc.kind == "text":
            inputs = [f"title: {doc.title} | text: {passage}" for passage in doc.segments]
            if not inputs:
                return []
            encoded = np.asarray(self.model.encode(inputs, normalize_embeddings=True,
                                    show_progress_bar=False, batch_size=8), dtype=np.float32)
            return [row for row in encoded]
        # Official EmbeddingGemma 2 SentenceTransformer input format for media.
        encoded = self.model.encode({doc.kind: doc.path}, normalize_embeddings=True)
        return [np.asarray(encoded, dtype=np.float32).reshape(-1)]

    def load(self) -> None:
        with self.lock:
            if self.state == "ready":
                return
            if self.state in ("loading", "indexing"):
                raise ValueError("The model is already loading")
            self.state, self.error = "loading", None
        try:
            # Lazy imports: HTML demo and unit tests need no ML dependencies.
            import torch
            from sentence_transformers import SentenceTransformer
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            model = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": dtype})
            with self.lock:
                self.model = model
                self.state = "indexing"
            self.reindex()
            with self.lock:
                self.state = "ready"
        except Exception as e:
            with self.lock:
                self.state = "error"
                self.error = f"{type(e).__name__}: {e}"[:500]
            raise

    def reindex(self) -> None:
        with self.inference_lock:
            with self.lock:
                docs = list(self.docs)
            fresh = []
            for doc in docs:
                for i, emb in enumerate(self._encode(doc)):
                    fresh.append((doc.id, i, unit(emb, 768)))
            with self.lock:
                self.index = fresh

    def index_one(self, doc: Document) -> None:
        if self.state != "ready":
            return
        with self.inference_lock:
            embeddings = self._encode(doc)
            with self.lock:
                self.index.extend((doc.id, i, unit(e, 768)) for i, e in enumerate(embeddings))

    def search(self, query: str, dimension: int, top_k: int) -> dict:
        if self.state != "ready" or self.model is None:
            raise ValueError("Load the real model before searching")
        with self.inference_lock:
            q768 = np.asarray(self.model.encode(query, prompt_name="SearchQuery",
                                         normalize_embeddings=True), dtype=np.float32)
            qvec = unit(q768, dimension)
            with self.lock:
                index, docs = list(self.index), {d.id: d for d in self.docs}
            found = []
            for did, chunk_number, emb in index:
                if did not in docs:
                    continue
                doc = docs[did]
                score = float(np.dot(qvec, unit(emb, dimension)))
                found.append({"id": did, "title": doc.title, "kind": doc.kind,
                              "passage": doc.segments[chunk_number] if doc.kind == "text" else f"{doc.kind.title()} match: {doc.title}",
                              "score": round(score, 4), "chunk": chunk_number + 1,
                              "media_url": f"/api/media/{did}" if doc.kind == "image" else None})
            found.sort(key=lambda x: x["score"], reverse=True)
            return {"mode": "real", "dimension": dimension, "results": found[:top_k],
                    "vector_preview": [round(float(x), 4) for x in qvec[:16]],
                    "indexed_chunks": len(index)}

    def similarity(self, a: str, b: str, dimension: int) -> dict:
        if self.state != "ready" or self.model is None:
            raise ValueError("Load the real model before comparing")
        with self.inference_lock:
            # Same symmetric instruction for both inputs; avoids retrieval-task bias.
            embeddings = np.asarray(self.model.encode([a, b],
                prompt="task: sentence similarity | query: ", normalize_embeddings=True), dtype=np.float32)
            score = float(np.dot(unit(embeddings[0], dimension), unit(embeddings[1], dimension)))
        return {"mode": "real", "score": round(score, 5), "dimension": dimension}

    def delete(self, did: str) -> bool:
        with self.inference_lock:
            with self.lock:
                doc = next((d for d in self.docs if d.id == did), None)
                if doc is None:
                    return False
                self.docs = [d for d in self.docs if d.id != did]
                self.index = [entry for entry in self.index if entry[0] != did]
            if doc.path:
                Path(doc.path).unlink(missing_ok=True)
            return True


store = Store()
app = FastAPI(title="EmbeddingGemma 2 Playground", version="1.0")


class TextInput(BaseModel):
    title: str = Field(min_length=1, max_length=140)
    content: str = Field(min_length=1, max_length=100000)


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    dimension: Literal[128, 256, 512, 768] = 768
    top_k: int = Field(default=4, ge=1, le=10)


class SimilarityInput(BaseModel):
    a: str = Field(min_length=1, max_length=2000)
    b: str = Field(min_length=1, max_length=2000)
    dimension: Literal[128, 256, 512, 768] = 768


@app.get("/api/status")
def status():
    with store.lock:
        return {"model": "google/embeddinggemma-2", "state": store.state,
                "error": store.error, "documents": len(store.docs), "indexed_chunks": len(store.index)}


@app.get("/api/documents")
def documents():
    with store.lock:
        return {"documents": [d.public() for d in store.docs]}


@app.post("/api/model/load")
async def load_model():
    with store.lock:
        if store.state in ("loading", "indexing"):
            return {"state": store.state}
    try:
        await asyncio.to_thread(store.load)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    except Exception as e:
        raise HTTPException(503, "Model initialization failed. See /api/status for details.") from e
    return {"state": store.state, "indexed_chunks": len(store.index)}


@app.post("/api/documents/text")
async def add_text(payload: TextInput):
    if store.state in ("loading", "indexing"):
        raise HTTPException(409, "Wait for model initialization to finish before adding sources")
    doc = store.add_text(payload.title.strip(), payload.content.strip())
    try:
        await asyncio.to_thread(store.index_one, doc)
    except Exception as e:
        store.delete(doc.id)
        raise HTTPException(422, f"Text could not be embedded: {type(e).__name__}: {e}") from e
    return {"document": doc.public()}


@app.post("/api/documents/upload")
async def upload(file: UploadFile = File(...)):
    if store.state in ("loading", "indexing"):
        raise HTTPException(409, "Wait for model initialization to finish before uploading")
    name = (file.filename or "upload").split("/")[-1].split("\\")[-1]
    ext = Path(name).suffix.lower()
    kind = ALLOWED.get(ext)
    if kind is None:
        raise HTTPException(415, "Supported: .txt, .md, .pdf, .png, .jpg, .jpeg, .webp, .wav, .flac, .mp3, .mp4, .mov, .webm")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File exceeds 15 MB demo limit")
    if kind == "text":
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                raise HTTPException(422, f"Cannot extract PDF text: {e}") from e
        else:
            content = data.decode("utf-8-sig", errors="replace")
        if not content.strip():
            raise HTTPException(422, "This file contains no extractable text (scanned PDFs need OCR)")
        return await add_text(TextInput(title=name, content=content[:100000]))
    if kind == "image":
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
        except Exception as e:
            raise HTTPException(422, "Invalid image data") from e
    path = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    doc = store.add_media(name, kind, path)
    try:
        await asyncio.to_thread(store.index_one, doc)
    except Exception as e:
        store.delete(doc.id)
        raise HTTPException(422, f"Media encoding failed: {type(e).__name__}: {e}") from e
    return {"document": doc.public()}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if store.state in ("loading", "indexing"):
        raise HTTPException(409, "Wait for model initialization to finish before deleting")
    if not store.delete(doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}


@app.get("/api/media/{doc_id}")
def media(doc_id: str):
    with store.lock:
        doc = next((d for d in store.docs if d.id == doc_id and d.kind == "image"), None)
    if doc is None or doc.path is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(doc.path)


@app.post("/api/search")
async def search(payload: SearchInput):
    try:
        return await asyncio.to_thread(store.search, payload.query, payload.dimension, payload.top_k)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/similarity")
async def similarity(payload: SimilarityInput):
    try:
        return await asyncio.to_thread(store.similarity, payload.a, payload.b, payload.dimension)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


# Static files last, so the /api routes above take precedence.
app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
