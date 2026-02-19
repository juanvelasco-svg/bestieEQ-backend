"""
rag.py - Motor RAG usando HuggingFace Inference API para embeddings
Sin sentence-transformers: elimina el problema de RAM en Render free tier.
Los embeddings se generan via API gratuita de HuggingFace, no localmente.
"""

import json
import os
import numpy as np
import httpx
import faiss

from config import INDICE_DIR, TOP_K

# Token de HuggingFace (gratis, se configura en Render como variable de entorno)
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

_indice: faiss.Index | None = None
_metadata: list[dict] | None = None


def cargar_indice() -> bool:
    global _indice, _metadata

    ruta_vectores = INDICE_DIR / "vectores.index"
    ruta_metadata = INDICE_DIR / "metadata.json"

    if not ruta_vectores.exists() or not ruta_metadata.exists():
        print("⚠️  Índice FAISS no encontrado.")
        return False

    print("🔄 Cargando índice FAISS...")
    _indice = faiss.read_index(str(ruta_vectores))
    print(f"✅ Índice cargado: {_indice.ntotal} vectores")

    with open(ruta_metadata, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    print(f"✅ Metadata cargada: {len(_metadata)} fragmentos")
    print("✅ Servidor listo")
    return True


def generar_embedding(texto: str) -> np.ndarray | None:
    """
    Genera embedding via HuggingFace Inference API (gratuita).
    No requiere cargar el modelo en RAM.
    """
    try:
        respuesta = httpx.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": texto},
            timeout=15.0
        )
        respuesta.raise_for_status()
        embedding = np.array(respuesta.json(), dtype=np.float32)
        # Normalizar para similitud coseno
        norma = np.linalg.norm(embedding)
        if norma > 0:
            embedding = embedding / norma
        return embedding
    except Exception as e:
        print(f"⚠️  Error generando embedding: {e}")
        return None


def buscar_contexto(pregunta: str) -> tuple[str, int]:
    if _indice is None or _metadata is None:
        return "[No hay índice de documentos cargado]", 0

    embedding = generar_embedding(pregunta)
    if embedding is None:
        return "[Error al generar embedding para la búsqueda]", 0

    distancias, indices = _indice.search(
        np.array([embedding], dtype=np.float32),
        TOP_K
    )

    fragmentos = []
    for idx in indices[0]:
        if idx == -1:
            continue
        chunk = _metadata[idx]
        encabezado = f"--- Fragmento de: {chunk['fuente']} (pág. {chunk.get('pagina', '?')}) ---"
        fragmentos.append(f"{encabezado}\n{chunk['texto']}")

    if not fragmentos:
        return "[No se encontraron fragmentos relevantes]", 0

    contexto = "\n\n".join(fragmentos)
    return contexto, len(contexto) // 4


def indice_listo() -> bool:
    return _indice is not None