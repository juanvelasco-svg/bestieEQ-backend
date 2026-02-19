"""
rag.py - Motor de búsqueda semántica con FAISS
Depende de: config.py, carpeta indice_faiss/ (generada por el script offline)
Es usado por: main.py

Flujo:
    Al arrancar el servidor  → cargar_indice() carga vectores.index + metadata.json en RAM
    Por cada pregunta        → buscar_contexto() genera embedding y busca los TOP_K más cercanos
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

from config import INDICE_DIR, EMBEDDING_MODEL, TOP_K

# Variables globales: se cargan UNA vez al arrancar, se reutilizan en cada consulta
_modelo: SentenceTransformer | None = None
_indice: faiss.Index | None = None
_metadata: list[dict] | None = None


def cargar_indice() -> bool:
    """
    Carga en RAM el modelo de embeddings, el índice FAISS y la metadata.
    Llamada una sola vez desde el lifespan de FastAPI en main.py.

    Retorna True si la carga fue exitosa, False si no hay índice disponible.
    Si retorna False el chatbot funciona pero sin contexto de documentos.
    """
    global _modelo, _indice, _metadata

    ruta_vectores = INDICE_DIR / "vectores.index"
    ruta_metadata = INDICE_DIR / "metadata.json"

    if not ruta_vectores.exists() or not ruta_metadata.exists():
        print("⚠️  Índice FAISS no encontrado.")
        print(f"   Esperado en: {INDICE_DIR}")
        print("   Ejecuta primero: python scripts/indexar_pdfs.py")
        return False

    print("🔄 Cargando modelo de embeddings (puede tardar 30s la primera vez)...")
    _modelo = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Modelo cargado: {EMBEDDING_MODEL}")

    print("🔄 Cargando índice FAISS...")
    _indice = faiss.read_index(str(ruta_vectores))
    print(f"✅ Índice cargado: {_indice.ntotal} vectores")

    with open(ruta_metadata, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    print(f"✅ Metadata cargada: {len(_metadata)} fragmentos")

    return True


def buscar_contexto(pregunta: str) -> tuple[str, int]:
    """
    Busca los fragmentos de texto más relevantes para la pregunta.

    Args:
        pregunta: Texto de la pregunta del estudiante

    Retorna:
        (contexto_str, tokens_estimados)
        contexto_str: Texto de los TOP_K fragmentos más relevantes, listos para el prompt
        tokens_estimados: Aproximación de tokens (para métricas). 1 token ≈ 4 caracteres.
    """
    if _indice is None or _modelo is None or _metadata is None:
        return "[No hay índice de documentos cargado]", 0

    # Generar embedding de la pregunta en el mismo espacio vectorial que los chunks
    embedding = _modelo.encode(
        [pregunta],
        normalize_embeddings=True,   # Normalizar para que el producto interno = similitud coseno
        show_progress_bar=False
    )

    # Buscar los TOP_K vectores más cercanos
    # _indice.search retorna (distancias, índices) — ambos de forma (1, TOP_K)
    distancias, indices = _indice.search(
        np.array(embedding, dtype=np.float32),
        TOP_K
    )

    fragmentos = []
    for idx in indices[0]:
        if idx == -1:
            # FAISS devuelve -1 cuando el índice tiene menos vectores que TOP_K
            continue
        chunk = _metadata[idx]
        # Encabezado para que el modelo sepa de dónde viene el fragmento
        encabezado = f"--- Fragmento de: {chunk['fuente']} (pág. {chunk.get('pagina', '?')}) ---"
        fragmentos.append(f"{encabezado}\n{chunk['texto']}")

    if not fragmentos:
        return "[No se encontraron fragmentos relevantes en los documentos]", 0

    contexto = "\n\n".join(fragmentos)
    tokens_estimados = len(contexto) // 4  # Estimación simple: 4 caracteres ≈ 1 token

    return contexto, tokens_estimados


def indice_listo() -> bool:
    """Retorna True si el índice está cargado y listo para buscar."""
    return _indice is not None
