"""
main.py - Servidor principal de BestieEQ
Depende de: config.py, rag.py, logger.py, Groq API
Es usado por: Render (lo ejecuta), frontend (llama a sus endpoints)

Endpoints:
    GET  /api/health    → Estado del servidor (Render lo usa para verificar que vive)
    GET  /api/metricas  → Resumen de uso del día (acceso directo desde el navegador)
    POST /api/chat      → Recibe pregunta, retorna respuesta del chatbot
"""

import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import Field

from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL,
    TEMPERATURA, MAX_TOKENS_RESPUESTA,
    CORS_ORIGINS, SYSTEM_PROMPT
)
from rag import cargar_indice, buscar_contexto, indice_listo
from logger import registrar_consulta, obtener_resumen_hoy


# ─── Ciclo de vida: cargar índice al arrancar ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Código que corre AL INICIAR el servidor (antes de recibir peticiones).
    cargar_indice() es costoso (carga modelo + vectores) así que se hace una sola vez.
    """
    print("🚀 Iniciando BestieEQ...")
    cargar_indice()
    print("✅ Servidor listo")
    yield
    # Código que correría al APAGAR (no necesitamos hacer nada especial)


# ─── Aplicación FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="BestieEQ API",
    description="Backend del chatbot educativo de Química",
    version="1.0.0",
    lifespan=lifespan
)

# CORS: permite que el frontend en Vercel llame a esta API
# allow_origin_regex cubre cualquier subdominio de vercel.app durante el desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in CORS_ORIGINS if o],   # Filtrar strings vacíos
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Modelos de datos Pydantic ────────────────────────────────────────────────
class PreguntaRequest(BaseModel):
    """Cuerpo del POST /api/chat que envía el frontend."""
    pregunta: str = Field(..., min_length=3, max_length=1000)
    es_regeneracion: bool = Field(default=False)
    # Últimos mensajes para que el bot recuerde el hilo de la conversación
    historial: list = Field(default=[])


class RespuestaResponse(BaseModel):
    """Cuerpo de la respuesta que recibe el frontend."""
    respuesta: str
    fuentes: list[str]       # Nombres de PDFs usados como fuente
    tiempo_ms: float
    sin_documentos: bool     # True si no había contexto disponible en el índice


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """
    Render llama a este endpoint cada 30s para verificar que el servidor vive.
    Debe responder rápido y con status 200.
    """
    return {
        "status": "ok",
        "indice_cargado": indice_listo(),
        "modelo": GROQ_MODEL
    }


@app.get("/api/metricas")
async def metricas():
    """
    Resumen de métricas del día actual.
    Para ver estadísticas accede directamente a:
    https://tu-backend.onrender.com/api/metricas
    """
    return obtener_resumen_hoy()


@app.post("/api/chat", response_model=RespuestaResponse)
async def chat(request: PreguntaRequest):
    """
    Endpoint principal del chatbot.

    Flujo:
        1. Buscar fragmentos relevantes en FAISS (RAG)
        2. Construir prompt con contexto recuperado
        3. Llamar a Groq API
        4. Registrar métricas
        5. Retornar respuesta al frontend
    """
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY no configurada en las variables de entorno de Render."
        )

    inicio = time.time()

    # ── Paso 1: RAG ───────────────────────────────────────────────────────
    contexto, tokens_contexto = buscar_contexto(request.pregunta)
    sin_documentos = contexto.startswith("[No")

    # Extraer nombres de PDFs del encabezado de cada fragmento
    fuentes: list[str] = []
    if not sin_documentos:
        for linea in contexto.split("\n"):
            if linea.startswith("--- Fragmento de:"):
                nombre = linea.replace("--- Fragmento de:", "").split("(pág.")[0].strip()
                if nombre and nombre not in fuentes:
                    fuentes.append(nombre)

    # ── Paso 2: Construir prompt ──────────────────────────────────────────
    system_prompt = SYSTEM_PROMPT.format(
        contexto=contexto,
        pregunta=request.pregunta
    )

    # Incluir solo los últimos 6 mensajes del historial para no exceder el contexto de Groq
    mensajes_historial = [
        {"role": m["role"], "content": m["content"]}
        for m in request.historial[-6:]
        if m.get("role") in ["user", "assistant"] and m.get("content")
    ]
    mensajes_historial.append({"role": "user", "content": request.pregunta})

    # ── Paso 3: Llamar a Groq ─────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=30.0) as cliente:
        try:
            respuesta_groq = await cliente.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *mensajes_historial
                    ],
                    "temperature": TEMPERATURA,
                    "max_tokens": MAX_TOKENS_RESPUESTA,
                }
            )
            respuesta_groq.raise_for_status()

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Groq tardó demasiado. Intenta de nuevo.")
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Error de Groq: {e.response.status_code}"
            )

    texto_respuesta = respuesta_groq.json()["choices"][0]["message"]["content"]

    # ── Paso 4: Métricas ──────────────────────────────────────────────────
    tiempo_ms = (time.time() - inicio) * 1000
    registrar_consulta(
        pregunta=request.pregunta,
        tiempo_respuesta_ms=tiempo_ms,
        tokens_contexto=tokens_contexto,
        es_regeneracion=request.es_regeneracion,
        sin_contexto=sin_documentos
    )

    # ── Paso 5: Retornar ──────────────────────────────────────────────────
    return RespuestaResponse(
        respuesta=texto_respuesta,
        fuentes=fuentes,
        tiempo_ms=round(tiempo_ms, 2),
        sin_documentos=sin_documentos
    )
