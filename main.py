import time
import os
import httpx
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL,
    TEMPERATURA, MAX_TOKENS_RESPUESTA,
    FRONTEND_URL, SYSTEM_PROMPT
)
from rag import cargar_indice, buscar_contexto, indice_listo
from logger import registrar_consulta, obtener_resumen_hoy

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=False)

with app.app_context():
    cargar_indice()

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "indice_cargado": indice_listo(), "modelo": GROQ_MODEL})

@app.get("/api/metricas")
def metricas():
    return jsonify(obtener_resumen_hoy())

@app.post("/api/chat")
def chat():
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY no configurada"}), 500

    data = request.get_json()
    pregunta = data.get("pregunta", "").strip()
    es_regeneracion = data.get("es_regeneracion", False)
    historial = data.get("historial", [])

    if not pregunta or len(pregunta) < 3:
        return jsonify({"error": "Pregunta muy corta"}), 400

    inicio = time.time()

    contexto, tokens_contexto = buscar_contexto(pregunta)
    sin_documentos = contexto.startswith("[No")

    fuentes = []
    if not sin_documentos:
        for linea in contexto.split("\n"):
            if linea.startswith("--- Fragmento de:"):
                nombre = linea.replace("--- Fragmento de:", "").split("(pág.")[0].strip()
                if nombre and nombre not in fuentes:
                    fuentes.append(nombre)

    system_prompt = SYSTEM_PROMPT.format(contexto=contexto, pregunta=pregunta)

    mensajes = [
        {"role": m["role"], "content": m["content"]}
        for m in historial[-6:]
        if m.get("role") in ["user", "assistant"] and m.get("content")
    ]
    mensajes.append({"role": "user", "content": pregunta})

    try:
        respuesta_groq = httpx.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "system", "content": system_prompt}, *mensajes],
                "temperature": TEMPERATURA,
                "max_tokens": MAX_TOKENS_RESPUESTA,
            },
            timeout=30.0
        )
        respuesta_groq.raise_for_status()
    except httpx.TimeoutException:
        return jsonify({"error": "Timeout de Groq"}), 504
    except httpx.HTTPStatusError as e:
        return jsonify({"error": f"Error de Groq: {e.response.status_code}"}), 502

    texto = respuesta_groq.json()["choices"][0]["message"]["content"]
    tiempo_ms = (time.time() - inicio) * 1000

    registrar_consulta(
        pregunta=pregunta,
        tiempo_respuesta_ms=tiempo_ms,
        tokens_contexto=tokens_contexto,
        es_regeneracion=es_regeneracion,
        sin_contexto=sin_documentos
    )

    return jsonify({
        "respuesta": texto,
        "fuentes": fuentes,
        "tiempo_ms": round(tiempo_ms, 2),
        "sin_documentos": sin_documentos
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)