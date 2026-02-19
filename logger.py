"""
logger.py - Registro de métricas de uso de BestieEQ
Depende de: config.py (solo usa LOG_FILE)
Es usado por: main.py

Registra en formato JSON Lines (una entrada por línea).
Cada línea es un JSON válido independiente, fácil de leer con cualquier herramienta.
"""

import json
from datetime import datetime, timezone
from config import LOG_FILE

# Palabras clave de química para detectar temas automáticamente
TEMAS_QUIMICA = [
    "átomo", "molécula", "enlace", "tabla periódica", "reacción",
    "estequiometría", "mol", "gas", "presión", "temperatura",
    "solución", "concentración", "ácido", "base", "ph",
    "redox", "oxidación", "reducción", "termoquímica", "entalpía",
    "cinética", "equilibrio", "orgánica", "inorgánica", "electroquímica",
    "orbital", "electrón", "protón", "neutrón", "isótopo",
    "nomenclatura", "formulación", "valencia", "ión", "sal"
]


def detectar_temas(pregunta: str) -> list[str]:
    """
    Detecta qué temas de química menciona la pregunta.
    Retorna lista de temas encontrados (puede ser vacía).
    """
    pregunta_lower = pregunta.lower()
    return [tema for tema in TEMAS_QUIMICA if tema in pregunta_lower]


def registrar_consulta(
    pregunta: str,
    tiempo_respuesta_ms: float,
    tokens_contexto: int,
    es_regeneracion: bool = False,
    sin_contexto: bool = False
) -> None:
    """
    Escribe una línea en metricas.jsonl con los datos de la consulta.

    Args:
        pregunta: Texto de la pregunta del estudiante
        tiempo_respuesta_ms: Milisegundos que tardó la respuesta completa
        tokens_contexto: Estimación de tokens enviados como contexto a Groq
        es_regeneracion: True si el usuario presionó el botón Regenerar
        sin_contexto: True si el índice FAISS no devolvió fragmentos relevantes
    """
    entrada = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fecha": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "hora": datetime.now(timezone.utc).strftime("%H:%M"),
        "temas_detectados": detectar_temas(pregunta),
        "longitud_pregunta": len(pregunta),
        "tiempo_respuesta_ms": round(tiempo_respuesta_ms, 2),
        "tokens_contexto": tokens_contexto,
        "es_regeneracion": es_regeneracion,
        "sin_contexto": sin_contexto,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entrada, ensure_ascii=False) + "\n")


def obtener_resumen_hoy() -> dict:
    """
    Lee metricas.jsonl y retorna estadísticas del día actual.
    Usado por el endpoint GET /api/metricas en main.py.
    """
    if not LOG_FILE.exists():
        return {
            "preguntas_hoy": 0,
            "temas_populares": [],
            "tiempo_promedio_ms": 0,
            "regeneraciones": 0
        }

    hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    consultas_hoy = []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for linea in f:
            try:
                entrada = json.loads(linea.strip())
                if entrada.get("fecha") == hoy:
                    consultas_hoy.append(entrada)
            except json.JSONDecodeError:
                continue  # Ignorar líneas corruptas

    if not consultas_hoy:
        return {
            "preguntas_hoy": 0,
            "temas_populares": [],
            "tiempo_promedio_ms": 0,
            "regeneraciones": 0
        }

    # Contar frecuencia de cada tema
    conteo_temas: dict[str, int] = {}
    for c in consultas_hoy:
        for tema in c.get("temas_detectados", []):
            conteo_temas[tema] = conteo_temas.get(tema, 0) + 1

    temas_ordenados = sorted(conteo_temas.items(), key=lambda x: x[1], reverse=True)

    return {
        "preguntas_hoy": len(consultas_hoy),
        "temas_populares": temas_ordenados[:5],
        "tiempo_promedio_ms": round(
            sum(c["tiempo_respuesta_ms"] for c in consultas_hoy) / len(consultas_hoy), 2
        ),
        "regeneraciones": sum(1 for c in consultas_hoy if c.get("es_regeneracion"))
    }
