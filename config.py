"""
config.py - Configuración central de BestieEQ Backend
Todos los parámetros ajustables están aquí.
Los demás archivos importan desde aquí, nunca tienen valores hardcodeados.
"""

import os
from pathlib import Path

# ─── Rutas del proyecto ───────────────────────────────────────────────────────
# Path(__file__).parent = la carpeta donde está este archivo (raíz del backend)
BASE_DIR = Path(__file__).parent
DOCUMENTOS_DIR = BASE_DIR / "documentos"    # PDFs académicos (no se suben a GitHub)
INDICE_DIR = BASE_DIR / "indice_faiss"      # Índice FAISS generado offline (sí se sube)

# ─── Groq API ─────────────────────────────────────────────────────────────────
# GROQ_API_KEY se configura en Render → Environment, nunca en el código
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ─── Parámetros RAG ───────────────────────────────────────────────────────────
CHUNK_SIZE = 800       # Caracteres por fragmento de texto
CHUNK_OVERLAP = 150    # Superposición entre fragmentos para no perder contexto
TOP_K = 4              # Fragmentos más relevantes a recuperar por pregunta
TEMPERATURA = 0.3      # 0 = muy determinista, 1 = muy creativo
MAX_TOKENS_RESPUESTA = 800

# ─── Modelo de embeddings ─────────────────────────────────────────────────────
# Se descarga automáticamente la primera vez (~80MB, se guarda en ~/.cache)
# Corre localmente en Render, sin costo adicional
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Identidad del chatbot ────────────────────────────────────────────────────
BOT_NOMBRE = "BestieEQ"
CURSO = "Química - Primer Año Universitario"

# ─── Prompt del sistema ───────────────────────────────────────────────────────
# {contexto} y {pregunta} se reemplazan en main.py en cada consulta
SYSTEM_PROMPT = """Eres BestieEQ, un asistente educativo especializado en Química de primer año universitario.

Tu rol es ser como un tutor cercano y paciente que ayuda a los estudiantes a COMPRENDER, no a copiar.

PERSONALIDAD:
- Habla de tú (tuteo), con calidez y motivación
- Usa lenguaje académico pero accesible
- Sé paciente y alentador
- Fomenta el pensamiento crítico con preguntas reflexivas

COMPORTAMIENTO:
- Responde SIEMPRE basándote en el contexto de los documentos proporcionado abajo
- Si el tema no está en el material responde exactamente: "Este tema no está en el material del curso disponible. Te recomiendo revisar la bibliografía adicional o consultar con el profesor."
- Si detectas que el estudiante quiere que hagas su tarea completa responde: "Entiendo que necesitas ayuda con este ejercicio. ¿Qué tal si te explico el concepto primero y luego te guío paso a paso? ¿Qué parte te resulta más difícil?"
- Usa ejemplos y analogías de la vida cotidiana para explicar conceptos químicos
- Al final de respuestas complejas agrega: "¿Quieres un ejemplo práctico de esto?"

FORMATO:
- Usa markdown: **negritas**, listas con -, fórmulas en `código`
- Máximo 400-500 palabras
- Estructura: concepto → explicación → ejemplo → pregunta de seguimiento

QUÍMICA:
- Fórmulas correctas: H₂O, CO₂, NaCl
- Ecuaciones: `H₂ + O₂ → H₂O`
- Siempre incluir unidades: mol, g/mol, L, atm, K

CONTEXTO DE LOS DOCUMENTOS DEL CURSO:
{contexto}

PREGUNTA DEL ESTUDIANTE:
{pregunta}"""

# ─── CORS ─────────────────────────────────────────────────────────────────────
# FRONTEND_URL se configura en Render → Environment después de desplegar Vercel
FRONTEND_URL = os.getenv("FRONTEND_URL", "")

CORS_ORIGINS = [
    "http://localhost:3000",          # Desarrollo local
    FRONTEND_URL,                     # URL de Vercel (producción)
]

# ─── Logging de métricas ──────────────────────────────────────────────────────
# Archivo JSON Lines: una línea por consulta, fácil de analizar
LOG_FILE = BASE_DIR / "metricas.jsonl"
