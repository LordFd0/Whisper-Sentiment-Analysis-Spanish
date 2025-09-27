import gradio as gr
import whisper
import spacy
from transformers import pipeline
from textblob import TextBlob
import torch

# ======================
# Carga de Modelos
# ======================

print("🔍 Cargando Whisper (small)...")
whisper_model = whisper.load_model("small")

print("🔍 Cargando modelos de sentimiento...")
sentiment_pipelines = {}

# Modelo multilingüe especializado
try:
    sentiment_pipelines["multilingual"] = pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis",
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ Modelo multilingüe cargado")
except Exception as e:
    print(f"❌ Error al cargar modelo multilingüe: {e}")
    sentiment_pipelines["multilingual"] = None

# BERT multilingüe (estrellas)
try:
    sentiment_pipelines["bert"] = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ Modelo BERT cargado")
except Exception as e:
    print(f"❌ Error al cargar modelo BERT: {e}")
    sentiment_pipelines["bert"] = None

# spaCy para español
print("🔍 Cargando spaCy...")
try:
    nlp = spacy.load("es_core_news_sm")
    print("✅ spaCy cargado")
except OSError:
    print("❌ Modelo spaCy no encontrado. Análisis lingüístico desactivado.")
    nlp = None

# Diccionarios léxicos en español
PALABRAS_POSITIVAS = {
    'bueno', 'excelente', 'fantástico', 'maravilloso', 'perfecto', 'genial',
    'increíble', 'amo', 'encanta', 'feliz', 'contento', 'satisfecho', 'agradable',
    'recomiendo', 'magnífico', 'extraordinario', 'asombroso', 'estupendo',
    'óptimo', 'superior', 'mejor', 'inmejorable', 'ideal'
}

PALABRAS_NEGATIVAS = {
    'malo', 'terrible', 'horrible', 'pésimo', 'odio', 'decepcionado', 'fatal',
    'triste', 'enojado', 'frustrado', 'pobre', 'deficiente', 'desastroso',
    'insatisfecho', 'decepcionante', 'horroroso', 'inútil', 'defectuoso',
    'deplorable', 'lamentable', 'desagradable'
}

# ======================
# Funciones de análisis
# ======================

def transcribe_audio(audio_path):
    """Transcribe audio a texto usando Whisper"""
    if not audio_path:
        return ""
    result = whisper_model.transcribe(audio_path, language="es")
    return result["text"].strip()

def analyze_sentiment_multimodel(text):
    """Combina 4 métodos de análisis de sentimiento"""
    if not text.strip():
        return {}, "Texto vacío", 0.0, "gray"

    resultados = {}

    # 1. Modelo multilingüe
    if sentiment_pipelines["multilingual"]:
        try:
            out = sentiment_pipelines["multilingual"](text)[0]
            score = out["score"] if out["label"] == "POSITIVE" else -out["score"]
            resultados["multilingual"] = {"normalized_score": score, "label": out["label"], "score": out["score"]}
        except Exception as e:
            resultados["multilingual"] = {"error": str(e)}

    # 2. BERT multilingüe (estrellas → -1 a 1)
    if sentiment_pipelines["bert"]:
        try:
            out = sentiment_pipelines["bert"](text)[0]
            star_map = {'1 star': -1.0, '2 stars': -0.5, '3 stars': 0.0, '4 stars': 0.5, '5 stars': 1.0}
            score = star_map.get(out["label"], 0.0)
            resultados["bert"] = {"normalized_score": score, "label": out["label"], "score": out["score"]}
        except Exception as e:
            resultados["bert"] = {"error": str(e)}

    # 3. Análisis léxico con spaCy
    if nlp:
        try:
            doc = nlp(text.lower())
            lemmas = [token.lemma_ for token in doc if token.is_alpha and len(token.text) > 2]
            pos = sum(1 for w in lemmas if w in PALABRAS_POSITIVAS)
            neg = sum(1 for w in lemmas if w in PALABRAS_NEGATIVAS)
            total = len(lemmas)
            polarity = (pos - neg) / total if total > 0 else 0.0
            polarity = max(-1.0, min(1.0, polarity * 2))  # ajuste suave
            resultados["lexico"] = {"normalized_score": polarity, "positivas": pos, "negativas": neg}
        except Exception as e:
            resultados["lexico"] = {"error": str(e)}

    # 4. TextBlob (fallback)
    try:
        blob = TextBlob(text)
        resultados["textblob"] = {"normalized_score": blob.sentiment.polarity}
    except Exception as e:
        resultados["textblob"] = {"error": str(e)}

    # Ponderación final
    pesos = {"multilingual": 0.4, "bert": 0.3, "lexico": 0.2, "textblob": 0.1}
    scores = []
    for metodo, peso in pesos.items():
        if metodo in resultados and "normalized_score" in resultados[metodo]:
            scores.append(resultados[metodo]["normalized_score"] * peso)

    final_score = sum(scores) if scores else 0.0

    if final_score > 0.15:
        label, color = "😊 POSITIVO", "green"
    elif final_score < -0.15:
        label, color = "😠 NEGATIVO", "red"
    else:
        label, color = "😐 NEUTRO", "gray"

    return resultados, label, final_score, color

def linguistic_analysis(text):
    """Análisis lingüístico con spaCy"""
    if not nlp or not text.strip():
        return "❌ spaCy no disponible o texto vacío."

    doc = nlp(text)
    stats = {
        "tokens": len(doc),
        "palabras": len([t for t in doc if t.is_alpha]),
        "oraciones": len(list(doc.sents)),
        "entidades": len(doc.ents)
    }

    output = f"📊 Estadísticas: {stats['tokens']} tokens, {stats['palabras']} palabras, "
    output += f"{stats['oraciones']} oraciones, {stats['entidades']} entidades.\n\n"

    if doc.ents:
        output += "🏷 Entidades:\n"
        for ent in doc.ents[:10]:
            output += f"  • {ent.text} ({ent.label_}: {spacy.explain(ent.label_)})\n"
    else:
        output += "🏷 Sin entidades detectadas.\n"

    output += "\n📝 Tokens (primeros 15):\n"
    for token in doc[:15]:
        output += f"  '{token.text}' → lemma: '{token.lemma_}', POS: {token.pos_} ({spacy.explain(token.pos_)})\n"

    return output

# ======================
# Interfaz Gradio
# ======================

def process_audio(audio_file, audio_mic):
    source = audio_mic if audio_mic is not None else audio_file
    if source is None:
        return "❌ Por favor, sube un archivo o graba con el micrófono.", "", "", ""

    try:
        text = transcribe_audio(source)
        if not text:
            return "❌ No se pudo transcribir audio.", "", "", ""

        resultados, label, score, color = analyze_sentiment_multimodel(text)
        ling = linguistic_analysis(text)

        resumen = f"""
        <div style='background-color:{color}20; padding: 15px; border-radius: 8px; border-left: 4px solid {color};'>
            <h3 style='margin:0; color:{color};'>{label}</h3>
            <p><strong>Puntuación:</strong> {score:.3f}</p>
            <p><strong>Texto:</strong> {text[:200]}{'...' if len(text) > 200 else ''}</p>
        </div>
        """

        detalles = "<h4>📊 Métodos:</h4>"
        for metodo, res in resultados.items():
            detalles += f"<strong>{metodo}:</strong> "
            if "error" in res:
                detalles += f"<span style='color:red'>Error: {res['error']}</span><br>"
            else:
                detalles += f"Puntuación: {res.get('normalized_score', 'N/A'):.3f}<br>"

        return resumen, detalles, ling, text

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""

with gr.Blocks(title="Análisis de Sentimiento desde Audio") as demo:
    gr.Markdown("# 🎙️➡️🧠 Análisis de Sentimiento desde Audio")
    gr.Markdown("Sube un archivo MP3 **o** graba con tu micrófono. El sistema transcribe y analiza el sentimiento en español.")

    with gr.Row():
        with gr.Column():
            audio_file = gr.Audio(sources=["upload"], type="filepath", label="📁 Subir archivo MP3")
            audio_mic = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Grabar con micrófono")
            btn = gr.Button("🚀 Analizar", variant="primary")

        with gr.Column():
            resultado_html = gr.HTML(label="🎯 Resultado Principal")
            detalles_html = gr.HTML(label="🔍 Detalles por Método")
            ling_text = gr.Textbox(label="📝 Análisis Lingüístico", lines=10, interactive=False)
            transcripcion = gr.Textbox(label="📜 Transcripción", lines=3, interactive=False)

    btn.click(
        fn=process_audio,
        inputs=[audio_file, audio_mic],
        outputs=[resultado_html, detalles_html, ling_text, transcripcion]
    )

if __name__ == "__main__":
    demo.launch()