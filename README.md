---
title: Análisis de Sentimiento desde Audio (Español)
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.47.0
app_file: app.py
pinned: false
---

# 🎙️➡️🧠 Análisis de Sentimiento desde Audio

Esta aplicación permite **subir un archivo MP3** o **grabar con el micrófono**, transcribe el audio a texto en español usando **Whisper**, y realiza un **análisis de sentimiento multimodelo** sin traducción automática.

## 🔍 Características

- ✅ **Transcripción de audio** con Whisper (`small`)
- ✅ **Análisis de sentimiento en español** combinando:
  - Modelo multilingüe especializado
  - BERT multilingüe
  - Análisis léxico con spaCy + diccionarios en español
  - TextBlob (fallback)
- ✅ **Análisis lingüístico**: tokens, POS, entidades nombradas
- ✅ **Sin traducción automática** → más preciso y rápido
- ✅ Listo para **Hugging Face Spaces** (GPU compatible)

## 🛠️ Tecnologías

- `whisper` (OpenAI)
- `transformers` (Hugging Face)
- `spaCy` + modelo `es_core_news_sm`
- `gradio` para interfaz web
- `textblob` como respaldo

## 📥 Cómo usar

1. Sube un archivo MP3 **o** haz clic en "Grabar" y habla.
2. Haz clic en **"Analizar"**.
3. ¡Obtén el sentimiento, la transcripción y el análisis lingüístico!

> Desarrollado por Edwin Merchán — Bootcamp IA Intermedio Talent Tech 

https://github.com/LordFd0/Whisper-Sentiment-Analysis-Spanish

https://huggingface.co/spaces/EdwinFdo/Whisper-Sentiment-Analysis-Spanish