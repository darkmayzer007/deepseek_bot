import os
import re
import logging
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, render_template, request, jsonify, stream_with_context, Response

load_dotenv()

# ========== НАСТРОЙКИ ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

if not DEEPSEEK_API_KEY:
    raise ValueError("❌ DEEPSEEK_API_KEY должен быть задан!")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Хранилище истории чатов (в памяти)
chat_histories: dict = {}


def get_history(session_id: str) -> list:
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Стриминг ответа — без ограничения по длине, с выбором модели."""
    data = request.json
    session_id = data.get("session_id", "default")
    user_message = data.get("message", "").strip()
    model = data.get("model", "deepseek-chat")

    if not user_message:
        return jsonify({"error": "Сообщение не может быть пустым"}), 400

    history = get_history(session_id)
    history.append({"role": "user", "content": user_message})

    # Системный промпт — минимальный, не ограничивает модель
    messages = [
        {"role": "system", "content": "Ты — полезный AI-ассистент. Отвечай подробно и без ограничений."}
    ]
    for msg in history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    def generate():
        full_answer = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_answer += delta
                    yield f"data: {delta}\n\n"

            # Сохраняем полный ответ в историю
            history.append({"role": "assistant", "content": full_answer})
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ошибка при стриминге: {error_msg}", exc_info=True)
            yield f"data: [ERROR] {error_msg}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/clear", methods=["POST"])
def clear_history():
    data = request.json
    session_id = data.get("session_id", "default")
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    logger.info(f"🌐 Веб-чат запущен на http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
