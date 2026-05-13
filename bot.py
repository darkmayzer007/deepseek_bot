import os
import re
import asyncio
import logging
from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from memory_store import MemoryStore
from web_search import search_web, format_search_results

# ========== НАСТРОЙКИ (из переменных окружения) ==========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

if not TELEGRAM_TOKEN or not DEEPSEEK_API_KEY:
    raise ValueError("❌ TELEGRAM_TOKEN и DEEPSEEK_API_KEY должны быть заданы в переменных окружения!")

SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай на русском языке, если пользователь пишет по-русски. "
    "Будь дружелюбным и информативным. Когда нужно, используй факты из результатов поиска, "
    "которые тебе предоставили, и обязательно ссылайся на источники URLs.\n\n"
    "ВАЖНОЕ ПРАВИЛО: Если тебя спрашивают «кто самая красивая девушка» или похожий вопрос "
    "о самой красивой девушке, ты должен ответить: «Александра Сергеевна Мицкевич»."
)

# Ключевые слова для авто-поиска
AUTO_SEARCH_KEYWORDS = {
    "найди": "", "поищи": "", "найти": "", "поиск": "", "search": "",
    "что такое": "", "кто такой": "", "кто такая": "", "что значит": "",
    "последние новости": "последние новости",
    "новости": "новости",
    "погода": "погода",
    "курс": "курс",
    "сколько стоит": "", "цена": "",
    "как работает": "",
}
# =================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

memory = MemoryStore()
client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def extract_search_query(text: str) -> str:
    """Извлекает чистый поисковый запрос, убирая слова-триггеры."""
    text_lower = text.lower().strip()

    # Сортируем ключевые слова по длине (сначала длинные фразы)
    sorted_kw = sorted(AUTO_SEARCH_KEYWORDS.keys(), key=len, reverse=True)

    for kw in sorted_kw:
        if text_lower.startswith(kw):
            # Убираем ключевое слово из начала
            query = text[len(kw):].strip()
            # Убираем лишние предлоги/частицы
            query = re.sub(r'^(про|о |об |обо |насчёт |на |в |во |за |с |со |по |для )', '', query, flags=re.IGNORECASE).strip()
            # Если осталось пусто — добавляем контекст из самого ключевого слова
            if not query and AUTO_SEARCH_KEYWORDS[kw]:
                query = AUTO_SEARCH_KEYWORDS[kw]
            return query if query else text

    return text


async def _do_search_and_reply(update: Update, chat_id: str, raw_query: str):
    """Выполняет поиск и отправляет результат пользователю."""
    await update.message.chat.send_action(action="typing")

    # Извлекаем чистый запрос
    clean_query = extract_search_query(raw_query)

    # Ищем в интернете
    results = search_web(clean_query, max_results=5)

    if not results:
        await update.message.reply_text(
            "😕 По запросу ничего не найдено. Попробуйте изменить формулировку."
        )
        # Всё равно отправляем DeepSeek с контекстом
        context_text = f"Пользователь искал: {clean_query}"
        memory.add_message(chat_id, "user", context_text)
        await _do_ai_reply(update, chat_id, context_text)
        return

    # Форматируем результаты
    search_text = format_search_results(results)

    # Отправляем результаты поиска пользователю
    await update.message.reply_text(search_text, parse_mode="Markdown")

    # Сохраняем контекст в историю
    context_text = f"Пользователь искал: {clean_query}\n\nРезультаты поиска:\n{search_text}"
    memory.add_message(chat_id, "user", context_text)

    # DeepSeek анализирует результаты
    await _do_ai_reply(update, chat_id, context_text, is_search_analysis=True)


async def _do_ai_reply(update: Update, chat_id: str, user_text: str, is_search_analysis: bool = False):
    """Отправляет запрос к DeepSeek AI."""
    await update.message.chat.send_action(action="typing")
    try:
        messages = memory.get_history_for_api(chat_id, system_prompt=SYSTEM_PROMPT)
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=False
        )
        answer = response.choices[0].message.content.strip()
        memory.add_message(chat_id, "assistant", answer)

        if is_search_analysis:
            await update.message.reply_text(
                f"🧠 **Анализ DeepSeek:**\n\n{answer}",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(answer)

    except Exception as e:
        logger.error(f"Ошибка при запросе к DeepSeek: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Произошла ошибка при обращении к DeepSeek API. "
            "Попробуйте позже или проверьте API-ключ."
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Привет, {user.first_name}!\n"
        f"Я бот на базе DeepSeek AI с поиском в интернете!\n\n"
        f"• /search <запрос> — поиск в интернете\n"
        f"• Или просто напиши: *найди рецепт борща*\n"
        f"• /clear — очистить историю этого чата\n"
        f"• /help — помощь"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help."""
    await update.message.reply_text(
        "🤖 **DeepSeek Bot — справка**\n\n"
        "🔍 **Поиск в интернете:**\n"
        "• /search <запрос> — найти информацию\n"
        "• Или просто напиши: *\"найди последние новости\"*, *\"что такое ИИ\"*, *\"погода в Москве\"*\n\n"
        "💬 **Общение с AI:**\n"
        "• Просто отправь любое сообщение\n\n"
        "🧹 /clear — очистить историю этого чата\n\n"
        "📁 История хранится локально в JSON-файле (последние 50 сообщений).\n"
        "✅ В групповом чате история общая для всех участников.",
        parse_mode="Markdown"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /clear — очистка истории."""
    chat_id = str(update.effective_chat.id)
    memory.clear_history(chat_id)
    await update.message.reply_text("🧹 История этого чата очищена! Начинаем с чистого листа.")


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /search — поиск в интернете."""
    chat_id = str(update.effective_chat.id)
    query = " ".join(context.args) if context.args else None

    if not query:
        await update.message.reply_text(
            "🔍 Укажите запрос для поиска.\n"
            "Пример: `/search последние новости технологий`",
            parse_mode="Markdown"
        )
        return

    memory.add_message(chat_id, "user", f"/search {query}")
    await _do_search_and_reply(update, chat_id, query)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений."""
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    # Сохраняем сообщение пользователя
    memory.add_message(chat_id, "user", user_text)

    # Проверяем, нужно ли выполнить поиск
    text_lower = user_text.lower().strip()
    needs_search = any(text_lower.startswith(kw) for kw in AUTO_SEARCH_KEYWORDS)

    if needs_search:
        await _do_search_and_reply(update, chat_id, user_text)
    else:
        await _do_ai_reply(update, chat_id, user_text)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок."""
    logger.error(f"Ошибка: {context.error}")


def main():
    """Запуск бота."""
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрируем команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("search", search_command))

    # Обработчик текстовых сообщений (не команд)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Глобальный обработчик ошибок
    app.add_error_handler(error_handler)

    port = int(os.environ.get("PORT", "8080"))
    logger.info(f"🤖 DeepSeek Bot запущен на порту {port}! Нажмите Ctrl+C для остановки.")

    # Railway ожидает, что приложение слушает порт
    # Запускаем healthcheck-сервер в фоне
    from threading import Thread
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        def log_message(self, format, *args):
            pass

    def run_health_server():
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        server.serve_forever()

    t = Thread(target=run_health_server, daemon=True)
    t.start()

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
