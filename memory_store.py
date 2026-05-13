import json
import os
from datetime import datetime
from typing import Dict, List

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "chat_memory.json")
MAX_HISTORY = 50  # Максимум сообщений в истории на пользователя


class MemoryStore:
    """Локальное JSON-хранилище историй чатов."""

    def __init__(self):
        self._data: Dict[str, List[dict]] = {}
        self._load()

    def _load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_history(self, user_id: str) -> List[dict]:
        """Возвращает историю сообщений пользователя."""
        return self._data.get(user_id, [])

    def add_message(self, user_id: str, role: str, content: str):
        """Добавляет сообщение в историю пользователя."""
        if user_id not in self._data:
            self._data[user_id] = []
        self._data[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Обрезаем историю до MAX_HISTORY
        if len(self._data[user_id]) > MAX_HISTORY:
            self._data[user_id] = self._data[user_id][-MAX_HISTORY:]
        self._save()

    def clear_history(self, user_id: str):
        """Очищает историю пользователя."""
        if user_id in self._data:
            del self._data[user_id]
            self._save()

    def get_all_users(self) -> List[str]:
        """Возвращает список всех пользователей с историей."""
        return list(self._data.keys())

    def get_history_for_api(self, user_id: str, system_prompt: str = None) -> List[dict]:
        """
        Возвращает историю в формате, пригодном для OpenAI API.
        Системный промпт добавляется первым сообщением.
        """
        history = self.get_history(user_id)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        return messages
