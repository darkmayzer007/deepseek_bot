from duckduckgo_search import DDGS
from typing import List, Dict, Optional


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Выполняет поиск в интернете через DuckDuckGo.
    Возвращает список результатов: заголовок, ссылка, сниппет.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        print(f"Ошибка поиска: {e}")
    return results


def format_search_results(results: List[Dict[str, str]]) -> str:
    """Форматирует результаты поиска в читаемый текст."""
    if not results:
        return "Поиск не дал результатов."

    text = "📊 **Результаты поиска:**\n\n"
    for i, r in enumerate(results, 1):
        text += f"{i}. **{r['title']}**\n"
        text += f"   {r['snippet'][:200]}{'...' if len(r['snippet']) > 200 else ''}\n"
        text += f"   🔗 {r['link']}\n\n"
    return text
