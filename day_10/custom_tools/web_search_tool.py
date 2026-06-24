from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

web_search_calls = 0


@tool
def web_search(query: str) -> str:
    """
    Search the internet for current news and information.
    Use this when the user asks about recent events, current news,
    live data, or anything that may have changed recently.

    Args:
        query: The search query string to look up on the web.

    Returns:
        A string containing relevant search results.
    """

    global web_search_calls
    web_search_calls += 1

    try:
        results = search.invoke(query)
        if not results or not results.strip():
            return "No results found for this query. Try rephrasing."
        return results

    except Exception as e:
        return f"Search Error: {str(e)}"