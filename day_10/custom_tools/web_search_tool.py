from langchain_core.tools import tool

web_search_calls = 0


@tool
def web_search(query: str) -> str:
    """
    Search the web and return summarized results.
    """

    global web_search_calls
    web_search_calls += 1

    try:
        return f"Summary of search results for: {query}"

    except Exception as e:
        return f"Search Error: {e}"