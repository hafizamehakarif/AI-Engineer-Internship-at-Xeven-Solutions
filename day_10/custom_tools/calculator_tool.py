from langchain_core.tools import tool
calculator_calls = 0

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression and return the result.
    """

    global calculator_calls
    calculator_calls += 1

    try:
        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"Calculation Error: {e}"


