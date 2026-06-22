from datetime import datetime
from langchain_core.tools import tool


datetime_calls = 0
@tool 
def get_current_datetime()->str:
    """"Get current date and time"""
    global datetime_calls
    datetime_calls +=1
    return str(datetime.now())

