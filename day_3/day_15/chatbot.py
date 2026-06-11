from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input="In one sentence, explain what an LLM is."
    )

    print("\nResponse:")
    print(response.output_text)

except Exception as e:
    print("Error:", e)