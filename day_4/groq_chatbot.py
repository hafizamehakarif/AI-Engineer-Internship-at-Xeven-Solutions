from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# System message
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Give clear and concise answers."
        )
    }
]

print("Chatbot started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
        )

        assistant_reply = response.choices[0].message.content

        print(f"\nChatbot: {assistant_reply}\n")

        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

    except Exception as e:
        print("Error:", e)