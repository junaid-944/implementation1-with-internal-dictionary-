import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

api_key = os.getenv("GEMINI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

for model in ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-2.0-flash", "models/gemini-2.0-flash"]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"SUCCESS with {model}: {response.choices[0].message.content}")
    except Exception as e:
        print(f"FAILED with {model}: {e}")
