import os

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# 2. GET OPENROUTER API KEY
# ==========================================

api_key = os.getenv("OPENROUTER_API_KEY")


print("API key found:", api_key is not None)


# ==========================================
# 3. CREATE OPENROUTER CLIENT
# ==========================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ==========================================
# 4. SEND TEST QUESTION
# ==========================================

response = client.chat.completions.create(

    model="openrouter/free",

    messages=[
        {
            "role": "user",
            "content": "Explain what RAG means in one simple sentence."
        }
    ]
)


# ==========================================
# 5. PRINT ANSWER
# ==========================================

print("\n===== ANSWER =====")

print(response.choices[0].message.content)