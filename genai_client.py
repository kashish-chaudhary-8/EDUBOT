import os
from typing import Union

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def get_groq_client(api_key: str = None):
    """Initialize and return a Groq client."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if key is None:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    if Groq is None:
        raise RuntimeError("groq package not available")

    client = Groq(api_key=key)
    return client


def get_google_client(api_key: str = None):
    """Initialize and configure Google GenAI client."""
    key = api_key or os.getenv("GENAI_API_KEY")
    if key is None:
        raise RuntimeError("GENAI_API_KEY not set in environment")

    if genai is None:
        raise RuntimeError("google-generativeai package not available")

    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-pro")


def get_client(api_key: str = None, provider: str = "groq"):
    """Get client for specified provider (groq or google)."""
    if provider.lower() == "google":
        return get_google_client(api_key)
    elif provider.lower() == "groq":
        return get_groq_client(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def ask_groq_from_client(client, context: str, question: str, model: str = "mixtral-8x7b-32768") -> str:
    """Ask a question using Groq client."""
    prompt = f"""You are EDUBOT, an educational assistant.

Use ONLY the provided context.

Answer the question in clear student-friendly language.

Do not repeat information.
Do not mention 'Based on the provided context'.
Keep the answer concise.

Context:
{context}

Question:
{question}"""

    message = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )

    return message.choices[0].message.content


def ask_google_from_client(client, context: str, question: str) -> str:
    """Ask a question using Google GenAI client."""
    prompt = f"""You are EDUBOT, an educational assistant.

Use ONLY the provided context.

Answer the question in clear student-friendly language.

Do not repeat information.
Do not mention 'Based on the provided context'.
Keep the answer concise.

Context:
{context}

Question:
{question}"""

    response = client.generate_content(prompt)
    return response.text
