
import os
import time
import requests
import mock_llm

API_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_llm(messages, temperature=0.0, max_tokens=300, retries=4):
    payload = {
        "model": "mock-model",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    mode = os.getenv("LLM_MODE", "mock").lower()

    if mode == "live":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY before using live mode.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        def send():
            return requests.post(
                API_URL, headers=headers, json=payload, timeout=60
            )
    else:
        headers = {
            "Authorization": "Bearer local-mock-key",
            "Content-Type": "application/json"
        }

        def send():
            return mock_llm.post(
                mock_llm.URL, headers=headers, json=payload, timeout=60
            )

    for attempt in range(retries):
        response = send()

        if response.status_code == 200:
            body = response.json()
            return body["choices"][0]["message"]["content"]

        if response.status_code == 429 and attempt < retries - 1:
            time.sleep(0.1 * (attempt + 1))
            continue

        raise RuntimeError(
            f"LLM request failed: HTTP {response.status_code} - "
            f"{response.text}"
        )

    raise RuntimeError("LLM request failed after retries.")
