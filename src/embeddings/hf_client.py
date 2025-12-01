from __future__ import annotations

import os
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()


class HuggingFaceEmbeddingClient:
    """
    Simple Hugging Face Inference API embedding client.
    Uses MODEL_BACKEND=huggingface_api and HF_API_TOKEN from environment.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        api_url: str | None = None,
    ):
        self.model_name = model_name
        self.api_token = os.getenv("HF_API_TOKEN")
        if not self.api_token:
            raise RuntimeError("HF_API_TOKEN is not set in environment")

        if api_url is None:
            # Standard HF Inference endpoint for embeddings
            api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.api_url = api_url

        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string and return its vector."""
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # HF feature-extraction returns [ [vector] ] or [vector]; normalize to 1D
        if isinstance(data[0], list):
            return data[0]
        return data

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings."""
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data
