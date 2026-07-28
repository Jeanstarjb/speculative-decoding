import requests
from typing import Optional, Callable
from pydantic import BaseModel

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    spec_config: Optional[dict] = None

class SpecDecodingClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        })
        
    def generate(self, **kwargs) -> dict:
        response = self.session.post(
            f"{self.base_url}/generate",
            json=GenerationRequest(**kwargs).dict()
        )
        response.raise_for_status()
        return response.json()

    def stream_generate(self, **kwargs):
        with self.session.post(
            f"{self.base_url}/stream",
            json=kwargs,
            stream=True
        ) as response:
            for chunk in response.iter_content(chunk_size=None):
                yield chunk.decode()

    def get_metrics(self) -> dict:
        response = self.session.get(f"{self.base_url}/metrics")
        return response.json()

    def set_vocab_adapter(self, adapter: Callable[[str], dict]):
        self.vocab_adapter = adapter