"""
Multi-provider embedding generation service
Supports: OpenAI, Cohere, HuggingFace, Ollama, Google
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import json
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import cohere
except ImportError:
    cohere = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class EmbeddingProviderBase(ABC):
    """Base class for embedding providers"""

    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass


class OpenAIEmbeddingProvider(EmbeddingProviderBase):
    """OpenAI embedding provider"""

    DIMENSION_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small", config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        if OpenAI is None:
            raise ImportError("openai package required")
        api_key = config.get("api_key") if config else None
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def get_dimension(self) -> int:
        return self.DIMENSION_MAP.get(self.model_name, 1536)


class CohereEmbeddingProvider(EmbeddingProviderBase):
    """Cohere embedding provider"""

    DIMENSION_MAP = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
    }

    def __init__(self, model_name: str = "embed-english-v3.0", config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        if cohere is None:
            raise ImportError("cohere package required")
        api_key = config.get("api_key") if config else None
        self.client = cohere.Client(api_key=api_key or os.getenv("COHERE_API_KEY"))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embed(texts=texts, model=self.model_name, input_type="search_document")
        return response.embeddings

    def embed_query(self, query: str) -> List[float]:
        response = self.client.embed(texts=[query], model=self.model_name, input_type="search_query")
        return response.embeddings[0]

    def get_dimension(self) -> int:
        return self.DIMENSION_MAP.get(self.model_name, 1024)


class HuggingFaceEmbeddingProvider(EmbeddingProviderBase):
    """HuggingFace sentence-transformers provider"""

    DIMENSION_MAP = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers package required")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def get_dimension(self) -> int:
        return self.DIMENSION_MAP.get(self.model_name, 384)


class OllamaEmbeddingProvider(EmbeddingProviderBase):
    """Ollama local embedding provider"""

    DIMENSION_MAP = {"nomic-embed-text": 768}

    def __init__(self, model_name: str = "nomic-embed-text", config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        if ollama is None:
            raise ImportError("ollama package required")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def get_dimension(self) -> int:
        return self.DIMENSION_MAP.get(self.model_name, 768)


class GoogleEmbeddingProvider(EmbeddingProviderBase):
    """Google Vertex AI embedding provider"""

    DIMENSION_MAP = {"models/embedding-001": 768}

    def __init__(self, model_name: str = "models/embedding-001", config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        if genai is None:
            raise ImportError("google-generativeai package required")
        api_key = config.get("api_key") if config else None
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(model=self.model_name, content=text, task_type="retrieval_document")
            embeddings.append(result['embedding'])
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        result = genai.embed_content(model=self.model_name, content=query, task_type="retrieval_query")
        return result['embedding']

    def get_dimension(self) -> int:
        return self.DIMENSION_MAP.get(self.model_name, 768)


class EmbeddingProviderFactory:
    """Factory for creating embedding providers"""

    PROVIDERS = {
        "openai": OpenAIEmbeddingProvider,
        "cohere": CohereEmbeddingProvider,
        "huggingface": HuggingFaceEmbeddingProvider,
        "ollama": OllamaEmbeddingProvider,
        "google": GoogleEmbeddingProvider,
    }

    @classmethod
    def create(cls, provider_type: str, model_name: str, config: Dict[str, Any] = None) -> EmbeddingProviderBase:
        provider_class = cls.PROVIDERS.get(provider_type.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider type: {provider_type}")
        return provider_class(model_name=model_name, config=config)

    @classmethod
    def from_db_config(cls, embedding_provider) -> EmbeddingProviderBase:
        config = {}
        if embedding_provider.config_json:
            config = json.loads(embedding_provider.config_json)
        return cls.create(
            provider_type=embedding_provider.provider_type,
            model_name=embedding_provider.model_name,
            config=config
        )
