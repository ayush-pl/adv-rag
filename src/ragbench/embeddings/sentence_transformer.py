"""Embedder backed by Sentence Transformers. Requires the `ml` extra."""

import numpy as np

from ragbench.embeddings.base import Embedder, EmbedderConfig, TruncationStats


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, config: EmbedderConfig, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.config = config
        self._model = SentenceTransformer(config.model, revision=config.revision, device=device)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    @property
    def device(self) -> str:
        return str(self._model.device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.config.document_prefix + text for text in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.config.query_prefix + text for text in texts])

    def truncation_stats(self, texts: list[str]) -> TruncationStats:
        max_tokens = self._model.max_seq_length
        prefixed = [self.config.document_prefix + text for text in texts]
        encoded = self._model.tokenizer(prefixed, add_special_tokens=True, truncation=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        truncated = sum(length > max_tokens for length in lengths)
        return TruncationStats(
            max_tokens=max_tokens,
            texts=len(texts),
            truncated_texts=truncated,
            truncated_rate=truncated / len(texts) if texts else 0.0,
            longest_text_tokens=max(lengths, default=0),
        )

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)
