from chromadb.api.types import D, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions


class AinekoEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: D) -> Embeddings:
        # embed the documents somehow
        embeddings = embedding_functions.DefaultEmbeddingFunction()(input)
        return embeddings

