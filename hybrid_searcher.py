from qdrant_client import QdrantClient, models

class HybridSearcher:
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"
    
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url="http://localhost:6333")

    def search(self, text: str):
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
              query=models.FusionQuery(
                fusion=models.Fusion.RRF  # we are using reciprocal rank fusion here
            ),
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=text, model=self.DENSE_MODEL),
                    using=self.DENSE_VECTOR_NAME
                ),
                models.Prefetch(
                    query=models.Document(text=text, model=self.SPARSE_MODEL),
                    using=self.SPARSE_VECTOR_NAME
                ),
            ],
            query_filter=None,  # If you don't want any filters for now
            limit=5,  # 5 the closest results
        ).points
        # `search_result` contains models.QueryResponse structure
        # We can access list of scored points with the corresponding similarity scores,
        # vectors (if `with_vectors` was set to `True`), and payload via `points` attribute.

        # Select and return metadata
        metadata = [point.payload for point in search_result]
        return metadata