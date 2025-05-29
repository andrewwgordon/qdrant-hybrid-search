# Import client library
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import json

client = QdrantClient(url="http://localhost:6333")

dense_vector_name = "dense"
sparse_vector_name = "sparse"
dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sparse_model_name = "prithivida/Splade_PP_en_v1"
if not client.collection_exists("startups"):
    client.create_collection(
        collection_name="startups",
        vectors_config={
            dense_vector_name: models.VectorParams(
                size=client.get_embedding_size(dense_model_name), 
                distance=models.Distance.COSINE
            )
        },  # size and distance are model dependent
        sparse_vectors_config={sparse_vector_name: models.SparseVectorParams()},
    )

payload_path = "./data/startups_demo.json"
documents = []
metadata = []

with open(payload_path) as fd:
    for line in fd:
        obj = json.loads(line)
        description = obj["description"]
        dense_document = models.Document(text=description, model=dense_model_name)
        sparse_document = models.Document(text=description, model=sparse_model_name)
        documents.append(
            {
                dense_vector_name: dense_document,
                sparse_vector_name: sparse_document,
            }
        )
        metadata.append(obj)

client.upload_collection(
    collection_name="startups",
    vectors=documents,
    payload=metadata,
    ids=tqdm(range(len(documents))),
)