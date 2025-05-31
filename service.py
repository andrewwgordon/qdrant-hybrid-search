from fastapi import FastAPI
from os import environ
import dotenv

dotenv.load_dotenv()

# The file where HybridSearcher is stored
from hybrid_searcher import HybridSearcher

app = FastAPI()

# Create a neural searcher instance
hybrid_searcher = HybridSearcher(collection_name=environ["VECTOR_STORE_COLLECTION"])


@app.get("/api/search")
def search_startup(q: str):
    return {"result": hybrid_searcher.search(text=q)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
