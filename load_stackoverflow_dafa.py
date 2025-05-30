import csv
import dotenv
import logging
from os import environ
from qdrant_client import QdrantClient, models
from tqdm import tqdm

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

logging.info('Startng...')
logging.info('Loading environment variables from .env file')
dotenv.load_dotenv()

# Initialize Qdrant client and collection
logging.info('Initializing Qdrant client and collection...')
client = QdrantClient(environ['VECTOR_STORE_URL'])

# Create the collection if it does not exist
if not client.collection_exists(environ['VECTOR_STORE_COLLECTION']):
    logging.info(f"Creating collection '{environ['VECTOR_STORE_COLLECTION']}'...")
    client.create_collection(
        collection_name=environ['VECTOR_STORE_COLLECTION'],
        vectors_config={
            environ['DENSE_VECTOR_NAME']: models.VectorParams(
                size=client.get_embedding_size(environ['DENSE_MODEL_NAME']), 
                distance=models.Distance.COSINE
            )
        },  # size and distance are model dependent
        sparse_vectors_config={environ['SPARSE_VECTOR_NAME']: models.SparseVectorParams()},
    )

# Read questions into a dict: {Id: (Title, Body)}
questions = {}
logging.info('Reading questions from CSV file...')
with open(environ['QUESTIONS_PATH'], newline='', encoding='latin-1') as qfile:
    reader = csv.DictReader(qfile)
    for row in reader:
        # Stop reading if row limit is reached
        if reader.line_num > int(environ['ROW_LIMIT']):
            print(f"Reached row limit: {environ['ROW_LIMIT']}. Stopping reading questions.")
            break
        questions[row['Id']] = (row['Title'], row['Body'])

# Prepare documents and metadata for upload
documents = []
metadata = []
logging.info('Reading answers from CSV file...')
with open(environ['ANSWERS_PATH'], newline='', encoding='latin-1') as afile:
    areader = csv.DictReader(afile)
    for row in areader:
        parent_id = row['ParentId']
        answer_body = row['Body']
        # Only process answers that have a matching question
        if parent_id in questions:
            question_title, question = questions[parent_id]
            qa_pair = {'title': question_title,'question': question,'answer': answer_body}
            # Create dense and sparse document representations
            dense_document = models.Document(text=question, model=environ['DENSE_MODEL_NAME'])
            sparse_document = models.Document(text=question, model=environ['SPARSE_MODEL_NAME'])
            documents.append(
                {
                    environ['DENSE_VECTOR_NAME']: dense_document,
                    environ['SPARSE_VECTOR_NAME']: sparse_document,
                }
            )
            metadata.append(qa_pair)

# Upload documents and metadata to Qdrant
logging.info(f'Loading {len(documents)} documents from answers.')
client.upload_collection(
    collection_name=environ['VECTOR_STORE_COLLECTION'],
    vectors=documents,
    payload=metadata,
    ids=tqdm(range(len(documents))),
)