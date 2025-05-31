import csv
import dotenv
import logging
import re
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
        if (len(questions)) > int(environ['QUESTIONS_LIMIT']):
            logging.info(f'Reached questions limit of {environ["QUESTIONS_LIMIT"]}. Stopping read.')
            break
        # Strip HTML tags from the question body
        question_body = re.sub(r'<[^>]+>', '', row['Body'])
        questions[row['Id']] = (row['Title'], question_body)
# Log the number of questions loaded
logging.info(f'Loaded {len(questions)} questions from CSV file.')

# Build a dictionary mapping question IDs to their answers
logging.info('Building question-to-answers mapping...')
question_answers = {}
with open(environ['ANSWERS_PATH'], newline='', encoding='latin-1') as afile:
    areader = csv.DictReader(afile)
    for row in areader:
        parent_id = row['ParentId']
        # Strip HTML tags from the question body
        answer_body = re.sub(r'<[^>]+>', '', row['Body'])
        if parent_id in questions:
            if parent_id not in question_answers:
                question_answers[parent_id] = []
            question_answers[parent_id].append(answer_body)

# Prepare documents and metadata for upload
documents = []
metadata = []
# Now iterate over each question and build the documents/metadata
for qid, (question_title, question) in questions.items():
    answers = question_answers.get(qid, [])
    qa_pair = {'title': question_title, 'question': question, 'answers': answers}
    text_content = question_title + '. ' + question
    dense_document = models.Document(text=text_content, model=environ['DENSE_MODEL_NAME'])
    sparse_document = models.Document(text=text_content, model=environ['SPARSE_MODEL_NAME'])
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