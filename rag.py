import glob
import os

import chromadb
import requests
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()  # loads EMBEDDING_* vars from .env into os.environ; safe to call again if already loaded

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_documents"

CHUNK_SIZE = 1000   # characters per chunk
CHUNK_OVERLAP = 200  # characters shared between consecutive chunks


class RemoteEmbeddingFunction(EmbeddingFunction):
    # calls an external embeddings endpoint, batching requests and reusing one session
    def __init__(self, endpoint, model_name, model_version, api_key):
        self.endpoint = endpoint
        self.model_name = model_name
        self.model_version = model_version
        self.session = requests.Session()
        self.session.headers.update({
            "api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def __call__(self, input: Documents) -> Embeddings:
        response = self.session.post(
            self.endpoint,
            params={"api-version": self.model_version},
            json={"input": input, "model": self.model_name},
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(
                f"Embedding request failed: HTTP {response.status_code} from {response.url}\n"
                f"Response body: {response.text[:500]}"
            )
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                f"Embedding endpoint returned non-JSON response (HTTP {response.status_code}) from {response.url}\n"
                f"Response body: {response.text[:500]}"
            )
        return [item["embedding"] for item in data["data"]]


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embedding_fn = RemoteEmbeddingFunction(
        endpoint=os.environ["EMBEDDING_ENDPOINT"],
        model_name=os.environ["EMBEDDING_MODEL_NAME"],
        model_version=os.environ["EMBEDDING_MODEL_VERSION"],
        api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ["AZURE_API_KEY"],
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def ingest_pdfs(folder_path, collection):
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_paths:
        return 0, 0

    total_chunks = 0
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        if not chunks:
            continue

        ids = [f"{filename}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
        # remove any previous chunks for this file so re-ingesting doesn't duplicate them
        collection.delete(where={"source": filename})
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        total_chunks += len(chunks)

    return len(pdf_paths), total_chunks


def query_documents(collection, query_text, n_results=4):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return "No relevant documents found."

    return "\n\n".join(
        f"[{meta.get('source', 'unknown')}, chunk {meta.get('chunk', '?')}]\n{doc}"
        for doc, meta in zip(documents, metadatas)
    )
