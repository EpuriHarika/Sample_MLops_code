from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def build_vector_store(texts):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    return db

def retrieve_context(db, query):
    docs = db.similarity_search(query, k=2)
    return [doc.page_content for doc in docs]