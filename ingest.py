"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def ingest_docs():
    """Get documents from web pages."""

    print("ingest_docs")

    embeddings = OpenAIEmbeddings()
    print("embeddings")
    print(embeddings)

    loader = ReadTheDocsLoader("python.langchain.com/en/latest/")
    print("loader")
    print(loader)

    raw_documents = loader.load()
    print("raw_documents")
    print(raw_documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    documents = text_splitter.split_documents(raw_documents)
    print(documents)

    vectorstore = FAISS.from_documents(documents, embeddings)
    print(vectorstore)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        print('should be saving the pkl file')
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
