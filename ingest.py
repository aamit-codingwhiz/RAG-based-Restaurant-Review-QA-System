from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import pandas as pd
import config
import os


vectore_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=config.DB_PATH,
    embedding_function=OllamaEmbeddings(
        model=config.EMBEDDING_MODEL
    )
)

df = pd.read_csv(filepath_or_buffer=config.CSV_FILE)
add_documents = not os.path.exists(path=config.DB_PATH)

if add_documents:
    print("Document creation - start")
    documents = []
    ids = []

    for i, row in df.iterrows():
        doc = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={
                "Rating": row["Rating"],
                "Date": row["Date"]
            },
            id=str(i)
        )

        documents.append(doc)
        ids.append(str(i))

    vectore_store.add_documents(
        documents=documents,
        ids=ids
    )

retriever = vectore_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)