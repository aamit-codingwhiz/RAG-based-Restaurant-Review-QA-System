from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import pandas as pd
import config


vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=config.DB_PATH,
    embedding_function=OllamaEmbeddings(
        model=config.EMBEDDING_MODEL
    )
)

# Check the actual database, not just whether the folder exists
if vector_store._collection.count() == 0:
    print("Document creation - start")

    df = pd.read_csv(config.CSV_FILE)

    documents = []
    ids = []

    for i, row in df.iterrows():
        doc = Document(
            page_content=f"{row['Title']} {row['Review']}",
            metadata={
                "Rating": row["Rating"],
                "Date": row["Date"]
            }
        )

        documents.append(doc)
        ids.append(str(i))

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    print(f"Added {len(documents)} reviews to Chroma.")

else:
    print(
        f"Loaded {vector_store._collection.count()} reviews from Chroma."
    )


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
