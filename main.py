from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ingest import retriever
import config
 

template = """
You answer questions about restaurant reviews.

Use only the context below.
Summarize patterns across multiple reviews when possible.

Do not invent information.
If the context does not contain enough information, say: "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=OllamaLLM(
        model=config.LLM_MODEL,
        temperature = 0.1,
        verbose=True
    ), 
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

while True:
    query = input("Ask your question (q to quit): ").strip()
    if query.lower() == "q":
        break

    result = qa.invoke(query)
    
    print(f"Bot: {result["result"]}")
    print("-"*60)
