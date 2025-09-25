from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ingest import retriever
import config
 

template = """
You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, make your best guess based on the information provided.

Context:
{context}

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=OllamaLLM(
        model=config.LLM_MODEL,
        temperature = 0.3,
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
