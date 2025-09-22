from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from ingest import retriever
import config

model = OllamaLLM(
    model=config.LLM_MODEL,
    temperature = 0.7,
    verbose=True
)

prompt_template = """
You are an exeprt in answering questions about a pizza restaurant
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}

Ans clearly and consisely.
"""
prompt = ChatPromptTemplate.from_template(
    template=prompt_template
)

chain = prompt | model
while True:
    qn = input("Ask your question (q to quit): ")
    if qn.strip().lower() == "q":
        break

    reviews = retriever.invoke(qn)
    result = chain.invoke({
        "reviews": reviews, 
        "question": qn
    })
    print(result)
