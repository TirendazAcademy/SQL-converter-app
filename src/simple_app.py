from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="gemma3")

template = """Question: {question}
Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

response = chain.invoke({"question": "tell me a joke about a computer?"})

print(response)