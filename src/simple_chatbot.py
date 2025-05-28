from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage

class State(TypedDict):
  messages: Annotated[list, add_messages]

model = OllamaLLM(model="gemma3")

def chatbot(state: State):
  answer = model.invoke(state["messages"])
  return {"messages": [answer]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

"""
with open("src/my-graph.png", "wb") as f:
	f.write(graph.get_graph().draw_mermaid_png())

"""

input = {"messages": [HumanMessage('hi!')]}

for chunk in graph.stream(input):
    print(chunk)