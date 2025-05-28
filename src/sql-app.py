from langchain_ollama.llms import OllamaLLM
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# For generating the SQL code
model_low_temp = OllamaLLM(model="gemma3",temperature=0.1)

# For explaining the code
model_high_temp = OllamaLLM(model="gemma3",temperature=0.7)

# Add a memory to the app and set the type of the variables.
class State(TypedDict): 
  messages: Annotated[list, add_messages]
  user_query: str
  sql_query: str
  sql_explanation: str

# Set the types of input and outputs
class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    sql_query: str
    sql_explanation: str

# Create a system message to generate the SQL query
generate_prompt = SystemMessage(
    """You are a helpful data analyst who generates SQL queries for users based on their questions.""")

# Define a function to generate the SQL query
def generate_sql(state: State) -> State:
  user_message = HumanMessage(state["user_query"])
  messages = [generate_prompt, *state["messages"], user_message]
  res = model_low_temp.invoke(messages)
  return {"sql_query": res, "messages": [user_message, res]}

# Create an explain prompt
explain_prompt = SystemMessage(
    """You are a helpful data analyst who explains SQL queries to users.""") 

# Define an function to explain the SQL code
def explain_sql(state: State) -> State:
  messages = [explain_prompt, *state["messages"]]
  res = model_high_temp.invoke(messages)
  return {"sql_explanation": res, "messages": res}

# Initialize the graph
builder = StateGraph(State, input=Input, output=Output)

# Create the nodes
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)

# Add the edges
builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "explain_sql")
builder.add_edge("explain_sql", END)

# Compile the graph
graph = builder.compile()

"""
with open("graph-images/graph_chain.png", "wb") as f:
  f.write(graph.get_graph().draw_mermaid_png())

"""

# Get an input:
input = {"user_query": "What is the total revenue for the year 2024?"}

# Run the model with the input as a stream
for chunk in graph.stream(input):
    print(chunk)