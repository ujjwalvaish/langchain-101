from typing_extensions import TypedDict
from typing import Literal
import random
from langgraph.graph import StateGraph, START, END

'''
This file just illustrates the components of a LangGraph,
without interacting with an LLM. 

LangGraph is a specialized library designed to build stateful systems, 
often involving multi agent non linear workflows.

Agents rely on LLMs to evaluate real time inputs and DECIDE the next course of action
'''


# State is the shared memory - can also inherit Pydantic to define addtional validations
class State(TypedDict):
    graph_state : str
    toss_res : float

# Nodes are simple python functions that take the state, do something with it and return an updated state
def node1(state):
    print("Let's do the coin toss.\n")
    toss_res = random.random()
    return { 
        "graph_state" :"Coin is in the air.\n" ,
        "toss_res": toss_res
        }

def node2(state):
    print("India win the toss.\n")
    return { "graph_state" : state["graph_state"] + "India wins the toss.\n" }

def node3(state):
    print("Aus win the toss.\n")
    return { "graph_state" : state["graph_state"] + "Aus wins the toss.\n" }

# Edges are decision makers that evaluate which node to go to next
# aka which action to take next?
# It can be sometimes straight forward e.g. go to B alsways, or sometimes conditional
def decide_toss(state) -> Literal["node2", "node3"]:
    toss_res = state["toss_res"]
    print(f"Toss result is {toss_res}")
    if toss_res <= 0.5:
        return "node2"
    return "node3"

# Create an instance of our State Graph
builder = StateGraph(State)

# Add all nodes
builder.add_node("node1", node1) # can provide a name to the nodes
builder.add_node("node2", node2)
builder.add_node("node3", node3)

# Add all edges
# Types of edges - normal | conditional | entry | exit
builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", decide_toss)
builder.add_edge("node2", END)
builder.add_edge("node3", END)

# Compile the graph for sanity checks
graph =  builder.compile()

res = graph.invoke({"graph_state": ""})
print(res)