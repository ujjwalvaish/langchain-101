from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

'''
Use memory in state as a way to remmember the context of conversations to
pass on to the LLM.
'''

llm = ChatOllama(model = "llama3.1")

# Simple Memory
# Message State is a special kind of LangGraph state for storing messages
class Conversation(MessagesState):
    pass

@tool
def multiply(a: float, b:float) -> float:
    '''
    Takes in 2 float numbers, and returns their product
    '''
    return a * b

llm_with_tool = llm.bind_tools([multiply])

'''
Create a node that 
1) Reads the entire state(conversation)
2) Invokes LLM with tools to return the response
3) Adds the returned response to the state
'''
def llm_with_memory(state: Conversation):
    response = llm_with_tool.invoke(state["messages"])
    return {"messages" : [response]}

builder = StateGraph(Conversation)
builder.add_node("call_llm_with_memory", llm_with_memory)
builder.add_edge(START, "call_llm_with_memory")
builder.add_edge("call_llm_with_memory", END)
graph = builder.compile()


messages = graph.invoke({"messages" : HumanMessage(content="What is 17 * 5?")})
messages_2 = graph.invoke({"messages" : HumanMessage(content="Hi, how are you?")})

# Atleast uses tool correctly
for m in messages["messages"]:
    if m.content:
        print((m.content))
    else:
        print(multiply.invoke(m.tool_calls[0]["args"]))