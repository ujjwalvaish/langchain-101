from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage


'''
Creating a dynamic routing system that transforms the AI model into a kind of “router.”
The routing system uses conditional edges to determine which node to run next based on 
LLM’s response. By externalizing decision-making to these conditional edges, we achieve 
a branched and scalable workflow where multiple nodes collaborate, allowing the AI to 
intelligently control the workflow’s direction.
'''
llm = ChatOllama(model = "llama3.1")
class Conversation(MessagesState):
    pass

def tool_calling_llm(state: Conversation):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

'''
We will introduce a routing mechanism to examine the AI’s response and 
decide what to do next. If the response suggests that a tool should be 
called, e.g. , to handle a calculation we direct the workflow to a specialized 
node that executes this tool. Otherwise, we simply produce the AI’s direct 
answer and end the process

1. ToolNode: This is a prebuilt node provided by LangGraph that knows how to execute tools.
2. tools_condition: This is a prebuilt condition that examines the AI’s latest response to 
    determine if it includes a tool call.
'''
# NEW GRAPH
@tool
def multiply(a: float, b:float) -> float:
    '''
    Takes in 2 float numbers, and returns their product
    '''
    return a*b

@tool
def power(a: float, b:float) -> float:
    '''
    Takes in 2 float numbers, and returns a raised to the power b
    '''
    return a ^ b


builder = StateGraph(Conversation)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply, power]))
builder.add_edge(START, "tool_calling_llm")
# Add a conditional edge that uses 'tools_condition'
# If the LLM’s response indicates a tool call, it routes to "tools"
# Otherwise, it routes to END
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)
graph =  builder.compile()


messages = [HumanMessage("What is 8 * 13")]
messages = graph.invoke({"messages": messages})
for m in messages["messages"]:
    # Prints human and AI messages nicely
    m.pretty_print()


messages = [HumanMessage("Tell me a nice high protein low carb breakfast?")]
messages = graph.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

messages = [HumanMessage("What is 8 to the power 3")]
messages = graph.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()