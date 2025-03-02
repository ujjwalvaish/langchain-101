from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage


'''
Calling Local LLMs
Passing a single message or a list of messages
Types of messages - SystemMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
Types of Roles - system, user, assistant, tool
Streaming output
'''


# Calling local LLMs - make sure Ollama is running
llm = OllamaLLM(model="llama3.1")
# Regular chat
response = llm.invoke("Are you here?")
print(response)

# List of messages
'''
Available Roles: system, user, assistant, tool
'''
messages = [
    {"role": "system", "content": "You are an decorated and disciplined army officer."},
    {"role": "user", "content": "How can you serve me?"}
]

response = llm.invoke(messages)
print(response)
print("-------------------------------")

# Another way to define messages through 
'''
Message Types: SystemMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
'''
messages = [
    SystemMessage("You are a comedian"),
    HumanMessage("How are you feeling today?")
]

response = llm.invoke(messages)
print(response)
print("-------------------------------")


# Streaming output
messages = [
    {"role": "system", "content": "You are a sarcastic teacher."},
    {"role": "user", "content": "How can you serve me?"}
]

response = llm.stream(messages)
for chunk in response:
    # Added _ just to see how a chunk looks
    print(chunk, end="_")