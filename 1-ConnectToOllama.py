from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage


'''
Call Local LLMs
Pass a single message or a list of messages
Types of messages - SystemMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
Types of Roles - system, user, assistant, tool
Streaming output
'''


# Call local LLMs - make sure Ollama is running
llm = OllamaLLM(model="llama3.1")
# Regular message
print("--------------TEST-----------------")
response = llm.invoke("Are you here?")
print(response)

# Pass list of messages
'''
Available Roles: system, user, assistant, tool
'''
messages = [
    {"role": "system", "content": "You are a wellness expert."},
    {"role": "user", "content": "How can you serve me?"}
]

print("------------WELLNESS EXPERT-------------")
response = llm.invoke(messages)
print(response)


# Substiture for "role"
'''
Message Types: SystemMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
'''
messages = [
    SystemMessage("You are a comedian"),
    HumanMessage("How are you feeling today?")
]
print("-----------COMEDIAN--------------")
response = llm.invoke(messages)
print(response)



# Streaming output
messages = [
    {"role": "system", "content": "You are a sarcastic teacher."},
    {"role": "user", "content": "How can you serve me?"}
]

response = llm.stream(messages)
print("-----------SARCASTIC TEACHER CHUNKS--------------")
for chunk in response:
    # Added _ just to see how a chunk looks
    print(chunk, end="_")