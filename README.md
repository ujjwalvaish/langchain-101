# langchain-101
A step-by-step journey through LangChain with local LLMs, from basic connections to advanced agents.

This repository contains hands-on tutorials for mastering LangChain using local LLMs like Deepseek-R1 and Llama3.1 running on Ollama. Each Python file demonstrates a different concept, progressively building on previous skills to create a complete AI research assistant.

# Prerequisites
Python 3.8+
Ollama installed with Deepseek-R1 and Llama3.1 models
Dependencies from requirements.txt

# Tutorial Files
1. ConnectToOllama.py - Establish your first connection to local LLMs through Ollama
2. PromptTemplate.py - Create dynamic, reusable prompts with variables
3. OutputParsers.py - Transform raw text outputs into structured data
4. RunnablesAndLCEL.py - Build flexible pipelines with LangChain Expression Language
5. Custom_Runnables.py - Extend functionality with your own custom components
6. Tools.py - Add capabilities through external tools like web search and databases
7. Embeddings.py - Implement semantic understanding with vector representations
8. VectorStores.py - Create efficient knowledge bases for semantic search
9. SimpleLangGraphNoLLM.py - Develop sequential workflows that don't require LLM intervention
10. MemoryWithLangGraph.py - Enable conversational context with memory systems
11. RoutingAgentInLangGraph.py - Build autonomous agents that make decisions about tools and workflows

# Getting Started
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Make sure Ollama is running with the required models
4. Start with 1-ConnectToOllama.py and progress through each tutorial

#The Journey
These tutorials take you from a simple LLM chat interface to a sophisticated research assistant that can understand complex queries, access external knowledge, remember context, and make autonomous decisions about how to solve problems - all while maintaining privacy and control with local models.

# Contributing
Feedback and contributions are welcome! Feel free to submit issues or pull requests.

# License
MIT
