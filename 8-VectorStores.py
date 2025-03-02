
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4

'''
Vector stores are special databases designed to store 
vectors, instead of traditional rows and columns.

We can use vector stores to - 
1. Find the most (or k most) relevant text, for user's query
2. Recommend similar items
This approach is crucial in natural language tasks because we care about 
“closeness in meaning,” not identical string matches as in traditional 
querying e.g. SELECT * FROM TABLE where COLUMN_NAME = X 
'''

embeddings = OllamaEmbeddings(model = "llama3.1")
# Simples vector store
vector_store1 = InMemoryVectorStore(embeddings)

# A more sophisticated vector store - performs much better
vector_store2 = Chroma(embedding_function = embeddings)


'''
Now we will add text to our vector stores. This involves - 
1. Fetch documents (text)
2. Generate embeddings
3. Store it in a vector DB

Documents in langchain include both the text and their metadata e.g. source and id
'''
document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"}, # Optional
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 32 degrees.",
    metadata={"source": "news"}, # Optional
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"}, # Optional
    id=3,
)

document_4 = Document(
    page_content="I should wear warm clothes tomorrow since it is expected to snow!",
    metadata={"source": "news"}, # Optional
    id=4,
)

documents = [document_1, document_2, document_3, document_4]

# Next, we create unique Ids for all documents for tracking
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store1.add_documents(documents = documents, ids = uuids)
vector_store2.add_documents(documents = documents, ids = uuids)

# Now we can query the vector store
query = "What clothers should I wear tomorrow?"
result1 = vector_store1.similarity_search(query, k=2)
result2 = vector_store2.similarity_search(query, k=2)

print(result1)
print("------")
print(result2)