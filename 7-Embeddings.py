
from langchain_ollama import OllamaEmbeddings
'''
Embeddings are numerical representation of text.
text can be a word, a pragraph or a book. Embeddings
capture the meaning of the text i.e. words or sentences 
having similar meanings will be close to each other in the 
vector space of embeddings. Embeddings are usually dense
multi dimensional vectors.

This is a very powerful language modeling technique as 
machines do not speak numbers., so embeddings bridge this
gap very nicely.
'''

embeddings = OllamaEmbeddings(model = "llama3.1")
vec1 = embeddings.embed_query("Hey, what's up!")
print(type(vec1)) # list
print(f"Length  = {len(vec1)}") # 4096 for this embedding model
print(vec1[:100])