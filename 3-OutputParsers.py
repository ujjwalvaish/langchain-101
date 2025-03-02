from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

llm = OllamaLLM(model="llama3.1")


# Multiple parsers available - Str, CSV, JSON, Pydantic, DateTime, PandasDataframe

'''
Remember LLMs just generate text, we cannot call response["name] or response["age"], here
is where paresers help.

Most parsers work in 2 steps - 
0. Instantiate the output parser
1. Add the parsing instruction in the prompt, to instruct the model that output is
expected in a particular format using .get_format_instructions() method
2. Parse the model output(text) to the requested format - .parse()
'''

# CSV Parser
csv_parser = CommaSeparatedListOutputParser()
csv_template = PromptTemplate(
    template = "Answer the question. \n{format_instructions} \n{question}",
    input_variables = ["question"], # dynamic
    partial_variables = {"format_instructions" : csv_parser.get_format_instructions()} # static
)

csv_prompt = csv_template.invoke("List the 10 most popular car companies in the world without returning \
                            any other text.")
print(csv_prompt)

# Raw response
response = llm.invoke(csv_prompt)
print(type(response))
print(response)
# Parsed response
parsed_response = csv_parser.parse(response)
print(type(parsed_response))
print(response)
for res in parsed_response:
    print(res)
print("----------------------------------")
# Pydantic parser - to extract multiplt "properties" from the LLM output
'''
This parser parses the response into a defined TypedDict class, 
JSON schema or a Pydantic class
'''

# The descriptions should be extremely clear to understand
class Book(BaseModel):
    """ Information about a book."""
    name: str = Field(description="Name of the Book")
    author: str = Field(description="Name of the Author")
    year:int = Field(description="Year the book was published")

pydantic_parser = PydanticOutputParser(pydantic_object=Book)


pydantic_template = PromptTemplate(
    template = "Answer the question. \n{format_instructions} \n{question}",
    input_variables = ["question"],
    partial_variables = {"format_instructions": pydantic_parser.get_format_instructions()}
)


book_prompt = pydantic_template.invoke({"question" : "Tell me the most helpful and insightful Non fiction book \
                                     published in the last 10 years that I should read?"})

print(book_prompt)
response = llm.invoke(book_prompt)
print(response)
print(type(response))
parsed_response = pydantic_parser.parse(response)
print(type(parsed_response))
print(response)
print("----------------------------------")
''' 
For parsing outputs, we first need to instruct the model on how to return
the output, and then parse the returned output.

An easier way is to use the .with_structured_output() method to do this in 1 go.
We don not even have to use the output parsers here, but few models support this.
'''

llm = ChatOllama(model="llama3.1")

structured_llm = llm.with_structured_output(Book)
response = structured_llm.invoke("Tell me the most helpful non fiction book that would be help \
                                me create a strong healthy and productive mind that I should read?")

print(type(response))
print(response)
print(response.name)