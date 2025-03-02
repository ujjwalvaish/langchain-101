from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda


llm = OllamaLLM(model="llama3.1")

'''
Here are the concise definitions and examples:

# Runnable
A callable unit of work that encapsulates tasks like LLM calls, database queries, or API calls.
Each runable must implement methods such as invoke(), batch(), and stream().

# LCEL (LangChain Expression Language)
A declarative approach to simplify chaining operations.
Example: Used to manage complex workflows, such as chaining PromptTemplate → ChatModel → OutputParser.
Langchain has overloaded the pipe "|" operator. This enables runnables to be structured in a familiar way.
'''

# Get sentiment from a  product review
sentiment_prompt_template = PromptTemplate(
    input_variables=["feedback"],
    template="Determine the sentiment of this feedback and reply in one word as either \
        'Positive', 'Neutral', or 'Negative':\n\n{feedback}"
)

bad_feedback = "The fan was horrible."
good_feedback = "The fan was outstanding."
neutral_feedback = "There is nothing to write. It works."

# With StrOutputParser
chain = sentiment_prompt_template | llm | StrOutputParser() # LCEL
print(chain.invoke({"feedback": bad_feedback}))
print(chain.invoke({"feedback": good_feedback}))
print(chain.invoke({"feedback": neutral_feedback}))

# Without StrOutputParser
print("WITHOUT OUTPUT PARSER")
chain2 = sentiment_prompt_template | llm
print(chain2.invoke({"feedback": bad_feedback}))
print(chain2.invoke({"feedback": good_feedback}))
print(chain2.invoke({"feedback": neutral_feedback}))

# Creating a bigger chain
# Summarize the key info from the review and also generate a sentiment

'''
Raw feedback -> Parsed feedback (Only key info) -> Summary of feedback -> Sentiment 
'''
parse_template = PromptTemplate(
    input_variables = ["raw_feedback"],
    template = "Gather the key information from this user feedback. /n {raw_feedback} "
)

summary_template = PromptTemplate(
    input_variables = ["parsed_feedback"],
    template = "Create a one sentence summary for this feedback. /n {parsed_feedback} "
)

# Now we need to pass the output of one runnable to next, using the same argument names
'''
RunnableLambda: Turn custom functions into runnables
'''
# Just creates a dictionary from the returned string, to have the same argument name
format_parsed_output = RunnableLambda(lambda output: {"parsed_feedback" : output})
format_summary_output = RunnableLambda(lambda output: {"feedback" : output})


user_feedback = "The delivery was late, and the product was damaged when it arrived. \
    However, the customer support team was very helpful in resolving the issue quickly."

# Nested chain
# chain = sentiment_prompt_template | llm | StrOutputParser() as defined above
chain3 = parse_template | llm | format_parsed_output | summary_template | llm | format_summary_output | chain
chain4 = parse_template | llm | format_parsed_output | summary_template | llm | format_summary_output | sentiment_prompt_template | llm | StrOutputParser()
print(chain3.invoke({"raw_feedback": user_feedback}))