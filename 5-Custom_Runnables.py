from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda


llm = OllamaLLM(model="llama3.1")

user_feedback_neg = "The delivery was late, and the product was damaged when it arrived. \
    However, the customer support team was very helpful in resolving the issue quickly."

user_feedback_pos = "The delivery was early, and the product was perfect when it arrived. \
    Also, the customer support team was very helpful in keeping me up to date."

# Raw feedback -> Parsed feedback -> Summary -> Sentiment
parse_template = PromptTemplate(
    input_variables = ["raw_feedback"],
    template = "Gather the key information from this user feedback. /n {raw_feedback} "
)

summary_template = PromptTemplate(
    input_variables = ["parsed_feedback"],
    template = "Create a one sentence summary for this feedback. /n {parsed_feedback} "
)

sentiment_template = PromptTemplate(
    input_variables=["feedback"],
    template="Determine the sentiment of this feedback and reply in one word only as either \
        'Positive', 'Neutral', or 'Negative':\n\n{feedback}"
)


# Take different actions depending on the sentiment of feedback
positive_response_template = PromptTemplate(
    input_variables=["feedback"],
    template="Given the feedback, draft a thank you message for the user and request them to leave a positive rating on our webpage:\n\n{feedback}"
)
# Neutral feedback
neutral_response_template = PromptTemplate(
    input_variables=["feedback"],
    template="Given the feedback, draft a message for the user and request them provide more details about their concern:\n\n{feedback}"
)
negative_response_template = PromptTemplate(
    input_variables=["feedback"],
    template="Given the feedback, draft an apology message for the user and mention that their concern has been forwarded to the relevant department:\n\n{feedback}"
)

positive_chain  = positive_response_template | llm | StrOutputParser()
negative_chain  = negative_response_template | llm | StrOutputParser()
neutral_chain  = neutral_response_template | llm | StrOutputParser()

'''
Conditional Routing using custom functions
'''
# Manual routing 
def route(info):
    # print(f"Info: {info}")
    # {feedback : summary, sentiment: sentiment}
    if "positive" in info["sentiment"].lower():
        return positive_chain
    elif "negative" in info["sentiment"].lower():
        return negative_chain
    else: 
        return neutral_chain
    
# raw_feedback -> summary -> sentiment
format_parsed_feedback = RunnableLambda(lambda output: {"parsed_feedback": output})
summary_chain = parse_template | llm | format_parsed_feedback | summary_template | llm | StrOutputParser()
sentiment_chain = sentiment_template | llm | StrOutputParser()

# Manually pass differnet chains to see intermediate outputs
# Run individual chains first
summary = summary_chain.invoke({"raw_feedback": user_feedback_neg})
sentiment = sentiment_chain.invoke({"feedback": summary})

# print("The summary of the user's message is:", summary)
# print("The sentiment was classifed as:", sentiment)
# print("-----------------------------------------------------")
# full_chain = {"feedback": lambda x: x['feedback'], 'sentiment' : lambda x : x['sentiment']} | RunnableLambda(route)
full_chain = RunnableLambda(route)
print(full_chain.invoke({'feedback': summary, 'sentiment': sentiment}))


