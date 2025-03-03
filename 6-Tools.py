# from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.tools import tool

# llm = OllamaLLM(model = "llama3.1")
# Mistral returns a better response for this example
llm = ChatOllama(model = "mistral:7b") # NEW

'''
In LangChain, a "tool" is more than just a Python function - 
it's a function wrapped with metadata that helps the AI understand its purpose, 
usage, and required inputs. This makes it easier for the AI to use and interact 
with the tool just like following an instruction manual.


While we could manually create tools by subclassing BaseTool, 
LangChain provides a much easier way: the @tool decorator.
'''
@tool
def sales_after_discount(orig_price: float, discount_percentage: float ) -> float:
    """
    Takes the original price and discount percentage, and returns the final
    price after discount

    Args:
        discount_percentage(float) : Percentage discount on the item e.g. A value of 20% discount on 100 = a discount of 20
        orig_price(float) : Price before applying discounts

    Returns:
        Final Price
    """
    if discount_percentage > 100:
        raise ValueError("Discount percentage must be between 0 and 100")
    return orig_price * (1 - discount_percentage/100)

# Check tool parameters
# print(sales_after_discount.name) # Name of tool aka function
# print(sales_after_discount.description)
# print(sales_after_discount.args)

# Tools are also runnables
# final_price = sales_after_discount.invoke({ "orig_price": 100, "discount_percentage": 25})
# print(final_price)

'''
Tool Calling - combining these tools with LLMs
1. Model binding
2. Model when prompted makes a decision if it wants to use any of the bound tools
3. If a tool needs to be used, we can directly pass the value of "tool_calls[0]['args'] to the tool 
'''

llm_with_tools = llm.bind_tools([sales_after_discount])

print("TOOL IS NOT NEEDED")
response = llm_with_tools.invoke("How can you help me?")
print(f"Content - {response.content}") # should have content
print(f"tool_calls = {response.tool_calls}") # should be empty

print("TOOL IS NEEDED")
response = llm_with_tools.invoke("What would the final price for a member, if the original item  \
                           is priced at 200 and there is a 15%  discount?")
print(f"Content - {response.content}") # should be empty
print(f"tool_calls = {response.tool_calls}") # dictionary with everything needed to call a tool
args = response.tool_calls[0]['args']
print(sales_after_discount.invoke(args)) # manuall calling tools