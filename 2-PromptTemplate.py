from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="llama3.1")

# Prompt template - instructions, few shot example(s), placeholders for user inputs
instruction  = "Create an invitation email to the recipient that is {recipient_name}\
  for an event that is {event_type}\
  in a style that is {style}\
  Mention the event location that is {event_location}.\
  and event date that is {event_date}.\
  Also write few sentences about the event description that is {event_description}."

invitation_template = PromptTemplate.from_template(instruction)
# invitation_template is an object containing input_variables, template, 
# for key in invitation_template:
#     print(key)

event_info = {
    "recipient_name" : "SkyWalker", 
    "event_type" : "Fancy Dress Competition",
    "style" : "rude",
    "event_location" : "In the ocean",
    "event_date" : "Jan 30",
    "event_description": "joyful but competetive envoronment"
}

print(f"Invitation template: {invitation_template}")
print("-------------------------------")

# Prompt template can directly be invoked
invite = invitation_template.invoke(event_info)
print(invite)
print("-------------------------------")

response = llm.invoke(invite)
print(response)