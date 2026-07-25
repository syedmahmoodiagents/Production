import uid
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOllama(model="llama3.2", temperature=0)

BAD_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "developer prompt"
]

question = input("Question: ")

# Input Guardrail
for pattern in BAD_PATTERNS:
    if pattern in question.lower():
        print("Prompt Injection Detected")
        exit()

# LLM
response = llm.invoke(question).content

# Output Guardrail
if "password" in response.lower():
    print("Unsafe Output")
else:
    print(response)