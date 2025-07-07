from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key, streaming=True)

prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

# Stream output from the chain
for chunk in chain.stream({"input": "What is Machine Learning?"}):
    print(chunk, end="")