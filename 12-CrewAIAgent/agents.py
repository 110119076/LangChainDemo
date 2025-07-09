from crewai import Agent
from tools import yt_tool
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model_name = "gemma2-9b-it"

llm = ChatGroq(model=groq_model_name, groq_api_key=groq_api_key)

# Blog Content Researcher
blog_researcher = Agent(
    role="Blog researcher from youtube videos",
    goal="Get the relevant video transcription for the topic {topic} from the provided youtube channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos related to general knowledge stuff and provide suggestions"
    ),
    tools=[yt_tool],
    allow_delegation=True
)

# Blog Content Writer
blog_writer = Agent(
    role="Blog writer from the given content",
    goal="Narrate and summarize the {topic} from the youtube video",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex concepts, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner"
    ),
    tools=[yt_tool],
    allow_delegation=False
)