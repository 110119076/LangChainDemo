from crewai import Task
from tools import yt_tool
from agents import blog_researcher, blog_writer

# Research task
research_task = Task(
    description=(
        "Identify the video {topic}."
        "Get detailed information about the video from the channel videos."
    ),
    expected_output="A comprehensive three paragraphs long report based on the {topic} of video content.",
    tools=[yt_tool],
    agent=blog_researcher
)

# Writing task
writing_task = Task(
    description=(
        "Get the info from the youtube channel from the following {topic}"
    ),
    expected_output="Sumarize the info from the youtube channel video on the {topic} and create the content for the blog",
    tools=[yt_tool],
    agent=blog_writer,
    async_execution=False,
    output_file='new-blog-post.md' # Output Customization
)