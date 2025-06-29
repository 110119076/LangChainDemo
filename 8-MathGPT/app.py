import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Streamlit App
st.set_page_config(page_title="MathGPT",page_icon="🧮")
st.title("🧮 MathGPT")

groq_api_key = st.sidebar.text_input("GROQ API KEY", type="password")

if not groq_api_key:
    st.info("Please add your GROQ API Key to proceed")
    st.stop()

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Initialize tools
wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="A tool to search the web to fetch various information that is required"
)

math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related queries. Only mathematical expression needs to be provided as input"
)

prompt = """
You are an agent expertised in mathematics, solved numerous problems. Logically arrive the solution and provide a detailed 
explanation and display it point wise for the given query
Query: {query}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=prompt
)

# Combine all the tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logical and reasoning questions"
)

# initialize agents
math_agent = initialize_agent(
    tools=[wiki_tool, math_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=False
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant", "content":"Hi, I am a Math Chatbot who can answer all your mathematical queries"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_area("Enter your query:", "I have 5 bananas and 7 grapes. I ate 2 bananas and gave away 3 grapes. Then I bought a dozen of apples and 2 packs of blueberries. Each pack contains 25 berries. How many total count of fruits do I have by the end?")

if st.button("Solution"):
    if question:
        with st.spinner("Loading..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = math_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant", "content":response})
            st.write("Response:")
            st.success(response)

    else:
        st.warning("Please enter the question")