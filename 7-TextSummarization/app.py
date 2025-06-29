import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit App
st.set_page_config(page_icon="🦜", page_title="Summarize text from YT & Websites")
st.title("🦜 Langchain Text Summarizer from Youtube & Websites")
st.subheader("Summarizer")

# Groq API Key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

url = st.text_input("URL", label_visibility="collapsed")

# Gemma LLM
llm = ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

prompt_template = """
Provide the summary for the following text in 300 words:
Content: {text}
"""

prompt = PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize"):
    # Validate all the inputs
    if not groq_api_key.strip() or not url.strip():
        st.error("Please enter the required fields")
    elif not validators.url(url):
        st.error("Unsupported URL! Please try with another URL")
    else:
        try:
            with st.spinner("Loading..."):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False,headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()

                # Chain for Summarization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)
                st.success(summary)
        
        except Exception as e:
            st.exception(f"Exception: {e}")