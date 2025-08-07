import streamlit as st
from basic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id":"1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask anything")

if user_input:
    st.session_state["message_history"].append({"role":"user","content":user_input})

    with st.chat_message("user"):
        st.text(user_input)
    
    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    chatbot_msg = response["messages"][-1].content

    st.session_state["message_history"].append({"role":"assistant","content":chatbot_msg})

    with st.chat_message("assistant"):
        st.text(chatbot_msg)