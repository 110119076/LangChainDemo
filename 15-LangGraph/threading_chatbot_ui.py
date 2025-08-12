import streamlit as st
from streaming_chatbot_backend import chatbot, llm
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# Utility functions

def generate_threadIds():
    thread_id = uuid.uuid4()
    return thread_id

def add_threads(thread_id):
    if thread_id not in st.session_state["threads_list"]:
        st.session_state["threads_list"].append(thread_id)

def reset_chat():
    st.session_state["message_history"] = []

def load_chat(thread_id):
    return chatbot.get_state(config={"configurable":{"thread_id":thread_id}}).values["messages"]


# Initial States

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_threadIds()

if "threads_list" not in st.session_state:
    st.session_state["threads_list"] = []

add_threads(st.session_state["thread_id"])

# Sidebar
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    st.session_state["thread_id"] = generate_threadIds()
    add_threads(st.session_state["thread_id"])
    reset_chat()
    
st.sidebar.header("My Conversations")

for thread in st.session_state["threads_list"][::-1]:
    if st.sidebar.button(str(thread)):
        st.session_state["thread_id"] = thread
        messages = load_chat(thread)
        temp_msgs = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_msgs.append({"role":role,"content":msg.content})
        
        st.session_state["message_history"] = temp_msgs

# Display chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask anything")

if user_input:
    # Store & display user input
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Build context from LangGraph memory
    history_state = chatbot.get_state(config={"configurable":{"thread_id":st.session_state["thread_id"]}})
    history_messages = history_state.values.get("messages", [])
    history_messages.append(HumanMessage(content=user_input))

    buffer = []

    # Stream tokens directly from LLM
    with st.chat_message("assistant"):
        def stream_tokens():
            for chunk in llm.stream(history_messages):
                if chunk.content:
                    buffer.append(chunk.content)
                    yield chunk.content  # live streaming to UI

        st.write_stream(stream_tokens())

    # Combine final assistant response
    full_response = "".join(buffer)
    st.session_state["message_history"].append({"role": "assistant", "content": full_response})

    # Push both messages into LangGraph memory
    chatbot.update_state(
        config={"configurable":{"thread_id":st.session_state["thread_id"]}},
        values={
            "messages": [
                HumanMessage(content=user_input),
                AIMessage(content=full_response)  # mark as assistant message
            ]
        }
    )
