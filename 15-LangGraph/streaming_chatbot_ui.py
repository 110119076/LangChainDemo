import streamlit as st
from streaming_chatbot_backend import chatbot, llm
from langchain_core.messages import HumanMessage, AIMessage

config = {"configurable": {"thread_id": "1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

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
    history_state = chatbot.get_state(config=config)
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
        config=config,
        values={
            "messages": [
                HumanMessage(content=user_input),
                AIMessage(content=full_response)  # mark as assistant message
            ]
        }
    )
