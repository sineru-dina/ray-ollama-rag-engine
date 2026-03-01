import streamlit as st
import requests

# UI Configuration
st.set_page_config(page_title="Chronos RAG Explorer", layout="wide")

# Sidebar Configuration for A/B Testing
with st.sidebar:
    st.header("Pipeline Controls")
    use_rag = st.toggle("Enable RAG (ChromaDB Retrieval)", value=True)
    st.markdown("---")
    if use_rag:
        st.success(
            "🟢 **RAG Active:** Grounding responses in downloaded research papers."
        )
    else:
        st.warning(
            "🔴 **RAG Disabled:** Relying strictly on Mistral's pre-trained weights."
        )

st.title("Ollama (Mistral Model) Chat Box: RAG Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about time series models..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Send HTTP POST request with our new flag
            response = requests.post(
                "http://127.0.0.1:8000/rag",
                json={
                    "query": prompt,
                    "use_rag": use_rag,  # Pass the toggle state to Ray Serve
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "No response generated.")
                sources = data.get("sources", [])

                # Format output differently based on the mode
                if use_rag:
                    full_response = (
                        f"{answer}\n\n**Sources Extracted (Pages):** {sources}"
                    )
                else:
                    full_response = (
                        f"{answer}\n\n*(Warning: Generated without verified context)*"
                    )

                message_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                st.error(f"Backend API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the Ray Serve API.")
