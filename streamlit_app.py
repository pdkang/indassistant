import os
import json
import streamlit as st
from ind_checklist_stlit import load_preprocessed_data, init_vector_store, create_rag_chain

# Prevent Streamlit from auto-reloading on file changes
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Define the preprocessed file path
PREPROCESSED_FILE = "preprocessed_docs.json"

def main():
    st.title("Appian IND Application Assistant")
    st.markdown("Chat about Investigational New Drug Applications")

    # Button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load preprocessed data and initialize the RAG chain
    if "rag_chain" not in st.session_state:
        if not os.path.exists(PREPROCESSED_FILE):
            st.error(f"❌ Preprocessed file '{PREPROCESSED_FILE}' not found. Please run preprocessing first.")
            return  # Stop execution if preprocessed data is missing

        with st.spinner("🔄 Initializing knowledge base..."):
            documents = load_preprocessed_data(PREPROCESSED_FILE)
            vectorstore = init_vector_store(documents)
            st.session_state.rag_chain = create_rag_chain(vectorstore.as_retriever())

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input and response handling
    if prompt := st.chat_input("Ask about IND requirements"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using the RAG chain
        with st.chat_message("assistant"):
            response = st.session_state.rag_chain.invoke({"question": prompt})
            st.markdown(response["response"])

        # Store bot response in chat history
        st.session_state.messages.append({"role": "assistant", "content": response["response"]})

if __name__ == "__main__":
    main()

