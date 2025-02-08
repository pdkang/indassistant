import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from ind_312_checklist import load_pdf, init_vector_store, create_rag_chain

def main():
    st.title("Appian IND ApplicationAssistant")
    st.markdown("Chat about Investigational New Drug Applications")
    
    # Add clear history button at the top
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load IND-312 PDF once
    if "rag_chain" not in st.session_state:
        with st.spinner("Initializing knowledge base..."):
            documents = load_pdf("IND-312.pdf")
            vectorstore = init_vector_store(documents)
            st.session_state.rag_chain = create_rag_chain(vectorstore.as_retriever())

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input with persistent history
    if prompt := st.chat_input("Ask about IND requirements"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display bot response
        with st.chat_message("assistant"):
            response = st.session_state.rag_chain.invoke({"question": prompt})
            st.markdown(response["response"])
        
        # Add bot response to history
        st.session_state.messages.append({"role": "assistant", "content": response["response"]})

if __name__ == "__main__":
    main() 