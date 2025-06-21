import streamlit as st
from RAG_Pipeline import rag_query

st.set_page_config(page_title="Quote RAG Generator", layout="centered")
st.title("RAG Quote Generator with FLAN-T5")

query = st.text_input("Enter your question (e.g., 'quotes about resilience by women')")

st.markdown(
    "**Top K Quotes:** Number of most relevant quotes to retrieve.\n\n"
    "Searches the FAISS index for the top K most similar quote embeddings.\n"
    "These retrieved quotes are used as context for the language model to generate an answer."
)

top_k = st.slider("Top K quotes", 1, 10, value=5)

if st.button("Generate Answer") and query.strip():
    with st.spinner("Thinking..."):
        summary, retrieved_quotes = rag_query(query, top_k=top_k)

    # Build structured JSON-style response
    response_json = {
        "summary": summary,
        "retrieved_quotes": retrieved_quotes
    }

    st.subheader("Output")
    st.json(response_json)
