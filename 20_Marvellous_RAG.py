import streamlit as st
from sentence_transformers import SentenceTransformer, util
import chromadb
from ollama import chat

# App Title and Description
st.title("Marvellous GenAI : RAG Application")
st.write("Search and chat with your local knowledge base using AI.")

# Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path="./MarvellousDB")
collection = client.get_collection(name="Marvellous_knowledge")

# Accept question from user
query = st.text_input("Enter your Question:")

# Button to trigger the RAG pipeline
if st.button("Generate Answer"):
    if query:
        # Generate query embedding
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Retrieve top 2 results from ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=2
        )

        # Create context from retrieved documents
        documents = results["documents"][0]
        context = "\n".join(documents)

        # Display retrieved context
        st.subheader("Retrieved Context:")
        st.write(context)

        # Create prompt for LLM
        prompt = f"""
        Answer the question based on the given context and more in 4-5 lines.

        Context : {context}.

        Question : {query}
        """

        # Send context and question to Ollama (Llama 3.2)
        response = chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.message.content

        # Display final RAG answer
        st.subheader("Marvellous GenAI : RAG Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question before clicking the button.")