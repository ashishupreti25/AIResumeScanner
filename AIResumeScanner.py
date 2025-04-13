import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import requests

# Constants
CHROMA_COLLECTION_NAME = "Resume_Collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-8b-8192"
groq_api_key = os.getenv("GROQ_API_KEY", "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")

# Load Sentence Transformer embeddings
@st.cache_resource
def get_vectorstore():
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(collection_name=CHROMA_COLLECTION_NAME, embedding_function=embedding_function)

# Load and process all documents from the given directory
def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith((".doc", ".docx")):
            loader = UnstructuredWordDocumentLoader(filepath)
        else:
            continue

        documents.extend(loader.load())

    return documents

# Split and store documents in Chroma DB
def process_and_store_documents(documents, vectorstore):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectorstore.add_documents(chunks)

# Call Groq API
def call_groq_api(query, api_key, selected_model):
    from groq import Groq  # Import the Groq client

    # Initialize the Groq client
    client = Groq(api_key = api_key)
    
    response = client.chat.completions.create(
     model = selected_model,  # Specify the alternate model
     messages = query,  # Provide the messages
    #     temperature = 0.5  # Set the temperature for response generation
     )
    
    return (response.choices[0].message.content)  # Print the response

# Run query using Groq + Chroma
def search_query(query, vectorstore, selected_model):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    print (f"Context: {context}")
    print (f"Query: {query}")
    groq_query = [
        {"role": "system", "content"
         : "You are a professional resume analyzer. List down matching candidate names to the job description, highlight strengths/weaknesses, and recommend the best fit."},
        {"role": "user", "content": f"Here are the matching resumes:\n\n{context}"},
        {"role": "user", "content": f"{query}"}
    ]
    # "You are a professional resume analyzer. Match candidates to the job description, highlight strengths/weaknesses, and recommend the best fit."
    # "You are a professional resume analyzer that evaluates matching candidates for the job description provided by user. "
    #      "List down the candidates with their names and the reason why they are best fit for the job. Also try to mention their strength or weekness"
    #      "Finally list down your best pick for the job and why"
    return call_groq_api(groq_query, groq_api_key, selected_model)

# Streamlit app
def main():
    st.title("🔍 AI Resume Scanner")

    # Input directory
    directory_path = st.text_input("Enter directory path with word/pdf files:")

    if directory_path and os.path.isdir(directory_path):
        vectorstore = get_vectorstore()
        with st.spinner("Loading and embedding documents..."):
            documents = load_documents_from_directory(directory_path)
            process_and_store_documents(documents, vectorstore)
        st.success("Documents processed and stored!")

        # Dropdown for switching LLM
        models = ["llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b"]
        # col1, col2 = st.columns(2)
        # with col1:
        selected_model = st.selectbox("Select a Language Model:", models)
        # with col2:
        #     num_results = st.number_input("Select candidatest needed:", min_value=1, max_value=10, value=1)

        query = st.text_input("Please provide the skills or job discription:")
        if query:
            with st.spinner("Retrieving answer..."):
                answer = search_query(query, vectorstore, selected_model)
            st.subheader("🧠 Answer:")
            st.write(answer)
    elif directory_path:
        st.error("Provided path is not a valid directory.")

if __name__ == "__main__":
    main()
