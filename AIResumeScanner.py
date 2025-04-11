# ----------------------------- Imports -----------------------------
import argparse  # For parsing command-line arguments
import os  # For file and directory operations
import shutil  # For file and directory removal
import nltk  # For natural language processing utilities
import chromadb  # For Chroma database operations
from langchain.document_loaders.pdf import PyPDFDirectoryLoader  # For loading PDF documents
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain.schema.document import Document  # For document schema
from langchain.embeddings import HuggingFaceEmbeddings  # For embedding generation
from langchain.vectorstores.chroma import Chroma  # For vector database operations
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader  # For loading Word documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain.schema import Document  # Ensure the Document class is imported
import streamlit as st  # For web app development (if needed)


# ----------------------------- NLTK Setup -----------------------------
# Download necessary NLTK resources
nltk.download('punkt_tab')  # Tokenizer models
nltk.download('averaged_perceptron_tagger_eng')  # POS tagger models

# ----------------------------- Constants -----------------------------
DATA_PATH = r"C:\Mine\GItHub\GenAI\RAGEnhancedSearch\Docs" # Path to the directory containing documents
CHROMA_PATH = "db/chroma_db"  # Path to the Chroma database

# ----------------------------- Argument Parsing and Database Reset -----------------------------
def handle_database_reset():
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")  # Add a flag to reset the database
    args = parser.parse_args([])  # Parse arguments (empty list for environments without CLI)

    # Database Reset
    if args.reset:
        print("✨ Clearing Database")  # Notify the user
        shutil.rmtree(DATA_PATH)  # Remove the database directory

def load_documents(data_path):
    # ----------------------------- PDF Document Loading -----------------------------
    # Load all PDF documents from the specified directory
    pdf_doc_loader = PyPDFDirectoryLoader(data_path)
    documents = pdf_doc_loader.load()  # Load documents into memory
    print(f"Number of pdf documents created: {len(documents)}")  # Print the number of loaded documents

    # ----------------------------- Word Document Loading -----------------------------
    wordx_docs = []  # Initialize an empty list for Word documents
    files = [ext for ext in os.listdir(data_path)]  # List all files in the data directory

    # Iterate through files and load Word documents
    for file in files:    
        if '.docx' in file or '.doc' in file:  # Check for Word document extensions
            full_path = os.path.join(data_path, file)  # Get the full file path
            docx_loader = UnstructuredWordDocumentLoader(full_path)  # Initialize the loader
            docx_document = docx_loader.load()  # Load the document
            wordx_docs += docx_document  # Append the loaded document to the list

    print(f"Word Document loaded with {len(wordx_docs)} sections.")  # Print the number of loaded Word document sections

    # ----------------------------- Combine Documents -----------------------------
    for doc in wordx_docs:
        documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))  # Wrap content in Document object

    print(f"Total Document loaded with {len(documents)} sections.")  # Print the total number of loaded documents
    return documents

def split_documents_into_chunks(documents):
    # ----------------------------- Text Splitting -----------------------------
    # Initialize the text splitter with chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    # Split the documents into smaller chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")  # Print the number of created chunks
    return chunks

def initialize_chromadb_and_add_data(documents):
    # ----------------------------- ChromaDB Initialization -----------------------------
    # Initialize ChromaDB in persistent mode
    chroma_client = chromadb.PersistentClient(path="db/chroma_db")

    # Check if the collection exists before attempting to delete it
    if "Resumes" in [col.name for col in chroma_client.list_collections()]:
        chroma_client.delete_collection(name="Resumes")

    # Create or get a collection for storing documents
    collection = chroma_client.get_or_create_collection(name="Resumes")

    # ----------------------------- Embedding Model Setup -----------------------------
    from sentence_transformers import SentenceTransformer  # Import the embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Load a pre-trained model
    print(collection.name)  # Print the collection name

    # ----------------------------- Add Data to ChromaDB -----------------------------
    # Generate embeddings for the documents
    embeddings = model.encode([doc.page_content for doc in documents])

    # Add documents, embeddings, and metadata to the ChromaDB collection
    collection.add(
        documents=[doc.page_content for doc in documents],  # Document content
        embeddings=embeddings,  # Generated embeddings
        metadatas=[doc.metadata for doc in documents],  # Metadata
        ids=[str(i) for i in range(len(documents))]  # Unique IDs for documents
    )

    print(f"Successfully added resume to ChromaDB.")  # Notify the user
    return model, collection  # Return the model and collection for further use

def perform_similarity_search(query, model, collection):
    # ----------------------------- Similarity Search -----------------------------
    # Perform a similarity search on the vector database

    query_embedding = model.encode(query)  # Generate embedding for the query

    results = collection.query(
        query_embeddings=query_embedding,  # Query embedding
        n_results=2  # Return top 2 results
    )

    # Display the search results
    print(f"Query: {query}")
    for i in range(len(results['documents'][0])):
        print(f"\nResult {i + 1}:")
        print(f"  Document: {results['documents'][0][i][:200]}...")  # Print first 200 characters
        print(f"  Metadata: {results['metadatas'][0][i]}")  # Print metadata
        print(f"  Distance: {results['distances'][0][i]}")  # Print distance

    return results

def integrate_chat_model(results, query):
    # ----------------------------- Chat Model Integration -----------------------------
    from groq import Groq  # Import the Groq client

    # Initialize the Groq client
    client = Groq(api_key='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')

    # Prepare context for the chat model
    system_prompt = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    # Create messages for the chat model
    messages = [
        {"role": "system", "content": str(results['documents'][0])},
        {"role": "user", "content": f"Extract content from this query: '{query}'."}
    ]

    # # Generate a response using the chat model
    # response = client.chat.completions.create(
    #     model="llama-3.1-8b-instant",  # Specify the model
    #     messages=messages,  # Provide the messages
    #     temperature=0.5  # Set the temperature for response generation
    # )

    # print(response.choices[0].message.content)  # Print the response

    # ----------------------------- Alternate Chat Model -----------------------------
    # Use a different model for generating a response
    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",  # Specify the alternate model
        messages=messages,  # Provide the messages
        temperature=0.5  # Set the temperature for response generation
    )

    return (response.choices[0].message.content)  # Print the response


# ----------------------------- Streamlit Setup -----------------------------
# Initialize Streamlit app
# App title
st.title("Simple Streamlit Input App")

# File path input
DATA_PATH = st.text_input("Enter file path")
if DATA_PATH:
    st.write(f"You entered: {DATA_PATH}")  # Display the entered file path
else:
    st.write("Please enter a file path.")

# Upload button
if st.button("Upload"):
    if DATA_PATH:
        st.success(f"File path '{DATA_PATH}' received successfully!")
        handle_database_reset()
        documents = load_documents(DATA_PATH)
        chunks = split_documents_into_chunks(documents)  # Split documents into chunks
        model, collection = initialize_chromadb_and_add_data(documents)  # Initialize ChromaDB and add data

    else:
        st.warning("Please enter a file path before uploading.")

st.markdown("---")

st.subheader("Query Input")
st.write("Enter your query below:")

# query = "Expirence in AWS Cloud ?"  # Example query

# Query input
query = st.text_input("Enter your query")

# Find button
if st.button("Find"):
    if query:
        results = perform_similarity_search(query, model, collection)  # Perform similarity search
        response = integrate_chat_model(results, query)  # Call the chat model integration
        st.write("Chat Model Response:")
        st.write(response)  # Display the response from the chat model
    
    else:
        st.warning("Please enter a query before clicking Find.")



