# AI Resume Scanner

Overview
The AI Resume Scanner is a Streamlit-based application designed to streamline the resume screening process using AI-powered tools. It allows users to upload resumes (in PDF or Word format), process them into searchable embeddings, and query the data using advanced language models (LLMs). The application is ideal for recruiters, hiring managers, and HR professionals looking to efficiently analyze resumes and match candidates to job descriptions.

Features
1. Resume Processing
Upload resumes in .pdf, .doc, or .docx formats.
Automatically processes and stores resumes in a vector database for efficient querying.
2. Query-Based Candidate Search
Input a job description or required skills as a query.
Retrieve relevant candidates based on their resumes.
Highlights strengths, weaknesses, and provides recommendations for the best fit.
3. Language Model Integration
Supports multiple LLMs for query processing:
llama-3.1-8b-instant
deepseek-r1-distill-llama-70b
Allows users to switch between models for flexibility.
4. Interactive User Interface
Simple and intuitive interface built with Streamlit.
Dropdown menus, text inputs, and spinners for a seamless user experience.
Project Structure
Main File
AIResumeScanner.py: The main application file that handles the Streamlit UI, document processing, and integration with the Groq API.
Key Functions
get_vectorstore():
Initializes a Chroma vector store for storing and retrieving document embeddings.
load_documents_from_directory(directory_path):
Loads resumes from the specified directory.
Supports .pdf, .doc, and .docx formats.
process_and_store_documents(documents, vectorstore):
Splits documents into chunks and stores them in the vector store.
call_groq_api(query, api_key, selected_model):
Sends a query to the Groq API and retrieves the response.
search_query(query, vectorstore, selected_model):
Retrieves relevant documents from the vector store and queries the Groq API for a response.
Setup and Installation
Prerequisites
Python 3.8 or higher
pip (Python package manager)
Installation Steps
Clone the repository:

Install the required dependencies:

Set up the environment variables:

Create a .env file in the root directory.
Add your Groq API key:
Run the application:

Usage
1. Launch the App
Open the app in your browser after running the streamlit command.
The home page will display the title "🔍 AI Resume Scanner."
2. Upload Resumes
Enter the directory path containing .pdf, .doc, or .docx files.
The app will process and store the resumes in a vector database.
3. Select a Language Model
Choose a language model from the dropdown menu:
llama-3.1-8b-instant
deepseek-r1-distill-llama-70b
4. Query the Resumes
Enter a job description or required skills in the query input box.
The app will retrieve relevant candidates and display:
Matching candidate names.
Strengths and weaknesses.
Recommended best fit for the job.
Example Use Cases
1. Candidate Screening
Scenario: A recruiter wants to find the best candidates for a Python developer role.
Steps:
Upload a folder containing resumes.
Enter the query: "Looking for a Python developer with experience in machine learning."
View the ranked candidates and their strengths/weaknesses.
Output:
Candidate 1: John Doe - Strength: Python expertise, Weakness: Limited ML experience.
Candidate 2: Jane Smith - Strength: Strong ML background, Weakness: Beginner in Python.
2. Candidate Comparison
Scenario: A hiring manager wants to compare candidates for a data analysis role.
Steps:
Upload resumes.
Query for specific skills or qualifications.
Analyze the strengths and weaknesses of each candidate.
Output:
Candidate 1: Proficient in SQL and Excel, lacks Python skills.
Candidate 2: Strong Python and Tableau skills, limited SQL experience.
Styling and Aesthetics
Custom Styling
The app uses Streamlit's built-in components for a clean and modern look.
Dropdown menus and spinners enhance interactivity.
User-Friendly Interface
Clear error messages for invalid directory paths.
Loading spinners for long-running tasks.
Future Enhancements
Add support for additional file formats (e.g., .txt, .csv).
Integrate more advanced LLMs for better accuracy.
Provide detailed analytics, such as skill distribution across candidates.
Add a feature for exporting results to a .csv or .pdf file.
Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or feedback, please contact:

Ashish: ashish_upreti7@yahoo.com
Enjoy using the AI Resume Scanner! 🚀---
