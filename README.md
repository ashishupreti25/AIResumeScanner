# AIResumeScanner

Code Overview
The provided code snippet is part of an AI-powered resume scanning application built using Streamlit. The application allows users to upload resumes (in Word or PDF format) from a directory, process them, and query the data using a selected language model (LLM). Here's a breakdown of the key components:

Key Functions and Components
main() Function:

This is the entry point for the Streamlit app.
It defines the user interface (UI) and handles user inputs.
Directory Input:

The user provides a directory path containing resumes.
The app verifies if the directory exists using os.path.isdir(directory_path).
Document Processing:

Resumes are loaded and embedded using helper functions:
load_documents_from_directory(directory_path): Loads resumes from the specified directory.
process_and_store_documents(documents, vectorstore): Processes and stores the documents in a vector database for efficient querying.
Language Model Selection:

Users can select a language model from a dropdown menu (st.selectbox).
Available models:
llama-3.1-8b-instant
deepseek-r1-distill-llama-70b
Query Input:

Users provide a job description or required skills as a query.
The app retrieves relevant information using:
search_query(query, vectorstore, selected_model): Queries the vector database using the selected LLM.
Answer Display:

The app displays the retrieved answer in a user-friendly format.
Use Cases
Resume Screening for Job Applications:

Scenario: A recruiter wants to find the best candidates for a job based on specific skills or job descriptions.
Steps:
Upload a folder containing resumes.
Enter the job description or required skills in the query input.
Select a language model.
View the ranked candidates and their strengths/weaknesses.
Example:

Input:
Directory: /resumes/
Query: "Looking for a Python developer with experience in machine learning."
Output:
Candidate 1: John Doe - Strength: Python expertise, Weakness: Limited ML experience.
Candidate 2: Jane Smith - Strength: Strong ML background, Weakness: Beginner in Python.
Candidate Comparison:

Scenario: A hiring manager wants to compare candidates for a specific role.
Steps:
Upload resumes.
Query for specific skills or qualifications.
Analyze the strengths and weaknesses of each candidate.
Example:

Input:
Query: "Compare candidates for data analysis roles."
Output:
Candidate 1: Proficient in SQL and Excel, lacks Python skills.
Candidate 2: Strong Python and Tableau skills, limited SQL experience.
Best Candidate Recommendation:

Scenario: The app recommends the best candidate for a role based on the provided job description.
Steps:
Provide a job description.
The app processes resumes and ranks candidates.
The app highlights the best candidate and explains why.
Example:

Input:
Query: "Find the best candidate for a software engineering role."
Output:
Best Pick: Jane Smith - Strong coding skills, excellent problem-solving abilities.
Future Enhancements
Add support for more file formats (e.g., plain text, HTML).
Allow users to specify the number of candidates to retrieve.
Integrate additional LLMs for better accuracy and flexibility.
Provide detailed analytics, such as skill distribution across candidates.
This application streamlines the resume screening process, saving time and improving hiring decisions.
