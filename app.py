import streamlit as st
from PyPDF2 import PdfReader
import spacy
from pyresparser import ResumeParser
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the spaCy model for natural language processing
nlp = spacy.load('en_core_web_sm')

# Function to extract text from PDF resumes
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to preprocess text using spaCy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Function to preprocess and combine the relevant resume fields
def preprocess_resume_data(resume_data):
    skills = " ".join(resume_data.get('skills', [])) if resume_data.get('skills') else ""
    experience = " ".join(resume_data.get('experience', [])) if resume_data.get('experience') else ""
    degree = " ".join(resume_data.get('degree', [])) if resume_data.get('degree') else ""
    combined_data = f"{skills} {experience} {degree}"
    return preprocess_text(combined_data)

# Main function to create the Streamlit app
def main():
    st.title("Resume Ranker and Prescreening Software")
    st.write("Upload resumes (in PDF format) and enter job descriptions or keywords to filter and rank them.")

    # Upload resumes
    uploaded_files = st.file_uploader("Upload Resumes (PDF files)", accept_multiple_files=True)
    
    # Input field for job description or keywords
    job_description = st.text_area("Enter Job Description or Keywords")

    if st.button("Rank Resumes"):
        if not uploaded_files:
            st.warning("Please upload one or more resumes.")
            return
        
        if not job_description:
            st.warning("Please enter a job description or keywords.")
            return
        
        # Preprocess the job description
        job_description_processed = preprocess_text(job_description)
        
        # Vectorize the job description and resumes
        vectorizer = TfidfVectorizer()
        job_vec = vectorizer.fit_transform([job_description_processed])
        
        # List to store responses along with file names and their matching percentages
        file_responses = []

        # Loop through uploaded resumes
        for file in uploaded_files:
            # Save the uploaded file to a temporary location
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())
            
            # Parse and preprocess resumes
            data = ResumeParser(file.name).get_extracted_data()
            if data:
                combined_resume_data = preprocess_resume_data(data)
                resume_vec = vectorizer.transform([combined_resume_data])
                similarity = (resume_vec * job_vec.T).A[0][0] * 100

                # Append file name and similarity to the list
                file_responses.append((file.name, similarity))
        
        # Sort file responses based on the similarity in descending order
        file_responses.sort(key=lambda x: x[1], reverse=True)
        
        # Display sorted file names and similarity percentages
        st.header("Ranked Resumes")
        for file_name, similarity in file_responses:
            st.write(f"Resume: {file_name}, Match Percentage: {similarity:.2f}%")

if __name__ == "__main__":
    main()