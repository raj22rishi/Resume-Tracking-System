import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load spaCy model
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
        
        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer()
        job_vec = vectorizer.fit_transform([job_description_processed])
        resume_texts = []
        resume_vecs = []
        for file in uploaded_files:
            # Parse and preprocess resumes
            text = extract_text_from_pdf(file)
            text_processed = preprocess_text(text)
            resume_texts.append(text_processed)
            resume_vecs.append(vectorizer.transform([text_processed]))

        # Calculate similarity scores
        similarities = []
        for i, resume_vec in enumerate(resume_vecs):
            similarity = (resume_vec * job_vec.T).A[0][0]
            similarities.append((uploaded_files[i].name, similarity))

        # Sort resumes by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Display sorted resumes with matching percentage
        st.header("Ranked Resumes")
        for resume, similarity in similarities:
            st.write(f"Resume: {resume}, Match Percentage: {similarity * 100:.2f}%")

if __name__ == "__main__":
    main()
