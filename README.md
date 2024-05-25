# Resume-Tracking-System

This Streamlit application allows you to upload resumes (in PDF format) and enter job descriptions or keywords to filter and rank the resumes based on their relevance to the job description. It uses natural language processing (NLP) and machine learning to match the content of the resumes with the job description and provide a similarity score for each resume.

## Features

- Upload multiple resumes in PDF format.
- Enter a job description or keywords to filter and rank resumes.
- Displays a ranked list of resumes based on their match percentage with the job description.
- Uses spaCy for text preprocessing and scikit-learn's TF-IDF Vectorizer for similarity calculation

## Requirements

- Python 3.6 or higher
- Required Python packages (listed in requirements.txt):
- streamlit
- PyPDF2
- spacy
- scikit-learn

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/raj22rishi/Resume-Tracking-System.git
   cd Resume-Tracking-System 

2. Install the required packages:
   
   ```
   pip install -r requirements.txt

3.  Run Application

   ```
   streamlit run app.py
``` 
## How It Works

Text Extraction: The application extracts text from the uploaded PDF resumes using PyPDF2.

Text Preprocessing: The extracted text is preprocessed using the spaCy NLP model to remove stop words and perform lemmatization.

Resume Parsing: The resumes are parsed to extract relevant fields such as skills, experience, and degree.

Similarity Calculation: The job description and the relevant fields from the resumes are vectorized using TF-IDF. The similarity between the job description and each resume is calculated using cosine similarity.

Ranking: The resumes are ranked based on their similarity scores and displayed in descending order.

