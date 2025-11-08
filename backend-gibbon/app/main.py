from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PyPDF2 import PdfReader
import spacy
import re

# Initialize FastAPI app
app = FastAPI(title="Groovy Gibbon - CV Ranking API", version="1.0.0")

# Allow your React app to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend dev URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NLP model
nlp = spacy.load("en_core_web_md")

# --- Landing Page ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Groovy Gibbon Backend</title>
            <style>
                body {
                    font-family: 'Segoe UI', sans-serif;
                    background: linear-gradient(135deg, #101820, #1b1f29);
                    color: #f8f8f8;
                    text-align: center;
                    height: 100vh;
                    margin: 0;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }
                h1 {
                    font-size: 3em;
                    color: #00ffcc;
                    margin-bottom: 0.2em;
                }
                p {
                    font-size: 1.2em;
                    color: #bbb;
                }
                .badge {
                    background-color: #00ffcc;
                    color: #101820;
                    border-radius: 5px;
                    padding: 6px 12px;
                    margin-top: 10px;
                    font-weight: bold;
                    text-decoration: none;
                }
                .badge:hover {
                    background-color: #00cc99;
                }
            </style>
        </head>
        <body>
            <h1>🦧 Groovy Gibbon Backend</h1>
            <p>FastAPI server is running successfully.</p>
            <a href="/docs" class="badge">Go to API Docs</a>
        </body>
    </html>
    """

# --- Helper functions ---
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

def calculate_similarity(cv_text, job_text):
    doc1 = nlp(cv_text)
    doc2 = nlp(job_text)
    return doc1.similarity(doc2) * 100

# --- Main endpoint ---
@app.post("/api/rank")
async def rank_jobs(files: list[UploadFile] = File(...)):
    job_descriptions = [
        "We are looking for a software engineer with experience in Python.",
        "Data analyst with strong Excel and visualization skills.",
        "Marketing manager with social media expertise."
    ]

    results = []
    for file in files:
        text = extract_text_from_pdf(file.file)
        cv_cleaned = preprocess_text(text)
        job_scores = []
        for job in job_descriptions:
            job_cleaned = preprocess_text(job)
            similarity = calculate_similarity(cv_cleaned, job_cleaned)
            job_scores.append({"job": job, "score": round(similarity, 2)})
        results.append({"filename": file.filename, "scores": job_scores})

    return {"results": results}
