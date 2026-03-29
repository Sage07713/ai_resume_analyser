from fastapi import FastAPI, UploadFile, File
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from pydantic import BaseModel
app = FastAPI()
@app.get("/")
def home():
 return { "message": "server running"}   


class ATSRequest(BaseModel):
    resume_text: str
    job_desc: str

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    contents = await file.read()

    # Save temporarily
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    text = ""

    with pdfplumber.open("temp.pdf") as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            

    return {
        "filename": file.filename,
        "preview": text[:500] , # first 500 chars
        "word_count": len(text.split()),
    }
@app.post("/ats-score")
async def ats_score(data: ATSRequest):
    
    resume_text = data.resume_text
    job_desc = data.job_desc

    # (rest of your code stays same)
    # --- STEP 1: Define skills ---
    skills = [
        "python", "java", "c++", "machine learning",
        "data analysis", "sql", "deep learning"
    ]

    resume_text_lower = resume_text.lower()
    job_desc_lower = job_desc.lower()

    # --- STEP 2: Extract skills ---
    resume_skills = [s for s in skills if s in resume_text_lower]
    job_skills = [s for s in skills if s in job_desc_lower]

    # --- STEP 3: Keyword Score ---
    matched = len(set(resume_skills) & set(job_skills))
    total = len(job_skills) or 1

    keyword_score = (matched / total) * 100

    # --- STEP 4: ML Similarity (TF-IDF) ---
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    ml_score = similarity * 100

    # --- STEP 5: Resume Quality ---
    word_count = len(resume_text.split())

    if word_count < 200:
        quality_score = 40
    elif word_count < 500:
        quality_score = 70
    else:
        quality_score = 90

    # --- STEP 6: Final ATS Score ---
    ats_score = (
        0.4 * keyword_score +
        0.4 * ml_score +
        0.2 * quality_score
    )

    # --- STEP 7: Suggestions ---
    missing_skills = list(set(job_skills) - set(resume_skills))

    suggestions = []

    if missing_skills:
        suggestions.append(f"Add skills: {', '.join(missing_skills)}")

    if word_count < 300:
        suggestions.append("Increase resume content")

    # --- RESPONSE ---
    return {
        "ats_score": round(ats_score, 2),
        "keyword_score": round(keyword_score, 2),
        "ml_score": round(ml_score, 2),
        "quality_score": quality_score,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "suggestions": suggestions
    }