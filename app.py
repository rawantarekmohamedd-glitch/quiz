from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import json
import fitz
import re
import os
from processing import sanitize_for_llm

load_dotenv()

# ── Multi-key setup ─────────────────────────────────────────────────────
# Put your keys in .env like this:
# GEMINI_API_KEYS=key1,key2,key3
# (comma-separated, no spaces or quotes)

raw_keys = os.getenv("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not API_KEYS:
    raise RuntimeError(
        "You must set at least one key in the GEMINI_API_KEYS environment variable "
        "(comma-separated if you have more than one)."
    )

MODEL = "gemini-2.5-flash"

# Status codes that mean "this key is exhausted/invalid, try the next one"
RETRYABLE_STATUS_CODES = {429, 403, 401}


def call_gemini_with_fallback(prompt: str):
    """
    Tries each API key in order.
    Returns the response as soon as one key succeeds.
    If a key fails due to quota/rate-limit/auth, tries the next key.
    If all keys fail, raises the last error encountered.
    """
    last_error = None

    for index, key in enumerate(API_KEYS):
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            return response  # this key worked, stop here

        except genai_errors.APIError as e:
            status_code = getattr(e, "code", None)
            last_error = e

            if status_code in RETRYABLE_STATUS_CODES:
                print(f"[warning] Key #{index + 1} failed (status={status_code}). Trying next key...")
                continue
            else:
                # Not a quota/rate-limit issue (e.g. invalid prompt) — no point trying another key
                raise

        except Exception as e:
            # Any other unexpected error (network, timeout, etc.) — try the next key too, just in case
            last_error = e
            print(f"[warning] Key #{index + 1} failed with an unexpected error: {e}. Trying next key...")
            continue

    # If we get here, every key failed
    raise RuntimeError(f"All API keys failed. Last error: {last_error}")


app = FastAPI(title="PDF Quiz Generator API")

DIFFICULTY_MAP = {
    "سهل": "easy", "سهلة": "easy", "سهلين": "easy", "بسيط": "easy", "بسيطة": "easy",
    "متوسط": "medium", "متوسطة": "medium", "متوسطين": "medium",
    "صعب": "hard", "صعبة": "hard", "صعبين": "hard", "صعاب": "hard",
    "easy": "easy", "simple": "easy",
    "medium": "medium", "moderate": "medium", "normal": "medium",
    "hard": "hard", "difficult": "hard", "tough": "hard",
}


def parse_user_intent(user_text: str) -> dict:
    text = user_text.lower().strip()
    arabic_indic = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    normalized = text.translate(arabic_indic)

    num_questions = 10
    match = re.search(r"\b(\d+)\b", normalized)
    if match:
        num_questions = max(1, min(30, int(match.group(1))))

    difficulty = "medium"
    for keyword, level in DIFFICULTY_MAP.items():
        if keyword in text:
            difficulty = level
            break

    return {"num_questions": num_questions, "difficulty": difficulty}


@app.get("/")
async def home():
    return {"message": "PDF Question Generator API", "available_keys": len(API_KEYS)}


@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    request: str = Form(..., description="Natural language request, e.g. 'make 15 hard questions'"),
):
    # ── 1. Validate file ────────────────────────────────────────────────
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed"}

    # ── 2. Parse intent ─────────────────────────────────────────────────
    intent = parse_user_intent(request)
    num_questions = intent["num_questions"]
    difficulty = intent["difficulty"]

    difficulty_guide = {
        "easy":   "straightforward recall questions, basic definitions, simple facts",
        "medium": "conceptual understanding, cause-and-effect, application of ideas",
        "hard":   "deep analysis, synthesis across topics, edge cases, tricky distractors",
    }

    # ── 3. Extract text from PDF ────────────────────────────────────────
    try:
        pdf_bytes = await file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        raw_text = ""
        for page in pdf_doc:
            raw_text += page.get_text() + "\n"
        pdf_doc.close()
    except Exception as e:
        return {"error": f"Failed to read PDF: {str(e)}"}

    if not raw_text.strip():
        return {"error": "No text found in PDF"}

    # ── 4. Clean & chunk ────────────────────────────────────────────────
    try:
        chunks = sanitize_for_llm(
            raw_text=raw_text,
            source=file.filename,
            redact_pii_flag=False,
            max_tokens_per_chunk=600,
        )
        context = "\n\n".join(c["text"] for c in chunks)[:12000]
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

    # ── 5. Generate questions via Gemini (with fallback across keys) ────
    prompt = f"""
You are a university exam question generator.

Generate exactly {num_questions} MCQ questions from the lecture notes below.
Difficulty level: **{difficulty.upper()}** — {difficulty_guide[difficulty]}

Return ONLY valid JSON — no markdown, no explanation, no extra text.

Format:
[
  {{
    "question": "Question text",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "difficulty": "{difficulty}"
  }}
]

Rules:
- Exactly 4 options per question
- Exactly one correct answer
- No repeated questions
- Base questions ONLY on the provided content
- Match the requested difficulty level strictly

Lecture Notes:
{context}
"""

    try:
        response = call_gemini_with_fallback(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        questions = json.loads(clean)

        return {
            "filename": file.filename,
            "parsed_request": {
                "original_text": request,
                "num_questions": num_questions,
                "difficulty": difficulty,
            },
            "total_questions": len(questions),
            "mcq_questions": questions,
        }

    except json.JSONDecodeError:
        return {"error": "Gemini returned invalid JSON", "raw": response.text}
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
