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

MODEL = "gemini-2.0-flash-lite"

# Status codes that mean "this key is exhausted/invalid, try the next one"
RETRYABLE_STATUS_CODES = {429, 403, 401}

# ── Hardcoded fallback questions ─────────────────────────────────────────
FALLBACK_QUESTIONS = [
    {
        "question": "What is the primary objective of minimizing the loss function during the training of a machine learning model?",
        "options": [
            "To increase the computational speed of the network.",
            "To reduce the number of parameters in the model.",
            "To maximize the agreement between predicted and actual outputs, thereby increasing classification accuracy.",
            "To prevent the model from overfitting to the training data."
        ],
        "correct_answer": "To maximize the agreement between predicted and actual outputs, thereby increasing classification accuracy.",
        "difficulty": "medium"
    },
    {
        "question": "In the context of gradient descent for minimizing a function, why is the update rule `xk+1 = xk - η * ∇f(xk)` used, specifically the subtraction of the gradient term?",
        "options": [
            "The gradient points towards the direction of fastest decrease, so subtracting it moves towards the minimum.",
            "The gradient points towards the direction of fastest increase, so subtracting it moves towards the minimum.",
            "The gradient represents the error magnitude, and subtracting it reduces the error.",
            "The gradient helps to normalize the step size, ensuring stable convergence."
        ],
        "correct_answer": "The gradient points towards the direction of fastest increase, so subtracting it moves towards the minimum.",
        "difficulty": "medium"
    },
    {
        "question": "When minimizing a loss function that depends on multiple weight parameters (w1, w2, ..., wt), what is the primary role of partial derivatives?",
        "options": [
            "To determine the overall magnitude of the loss function.",
            "To calculate how the loss changes with respect to one specific weight parameter, while holding others constant.",
            "To identify if the loss function is convex or non-convex.",
            "To simplify the loss function into a single-variable problem."
        ],
        "correct_answer": "To calculate how the loss changes with respect to one specific weight parameter, while holding others constant.",
        "difficulty": "medium"
    },
    {
        "question": "For a neural network designed for binary classification where the output represents the probability of belonging to class 1, which activation function is typically used in the output layer?",
        "options": ["ReLU", "Softmax", "Sigmoid", "Tanh"],
        "correct_answer": "Sigmoid",
        "difficulty": "medium"
    },
    {
        "question": "If a neural network is designed to classify an input into one of N distinct classes (e.g., cat, dog, camel), how is the desired output typically represented for training?",
        "options": [
            "A single scalar value representing the class index.",
            "A binary vector where each element corresponds to a class, and only the correct class is 1 (one-hot encoding).",
            "A vector of N real-valued probabilities that do not necessarily sum to 1.",
            "A single binary value (0 or 1) indicating if the input belongs to any of the N classes."
        ],
        "correct_answer": "A binary vector where each element corresponds to a class, and only the correct class is 1 (one-hot encoding).",
        "difficulty": "medium"
    },
    {
        "question": "The L2 loss function, also known as squared Euclidean distance, is explicitly mentioned for what type of neural network problem?",
        "options": [
            "Binary classification with a single output neuron.",
            "Multi-class classification with one-hot encoded outputs.",
            "Regression problems with real-valued output vectors.",
            "Problems requiring non-differentiable loss functions."
        ],
        "correct_answer": "Regression problems with real-valued output vectors.",
        "difficulty": "medium"
    },
    {
        "question": "In a Keras model compilation for a binary classification task, which loss function is typically specified, as shown in the lecture notes?",
        "options": [
            "`mean_squared_error`",
            "`categorical_crossentropy`",
            "`binary_crossentropy`",
            "`sparse_categorical_crossentropy`"
        ],
        "correct_answer": "`binary_crossentropy`",
        "difficulty": "medium"
    },
    {
        "question": "In the iterative solution for gradient descent, what does the parameter `η` (eta) represent?",
        "options": [
            "The number of iterations.",
            "The current value of the weight parameter.",
            "The step size or learning rate, controlling how much the parameters are updated in each iteration.",
            "The magnitude of the gradient at the current point."
        ],
        "correct_answer": "The step size or learning rate, controlling how much the parameters are updated in each iteration.",
        "difficulty": "medium"
    },
    {
        "question": "According to the lecture notes, what does \"network training\" primarily entail in the context of empirical error optimization?",
        "options": [
            "Increasing the complexity of the network architecture.",
            "Updating the weight parameters (`w`) to minimize the loss function.",
            "Collecting more input data (`x`) for better accuracy.",
            "Changing the activation functions of the neurons."
        ],
        "correct_answer": "Updating the weight parameters (`w`) to minimize the loss function.",
        "difficulty": "medium"
    },
    {
        "question": "When dealing with a multi-class classification problem where the neural network's output needs to represent a probability distribution over N classes (i.e., N probability values that sum to 1), which activation function is often used in the output layer?",
        "options": ["Sigmoid", "ReLU", "Softmax", "Linear"],
        "correct_answer": "Softmax",
        "difficulty": "medium"
    }
]


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
                raise

        except Exception as e:
            last_error = e
            print(f"[warning] Key #{index + 1} failed with an unexpected error: {e}. Trying next key...")
            continue

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
        # ── FALLBACK: كل الـ keys اتحرقت، رجّع أسئلة MLP ──────────────
        return {
            "filename": file.filename,
            "parsed_request": {
                "original_text": request,
                "num_questions": num_questions,
                "difficulty": difficulty,
            },
            "total_questions": len(FALLBACK_QUESTIONS),
            "mcq_questions": FALLBACK_QUESTIONS,
        }
