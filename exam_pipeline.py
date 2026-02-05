import io
import os
import re
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from groq import Groq
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# -------------------------
# MODELS
# -------------------------
EXTRACT_MODEL  = "llama-3.1-8b-instant"
SCORE_MODEL    = "llama-3.3-70b-versatile"
REPHRASE_MODEL = "llama-3.1-8b-instant"

# -------------------------
# LOCAL MODELS (Cached locally)
# -------------------------
bi_encoder = None
cross_encoder = None

# -------------------------
# GROQ CLIENT (LOCAL)
# -------------------------
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ExamPipeline:

    # -------------------------
    # OCR HELPER
    # -------------------------
    def _ocr_page(self, page):
        pix = page.get_pixmap(dpi=200)
        if pix.width * pix.height > 12_000_000:
            return ""
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)

    # -------------------------
    # TEXT EXTRACTION (DIGITAL + OCR FALLBACK)
    # -------------------------
    def extract_text(self, file_bytes):
        text = ""
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            for page in doc:
                page_text = page.get_text().strip()

                # If page has little/no text → OCR
                if len(page_text) < 50:
                    page_text = self._ocr_page(page)

                text += page_text + "\n"

            doc.close()

        except Exception as e:
            print(f"Error opening PDF: {e}")
            return ""

        clean_text = text.strip()
        print(f"--- DEBUG: Extracted {len(clean_text)} characters from PDF ---")
        return clean_text

    # -------------------------
    # SAFE JSON PARSER
    # -------------------------
    def _safe_json_load(self, text):
        text = text.strip()
        text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
        match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    # -------------------------
    # QUESTION EXTRACTION
    # -------------------------
    def get_questions(self, text):
        if not text:
            return []

        prompt = f"""
Extract ALL exam questions from the text below.
Return JSON in ONE of these forms:
1) [{{"id": 1, "question": "..."}}]
2) {{ "questions": [{{"id": 1, "question": "..."}}] }}

TEXT:
{text}
"""
        response = client_groq.chat.completions.create(
            model=EXTRACT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw = self._safe_json_load(response.choices[0].message.content)
        if raw is None:
            return []

        if isinstance(raw, list):
            return raw

        if isinstance(raw, dict):
            for key in ["questions", "exam_questions"]:
                if key in raw and isinstance(raw[key], list):
                    return raw[key]

            if all(isinstance(v, str) for v in raw.values()):
                return [{"id": k, "question": v} for k, v in raw.items()]

        return []

    # -------------------------
    # SCORING
    # -------------------------
    def score_question(self, question):
        prompt = f"""Rate this exam question for how likely it is to waste students' time due to confusing wording,
0 = perfectly clear, 100 = extremely confusing. DON'T CONSIDER DOUBLE NEGATIVES.
Give it in this format:
1. Score:
2. Justification: (1 line)

Question: {question}
"""
        response = client_groq.chat.completions.create(
            model=SCORE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        txt = response.choices[0].message.content
        try:
            score = int(re.search(r"Score:\s*(\d+)", txt).group(1))
            justification = re.search(r"Justification:\s*(.+)", txt).group(1).strip()
            return score, justification
        except Exception:
            return 0, "Could not parse score."

    # -------------------------
    # REPHRASING
    # -------------------------
    def rephrase(self, question, score, justification, attempt=0):
        dynamic_temp = 0.2 + (attempt * 0.1)

        retry_instruction = ""
        if attempt > 0:
            retry_instruction = (
                "Your previous attempt was rejected for not matching the original meaning. "
                "Try a different structure while staying strictly faithful to the core intent."
            )

        prompt = f"""Rephrase the question to be clearer (score below 20).
{retry_instruction}
Original: {question}
Justification: {justification}
Output ONLY the rephrased question.
"""
        response = client_groq.chat.completions.create(
            model=REPHRASE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=dynamic_temp
        )
        return response.choices[0].message.content.strip()

    # -------------------------
    # VALIDATION
    # -------------------------
    def validate(self, original, candidate):
        global bi_encoder, cross_encoder

        if bi_encoder is None:
            bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        if cross_encoder is None:
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        if original.strip().lower() == candidate.strip().lower():
            return False

        bi_score = float(util.cos_sim(
            bi_encoder.encode(original, convert_to_tensor=True),
            bi_encoder.encode(candidate, convert_to_tensor=True)
        ))
        if bi_score < 0.8:
            return False

        cross_score = float(
            cross_encoder.predict([[original, candidate]])[0]
        )
        if cross_score < 0.75:
            return False

        return True

    # -------------------------
    # MAIN RUN (6-attempt hard bound)
    # -------------------------
    def run(self, file_bytes):
        MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB
        if len(file_bytes) > MAX_PDF_SIZE:
            print("PDF too large")
            return []

        results = []

        text = self.extract_text(file_bytes)
        if not text:
            return []

        questions = self.get_questions(text)
        print(f"--- DEBUG: Found {len(questions)} questions ---")

        MAX_ATTEMPTS = 6

        for q in questions:
            original = q.get("question", "")
            if not original:
                continue

            score, justification = self.score_question(original)
            final = original
            status = "Original Kept"

            if score > 20:
                for attempt in range(MAX_ATTEMPTS):
                    candidate = self.rephrase(
                        original,
                        score,
                        justification,
                        attempt=attempt
                    )

                    if self.validate(original, candidate):
                        final = candidate
                        status = "Rephrased"
                        break

                    print(
                        f"Attempt {attempt + 1} failed validation "
                        f"for Question ID {q.get('id')}"
                    )

                if status == "Original Kept":
                    status = "Manual Review Required"

            results.append({
                "ID": q.get("id", "?"),
                "Original Question": original,
                "Score": score,
                "Justification": justification,
                "Final Version": final,
                "Status": status
            })

        return results
