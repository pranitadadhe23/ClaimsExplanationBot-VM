# claim_explainer_chatbot_app.py
# ============================================================
# 🧾 Insurance Claim Explanation Chatbot — Smart Q&A Version
# (Same logic + download feature, only colors changed)
# ============================================================

import re
import os
import tempfile
from pathlib import Path

import streamlit as st
import pdfplumber

# ---------- Optional OCR (DocTR) ----------
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except Exception:
    DocumentFile = None
    ocr_predictor = None
    DOCTR_AVAILABLE = False

# ---------- Gen AI imports ----------
from langdetect import detect
from transformers import pipeline, MarianTokenizer, MarianMTModel

# ------------------------------------------------------------
# Streamlit page setup & custom styles
# ------------------------------------------------------------
st.set_page_config(page_title="Claim Explanation Chatbot", page_icon="🧾", layout="wide")

st.markdown(
    """
<style>
:root {
    --bg-main: #f4f4fb;
    --bg-gradient-top: #eef2ff;
    --bg-gradient-mid: #f9fafb;
    --bg-gradient-bottom: #fdf2ff;
    --card-bg: #ffffff;
    --card-border: #e5e7eb;
    --accent: #6366f1;
    --accent-soft: #e0e7ff;
    --text-main: #111827;
    --text-muted: #6b7280;
}

/* App Background */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, var(--bg-gradient-top) 0, var(--bg-gradient-mid) 45%, var(--bg-gradient-bottom) 100%);
    color: var(--text-main);
}

/* Remove Streamlit header bar */
[data-testid="stHeader"] {
    background: transparent;
}

/* Main container */
.block-container {
    max-width: 1000px;
    padding-top: 2.5rem;
}

/* Title */
.app-title{
    text-align:center;
    font-size:32px;
    color:var(--text-main);
    font-weight:800;
    margin-bottom:4px;
}
.app-sub{
    text-align:center;
    color:var(--text-muted);
    margin-bottom:12px;
}

/* Upload pill */
.chat-upload-wrapper{
    display:flex;
    justify-content:center;
    margin-bottom:10px;
}
.chat-upload{
    background:var(--card-bg);
    border-radius:999px;
    padding:8px 18px;
    border:1px solid var(--card-border);
    box-shadow:0 10px 25px rgba(15,23,42,0.06);
    display:flex;
    align-items:center;
    gap:8px;
}
.chat-upload-label{
    font-size:14px;
    color:var(--accent);
    font-weight:600;
}

/* Make file uploader invisible & clean */
.chat-upload .stFileUploader > label { display:none; }
.chat-upload .stFileUploader [data-baseweb="file-uploader"] {
    border:none;
    background:transparent;
}

/* Chat helper text */
.chat-helper{
    font-size:13px;
    color:var(--text-muted);
}

/* Chat bubble width */
.stChatMessage {
    max-width: 780px;
    margin-left:auto;
    margin-right:auto;
}

/* Assistant bubble style */
.stChatMessage[data-testid="assistant"] [data-testid="stMarkdownContainer"] {
    background: var(--card-bg);
    border:1px solid var(--card-border);
    border-radius:16px;
    padding:10px 14px;
}

/* User bubble */
.stChatMessage[data-testid="user"] [data-testid="stMarkdownContainer"] {
    background: var(--accent-soft);
    border-radius:16px;
    padding:10px 14px;
}

/*  Chat input (BOTTOM BAR LIKE CHATGPT) */
.stChatInput {
    background: var(--card-bg) !important;
    border-top: 1px solid var(--card-border);
    padding: 10px !important;
}

/* Text inside input */
textarea, input, .stChatInput textarea {
    color: #000000 !important;
    font-weight: 500;
}

/* Placeholder color */
textarea::placeholder {
    color: #9ca3af;
}

/*  Send button */
.stChatInput button {
    background: var(--accent) !important;
    color: white !important;
    border-radius: 50% !important;
}

/* Download button polish */
button[kind="secondary"] {
    background: var(--accent);
    color: white;
    border-radius: 10px;
}

/* Footer */
.footer{
    text-align:center;
    color:#9ca3af;
    font-size:12px;
    margin-top:18px;
}

""",
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">🧾 Claim Explanation Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Upload a claim report and ask anything about its status, amount, or reason.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# OCR model loader (cached)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ocr_model():
    if not DOCTR_AVAILABLE:
        return None
    try:
        return ocr_predictor(pretrained=True)
    except Exception:
        return None

ocr_model = load_ocr_model()

# ------------------------------------------------------------
# Gen AI: summarizer loader (cached)
# ------------------------------------------------------------
SUMMARIZATION_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # small, CPU-friendly

@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model=SUMMARIZATION_MODEL_NAME)

summarizer = load_summarizer()

# ------------------------------------------------------------
# Text cleaning and extraction helpers
# ------------------------------------------------------------
def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r'(?m)^[\-\=_\*]{3,}\s*$', '\n', t)
    t = re.sub(r'[ \t]+$', '', t, flags=re.M)
    t = re.sub(r'\n{3,}', '\n\n', t)

    cutoff_headers = [
        r'FINAL REMARKS', r'SIGN-?OFF', r'END OF REPORT', r'REVIEWED & SIGNED BY',
        r'APPROVAL & FINANCIAL ASSESSMENT', r'SIGN-OFF — CLAIMS DEPARTMENT', r'FINAL REMARKS:'
    ]
    for h in cutoff_headers:
        m = re.search(fr'(?im)\n{h}.*', t)
        if m:
            t = t[:m.start()].strip()
            break
    return t.strip()

def _extract_section(txt: str, header_patterns, stop_patterns=None):
    if not txt:
        return None
    if isinstance(header_patterns, (list, tuple)):
        header_rx = r'(?:' + r'|'.join(header_patterns) + r')'
    else:
        header_rx = r'(?:' + header_patterns + r')'

    m = re.search(fr'(?im){header_rx}\s*[:\-]?\s*\n', txt)
    if m:
        start = m.end()
        end = len(txt)
        if stop_patterns:
            stops = stop_patterns if isinstance(stop_patterns, (list, tuple)) else [stop_patterns]
            nearest = None
            for s in stops:
                ms = re.search(fr'(?im)\n{s}\s*[:\-]?\s*\n', txt[start:])
                if ms:
                    end_idx = start + ms.start()
                    if nearest is None or end_idx < nearest:
                        nearest = end_idx
            if nearest is not None:
                end = nearest
        section = txt[start:end].strip()
        return section or None

    m2 = re.search(fr'(?im){header_rx}\s*[:\-]\s*(.+)', txt)
    if m2:
        val = m2.group(1)
        return val.strip() if isinstance(val, str) and val.strip() else None
    return None

def resolve_status(txt: str, info_status: str = None) -> str:
    if not txt:
        return info_status
    candidates = []
    for m in re.finditer(r'(?im)(?:Claim\s*Status|Status)\s*[:\-]\s*([A-Za-z \-]+)', txt):
        val = m.group(1).strip()
        if val:
            candidates.append(val)
    for m in re.finditer(r'(?im)(?:Automatically updated status|Automatically updated)\s*[:\-]?\s*([A-Za-z \-]+)', txt):
        val = m.group(1).strip()
        if val:
            candidates.append(val)
    for m in re.finditer(r'(?im)\b(Approved with Reduction|Partially Approved|Approved|Denied|Rejected|Declined|Full Approval|FULL APPROVAL|APPROVED|DECLINED)\b', txt):
        candidates.append(m.group(1).strip())
    if candidates:
        return candidates[-1]
    return info_status

def extract_info(txt: str) -> dict:
    info = {}
    t = txt or ""

    def single_line(pat):
        m = re.search(fr'(?im)(?:{pat})\s*[:\-]\s*(.+)', t)
        if not m:
            return None
        v = m.group(1)
        return v.strip() if isinstance(v, str) and v.strip() else None

    info['claim_id'] = single_line(r'Claim\s*ID|Claim\s*No|Claim\s*Number|CLAIM\s*:|CLAIM\s*:')
    info['policy_number'] = single_line(r'Policy\s*Number|Policy\s*No|POLICY\s*NO|Policy\s*:')
    info['patient'] = single_line(r'Patient\s*Name|PATIENT\s*NAME|Name\b')

    ag = single_line(r'Age\s*/\s*Gender|Age\s*\(Years\)\s*/\s*Gender|Age/Gender|AGE/GENDER|Age[:]')
    if ag:
        m = re.search(r'(\d{1,3})\s*/\s*([A-Za-z]+)', ag)
        if m:
            info['age'] = m.group(1)
            info['gender'] = m.group(2)
        else:
            parts = [p.strip() for p in re.split(r'[/,|]', ag) if p.strip()]
            for p in parts:
                if re.match(r'^\d{1,3}$', p):
                    info['age'] = p
                elif p.isalpha():
                    info['gender'] = p

    info['uhid'] = single_line(r'UHID|UHID[: ]')
    info['hospital'] = single_line(r'Hospital\s*Name|HOSPITAL|Hospital[:]')
    info['hospital_city'] = single_line(r'Hospital\s*City|HOSPITAL\s*LOCATION|City\s*:')
    info['admission_date'] = single_line(r'Admission\s*Date|Admit[:]')
    info['discharge_date'] = single_line(r'Discharge\s*Date|Discharge[:]')

    m = re.search(r'(?im)Claim\s*Amount\s*(?:Submitted)?\s*[:\-]?\s*[₹INR\s]*([0-9,]+(?:\.\d+)?)', t)
    if m:
        info['claim_amount'] = m.group(1).strip()
    else:
        m2 = re.search(
            r'(?im)(?:total (?:hospital )?expenses|total billed|total claimed|expenses came to|expenses were|Bill paid by patient|expenses paid)\s*(?:[:\-]?)\s*[₹INR\s]*([0-9,]+(?:\.\d+)?)',
            t
        )
        if m2:
            info['claim_amount'] = m2.group(1).strip()

    m3 = re.search(r'(?im)Approved\s*Amount\s*[:\-]?\s*[₹INR\s]*([0-9,]+(?:\.\d+)?)', t)
    if m3:
        info['approved_amount'] = m3.group(1).strip()

    inline_status = single_line(r'Claim\s*Status|Status')
    resolved = resolve_status(t, inline_status)
    info['status'] = resolved

    reason_section = _extract_section(
        t,
        header_patterns=[
            r'REASON\s*FOR\s*REJECTION',
            r'REASONS?\s*FOR\s*REJECTION',
            r'REASON\s*FOR\s*PARTIAL\s*APPROVAL',
            r'REASONS?\s*FOR\s*PARTIAL\s*APPROVAL',
            r'REASONS?\s*FOR\s*DECLINE',
            r'REASONS?\s*FOR\s*DENIAL',
        ],
        stop_patterns=[
            r'FINAL\s+REMARKS',
            r'SIGN-?OFF',
            r'APPROVAL\s*&\s*FINANCIAL',
            r'HOSPITAL\s*BILL\s*SUMMARY',
            r'FINAL\s+REMARKS:'
        ]
    )
    if reason_section:
        reason = re.sub(r'\s+', ' ', reason_section).strip()
        info['reason'] = (reason[:800] + '...') if len(reason) > 800 else reason
    else:
        m = re.search(r'(?im)(policy (?:was )?inactive.*?grace period.*?\.?)', t)
        if m:
            info['reason'] = m.group(1).strip()
        else:
            m2 = re.search(r'(?im)Reason\s*[:\-]?\s*(.+)', t)
            if m2 and m2.group(1):
                info['reason'] = m2.group(1).strip()

    return info

def quick_summary(txt: str, info: dict) -> str:
    status_text = (info.get('status') or '') or ''
    stx = (status_text or '').lower()

    decision = "Unknown"
    if stx:
        if 'approved' in stx and ('reduction' in stx or 'partial' in stx):
            decision = "Approved with Reduction"
        elif 'approved' in stx or 'full approval' in stx or ('full' in stx and 'approval' in stx):
            decision = "Approved"
        elif 'rejected' in stx or 'denied' in stx or 'declined' in stx or 'decline' in stx:
            decision = "Rejected"
        else:
            if 'reject' in stx or 'deni' in stx:
                decision = "Rejected"
            elif 'approv' in stx:
                decision = "Approved"

    if decision == "Unknown":
        final_status = resolve_status(txt, None)
        if final_status:
            fs = final_status.lower()
            if 'reject' in fs or 'deni' in fs:
                decision = "Rejected"
            elif 'approv' in fs:
                decision = "Approved"

    info['decision'] = decision

    lines = [f"Decision: {decision}"]
    if info.get('claim_id'):
        lines.append(f"Claim ID: {info['claim_id']}")
    if info.get('policy_number'):
        lines.append(f"Policy: {info['policy_number']}")

    patient_parts = []
    if info.get('patient'):
        patient_parts.append(info['patient'])
    if info.get('age'):
        patient_parts.append(f"Age: {info['age']}")
    if info.get('gender'):
        patient_parts.append(f"Gender: {info['gender']}")
    if patient_parts:
        lines.append("Patient: " + " | ".join(patient_parts))

    if info.get('hospital'):
        hosp = info['hospital']
        if info.get('hospital_city'):
            hosp = f"{hosp} — {info['hospital_city']}"
        lines.append(f"Hospital: {hosp}")

    if info.get('claim_amount'):
        lines.append(f"Claimed: ₹{info['claim_amount']}")
    if info.get('approved_amount'):
        lines.append(f"Approved: ₹{info['approved_amount']}")

    if info.get('reason'):
        reason = info['reason'].strip()
        if len(reason) > 500:
            reason = reason[:500].rstrip() + "..."
        lines.append(f"Reason: {reason}")

    if decision.startswith("Approved"):
        lines.append("Outcome: Your claim has been accepted. Amount may differ due to deductions or sub-limits.")
    elif decision == "Rejected":
        lines.append("Outcome: Claim not approved. Please review reason and contact insurer if clarification needed.")
    else:
        lines.append("Outcome: Decision unclear. Please verify with insurer or upload the original PDF for manual review.")

    return "\n".join(lines)

# ------------------------------------------------------------
# Gen AI: translation + long-text summarization
# ------------------------------------------------------------
def auto_detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

_translation_models = {}

def get_translation_model(src_lang: str, tgt_lang: str):
    key = f"{src_lang}-{tgt_lang}"
    if key in _translation_models:
        return _translation_models[key]
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    _translation_models[key] = (tokenizer, model)
    return tokenizer, model

def translate_text(text: str, src_lang: str, tgt_lang: str, max_chunk_chars: int = 1500) -> str:
    try:
        tokenizer, model = get_translation_model(src_lang, tgt_lang)
    except Exception as e:
        print(f"Translation model for {src_lang}->{tgt_lang} not found ({e}). Using original text.")
        return text

    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return text

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chunk_chars, n)
        split = text.rfind(".", start, end)
        if split == -1 or split <= start + 100:
            split = end
        else:
            split += 1
        chunks.append(text[start:split].strip())
        start = split

    translated_chunks = []
    for c in chunks:
        if not c.strip():
            continue
        inputs = tokenizer(
            c,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        out_ids = model.generate(**inputs, max_length=512)
        translated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        translated_chunks.append(translated.strip())

    return " ".join(translated_chunks).strip()

def _summarize_chunk(chunk: str, max_length: int = 180, min_length: int = 50) -> str:
    if not chunk.strip():
        return ""
    result = summarizer(
        chunk,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return result[0]["summary_text"].strip()

def summarize_long_text(
    text: str,
    max_chunk_chars: int = 2500,
    chunk_summary_max_len: int = 180,
    chunk_summary_min_len: int = 50,
) -> str:
    if not text or not text.strip():
        return "⚠️ No text detected in the document."

    text = re.sub(r"\s+", " ", text.strip())

    if len(text) <= max_chunk_chars:
        return _summarize_chunk(
            text,
            max_length=chunk_summary_max_len,
            min_length=chunk_summary_min_len,
        )

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chunk_chars, n)
        split = text.rfind(".", start, end)
        if split == -1 or split <= start + 400:
            split = end
        else:
            split += 1
        chunk = text[start:split].strip()
        if chunk:
            chunks.append(chunk)
        start = split

    partial_summaries = []
    for i, c in enumerate(chunks, start=1):
        print(f"Summarizing chunk {i}/{len(chunks)} (len={len(c)} chars)")
        if len(c.split()) < 40:
            partial_summaries.append(c)
        else:
            s = _summarize_chunk(
                c,
                max_length=chunk_summary_max_len,
                min_length=chunk_summary_min_len,
            )
            partial_summaries.append(s)

    combined = " ".join(partial_summaries)
    combined = re.sub(r"\s+", " ", combined).strip()

    if len(combined) <= max_chunk_chars:
        final = _summarize_chunk(
            combined,
            max_length=chunk_summary_max_len,
            min_length=chunk_summary_min_len,
        )
        return final

    return combined

def generate_claim_summary(raw_text: str, back_translate: bool = True) -> str:
    if not raw_text or not raw_text.strip():
        return "⚠️ No text detected to summarize."

    src_lang = auto_detect_language(raw_text)
    print(f"Detected language for summary: {src_lang}")

    if src_lang != "en":
        text_en = translate_text(raw_text, src_lang, "en")
    else:
        text_en = raw_text

    summary_en = summarize_long_text(text_en)

    if back_translate and src_lang != "en":
        final_summary = translate_text(summary_en, "en", src_lang)
    else:
        final_summary = summary_en

    return final_summary.strip()

# ------------------------------------------------------------
# File text extraction (pdf / image / txt)
# ------------------------------------------------------------
def extract_text(file) -> str:
    suffix = Path(file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.read()); tmp.flush(); tmp.close()
    text = ""
    try:
        if suffix == ".pdf":
            with pdfplumber.open(tmp.name) as pdf:
                for p in pdf.pages:
                    page_text = p.extract_text() or ""
                    text += page_text + "\n"
        elif suffix in (".jpg", ".jpeg", ".png"):
            if ocr_model is None:
                raise RuntimeError("OCR model not available in this environment.")
            doc = DocumentFile.from_images(tmp.name)
            res = ocr_model(doc)
            try:
                for pg in res.pages:
                    try:
                        text += pg.get_text() + "\n"
                    except Exception:
                        pass
                if not text.strip():
                    text = res.render()
            except Exception:
                text = res.render()
        elif suffix == ".txt":
            with open(tmp.name, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        else:
            with open(tmp.name, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return clean_text(text)

# ------------------------------------------------------------
# Smart answer builder based on question intent
# ------------------------------------------------------------
def answer_question(question: str, info: dict, structured: str, nlp_summary: str) -> str:
    q = (question or "").lower()
    decision = info.get("decision") or (info.get("status") or "Unknown")
    claim_amt = info.get("claim_amount")
    approved_amt = info.get("approved_amount")
    reason = info.get("reason")
    patient = info.get("patient")
    hosp = info.get("hospital")
    hosp_city = info.get("hospital_city")

    # Status / decision
    if any(k in q for k in ["status", "approved", "rejected", "reject", "declined", "decision", "accept", "approve my claim"]):
        msg = []
        if decision and decision != "Unknown":
            msg.append(f"**Decision:** Your claim is **{decision}**.")
        elif info.get("status"):
            msg.append(f"**Status mentioned in report:** {info['status']}.")
        else:
            msg.append("I couldn't clearly find the final status in the report.")

        if approved_amt:
            msg.append(f"**Approved Amount:** ₹{approved_amt}.")
        if claim_amt and approved_amt:
            try:
                ca = float(claim_amt.replace(",", ""))
                aa = float(approved_amt.replace(",", ""))
                if aa < ca:
                    diff = ca - aa
                    msg.append(f"This is lower than the claimed amount of ₹{claim_amt} by about ₹{int(diff)}.")
            except Exception:
                pass

        if reason:
            msg.append(f"**Reason (short):** {reason}")

        return "\n\n".join(msg)

    # Amount
    if any(k in q for k in ["amount", "money", "how much", "rupees", "₹"]):
        parts = []
        if claim_amt:
            parts.append(f"🔹 **Claimed Amount:** ₹{claim_amt}")
        if approved_amt:
            parts.append(f"🔹 **Approved Amount:** ₹{approved_amt}")
        if claim_amt and approved_amt:
            try:
                ca = float(claim_amt.replace(",", ""))
                aa = float(approved_amt.replace(",", ""))
                if aa < ca:
                    diff = ca - aa
                    parts.append(f"➡ The approved amount is lower than claimed by about ₹{int(diff)} due to policy deductions.")
            except Exception:
                pass

        if not parts:
            parts.append("I couldn't find clear claimed or approved amounts in this report.")

        if decision and decision != "Unknown":
            parts.append(f"**Overall Decision:** {decision}")

        return "\n\n".join(parts)

    # Reason
    if any(k in q for k in ["why", "reason", "deduct", "reduction", "cut", "declined", "rejected"]):
        msg = []
        if reason:
            msg.append(f"**Main Reason from report:** {reason}")
        else:
            msg.append("The report does not contain a clearly labeled rejection/deduction reason section, or it is hard to detect.")

        if nlp_summary:
            msg.append("Here is a brief summary of the claim to give more context:\n\n" + nlp_summary)

        return "\n\n".join(msg)

    # Hospital / where
    if any(k in q for k in ["hospital", "where", "which hospital", "city", "location"]):
        lines = []
        if patient:
            lines.append(f"**Patient:** {patient}")
        if hosp:
            if hosp_city:
                lines.append(f"**Hospital:** {hosp} — {hosp_city}")
            else:
                lines.append(f"**Hospital:** {hosp}")
        if info.get("admission_date") or info.get("discharge_date"):
            ad = info.get("admission_date") or "N/A"
            dd = info.get("discharge_date") or "N/A"
            lines.append(f"**Hospitalization Period:** {ad} to {dd}")
        if not lines:
            lines.append("I couldn't clearly detect hospital and patient details from the report.")
        return "\n\n".join(lines)

    # Patient details
    if any(k in q for k in ["name", "patient", "age", "gender"]):
        lines = []
        if patient:
            lines.append(f"**Patient:** {patient}")
        if info.get("age"):
            lines.append(f"**Age:** {info['age']}")
        if info.get("gender"):
            lines.append(f"**Gender:** {info['gender']}")
        if not lines:
            lines.append("I couldn't clearly extract the patient details from this report.")
        return "\n\n".join(lines)

    # Explain / summary
    if any(k in q for k in ["explain", "simple words", "summary", "summarise", "summarize", "what is this claim", "tell me about this claim"]):
        out = []
        if decision and decision != "Unknown":
            out.append(f"**High-level decision:** Your claim is **{decision}**.")
        if nlp_summary:
            out.append("**In simple words, here's what the report says:**\n\n" + nlp_summary)
        else:
            out.append("I couldn't generate a clear AI summary, but here is the structured explanation:\n\n" + (structured or ""))
        return "\n\n".join(out)

    # Default
    out = []
    out.append("Here's an overview of your claim based on the uploaded report:")
    if structured:
        out.append("**Key points (structured):**\n```text\n" + structured + "\n```")
    if nlp_summary:
        out.append("**AI summary of the report:**\n```text\n" + nlp_summary + "\n```")
    return "\n\n".join(out)

# ------------------------------------------------------------
# Download summary helper (already added in previous step)
# ------------------------------------------------------------
def build_downloadable_summary(info, structured, nlp_summary):
    lines = ["===== CLAIM SUMMARY REPORT =====", ""]
    if info:
        for k, v in info.items():
            if v:
                lines.append(f"{k.replace('_',' ').title()}: {v}")
    lines.append("")
    lines.append("----- Structured Explanation -----")
    lines.append(structured or "Not available.")
    lines.append("")
    lines.append("----- AI Summary -----")
    lines.append(nlp_summary or "Not available.")
    return "\n".join(lines)

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! 👋 First, attach a claim report using the `➕ Upload` button above.\n\n"
                "Then ask me things like:\n"
                "- \"Is my claim approved or rejected?\"\n"
                "- \"How much amount is approved?\"\n"
                "- \"Why was my claim rejected?\"\n"
                "- \"Explain this claim in simple words.\""
            ),
        }
    ]

if "raw_text" not in st.session_state:
    st.session_state.raw_text = None
    st.session_state.file_name = None
    st.session_state.info = None
    st.session_state.structured_summary = None
    st.session_state.nlp_summary = None

if "show_download" not in st.session_state:
    st.session_state.show_download = False

# ------------------------------------------------------------
# Compact upload (like "+" attach) just above chat
# ------------------------------------------------------------
st.markdown('<div class="chat-upload-wrapper"><div class="chat-upload">', unsafe_allow_html=True)
st.markdown('<span class="chat-upload-label">➕ Upload claim report (PDF, image, or text)</span>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "",
    type=["pdf", "jpg", "jpeg", "png", "txt"],
    label_visibility="collapsed",
    help="Attach a claim document here."
)
st.markdown('</div></div>', unsafe_allow_html=True)

# When a new file is uploaded, process it once
if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        with st.spinner("Reading and analyzing the uploaded claim report…"):
            try:
                raw_text = extract_text(uploaded_file)
                info = extract_info(raw_text)
                structured_summary = quick_summary(raw_text, info)
                nlp_summary = generate_claim_summary(raw_text, back_translate=True)

                st.session_state.raw_text = raw_text
                st.session_state.info = info
                st.session_state.structured_summary = structured_summary
                st.session_state.nlp_summary = nlp_summary
                st.session_state.file_name = uploaded_file.name
                st.session_state.show_download = False

            except Exception as e:
                st.error(f"Failed to process file: {e}")
                st.session_state.raw_text = None
                st.session_state.info = None
                st.session_state.structured_summary = None
                st.session_state.nlp_summary = None
                st.session_state.file_name = None
                st.session_state.show_download = False

# ------------------------------------------------------------
# Chat UI
# ------------------------------------------------------------
st.markdown('<p class="chat-helper">💬 Ask anything about the uploaded claim below.</p>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Type your question about this claim...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        lower_q = user_prompt.lower()

        # Download trigger
        if "download" in lower_q:
            if st.session_state.raw_text is None:
                reply = (
                    "I don't see any claim document yet. "
                    "Please upload a report first, then ask to download the summary."
                )
            else:
                st.session_state.show_download = True
                reply = "✅ You can download your claim summary using the button below."
        else:
            if st.session_state.raw_text is None:
                reply = (
                    "I don't see any claim document yet. "
                    "Please attach a PDF / image / text report using the `➕ Upload` button above, then ask again."
                )
            else:
                reply = answer_question(
                    user_prompt,
                    st.session_state.info or {},
                    st.session_state.structured_summary or "",
                    st.session_state.nlp_summary or "",
                )

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# Download button
if st.session_state.show_download and st.session_state.raw_text:
    download_text = build_downloadable_summary(
        st.session_state.info or {},
        st.session_state.structured_summary or "",
        st.session_state.nlp_summary or "",
    )
    st.download_button(
        label="⬇️ Download Claim Summary",
        data=download_text,
        file_name="claim_summary.txt",
        mime="text/plain",
    )

st.markdown('<div class="footer">Claim Explanation Chatbot · OCR + NLP powered.</div>', unsafe_allow_html=True)

# ============================================================
# End of file
# ============================================================
