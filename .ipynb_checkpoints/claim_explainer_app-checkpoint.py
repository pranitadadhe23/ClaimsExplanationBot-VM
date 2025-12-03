# ============================================================
# ⚡ Insurance Claim Explainer — Fast, Clean & Clear (One-Click)
# ============================================================
# Simplified for instant, understandable explanations:
# - One button only — fast summarization & decision extraction
# - Removes white box / preview confusion
# - Outputs clear decision: Approved / Rejected / Reduced + reason
# - Uses lightweight extraction (no multi-pass delays)
# ============================================================

import re, os, io, tempfile
from pathlib import Path
import streamlit as st
import pdfplumber
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# ------------------------------------------------------------
# 🎨 Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Insurance Claim Explainer", page_icon="🧾", layout="centered")

st.markdown("""
<style>
.app-title{text-align:center;font-size:36px;color:#123b5a;font-weight:800;margin-bottom:4px}
.app-sub{text-align:center;color:#52606d;margin-bottom:20px}
.card{background:#fff;border:1px solid #e6edf5;border-radius:16px;padding:20px;box-shadow:0 5px 16px rgba(0,0,0,0.05)}
.result{background:#f7fafc;border:1px solid #e6edf5;border-radius:14px;padding:18px;margin-top:16px;white-space:pre-wrap;font-size:15px;color:#0f1a2d}
.button{display:flex;justify-content:center;margin-top:10px}
.footer{text-align:center;color:#738093;font-size:12px;margin-top:20px}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">🧾 Claim Explanation</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Instant, clear summary showing claim approval or rejection status.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# ⚙️ OCR Loader (cached)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ocr():
    return ocr_predictor(pretrained=True)

o = load_ocr()

# ------------------------------------------------------------
# 🧠 Helper Functions
# ------------------------------------------------------------
def clean_text(t:str)->str:
    return re.sub(r"\s+"," ",t or "").strip()

def extract_text(file)->str:
    suffix=Path(file.name).suffix.lower()
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=suffix)
    tmp.write(file.read());tmp.flush();tmp.close()
    text=""
    try:
        if suffix==".pdf":
            with pdfplumber.open(tmp.name) as pdf:
                for p in pdf.pages:
                    text+=p.extract_text() or ""
        elif suffix in (".jpg",".jpeg",".png"):
            doc=DocumentFile.from_images(tmp.name)
            res=o(doc);text=res.render()
        elif suffix==".txt":
            text=open(tmp.name,encoding="utf-8",errors="ignore").read()
    finally:
        os.unlink(tmp.name)
    return clean_text(text)


def extract_info(txt:str)->dict:
    info={}
    def f(p):
        m=re.search(p,txt,flags=re.I)
        return m.group(1).strip() if m else None
    info['claim_id']=f(r"claim\s*id\s*[:#]\s*([A-Za-z0-9-]+)")
    info['patient']=f(r"patient\s*name\s*[:]\s*([A-Za-z .'-]+)")
    info['hospital']=f(r"hospital\s*(?:name)?\s*[:]\s*([A-Za-z0-9 .'-,&]+)")
    info['claim_amount']=f(r"claim\s*amount\s*[:₹]\s*([0-9,]+)")
    info['approved_amount']=f(r"approved\s*amount\s*[:₹]\s*([0-9,]+)")
    status=f(r"claim\s*status\s*[:]\s*([A-Z /-]+)")
    if not status: status=f(r"(approved with reduction|partially approved|approved|denied|rejected)")
    info['status']=status
    info['reason']=f(r"reason\s*[:]\s*(.+)") or f(r"remarks\s*[:]\s*(.+)")
    return info


def quick_summary(txt:str,info:dict)->str:
    s=txt.lower()
    decision="Unknown"
    if "approved" in (info.get('status') or '').lower():
        if "reduction" in (info.get('status') or '').lower():
            decision="Approved with Reduction"
        else:
            decision="Approved"
    elif "denied" in (info.get('status') or '').lower() or "rejected" in (info.get('status') or '').lower():
        decision="Rejected"
    elif "approved" in s: decision="Approved"
    elif "denied" in s or "rejected" in s: decision="Rejected"

    lines=[f"Decision: {decision}"]
    if info.get('claim_id'): lines.append(f"Claim ID: {info['claim_id']}")
    if info.get('patient'): lines.append(f"Patient: {info['patient']}")
    if info.get('hospital'): lines.append(f"Hospital: {info['hospital']}")
    if info.get('claim_amount'): lines.append(f"Claimed: ₹{info['claim_amount']}")
    if info.get('approved_amount'): lines.append(f"Approved: ₹{info['approved_amount']}")
    if info.get('reason'): lines.append(f"Reason: {info['reason']}")

    # Add contextual suggestion
    if decision.startswith("Approved"):
        lines.append("Outcome: Your claim has been accepted. Amount may differ due to deductions or sub-limits.")
    elif decision=="Rejected":
        lines.append("Outcome: Claim not approved. Please review reason and contact insurer if clarification needed.")
    else:
        lines.append("Outcome: Decision unclear. Please verify with insurer.")

    return "\n".join(lines)

# ------------------------------------------------------------
# 🖥️ UI
# ------------------------------------------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    file = st.file_uploader("Upload claim report (.pdf, .jpg, .jpeg, .png, .txt)", type=["pdf","jpg","jpeg","png","txt"])
    go = st.button("⚡ Summarize Claim", type="primary", disabled=not file)
    st.markdown('</div>', unsafe_allow_html=True)

if go and file:
    with st.spinner("Processing and summarizing…"):
        text=extract_text(file)
        info=extract_info(text)
        summary=quick_summary(text,info)
    st.markdown(f'<div class="result">{summary}</div>', unsafe_allow_html=True)
    st.download_button("⬇️ Download Explanation", data=summary.encode('utf-8'), file_name="claim_summary.txt", mime="text/plain")

st.markdown('<div class="footer">Simple. Fast. Understandable — instant claim explanations.</div>', unsafe_allow_html=True)
