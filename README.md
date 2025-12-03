# 🧾 Claim Explanation Chatbot (OCR + NLP Powered) 

## An AI-powered Insurance Claim Explanation Chatbot built using Streamlit, OCR, and Natural Language Processing (NLP). 
This application allows users to upload insurance claim documents (PDF, Image, or Text) and ask questions in natural language about: 
Claim approval or rejection  
Approved amount  
Reasons for rejection  
Patient and hospital details  
Simple claim summary  
Downloadable claim summary report  

🚀 Features 
✅ Upload claim reports in PDF, JPG, PNG, or TXT 
✅ Automatic OCR for scanned documents 
✅ AI-based summarization using Transformers 
✅ Natural language Q&A (Ask anything about the claim) 
✅ Structured claim extraction 
✅ Downloadable claim summary 
✅ Smart detection of: 
  Claim Status 
  Approved Amount 
  Rejection Reason 
  Patient Details 
  Hospital Details 
✅ Clean ChatGPT-style user interface 
✅ Fully deployable on Streamlit Cloud

🛠️ Tech Stack 
Frontend: Streamlit 
Backend: Python
OCR: DocTR
NLP & AI: HuggingFace Transformers
PDF Processing: pdfplumber
Translation & Language Detection: langdetect
Deep Learning: PyTorch

📁 Project Structure 
claim-explainer-chatbot/ 
│ 
├── claim_explainer_chatbot_app.py 
├── requirements.txt 
├── packages.txt 
├── test_files/ 
├── .gitignore 
└── README.md 

⚙️ Installation & Setup 
1️⃣ Clone the Repository 
``` git clone https://github.com/pranitadadhe23/ClaimsExplanationBot-VM  
cd claim-explainer-chatbot ```

2️⃣ Create Virtual Environment (Recommended) 
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run claim_explainer_chatbot_app.py

📄 How to Use

Launch the app.

Click ➕ Upload to upload a claim report.

Ask questions like:

Is my claim approved or rejected?

How much amount is approved?

Why was my claim rejected?

Explain this claim in simple words.

To download the summary:

Type Download summary

Click the Download button.

🌐 Deployment (Streamlit Cloud Ready)

You already have:

✅ requirements.txt

✅ packages.txt

✅ Main app file

Steps:

Push this repo to GitHub

Go to 👉 https://share.streamlit.io

Connect your GitHub

Select this repository

Set main file:

claim_explainer_chatbot_app.py


Click Deploy ✅

🎯 Use Cases

Insurance Companies

Hospital Billing Departments

Third-party Administrators (TPA)

Customer Support Automation

Digital Insurance Claim Portals

🔐 Privacy & Security Note

This application:

Does not store uploaded files permanently

Processes documents only in memory

Is meant for educational and demo purposes

👩‍💻 Developed By

Pranita Dadhe
Final Year Engineering Student
Project: AI-Based Insurance Claim Explanation System

📜 License

This project is for educational and demonstration purposes only.
You may modify and use it with proper credit.


