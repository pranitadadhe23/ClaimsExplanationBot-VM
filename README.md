# 🧾 Claim Explanation Chatbot (OCR + NLP Powered)

An AI-powered Insurance Claim Explanation Chatbot built using Streamlit, OCR, and Natural Language Processing (NLP).  
This application allows users to upload insurance claim documents (PDF, Image, or Text) and ask questions in natural language about:

- Claim approval or rejection  
- Approved amount  
- Reasons for rejection  
- Patient and hospital details  
- Simple claim summary  
- Downloadable claim summary report  

---

## 🚀 Features

- ✅ Upload claim reports in PDF, JPG, PNG, or TXT  
- ✅ Automatic OCR for scanned documents  
- ✅ AI-based summarization using Transformers  
- ✅ Natural language Q&A (Ask anything about the claim)  
- ✅ Structured claim extraction  
- ✅ Downloadable claim summary  
- ✅ Clean ChatGPT-style user interface  
- ✅ Ready for Streamlit Cloud deployment  

---

## 🛠️ Tech Stack

- Frontend: Streamlit  
- Backend: Python  
- OCR: DocTR  
- NLP & AI: HuggingFace Transformers  
- PDF Processing: pdfplumber  
- Translation & Language Detection: langdetect  
- Deep Learning: PyTorch  

---

## 📁 Project Structure

claim-explainer-chatbot/  
│  
├── claim_explainer_chatbot_app.py  
├── requirements.txt  
├── packages.txt  
├── test_files/  
├── .gitignore  
└── README.md  


---

## ⚙️ Installation & Setup
### Step 1: Clone the Repository
```bash
git clone https://github.com/pranitadadhe23/ClaimsExplanationBot-VM
cd claim-explainer-chatbot
``` 

### Step 2: Create Virtual Environment (Recommended)
``` python -m venv venv ``` 

# Windows
```bash venv\Scripts\activate``` 

# macOS/Linux
```bash source venv/bin/activate``` 

### Step 3: Install All Dependencies
```bash pip install -r requirements.txt``` 

### Step 4: Run the Application
streamlit run claim_explainer_chatbot_app.py

## 📄   How to Use the Application
1. Launch the app in your browser
2. Click ➕ Upload and select a claim report
3. Ask questions like:
  Is my claim approved or rejected?
  How much amount is approved?
  Why was my claim rejected?
  Explain this claim in simple words
4. To download the report:
  Type Download summary
  Click the Download Claim Summary button

---

## ☁️ Streamlit Cloud Deployment
This project is ready for Streamlit Deployment.
Steps:
1. Push this repo to GitHub ✔️
2. Visit → https://share.streamlit.io
3. Connect your GitHub account
4. Select this repository
5. Set main file as:
  claim_explainer_chatbot_app.py
Click Deploy ✅
---

### 🎯 Use Cases
🏥 Hospitals & Billing Departments
🧾 Insurance Companies
🧑‍💼 Third Party Administrators (TPA)
📞 Customer Support Automation
🎓 Academic & Demonstration Projects
---
### 🔐 Data Privacy & Security
Uploaded documents are processed in runtime memory only
No permanent file storage
No user data retention
Safe for demo and educational use
### 👩‍💻 Developer
Pranita Dadhe & Sakshi Parate
Final Year Engineering Student
Project: Claims Explanation Bot  
---

### GitHub Profile:
https://github.com/pranitadadhe23 , https://github.com/SakshiParate27
---

### 📜 License
This project is developed for educational and demonstration purposes.
Feel free to use, modify, and share with proper credit.


