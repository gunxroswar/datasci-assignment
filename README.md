# Datasci-assignment
This is the test for Data scientist application which belong to Thirachai Ngaoju

# Instruction: RAG + Streamlit Chatbot
1. เตรียม Environment

ไปที่โฟลเดอร์โปรเจค เช่น D:\CMU\ปี4\Project\datasci-assignment\

สร้าง virtual environment ใหม่ด้วยคำสั่ง
python -m venv venv

เปิดใช้งาน environment (Windows PowerShell)
venv\Scripts\activate

2. ติดตั้ง Dependencies

สร้างไฟล์ชื่อ requirements.txt ใส่เนื้อหาดังนี้

streamlit
langchain
langchain-community
langchain-ollama
faiss-cpu
chromadb
sentence-transformers

ติดตั้งทั้งหมดด้วยคำสั่ง
pip install -r requirements.txt

รัน Streamlit App

3. ใช้คำสั่ง
streamlit run appchat_ui.py

เปิด browser ที่ http://localhost:8501

อาจจะต้องรอ streamlit ไปรัน rag_chain.py ถ้าไม่อยากรอนานไปคอมเมนต์โค้ดในส่วนทดสอบการใช้งาน rag ก่อน