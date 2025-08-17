# Datasci-assignment
This is the assignment for Data scientist application which belong to Thirachai Ngaoju

# Intruction Web Scraper

1. เตรียม url ของ web ที่ต้องการจะ scrape ในทีนี้ผมใช้ get_links.ipynb รับ base url มาก่อน ในที่นี้คือ https://www.agnoshealth.com/forums/search?page=" จากนั้นบันทึก urls ที่ได้ลงไฟล์ .txt
2. ในไฟล์ scraper.py อ่าน urls จากไฟล์ .txt จากนั้นก็ for-loop รันฟังก์ชั่น web_scraper เก็บผลลัพธ์เป็นลิสต์ของ JSON แล้วเซฟผลลัพธ์เป็นไฟล์ .JSON

   
# Intruction RAG
1. ในการเรียกใช้ rag_chain ให้ใช้ rag_chain.invoke("Input ที่ต้องการใช้")
2. input ที่ rag_chain รับจะอยู่ในรูปแบบ JSON โดยมีข้อมูลดังนี้
  {    
      "subject": "...",
      "gender": "...",
      "age": "...",
      "symptoms": "...",
      "question": "..."
  }
  ซึ่งเป็นรูปแบบที่ผมดูมาจากกระทู้ถามคำถามในเว็บ agnos
  โดยจะใส่ข้อมูลครบหรือไม่ครบก็ได้ เช่นใส่แค่ส่วนของ question อย่างเดียวก็สามารถเรียกใช้ rag ได้
  แต่หากใส่แค่ข้อมูลที่ไม่น่าจะเอาไปวินิจฉัยได้มากเช่นใส่แค่เพศกับอายุ แล้วไม่ได้ใส่ในส่วนของหัวข้อ อาการ หรือคำถาม ณ ตอนนี้ตัวโมเดลเหมือนจะคิดไปเองว่าคนนี้เป็นโรคอะไร
3. หากมีลิสต์ที่ input ไว้หลาย ๆ ตัว ก็สามารถวนลูปเพื่อให้ rag รับ Input เหล่านั้นทีละตัวได้ 

# Instruction: Streamlit Chatbot
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

อาจจะต้องรอ streamlit ไปรัน rag_chain.py 
