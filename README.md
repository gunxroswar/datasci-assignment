# Datasci-assignment
This is the assignment for Data scientist application which belong to Thirachai Ngaoju

# Task 1
# Intruction Web Scraper

1. เตรียม url ของ web ที่ต้องการจะ scrape ในทีนี้ผมใช้ get_links.ipynb รับ base url มาก่อน ในที่นี้คือ "https://www.agnoshealth.com/forums/search?page=" จากนั้นบันทึก urls ที่ได้ลงไฟล์ .txt
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

# Task 2
เนื่องจากใน Dataset ไม่ได้มี label ว่าถ้ามีอาการแบบนี้สรุปแล้วแพทย์ตัดสินว่าเป็นโรคอะไร ผมจึงเลือกใช้วิธีแบบ Task 1 คือใช้ RAG เพื่อส่งprompt ไปถาม LLM ว่าถ้ามีอาการแบบนี้เป็นโรคอะไร
โดยอย่างแรกผสมรวบรวมข้อมูลในแต่ละคอลัมน์มารวมเป็นประโยคแล้วสร้างคอลัมน์ใหม่เพื่อประโยคนั้นไว้ ซึ่งผมใช้คอลัมน์ gender, age, summary มารวมไว้เป็นประโยคแบบนี้
ผู้ป่วยเพศ ชาย อายุ 28 มาด้วยโรค: ไม่ระบุ, อาการที่มี: เสมหะ (ลักษณะ เสมหะเปลี่ยนสีเหลือง/เขียว), ไอ (ระยะเวลา ไม่เกิน 1 สัปดาห์ (ไม่เกิน 7 วัน)), การรักษาก่อนหน้า (การรักษาก่อนหน้า ไม่เคย), อาการที่ไม่มี: ไม่ระบุ, อาการที่ไม่แน่ใจ: ไม่ระบุ

จากนั้นแปลงข้อมูลให้อยู่ในรูปแบบ Langchain Document
จากนั้นก็ทำ rag แบบ Task 1 เลย แต่มีการปรับเปลี่ยน Prompt ให้ LLM ตอบมาเป็นโรคที่น่าจะเป็น 3 โรคพร้อมคำอธิบายสั้นๆ

แล้วก็Task นี้ผมลองใช้เพิ่มอีกหนึ่งโมเดลคือ aaditya/OpenBioLLM-Llama3-8B-GGUF ที่เป็นโมเดลที่ใช้ในทางการแพทย์แต่ไม่ได้ระบุว่ารองรับภาษาไทย เพื่อเอาเปรียบเทียบกับ scb10x/llama3.1-typhoon2-8b-instruct ที่เป็นโมเดลที่ fine tune มาให้รองรับภาษาไทยได้ดี 

ซึ่งผลลัพธ์ก็ไม่ได้แตกต่างกันมาก ตัว OpenBioLLM-Llama3-8B-GGUF ก็เจนภาษาไทยออกมาได้ดีแม้ว่าเขาจะไม่ได้ระบุว่ารองรับภาษาไทย

note: หากใน dataset มี label ไว้ ผมน่าจะลองใช้เป็น classification model เพื่อทำนายโรคที่น่าจะเป็นจากอาการที่ระบุมา
