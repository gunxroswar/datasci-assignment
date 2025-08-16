# %%
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

import json
from langchain.docstore.document import Document


# %% [markdown]
# # Data Preparation

# %% [markdown]
# ## Import scraped data from scraper

# %%
# from scraper import json_list

# %%
# json_list[:5]

# %%
# # เซฟทั้งหมดเป็นไฟล์เดียว
# with open("data.json", "w", encoding="utf-8") as f:
#     json.dump(json_list, f, ensure_ascii=False, indent=2)


# %% [markdown]
# ## อ่านข้อมูลจาก json มาเก็บในลิสต์

# %%
json_list_x = []
with open("data.json", "r", encoding="utf-8") as f:
    json_list_x = json.load(f)


# %% [markdown]
# ## Change json_list to Document

# %%
documents = []
for item in json_list_x:
    content = f"subject: {item['subject']}\n gender: {item['gender']}\n age: {item['age']}\n symptoms: {', '.join(item['symptoms'])}\n question: {item['question']}\n answer: {item['answer']}"
    documents.append(Document(page_content=content, metadata={"source": "scraper"}))



# %% [markdown]
# ## Embeding and Store to VectorDB

# %%
embedder = OllamaEmbeddings(model="bge-m3")


# %%
# vector_store = FAISS.from_documents(documents, embedder)

vector_store = FAISS.load_local("D:/faiss_storage", 
                                embedder,
                                allow_dangerous_deserialization=True
)
print("Loaded vector store from disk")

# %%
retriever = vector_store.as_retriever()



# %% [markdown]
# # Create chat

# %%
llm = ChatOllama(model="scb10x/llama3.1-typhoon2-8b-instruct",
                    temperature=0.4)


# %%


# %%
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "คุณคือผู้แพทย์เชี่ยวชาญด้านสุขภาพที่มีความรู้ลึกซึ้งทั้งจากประสบการณ์และฐานข้อมูลทางการแพทย์ "
     "คุณจะต้องให้คำแนะนำที่แม่นยำและปลอดภัย โดยเริ่มต้นด้วยประโยค 'คำตอบจากแพทย์ผู้เชี่ยวชาญ' "
     "จากนั้นให้คำตอบแบ่งเป็นหัวข้อ: สาเหตุที่เป็นไปได้, แนวทางการดูแลเบื้องต้น, คำแนะนำเพิ่มเติม "
     " ใส่ข้อความ 'คนไข้สามารถค้นหาแพทย์เฉพาะทางที่ต้องการและทำนัดหมายได้ทางเว็บไซต์ของเราที่ลิ้งค์นี้ \nhttps://bit.ly/doctor-refer-agnos' ในช่วงท้ายของการสนทนา"),
    
    ("human", 
     "มีผู้ป่วยที่มีข้อมูลดังนี้:\n"
     "- หัวข้อ: {subject}\n"
     "- เพศ: {gender}\n"
     "- อายุ: {age} ปี\n"
     "- อาการ: {symptoms}\n"
     "- คำถาม: {question}\n"
     "ข้อมูลจากฐานข้อมูลที่เกี่ยวข้อง:\n{context}\n"
     "กรุณาวิเคราะห์และให้คำตอบอย่างละเอียด โดยอ้างอิงจากทั้งความรู้ของคุณและข้อมูลที่ให้มาในส่วนของ context อย่างรอบคอบ"
     "หากข้อมูลใน context ไม่เพียงพอ ให้ใช้ความรู้ทั่วไปของคุณในการตอบคำถามอย่างระมัดระวังและแนะนำให้ผู้ป่วยไปพบแพทย์ผู้เชี่ยวชาญ"
     )
])

# %%
from langchain.schema.runnable import RunnableLambda

def fill_missing_fields(inputs):
    return {
        "subject": inputs.get("subject", "ไม่ระบุ"),
        "gender": inputs.get("gender", "ไม่ระบุ"),
        "age": inputs.get("age", "ไม่ระบุ"),
        "symptoms": inputs.get("symptoms", "ไม่ระบุ"),
        "question": inputs.get("question", "ไม่ระบุ"),
        "context": inputs.get("context", "ไม่มีข้อมูลจากฐานข้อมูล")
    }

input_preprocessor = RunnableLambda(fill_missing_fields)


# %%
rag_chain = (
    {
        "context": retriever,
        "subject": RunnablePassthrough(),
        "gender": RunnablePassthrough(),
        "age": RunnablePassthrough(),
        "symptoms": RunnablePassthrough(),
        "question": RunnablePassthrough(),

    }
    | input_preprocessor
    | prompt
    | llm
    | StrOutputParser()
)

# %%
data_input =[ {
    "subject": "อาการปวดหัวเรื้อรัง",
    "gender": "หญิง",
    "age": "35",
    "symptoms": "ปวดหัวบริเวณขมับทั้งสองข้างมาเกือบทุกวัน โดยเฉพาะช่วงบ่าย รู้สึกตึงๆ และมีอาการคลื่นไส้ร่วมด้วย",
    "question": "อาการแบบนี้เกิดจากอะไร และควรดูแลตัวเองอย่างไร?"
},
{
    "subject": "การนอนหลับในผู้สูงอายุ",
    "gender": "ชาย",
    "age": "68",
    "symptoms": "นอนหลับไม่สนิท ตื่นบ่อยตอนกลางคืน และรู้สึกไม่สดชื่นตอนเช้า",
    "question": "ควรปรับพฤติกรรมหรือมีวิธีดูแลสุขภาพอย่างไรให้หลับดีขึ้น?"
},
{
    "subject": "ภาวะขาดวิตามินบี 12 (Vitamin B12 deficiency)",
    "gender": "หญิง",
    "age": "29",
    "symptoms": [
      "เหนื่อยง่าย",
      "มือเท้าชา",
      "เวียนหัว",
      "ลืมง่าย",
      "อ่อนแรง"
    ],
    "question": "ช่วงนี้รู้สึกเหนื่อยง่ายมากค่ะ เดินนิดเดียวก็หอบ มือเท้าชาบ่อย ๆ โดยเฉพาะตอนตื่นนอน รู้สึกเวียนหัวและลืมง่ายขึ้นเรื่อย ๆ อยากทราบว่าอาการแบบนี้เกิดจากอะไร และควรตรวจอะไรบ้างคะ?"
  },
  {
    "subject": "โรคกรดไหลย้อน (GERD)",
    "gender": "ชาย",
    "age": "34",
    "symptoms": [
      "แสบร้อนกลางอก",
      "เรอเปรี้ยว",
      "ไอเรื้อรัง",
      "แน่นหน้าอก",
      "กลืนลำบาก"
    ],
    "question": "ผมมีอาการแสบร้อนกลางอกหลังทานอาหาร โดยเฉพาะอาหารเผ็ดหรือมัน ๆ เรอเปรี้ยวบ่อยมาก และบางครั้งรู้สึกแน่นหน้าอกจนหายใจไม่สะดวก อยากทราบว่าเป็นกรดไหลย้อนหรือเปล่า และควรปรับพฤติกรรมยังไงครับ?"
  },
  {
    "subject": "ภาวะถุงน้ำรังไข่หลายใบ (PCOS)",
    "gender": "หญิง",
    "age": "26",
    "symptoms": [
      "ประจำเดือนมาไม่ปกติ",
      "สิวขึ้นเยอะ",
      "ขนดก",
      "น้ำหนักขึ้นง่าย",
      "ปวดท้องน้อย"
    ],
    "question": "ประจำเดือนมาไม่ตรงเวลาเลยค่ะ บางเดือนก็ไม่มาเลย สิวขึ้นเยอะมากทั้งที่ดูแลผิวดีแล้ว ขนตามตัวก็ดูเยอะขึ้น น้ำหนักขึ้นง่ายมากทั้งที่กินเท่าเดิม อยากทราบว่าอาการแบบนี้เกี่ยวกับฮอร์โมนหรือเปล่า และควรไปตรวจอะไรบ้างคะ?"
  },
  {
    "subject": "โรคภูมิแพ้อากาศ (Allergic Rhinitis)",
    "gender": "ชาย",
    "age": "18",
    "symptoms": [
      "จามบ่อย",
      "คัดจมูก",
      "น้ำมูกไหล",
      "คันจมูก",
      "ตาแดง"
    ],
    "question": "ผมมีอาการจามบ่อยมาก โดยเฉพาะตอนเช้า ๆ หรือเวลาอยู่ในห้องแอร์ น้ำมูกไหล คัดจมูก และบางครั้งตาแดงคันร่วมด้วย อยากทราบว่าเป็นภูมิแพ้อากาศหรือเปล่า และมีวิธีดูแลตัวเองยังไงบ้างครับ?"
  }
]

# %%


# %%
# for i, data in enumerate(data_input):
#     result = rag_chain.invoke(data)
#     print(f"หัวข้อที่ {i+1}\n")
#     print(result)




# %%
# my_question = rag_chain.invoke({    
#     "subject": "กรดไหลย้อน อาหารไม่ย่อย",
#     "gender": "ชาย",
#     "age": "24",
#     "symptoms": "แสบอก, แสบคอ, ท้องอืด, อาหารไม่ย่อย, จุกคอ",
#     "question": "มีคืนนึงตอนผมกำลังจะนอนจู่ๆผมก็รู้สึกแสบคอขึ้นมา แล้วพอตื่นมาก็รู้สึกว่าเหมือนมีก้อนอะไรมาจุกที่คอ ผมเลยค้นหาจากอินเตอร์เน็ตก็พบว่าผมเป็นกรดไหลย้อนจึงไปซื้อยาที่ร้านขายยามากินเป็นเวลา 1 อาทิตย์ จนพอผมเริ่มรู้สึกดีขึ้นผมจึงไปกินบุฟเฟ่ต์ชาบูแล้วทีนี้ผมรู้สึกว่าอาหารมันไม่ย่อยเลยแม้แต่ตอนกำลังจะนอนก็รู้สึกได้ว่ามีอาหารอยู่ในท้องจนถึงวัดถัดไป ควรรักษาต่อยังไงดีครับ"
# })

# print(my_question)

# # %%
# my_question2 = rag_chain.invoke({    
#     "question": "สวัสดีครับคือช่วงนี้ผมนอนไม่ค่อยหลับเวลานอนมันก็จะคิดฟุ้งซ่านตลอดเวลาหรือบางทีก็มีเพลงเล่นในหัวตลอดเวลาเลยทำให้นอนไม่พอ พอจะมีวิธีไหนช่วยจัดการไหมครับ"
# })

# print(my_question2)

# # %%
# my_question3 = rag_chain.invoke({    
#     "subject": "เข่าดังก๊อกแก๊กเวลาเดิน",
#     "symptoms": "เข่าดังก๊อกแก๊กเวลาเดิน, ปวดเข่า",
#     })
# print(my_question3)

# # %%
# my_question4 = rag_chain.invoke({    
#     "question": "รู้สึกเดินๆอยู่แล้วเข่าอ่อนบ่อยๆ ไม่ก็ข้อเท้าเกือบจะพลิกบ่อยๆ มันคืออาการอะไรครับ"
# })
# print(my_question4)

# %%



