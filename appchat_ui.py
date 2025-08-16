from rag_chain import rag_chain
import streamlit as st

st.title("แชทบอทให้คำปรึกษาสุขภาพ")

if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการสนทนาเก่า
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ถ้ามี user พิมพ์ prompt ใหม่
if prompt := st.chat_input("พิมพ์คำถามหรือข้อมูลผู้ป่วยที่ต้องการให้แชทบอทตอบ"):
    # เก็บ user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # เรียก rag_chain โดยส่งเป็น dict
    with st.chat_message("assistant"):
        response = rag_chain.invoke({"question": prompt})
        st.markdown(response)

    # เก็บ assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
