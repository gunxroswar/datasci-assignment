# %%
import requests
from bs4 import BeautifulSoup
import re
import json

# %%
url_list = []

with open("forum_links.txt", "r", encoding="utf-8") as f:
    thread_links = f.read().splitlines()
    url_list.extend(thread_links)




# %%
def web_scraper(url):
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # subject จาก meta name="disease"
    subject = soup.find("meta", attrs={"name": "disease"})
    subject = subject["content"].strip() if subject else ""

    # gender และ age จาก meta name="publisher"
    publisher = soup.find("meta", attrs={"name": "publisher"})
    gender = age = ""
    if publisher:
        pub_text = publisher["content"]
        gender_match = re.search(r"เพศ:\s*(ชาย|หญิง)", pub_text)
        age_match = re.search(r"อายุ:\s*(\d+)", pub_text)
        gender = gender_match.group(1) if gender_match else ""
        age = age_match.group(1) if age_match else ""

    # symptoms จาก meta name="keywords"
    keywords = soup.find("meta", attrs={"name": "keywords"})
    symptoms = []
    if keywords:
        symptoms = [kw.strip() for kw in keywords["content"].split(",")]

    # question จาก <span class="font-bold text-lg">
    question_tag = soup.find("span", class_="font-bold text-lg")
    question = question_tag.get_text(strip=True) if question_tag else ""

    # answer จาก <p class="mt-4">
    answer_tag = soup.find("p", class_="mt-4")
    answer = answer_tag.get_text(strip=True) if answer_tag else ""

    # สร้าง dictionary ตามที่คุณต้องการ
    data = {
        "subject": subject,
        "gender": gender.strip(),
        "age": age.strip(),
        "symptoms": symptoms,
        "question": question,
        "answer": answer
    }

    return data

# %%
json_list = []

for url in url_list:
    json_list.append(web_scraper(url))


# เซฟทั้งหมดเป็นไฟล์เดียว
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(json_list, f, ensure_ascii=False, indent=2)

# %%


