import os
import logging

from libs.Chroma import ChromaManager
from libs.ChatBot import ChatBot
from config import ENV, CHROMADB_PATH, CSV_SOURCE, CSV_NER_PATH, HFACE_TOKEN


db = ChromaManager(use_local_fs=ENV=="colab", path=CHROMADB_PATH)

bot = ChatBot( db=db, name="Rick Sanchez", hface_token=HFACE_TOKEN)

intro = bot.welcome( lng="es")
print(intro)

esp_response = bot.ask(collection="cve", question="Generar una lista de vulnerabilidades en kernel de linux", k=10)
fr_response = bot.ask(collection="cve", question="Générer une liste de vulnérabilités dans le noyau Linux", k=10)
en_response = bot.ask(collection="cve", question="Make a list of linux kernel vulnerabilities", k=10)
esp_response = bot.ask(collection="cve", question="Necesito que me des informacion sobre esta vulnerabilidad <_ID_>CVE-2017-11302</_ID_/>", k=10)

for chat in bot.history:
    print("-------------------------------------------------")
    i = chat["idiom"]
    q = chat["question"]
    a = chat["response"]

    print(f"Idiom: {i}")
    print(f"Question: {q}")
    print(f"Answer: {a}")
    print("-------------------------------------------------")

while True:

    q = input("You: ")

    if q == "exit":
        break

    r = bot.ask(collection="cve", question=q, k=10)
    last_dialog = bot.history[-1]

    i = last_dialog["idiom"]
    q = last_dialog["question"]
    a = last_dialog["response"]

    print(f"Idiom Detected: {i}")
    print(f"Question : {q}")
    print(f"Answer: {a}")
    print("-------------------------------------------------")