**🤖 Hosla Public Chatbot**

Welcome to the Hosla Public Chatbot – an intelligent FAQ assistant for Hosla users and guests. This chatbot helps users get answers to frequently asked questions about Hosla, view upcoming events, and log new queries not found in the system. Built using Python, NLP, and web scraping.

✨ Features
✅ Semantic matching of user queries using BERT-based Sentence Transformers

🔒 Guest access with restricted features

📅 Event fetcher from the official Hosla Events Page

🧠 Learns over time – logs unknown questions automatically

📂 Saves user interactions in timestamped logs

☎️ Unknown questions prompt contact with Hosla (clickable call link)

🛠️ Tech Stack
Python 3.9+

SentenceTransformers (all-MiniLM-L6-v2)

pandas

requests

BeautifulSoup (bs4)

urllib3

🚀 How It Works
Loads a dataset of Questions and Answers from a CSV file (FAQ - Sheet1.csv)

Uses sentence embeddings to semantically compare user queries

Responds with the most relevant answer or logs unknown ones

If the user asks about “event” or “events”, it fetches live data from the Hosla website

Guest users are restricted from accessing certain sensitive queries
