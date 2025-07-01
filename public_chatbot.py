import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load FAQ Data
FAQ_FILE = "FAQ - Sheet1.csv"
if not os.path.exists(FAQ_FILE):
    print(f"❌ File '{FAQ_FILE}' not found. Please make sure it's in the same directory.")
    exit()

df = pd.read_csv(FAQ_FILE, usecols=[0, 1], names=["Question", "Answer"], header=0, encoding='utf-8', on_bad_lines='skip')
df = df.dropna(subset=['Question', 'Answer'])

# Load Semantic Model
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(df['Question'].tolist(), convert_to_tensor=True)

# Fetch events from Hosla site
def fetch_hosla_events():
    url = "https://hosla.in/events.php"
    try:
        resp = requests.get(url, timeout=5, verify=False)
        resp.raise_for_status()
    except Exception as e:
        return f"❌ Unable to fetch events: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    events = []

    for card in soup.select(".event-card"):
        lines = [p.get_text(strip=True) for p in card.find_all("p")]
        if lines:
            event_text = " | ".join(lines)
            events.append(event_text)

    return "\n\n".join(events[:5]) if events else "📭 No upcoming events found on Hosla right now."


# Answering Function with guest restriction and logging unknown Qs
def get_answer_from_faq(user_input, user_is_guest):
    user_input_lower = user_input.lower()
    restricted_keywords = ["mental age", "happiness quotient", "mental health"]

    if user_is_guest:
        for keyword in restricted_keywords:
            if keyword in user_input_lower:
                return "🚫 Sorry, this feature is for members only.", False

    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings)[0]
    max_score = float(similarities.max())
    best_idx = int(similarities.argmax())

    if max_score > 0.55:
        return df.iloc[best_idx]['Answer'], False
    else:
        return "For further details contact Hosla.", True

# Append unknown question to CSV with timestamp and user name
def log_unknown_question(question_text, user_name):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_row = pd.DataFrame([{
            "Question": question_text,
            "Answer": "",
            "User": user_name,
            "Timestamp": timestamp
        }])
        new_row.to_csv(FAQ_FILE, mode='a', header=False, index=False)
        print("🆕 For Further Details, Contact Hosla : 📞7811009309")
    except Exception as e:
        print("❌ Failed to log unknown question:", e)

# Login
def check_user_login():
    user_name = input("Enter your name (or type 'guest'): ")
    user_is_guest = user_name.strip().lower() == 'guest'
    return user_is_guest, "Guest" if user_is_guest else user_name.strip()

# Save log
def save_chat_log(user_name, logs):
    filename = f"{user_name}_chat_log.txt"
    with open(filename, "a", encoding="utf-8") as f:
        for entry in logs:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {entry}\n")
    print(f"📝 Chat log saved to {filename}")

# Main Chat Loop
def faq_chat_loop():
    user_is_guest, user_name = check_user_login()
    chat_logs = []

    print(f"\n👋 Hello {user_name}! I'm your FAQ assistant. Type 'exit' to quit.")

    while True:
        try:
            user_input = input(f"\n{user_name}: ").strip()
        except EOFError:
            print("\n🚫 Input interrupted. Exiting...")
            break

        if not user_input:
            print("Assistant: Please ask something.")
            continue

        if user_input.lower() == 'exit':
            print("Assistant: Goodbye! Stay healthy 🌟")
            break

        if user_input.lower() in ['event', 'events']:
            response = fetch_hosla_events()
        else:
            response, log_unknown = get_answer_from_faq(user_input, user_is_guest)
            if log_unknown:
                log_unknown_question(user_input, user_name)

        print("Assistant:", response)
        chat_logs.append(f"{user_name}: {user_input}")
        chat_logs.append(f"Assistant: {response}")

    save_chat_log(user_name, chat_logs)

# Run the chatbot
if __name__ == "__main__":
    faq_chat_loop()