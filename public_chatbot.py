import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
import wikipedia
import os
import pathlib
import re, requests
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import urllib3
from googletrans import Translator
from PIL import Image
from IPython.display import display
from langdetect import detect
from difflib import SequenceMatcher
from functools import lru_cache
from difflib import get_close_matches
import xml.etree.ElementTree as ET

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load FAQ CSV
FAQ_FILE = "FAQ - Sheet1.csv"
if not os.path.exists(FAQ_FILE):
    print(f"❌ File '{FAQ_FILE}' not found.")
    exit()

df = pd.read_csv(FAQ_FILE, usecols=[0, 1], names=["Question", "Answer"], header=0, encoding='utf-8', on_bad_lines='skip')
df = df.dropna(subset=['Question', 'Answer'])

# Load BERT model and encode questions
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(df['Question'].tolist(), convert_to_tensor=True)

translator = Translator()
wikipedia.set_lang("en")

# Supported languages
SUPPORTED_LANGUAGES = ['en', 'bn', 'hi', 'ta', 'te', 'mr', 'pa', 'or']

upcoming_event_keywords = [
    "upcoming event", "upcoming events", "future events", "hosla events", "upcoming programs",
    "sessions happening", "event list", "আসন্ন ইভেন্ট", "আসন্ন অনুষ্ঠান", 
    "आगामी इवेंट", "आगामी कार्यक्रम", "आगामी घटना"
]

past_event_keywords = [
    "past events", "previous sessions", "earlier events", "events already held", "past programs",
    "পূর্ববর্তী ইভেন্ট", "গত অনুষ্ঠান", "পূর্ববর্তী ঘটনা", 
    "पिछले इवेंट", "पूर्व की घटनाएँ", "पिछले कार्यक्रम"
]

def build_medline_topic_index_api():
    print("📦 Fetching topics via MedlinePlus API...")
    try:
        url = "https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=health&retmax=1000"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(res.content, "xml")

        topics = {}
        for doc in soup.find_all("document"):
            url = doc.get("url")
            title_tag = doc.find("content", attrs={"name": "title"})
            if title_tag and url:
                clean_title = BeautifulSoup(title_tag.text, "html.parser").get_text()
                topics[clean_title.strip().lower()] = url
        return topics
    except Exception as e:
        print(f"❌ Failed to load MedlinePlus topics via API: {e}")
        return {}

medical_cache = {}
medline_topics = build_medline_topic_index_api()   

def is_health_query(user_input):
    # Check if the cleaned input loosely matches any MedlinePlus topic
    user_input_clean = user_input.strip().lower()
    matches = get_close_matches(user_input_clean, medline_topics.keys(), n=1, cutoff=0.6)
    return bool(matches)
 
def fetch_medline_info(user_query, user_lang="en"):
    query_key = user_query.strip().lower()

    # Check cached response
    if query_key in medical_cache:
        cached = medical_cache[query_key]
        return translator.translate(cached, dest=user_lang).text if user_lang != "en" else cached

    # Match query to Medline topics
    best_match = get_close_matches(query_key, medline_topics.keys(), n=1, cutoff=0.6)
    if not best_match:
        return None

    topic_title = best_match[0]
    topic_url = medline_topics[topic_title]

    # Scrape the topic page
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(topic_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Try to extract meaningful paragraph
        content_block = soup.select_one(".main-content p") or soup.select_one("p")
        if not content_block:
            return None

        content = content_block.get_text(strip=True)
        full_text = f"{content}\n\n🔗 More info: {topic_url}"

        # Cache it
        medical_cache[query_key] = full_text

        return translator.translate(full_text, dest=user_lang).text if user_lang != "en" else full_text

    except Exception as e:
        print(f"❌ Error fetching Medline topic: {e}")
        return None

def detect_language_safe(text):
    try:
        # If only ASCII characters, assume English
        if all(ord(c) < 128 for c in text):
            return "en"

        # Use langdetect
        lang = detect(text)

        # If not in supported list, try to infer from Unicode ranges
        if lang not in SUPPORTED_LANGUAGES:
            if re.search(r'[ঀ-৿]', text): return "bn"   # Bengali
            if re.search(r'[ऀ-ॿ]', text): return "hi"   # Hindi
            if re.search(r'[஀-௿]', text): return "ta"   # Tamil
            if re.search(r'[ఀ-౿]', text): return "te"   # Telugu
            if re.search(r'[਀-੿]', text): return "pa"   # Punjabi
            if re.search(r'[଀-୿]', text): return "or"   # Odia
            if re.search(r'[मराठी]', text): return "mr" # Marathi (fallback if needed)
            return "en"

        return lang
    except:
        return "en"

def clean_query(text):
    return re.sub(r'[^\w\s]', '', text).strip().lower()

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

@lru_cache(maxsize=128)
def search_wikipedia(query, user_lang='en'):
    def try_wikipedia_summary(lang):
        try:
            wikipedia.set_lang(lang)
            search_results = wikipedia.search(query)
            for title in search_results[:3]:
                try:
                    summary = wikipedia.summary(title, sentences=3)
                    if any(tag in summary.lower() for tag in ["film", "tv", "anime", "fictional", "band"]):
                        continue
                    return summary + f"\n🔗 Source: https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                except:
                    continue
        except:
            pass
        return None

    # 1. Try in detected language
    summary = try_wikipedia_summary(user_lang)
    # 2. If nothing found, try in English
    if not summary and user_lang != 'en':
        summary = try_wikipedia_summary('en')
    # 3. Fallback
    return summary if summary else "🤖 Sorry, I couldn't find anything relevant on that topic."

def show_event_feedback(user_lang="en"):
    try:
        feedback_file = "hosla_members_feedback.csv"
        if not os.path.exists(feedback_file):
            return "🙁 No members' feedback available right now."

        df_event = pd.read_csv(feedback_file)
        if df_event.empty:
            return "🙁 No members'feedback available right now."

        responses = []
        for _, row in df_event.iterrows():
            name = row.get("Name", "Someone")
            relation = row.get("Title (Relation to Hosla)", "")
            message = str(row.get("Description (Message)", "")).strip()

            if not message:
                continue

            base_line = f"{name} ({relation}) said: \"{message}\""
            if user_lang != "en":
                translated = translator.translate(message, dest=user_lang).text
                base_line += f"\n🗣️ Translated: \"{translated}\""
            responses.append(base_line)

        return "\n\n".join(responses[:3])  # Limit to top 3 for brevity

    except Exception as e:
        print(f"❌ Error reading event feedback: {e}")
        return "🙁 Sorry, couldn't load event feedback right now."

def translate_text(text, dest_language="en"):
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception:
        return text  # Fallback to original if translation fails


def show_events_info(user_input):
    events_path = os.path.join("assets", "Hosla Events.csv")

    if not os.path.exists(events_path):
        return "Sorry, event information is currently unavailable."

    df = pd.read_csv(events_path)

    lang = detect_language_safe(user_input.lower())

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Check required columns
    required_cols = ["event", "event description", "image path", "date", "time", "location"]
    if not all(col in df.columns for col in required_cols):
        return translate_text("⚠️ The events file is missing required columns.", lang)

    # Combine Date and Time into a single datetime column
    try:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    except Exception as e:
        return f"⚠️ Could not parse event dates: {e}"

    now = datetime.now()

    # Filter events
    if "past" in user_input.lower():
        filtered_df = df[df["datetime"] < now]
    elif "upcoming" in user_input.lower():
        filtered_df = df[df["datetime"] >= now]
    else:
        filtered_df = df

    if filtered_df.empty:
        return translate_text("No events found matching your query.", lang)

    responses = []

    for _, row in filtered_df.iterrows():
        event = translate_text(str(row.get("event", "Unnamed Event")), lang)
        desc = translate_text(str(row.get("event description", "")), lang)
        location = translate_text(str(row.get("location", "Unknown")), lang)
        datetime_str = row["datetime"].strftime("%B %d, %Y at %I:%M %p")

        response = f"📌 {event}\n📝 {desc}\n📍 Location: {location}\n🗓️ Date & Time: {datetime_str}"

        # Add clickable image path (for PowerShell or Terminal)
        image_filename = str(row.get("image path", "")).strip()
        img_path = os.path.join("assets", "images", image_filename)
        if os.path.exists(img_path):
            abs_path = pathlib.Path(img_path).absolute().as_uri()
            response += f"\n🖼️ View Image: {abs_path}"

        responses.append(response)

    return "\n\n".join(responses) + f"\n\n{translate_text('These are the events we found.', lang)}"

def get_answer_from_faq(user_input, user_is_guest):
    user_input_lower = user_input.lower()
    restricted_keywords = ["mental age", "happiness quotient", "mental health"]

    # 🔒 Restrict some features for guest users
    if user_is_guest:
        for keyword in restricted_keywords:
            if keyword in user_input_lower:
                return "🚫 Sorry, this feature is for members only.", False

    # 🌐 Detect input language
    try:
        user_lang = detect_language_safe(user_input)
        translated_input = translator.translate(user_input, dest="en").text if user_lang != "en" else user_input
    except:
        translated_input = user_input
        user_lang = "en"

    # 📣 Show feedback if user asks
        # 📣 Check if user asked for general or event feedback
    feedback_triggers = ["user feedback", "what members say", "hosla feedback", "event feedback", "audience feedback", "feedback"]
    if any(kw in translated_input.lower() for kw in feedback_triggers):
        feedback_response = show_event_feedback(user_lang)
        return feedback_response, False

    translated_input_lower = translated_input.lower()
    original_input_lower = user_input.lower()

    # ✅ Upcoming events logic
    if any(kw in original_input_lower for kw in upcoming_event_keywords) or "upcoming" in original_input_lower or "events" in original_input_lower:
       return show_events_info(original_input_lower), False

    # ✅ Past events logic
    if any(kw in original_input_lower for kw in past_event_keywords) or "past" in original_input_lower:
       return show_events_info(original_input_lower), False


    # 🏥 Check if it's a health-related query
    if is_health_query(translated_input):
        health_info = fetch_medline_info(translated_input, user_lang)
        if health_info:
            return health_info, False

    # ✅ Direct match for "founder of hosla"
    founder_phrases = [
        "founder of hosla", "hosla founder", "হোসলার প্রতিষ্ঠাতা", "হোসলার ফাউন্ডার",
        "হোসলা প্রতিষ্ঠাতা", "হোসলার ফাউন্ডার", "হোসলার প্রতিষ্ঠাতা",
        "होसला के संस्थापक", "संस्थापक होसला"
    ]

    if any(phrase in original_input_lower or phrase in translated_input_lower for phrase in founder_phrases):
        founder_answer = "The Founder of Hosla is Mr. Shantanu Mukhopadhyay"
        try:
            if user_lang != "en":
                founder_answer = translator.translate(founder_answer, dest=user_lang).text
        except Exception as e:
            print("⚠️ Translation error:", e)
        return founder_answer, False
    
    # 🤖 Try semantic match from FAQ
    user_embedding = model.encode(translated_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings)[0]
    max_score = float(similarities.max())
    best_idx = int(similarities.argmax())

    if max_score >= 0.45:
        answer = df.iloc[best_idx]["Answer"]
        try:
            if user_lang != "en":
                translated = translator.translate(answer, dest=user_lang)
                if translated.text.strip():
                    answer = translated.text
        except Exception as e:
            print("⚠️ Translation error:", e)
        return answer, False

    # 🌐 Wikipedia fallback for meaningful queries
    if len(translated_input.strip().split()) > 1:
        summary = search_wikipedia(user_input, user_lang)
        if summary and "🤖 Sorry" not in summary and "⚠️" not in summary:
           return summary, False
        else:
           log_unknown_question(user_input, "Guest" if user_is_guest else "Member")

    # 📝 Log unknown question for review
    log_unknown_question(user_input, "Guest" if user_is_guest else "Member")

    # ❌ Fallback response
    fallback = "🤖 I'm not sure about that yet. Please contact Hosla at 📞7811009309 for more information."
    try:
        if user_lang != "en":
            fallback = translator.translate(fallback, dest=user_lang).text
    except:
        pass
    return fallback, True

def log_unknown_question(question_text, user_name):
    try:
        new_row = pd.DataFrame([{
            "Question": question_text,
            "Answer": "",
            "User": user_name,
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        new_row.to_csv(FAQ_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
        print("📝 Question logged for review.")
    except Exception as e:
        print(f"❌ Failed to log: {e}")

def check_user_login():
    user_name = input("Enter your name (or type 'guest'): ")
    return user_name.strip().lower() == 'guest', user_name.strip().title()

def save_chat_log(user_name, logs):
    with open(f"{user_name}_chat_log.txt", "a", encoding="utf-8") as f:
        for entry in logs:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {entry}\n")
    print(f"💾 Chat log saved to {user_name}_chat_log.txt")

def faq_chat_loop():
    user_is_guest, user_name = check_user_login()
    chat_logs = []
    print(f"\n👋 Hello {user_name}! I'm your multilingual FAQ assistant. Type 'exit' to store the chat and quit.\n")

    while True:
        try:
            user_input = input(f"{user_name}: ").strip()
        except EOFError:
            print("\n❌ Input interrupted. Exiting.")
            break

        if not user_input:
            print("Assistant: Please ask something.")
            continue
        if user_input.lower() == 'exit':
            print("Assistant: Goodbye! Stay healthy 🌟")
            break

        response, should_log = get_answer_from_faq(user_input, user_is_guest)
        if should_log:
            log_unknown_question(user_input, user_name)

        print("Assistant:", response)
        chat_logs.append(f"{user_name}: {user_input}")
        chat_logs.append(f"Assistant: {response}")

    save_chat_log(user_name, chat_logs)

if __name__ == "__main__":
    faq_chat_loop()
