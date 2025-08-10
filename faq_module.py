import warnings
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
import os
import time 
import pathlib
import unicodedata
import re, requests
import pandas as pd
import pytz
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from datetime import datetime, timedelta
import urllib3
from googletrans import Translator
from langdetect import detect
from difflib import SequenceMatcher
from functools import lru_cache
from difflib import get_close_matches
import xml.etree.ElementTree as ET
from huggingface_hub import HfApi, HfFolder, Repository
from io import StringIO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================================================
# 📄 Load FAQ CSV from Google Sheet or fallback
# ================================================
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8IArJoxgQ2EL2fQJn_rUVozWqJbz-n0Qn42rTMDHHZezCbn5MEa-0TcvRfPiEGPyDj3W96LkRFwSH/pub?gid=0&single=true&output=csv"

def load_faq_from_google_sheet():
    try:
        response = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, usecols=[0, 1], names=["Question", "Answer"], header=0, encoding='utf-8', on_bad_lines='skip')
            print("✅ Loaded FAQ from live Google Sheet.")
            return df.dropna(subset=['Question', 'Answer'])
        else:
            print(f"⚠️ Failed to fetch from Google Sheet. Status: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Exception while loading from Google Sheet: {e}")
    
    return None

# Load from Google Sheet, or fallback to local file
faq_df = load_faq_from_google_sheet()
if faq_df is None:
    FAQ_FILE = "FAQ - Sheet1.csv"
    if not os.path.exists(FAQ_FILE):
        raise FileNotFoundError(f"❌ File '{FAQ_FILE}' not found.")
    faq_df = pd.read_csv(FAQ_FILE, usecols=[0, 1], names=["Question", "Answer"], header=0, encoding='utf-8', on_bad_lines='skip')
    faq_df = faq_df.dropna(subset=['Question', 'Answer'])
    print("✅ Loaded FAQ from local CSV.")

df = faq_df 

# Load BERT model and encode questions
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(faq_df['Question'].tolist(), convert_to_tensor=True)

translator = Translator()

# Supported languages
SUPPORTED_LANGUAGES = ['en', 'bn', 'hi', 'ta', 'te', 'mr', 'pa', 'or']

upcoming_event_keywords = [
    "upcoming event", "upcoming events", "future events", "hosla events", "upcoming programs",
    "sessions happening", "event list", "আসন্ন ইভেন্ট", "আসন্ন অনুষ্ঠান", "events",
    "आगामी इवेंट", "आगामी कार्यक्रम", "आगामी घटना"
]

past_event_keywords = [
    "past events", "previous sessions", "earlier events", "events already held", "past programs",
    "পূর্ববর্তী ইভেন্ট", "গত অনুষ্ঠান", "পূর্ববর্তী ঘটনা", 
    "पिछले इवेंट", "पूर्व की घटनाएँ", "पिछले कार्यक्रम"
]

def build_medline_topic_index_api():
    try:
        url = "https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=health&retmax=1000"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(res.content, "lxml")

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

def detect_language_safe(text: str) -> str:
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

def show_event_feedback(user_lang="en"):
    try:
        csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8IArJoxgQ2EL2fQJn_rUVozWqJbz-n0Qn42rTMDHHZezCbn5MEa-0TcvRfPiEGPyDj3W96LkRFwSH/pub?gid=1271142221&single=true&output=csv"

        response = requests.get(csv_url, timeout=5)  
        response.raise_for_status()  # Raise error for bad response

        df_event = pd.read_csv(StringIO(response.text))

        if df_event.empty:
            text = "🙁 No members' feedback available right now."
            return translator.translate(text, dest=user_lang).text if user_lang != "en" else text

        responses = []
        for _, row in df_event.iterrows():
            name = row.get("Name", "Someone")
            relation = row.get("Title (Relation to Hosla)", "")
            message = str(row.get("Description (Message)", "")).strip()
            if not message:
                continue

            try:
                said_translated = translator.translate("said", dest=user_lang).text if user_lang != "en" else "said"
            except:
                said_translated = "said"

            base_line = f"{name} ({relation}) {said_translated}: \"{message}\""

            if user_lang != "en":
                try:
                    translated = translator.translate(message, dest=user_lang).text
                    base_line += f"\n🗣️ Translated: \"{translated}\""
                except:
                    pass

            responses.append(base_line)

        return "\n\n".join(responses[:3])

    except Exception as e:
        print(f"❌ Error reading public sheet: {e}")
        text = "🙁 Sorry, couldn't load event feedback right now."
        return translator.translate(text, dest=user_lang).text if user_lang != "en" else text



def translate_text(text, dest_language="en"):
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception:
        return text  # Fallback to original if translation fails

def show_events_info(user_input):
    lang = detect_language_safe(user_input.lower())

    # Detect event type and choose correct sheet URL
    if any(kw in user_input.lower() for kw in upcoming_event_keywords):
        sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8IArJoxgQ2EL2fQJn_rUVozWqJbz-n0Qn42rTMDHHZezCbn5MEa-0TcvRfPiEGPyDj3W96LkRFwSH/pub?gid=1925531775&single=true&output=csv"
    elif any(kw in user_input.lower() for kw in past_event_keywords):
        sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8IArJoxgQ2EL2fQJn_rUVozWqJbz-n0Qn42rTMDHHZezCbn5MEa-0TcvRfPiEGPyDj3W96LkRFwSH/pub?gid=1834108950&single=true&output=csv"
    else:
        return translate_text("❓ Please specify whether you want to see past or upcoming events.", lang)

    # Read CSV data
    try:
        df = pd.read_csv(sheet_url)
    except Exception as e:
        return translate_text(f"⚠️ Could not load event data: {e}", lang)

    # Normalize columns
    df.columns = [col.strip().lower() for col in df.columns]
    required_cols = ["event", "event description", "date", "time", "location"]

    if not all(col in df.columns for col in required_cols):
        return translate_text("⚠️ The events sheet is missing required columns.", lang)

    # Parse datetime (used only for display formatting)
    try:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True, errors='coerce')
    except:
        df["datetime"] = None

    # Prepare responses
    responses = []
    for _, row in df.iterrows():
        event = translate_text(str(row.get("event", "Unnamed Event")), lang)
        desc = translate_text(str(row.get("event description", "")), lang)
        location = translate_text(str(row.get("location", "Unknown")), lang)

        if pd.notnull(row.get("datetime")):
            datetime_str = row["datetime"].strftime("%B %d, %Y at %I:%M %p")
        else:
            datetime_str = f"{row.get('date', '')} {row.get('time', '')}"

        response = f"📌 {event}\n📝 {desc}\n📍 {location}\n🗓️ {datetime_str}"
        responses.append(response)

    if not responses:
        return translate_text("No events found in the sheet.", lang)

    return "\n\n".join(responses) + f"\n\n{translate_text('These are the events we found.', lang)}"
    
def normalize_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def get_answer_from_faq(query: str, is_guest: bool = True) -> tuple:
    user_input = query

        # Emergency fuzzy check (multilingual + typo-tolerant)
    from difflib import get_close_matches

    def contains_emergency_keyword(text):
        keywords = ["emergency", "জরুরী", "आपातकाल","প্রয়োজন","help"]
        text_words = text.lower().split()
        for word in text_words:
            if get_close_matches(word, keywords, n=1, cutoff=0.8):
                return True
        return False

    if contains_emergency_keyword(user_input):
        emergency_msg = (
            "🚨 I understand you need an urgent help. Please stay calm.\n"
            "For any help during any emergency,Please dial +91 78110 09309.\n"
            "Hosla is always here for you. ❤️"
        )
        try:
            user_lang = detect_language_safe(user_input)
            if user_lang != "en":
                emergency_msg = translator.translate(emergency_msg, dest=user_lang).text
        except:
            pass
        return emergency_msg, False

    user_input_lower = normalize_text(user_input)
    user_is_guest = is_guest
    
    # 🌐 Multilingual restricted keywords
    restricted_keywords = [
        "mental age", "happiness quotient", "mental health",
        "मानसिक स्वास्थ्य", "मनोस्वास्थ्य", "ख़ुशी का स्तर", "आनंद स्तर",
        "मानसिक आरोग्य", "आनंद प्रमाण",
        "மனநலம்", "மனச் சுகாதாரம்", "மகிழ்ச்சி அளவீடு",
        "ਮਾਨਸਿਕ ਸਿਹਤ", "ਖੁਸ਼ੀ ਦਾ ਪੱਧਰ",
        "మానసిక ఆరోగ్యం", "ఆనంద స్థాయి",
        "মানসিক স্বাস্থ্য", "আনন্দমাত্রা", "মানসিক বয়স"
    ]

    try:
        user_lang = detect_language_safe(user_input)
    except:
        user_lang = "en"

    # 🔒 Guest restriction check
    if user_is_guest:
        for keyword in restricted_keywords:
            if normalize_text(keyword) in user_input_lower:
                response_text = "🚫 Sorry, this feature is for members only."
                try:
                    if user_lang != "en":
                        response_text = translator.translate(response_text, dest=user_lang).text
                except:
                    pass
                return response_text, False

    # 🌐 Translation for semantic matching
    try:
        translated_input = translator.translate(user_input, dest="en").text if user_lang != "en" else user_input
    except:
        translated_input = user_input

    translated_input_lower = normalize_text(translated_input)
    original_input_lower = user_input.lower().strip()
    # 👋 Greeting response
    greeting_inputs = [
        "hi", "hello", "hey", "হ্যালো", "নমस्तে", "হাই", "হে", "হেলো", "সুপ্রভাত", "শুভ সকাল",
        "good morning", "good evening", "good afternoon"
    ]
    normalized_input = original_input_lower.replace(",", "").replace("?", "")
    if normalized_input in [g.lower() for g in greeting_inputs]:
        greeting_response = "Hi there! 👋 I'm Hosla Public Chatbot. How can I assist you today?"
        try:
            if user_lang != "en":
                greeting_response = translator.translate(greeting_response, dest=user_lang).text
        except:
            pass
        return greeting_response, False

    # 🌐 Multilingual Feedback Trigger
    feedback_triggers_multilingual = {
        "en": ["user feedback", "what members say", "hosla feedback", "event feedback", "audience feedback", "feedback"],
        "hi": ["प्रतिक्रिया", "सदस्य क्या कहते हैं", "होसला प्रतिक्रिया", "कार्यक्रम प्रतिक्रिया", "दर्शकों की प्रतिक्रिया"],
        "bn": ["প্রতিক্রিয়া", "সদস্যরা কী বলেন", "হোসলা প্রতিক্রিয়া", "ইভেন্ট প্রতিক্রিয়া", "দর্শকদের প্রতিক্রিয়া"],
        "pa": ["ਪ੍ਰਤੀਕਿਰਿਆ", "ਮੈਂਬਰ ਕੀ ਕਹਿੰਦੇ ਹਨ", "ਹੌਸਲਾ ਫੀਡਬੈਕ", "ਈਵੈਂਟ ਫੀਡਬੈਕ", "ਦਰਸ਼ਕਾਂ ਦੀ ਪ੍ਰਤੀਕਿਰਿਆ"],
        "ta": ["கருத்து", "உறுப்பினர்கள் சொல்வது", "ஹொஸ்லா கருத்து", "நிகழ்ச்சி கருத்து", "பார்வையாளர்கள் கருத்து"],
        "te": ["అభిప్రాయం", "సభ్యులు ఏమంటున్నారు", "హోస్లా అభిప్రాయం", "ఈవెంట్ అభిప్రాయం", "ప్రేక్షకుల అభిప్రాయం"],
        "mr": ["प्रतिक्रिया", "सदस्य काय म्हणतात", "होसला अभिप्राय", "कार्यक्रम अभिप्राय", "प्रेक्षक अभिप्राय"],
        "or": ["ପ୍ରତିକ୍ରିୟା", "ସଦସ୍ୟମାନେ କଣ କହନ୍ତି", "ହୋସଲା ପ୍ରତିକ୍ରିୟା", "ଇଭେଣ୍ଟ ପ୍ରତିକ୍ରିୟା", "ଦର୍ଶକ ପ୍ରତିକ୍ରିୟା"]
    }

    def normalize(text):
        return unicodedata.normalize("NFKC", text).strip().lower()
    
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()
        
    def check_feedback_trigger(user_input, user_lang):
        triggers = feedback_triggers_multilingual.get(user_lang, feedback_triggers_multilingual["en"])
        user_input = normalize(user_input)
        for trigger in triggers:
            if similar(user_input, normalize(trigger)) > 0.85:
               return True
        return False

    user_input_lower = user_input.lower()

    if check_feedback_trigger(user_input_lower, user_lang):
       return show_event_feedback(user_lang), False


    if any(kw in query.lower() for kw in upcoming_event_keywords + past_event_keywords):
       return show_events_info(query), False

        
    # ✨ Multilingual internship keywords
    internship_keywords_by_lang = {
        "en": ["intern", "internship", "volunteer", "work with hosla", "join hosla", "career", "part of hosla", "can I work as an intern", "want to be a part of hosla"],
        "hi": ["इंटर्न", "होसला में काम", "स्वयंसेवक", "होसला में जुड़ें"],
        "bn": ["ইন্টার্ন", "হোসলাতে কাজ", "হোসলা তে কাজ", "স্বেচ্ছাসেবক", "হোসলা তে যুক্ত হই","হোসলার অংশ হতে চাই", "হোসলা তে যুক্ত হতে চাই"],
        "ta": ["இண்டர்ன்ஷிப்", "ஹோஸ்லாவில் வேலை", "விருப்ப சேவையாளர்"],
        "te": ["ఇంటర్న్", "ఇంటర్న్‌షిప్", "వాలంటీర్", "హోస్లాలో పని చేయాలి", "హోస్లాలో చేరాలనుకుంటున్నాను", "ఉద్యోగ అవకాశాలు", "హోస్లా భాగం కావాలనుకుంటున్నాను"],
        "pa": ["ਇੰਟਰਨ", "ਇੰਟਰਨਸ਼ਿਪ", "ਸੇਵਾ ਕਰਤਾ", "ਹੌਸਲਾ ਵਿੱਚ ਕੰਮ ਕਰਨਾ ਚਾਹੁੰਦਾ ਹਾਂ","ਹੌਸਲਾ ਵਿੱਚ ਸ਼ਾਮਿਲ ਹੋਣਾ", "ਕੈਰੀਅਰ ਮੌਕੇ", "ਹੌਸਲਾ ਦਾ ਹਿੱਸਾ ਬਣਨਾ"],
        "mr": ["इंटर्न", "इंटर्नशिप", "स्वयंसेवक", "होसला मध्ये काम करायचं आहे","होसला मध्ये सामील व्हायचं आहे", "करिअर संधी", "होसला चा भाग बनायचं आहे"]
    }

    detected_intern_lang = None
    for lang, keywords in internship_keywords_by_lang.items():
        for kw in keywords:
            match = get_close_matches(user_input_lower, [kw.lower()], n=1, cutoff=0.75)
            if match:
                detected_intern_lang = lang
                break
        if detected_intern_lang:
            break


    if detected_intern_lang:
        internship_responses = {
            "en": (
                "Great! We'd love to have you! 🙌\n"
                "Please share your interest and email your Resume to hosla.dalmadal@gmail.com or shantanumproductmanager@gmail.com or shraddhawelfareassociation@gmail.com.\n"
                "📞 You can also call us at +91-7811 009 309 for internship or volunteering opportunities."
            ),
            "hi": (
                "बहुत बढ़िया! हमें खुशी होगी कि आप हमारे साथ जुड़ें। 🙌\n"
                "कृपया अपना रुचि पत्र और रिज़्यूमे इस ईमेल पर भेजें: hosla.dalmadal@gmail.com या shantanumproductmanager@gmail.com या shraddhawelfareassociation@gmail.com\n"
                "📞 आप इस नंबर पर कॉल भी कर सकते हैं: +91-7811 009 309 (सोम-शुक्र, सुबह 10 – शाम 6)।"
            ),
            "bn": (
                "দারুন! আমরা খুব খুশি হব যদি আপনি আমাদের সাথে যুক্ত হন। 🙌\n"
                "অনুগ্রহ করে আপনার আগ্রহ এবং রেজুমে পাঠান এই ইমেইল ঠিকানায়: hosla.dalmadal@gmail.com অথবা shantanumproductmanager@gmail.com অথবা shraddhawelfareassociation@gmail.com\n"
                "📞 ফোন করুন: +91-7811 009 309 (সোম-শুক্র, সকাল ১০টা – সন্ধ্যা ৬টা)।"
            ),
            "ta": (
                "அருமை! நீங்கள் எங்களுடன் இணைய விரும்புகிறீர்கள் என்பதை கேட்டு மகிழ்ச்சி! 🙌\n"
                "தயவுசெய்து உங்கள் விருப்பம் மற்றும் பயோடேட்டா இமெயில் செய்யவும்: hosla.dalmadal@gmail.com அல்லது shantanumproductmanager@gmail.com அல்லது shraddhawelfareassociation@gmail.com\n"
                "📞 மேலும் தகவலுக்கு இந்த எண்ணில் அழைக்கவும்: +91-7811 009 309 (திங்கள்–வெள்ளி, காலை 10 முதல் மாலை 6 வரை)."
            ),
            "te": (
                "చాలా బాగుంది! మీరు మాకు చాలా ఇష్టం! 🙌\n"
                "దయచేసి మీ ఆసక్తిని పంచుకోండి మరియు మీ రెజ్యూమ్‌ను hosla.dalmadal@gmail.com లేదా shantanumproductmanager@gmail.com లేదా shraddhawelfareassociation@gmail.com కు ఇమెయిల్ చేయండి.\n"
                "📞 ఇంటర్న్‌షిప్ లేదా వాలంటీరింగ్ అవకాశాల కోసం మీరు +91-7811 009 309 కు కూడా కాల్ చేయవచ్చు."
            ),
            "pa": (
                "ਵਧੀਆ! ਅਸੀਂ ਖੁਸ਼ ਹਾਂ ਕਿ ਤੁਸੀਂ ਸਾਡੇ ਨਾਲ ਜੁੜਣਾ ਚਾਹੁੰਦੇ ਹੋ! 🙌\n"
                "ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੀ ਦਿਲਚਸਪੀ ਅਤੇ ਰਿਜ਼ਿਊਮ ਇਹਨਾਂ ਈਮੇਲ ਪਤੇ 'ਤੇ ਭੇਜੋ: hosla.dalmadal@gmail.com, shantanumproductmanager@gmail.com ਜਾਂ shraddhawelfareassociation@gmail.com\n"
                "📞 ਇੰਟਰਨਸ਼ਿਪ ਜਾਂ ਵੋਲੰਟੀਅਰ ਮੌਕਿਆਂ ਲਈ +91-7811 009 309 'ਤੇ ਸੰਪਰਕ ਕਰੋ।"
            ),
            "mr": (
                "छान! तुम्ही आमच्यासोबत जोडले जाल याचा आम्हाला खूप आनंद होईल! 🙌\n"
                "कृपया तुमची आवड आणि रिझ्युमे पुढील ईमेलवर पाठवा: hosla.dalmadal@gmail.com, shantanumproductmanager@gmail.com किंवा shraddhawelfareassociation@gmail.com\n"
                "📞 इंटर्नशिप किंवा स्वयंसेवक संधींसाठी कृपया +91-7811 009 309 या क्रमांकावर संपर्क करा."
            )

        }

        reply_text = internship_responses.get(detected_intern_lang, internship_responses["en"])
        return reply_text, False

    # ✅ Direct match for founder query
    founder_phrases = [
        "founder of hosla", "hosla founder", "হোসলার প্রতিষ্ঠাতা", "হোসলার ফাউন্ডার",
        "হোসলা প্রতিষ্ঠাতা", "होसला के संस्थापक", "संस्थापक होसला"
    ]

    if any(phrase in original_input_lower or phrase in translated_input_lower for phrase in founder_phrases):
        founder_answer = "The Founder of Hosla is Mr. Shantanu Mukhopadhyay"
        try:
            if user_lang != "en":
                founder_answer = translator.translate(founder_answer, dest=user_lang).text
        except Exception as e:
            print("⚠️ Translation error:", e)
        return founder_answer, False

    # 🤖 Semantic match from FAQ
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

    # 🧠 Fuzzy matching fallback for minor typos
    best_match = get_close_matches(translated_input_lower, df["Question"].apply(normalize_text).tolist(), n=1, cutoff=0.6)
    if best_match:
        match_index = df[df["Question"].apply(lambda q: normalize_text(q) == best_match[0])].index[0]
        answer = df.iloc[match_index]["Answer"]
        try:
            if user_lang != "en":
                translated = translator.translate(answer, dest=user_lang)
                if translated.text.strip():
                    answer = translated.text
        except Exception as e:
            print("⚠️ Translation error:", e)
        return answer, False


     # 🏥 Health-related query
    if is_health_query(translated_input):
        health_info = fetch_medline_info(translated_input, user_lang)
        if health_info:
            return health_info, False

    # ❌ Fallback response
    fallback = "🤖 I'm not sure about that yet. Please contact Hosla at 📞7811009309 for more information."
    try:
        if user_lang != "en":
            fallback = translator.translate(fallback, dest=user_lang).text
    except:
        pass

    log_unknown_question(user_input, "Guest" if user_is_guest else "Member")
    return fallback, True

last_logged = {}

def log_unknown_question(query, user_name):
    global last_logged
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)  # Current time in IST
    key = (query.strip().lower(), user_name.strip().lower())

    # Cooldown check (10 seconds)
    if key in last_logged and (now - last_logged[key]).total_seconds() < 10:
        print("[HUGGINGFACE LOG] Skipping duplicate log.")
        return

    last_logged[key] = now
    try:
        timestamp = now.strftime("%d-%m-%Y %H:%M:%S")
        data = {
            "question": query,
            "asked_by": user_name,
            "timestamp": timestamp
        }
        webhook_url = "https://script.google.com/macros/s/AKfycbxAohOvjPAnEYCEnaePL3u2FnQiUIGf347PTlP89vX7IW15kh3YOHbi7nCX_jKLlCpg/exec"
        print("[HUGGINGFACE LOG] Sending to Google Sheets:", data)
        res = requests.post(webhook_url, data=data, timeout=5)
        print("[HUGGINGFACE LOG] Google Sheets responded:", res.status_code, res.text)
    except Exception as e:
        print("[HUGGINGFACE LOG] Error logging to Google Sheet:", e)

        
__all__ = ["get_answer_from_faq", "detect_language_safe", "log_unknown_question"]
