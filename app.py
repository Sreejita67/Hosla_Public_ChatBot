from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# Import your entire chatbot logic here
from faq_module import get_answer_from_faq, detect_language_safe

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

@app.route("/")
def index():
    return "✅ Hosla Public Chatbot API is running."

@app.route("/get_answer", methods=["POST"])
def get_answer():
    try:
        data = request.get_json()

        user_query = data.get("query", "")
        user_is_guest = data.get("is_guest", True)

        response, should_log = get_answer_from_faq(user_query, user_is_guest)

        return jsonify({
            "response": response,
            "should_log": should_log,
            "language": detect_language_safe(user_query)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
