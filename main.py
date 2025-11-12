from flask import Flask, request, send_from_directory, jsonify, redirect
from werkzeug.utils import secure_filename
import os
import traceback
import json
import logging
from gemini_transcriber import transcribe_audio
from gemini_notes_generator import generate_structured_notes
from google import genai
from google.genai import types
from datetime import datetime, timedelta


# --- User Management ---
USER_FILE = "user_profile.json"

def load_user():
    if os.path.exists(USER_FILE):
        with open(USER_FILE) as f:
            return json.load(f)
    return {"name": "User", "email": "", "initials": "U"}

def save_user(data):
    # Ensure initials are calculated if not provided
    if 'name' in data and 'initials' not in data:
        name = data.get('name', 'User')
        data['initials'] = ''.join([n[0] for n in name.split(' ') if n]).upper() or 'U'
        
    with open(USER_FILE, "w") as f:
        json.dump(data, f)


# --- Configuration ---
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
DOCX_DIR = os.path.join(os.path.dirname(__file__), 'generated_docs')
PDF_DIR = os.path.join(os.path.dirname(__file__), 'generated_pdfs')
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'webm'}

# Ensure directories exist (CRITICAL STEP)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOCX_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# --- Initial Setup ---
API_KEY = os.getenv("GEMINI_API_KEY") or ""
if not API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set. API features will fail.")
else:
    # Masked print to avoid leaking the full key in logs
    masked = (API_KEY[:6] + '...' + API_KEY[-4:]) if len(API_KEY) > 10 else '<set>'
    print(f"GEMINI_API_KEY is set (masked): {masked}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    """Default route, redirect to login page."""
    return redirect('/login.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static HTML, JS, CSS files (like login.html and dashboard.html)."""
    return send_from_directory(os.getcwd(), filename)

@app.route('/api/transcribe', methods=['POST'])
def handle_transcription():
    """Handles file upload, transcription, and note generation."""
    
    # Check for API key access at the entry point of the API route
    if not API_KEY:
        return jsonify({"error": "GEMINI_API_KEY is missing. Please set it in your environment."}), 401

    # 1. Handle file upload
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save uploaded file
            file.seek(0)
            with open(audio_path, 'wb') as f:
                f.write(file.read())

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return jsonify({"error": "Failed to save audio file to disk."}), 500

        except Exception as e:
            return jsonify({"error": f"Failed to save file locally: {e}"}), 500

        # --- 🧩 Handle optional prompts and templates ---
        # **MODIFIED: Separate prompt and template**
        user_prompt_text = request.form.get('prompt', '')
        template_name = request.form.get('template', '')

        # 2. Transcribe audio
        try:
            transcript = transcribe_audio(audio_path)

            if transcript.startswith("ERROR:"):
                print(f"Transcription returned ERROR: {transcript}")
                lower = transcript.lower()
                os.remove(audio_path)
                if any(key in lower for key in ['api key', 'api_key', 'expired']):
                    return jsonify({"error": transcript}), 401
                return jsonify({"error": transcript}), 500

        except Exception as e:
            transcript = f"(Unhandled error during transcription: {e})"
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return jsonify({"error": transcript}), 500

        # 3. Generate structured notes (and save DOCX/PDF)
        try:
            base_title = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
            
            # **MODIFIED: Pass template_name to the generator**
            docx_path, pdf_path, final_title = generate_structured_notes(
                transcript_text=transcript,
                user_prompt=user_prompt_text or f"Generate notes on this transcript: {base_title}",
                template=template_name,
                custom_title=base_title
            )
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return jsonify({
                "message": "Transcription and notes generated.",
                "title": final_title,
                "transcript": transcript,
                "docx_path": docx_path
            }), 200

        except Exception as e:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return jsonify({"error": f"Note generation failed: {e}"}), 500
            
    return jsonify({"error": "File type not allowed or other internal file error"}), 400

@app.route('/api/notes', methods=['GET'])
def list_notes():
    """ 
    **MODIFIED: Fetches all notes and sorts them by modification time (newest first).**
    """
    notes = []
    try:
        for f in os.listdir(DOCX_DIR):
            if f.endswith(".docx"):
                path = os.path.join(DOCX_DIR, f)
                if os.path.isfile(path):
                    mtime = os.path.getmtime(path)
                    notes.append({
                        "title": f.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' '),
                        "filename": f,
                        "mtime": mtime
                    })
        
        # Sort by modification time, newest first
        notes.sort(key=lambda x: x['mtime'], reverse=True)
        
    except Exception as e:
        print(f"Error listing notes: {e}")
        return jsonify({"error": f"Failed to list notes: {e}"}), 500
        
    return jsonify(notes)

@app.route('/api/chat', methods=['POST'])
def handle_ai_chat():
    """Handles chat messages and uses Gemini to analyze notes/respond."""
    # Require the API key for the chat endpoint too; return 401 if missing so client can tell user
    if not API_KEY:
        return jsonify({"response": "GEMINI_API_KEY is missing. Please set it in your environment."}), 401
    # ... (chat logic remains mostly the same, removed for brevity)
    # The chat endpoint does not rely on reportlab and is less prone to dependency errors.
    data = request.get_json()
    user_message = data.get('message', '')
    attached_note = data.get('attached_note', None)
    
    if not user_message:
        return jsonify({"response": "Please type a message."}), 200

    client = genai.Client(api_key=API_KEY)
    
    system_prompt = "You are a friendly AI assistant for Transcripto. Your task is to process user requests, summarize notes, and pull action items. Be concise and helpful."
    
    try:
        # Simplified logic for chat as per the existing structure:
        if attached_note:
             full_prompt = (
                 f"{system_prompt}\n\n"
                 f"CONTEXT: The user has attached the note titled '{attached_note}'. Analyze the note to answer the question.\n\n"
                 f"USER: {user_message}"
             )
        else:
             full_prompt = f"{system_prompt}\n\nUSER: {user_message}"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[full_prompt]
        )
        
        return jsonify({"response": response.text}), 200

    except Exception as e:
        return jsonify({"response": f"An unexpected error occurred with the AI chat: {e}"}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Serve the generated DOCX or PDF files for download."""
    # Split filename to check if it's DOCX or PDF and which folder to look in
    name, ext = os.path.splitext(filename)
    
    if ext.lower() == '.docx':
        return send_from_directory(DOCX_DIR, filename, as_attachment=True)
    elif ext.lower() == '.pdf':
        # 🚨 FIX: Use the corrected PDF_DIR path
        return send_from_directory(PDF_DIR, filename, as_attachment=True)
        
    return jsonify({"error": "File not found or invalid format"}), 404

@app.route('/api/stats', methods=['GET'])
def get_stats():
    now = datetime.now()
    week_ago = now - timedelta(days=7)
    notes_this_week = 0

    try:
        for f in os.listdir(DOCX_DIR):
            path = os.path.join(DOCX_DIR, f)
            if os.path.isfile(path):
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                if mtime > week_ago:
                    notes_this_week += 1
    except Exception as e:
        print(f"Error calculating stats: {e}")

    return jsonify({"notes_this_week": notes_this_week})

@app.route('/api/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'GET':
        return jsonify(load_user())
    else:
        data = request.json
        save_user(data)
        return jsonify({"message": "Profile updated"})

# --- Run Server ---
if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000/")
    if API_KEY:
        print("API Key check passed.")
    else:
        print("Warning: GEMINI_API_KEY was not set at startup. API endpoints will return 401 until it's configured.")
    # Ensure this is running in a shell where GEMINI_API_KEY is set.
    app.run(debug=True)