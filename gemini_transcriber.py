from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
load_dotenv()

MIME_TYPE_MAP = {
    '.mp3': 'audio/mp3',
    '.wav': 'audio/wav',
    '.m4a': 'audio/m4a',
    '.flac': 'audio/flac',
    '.webm': 'audio/webm'
}

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file using the Gemini API.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment. Transcription aborted.")
        return "ERROR: GEMINI_API_KEY is missing or not set"
    
    client = genai.Client(api_key=api_key)
    masked = (api_key[:6] + '...' + api_key[-4:]) if api_key else '<missing>'
    
    _, ext = os.path.splitext(audio_path)
    mime_type = MIME_TYPE_MAP.get(ext.lower())
    if not mime_type:
        print(f"ERROR: Unsupported file type: {ext}")
        return f"ERROR: Unsupported file type: {ext}"

    print(f"🎧 Uploading '{audio_path}' to Gemini for transcription... (key={masked})")

    try:
        # Use the client.files.upload() method
        gemini_file = client.files.upload(file=audio_path)
    
    except Exception as e:
        print(f"Error during file upload to Gemini: {e}")
        return f"ERROR: Failed to upload audio file: {e}"

    try:
        text_part = types.Part(text="Please transcribe this audio clearly with punctuation.")
        
        # 🟢 FIX 1:
        # Using the FileData constructor, which we know works
        # and avoids the 'Part.from_uri()' bug.
        audio_part = types.Part(
            file_data=types.FileData(
                mime_type=gemini_file.mime_type,
                file_uri=gemini_file.uri
            )
        )
        
    except Exception as e:
         print(f"Error constructing types.Part: {e}")
         return f"ERROR: Failed to construct types.Part: {e}"

    parts = [text_part, audio_part]

    try:
        # 🟢 FIX 2:
        # Using the correct model name from your list: "models/gemini-2.5-flash"
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[types.Content(role="user", parts=parts)],
        )
    except Exception as e:
        try:
            client.files.delete(name=gemini_file.name)
            print(f"✅ Deleted uploaded file {gemini_file.name} after API failure.")
        except Exception:
            pass
        return f"ERROR: Gemini API call failed during transcription: {e}"
    
    try:
        client.files.delete(name=gemini_file.name)
        print(f"✅ Deleted uploaded file {gemini_file.name}")
    except Exception:
        print(f"Warning: failed to delete uploaded file {getattr(gemini_file, 'name', '<unknown>')} from Gemini")

    try:
        transcript = response.candidates[0].content.parts[0].text.strip()
        if not transcript:
            return "ERROR: Gemini returned an empty transcript."
        return transcript
    except Exception as e:
        return f"ERROR: Gemini returned an unparseable response structure: {e}"