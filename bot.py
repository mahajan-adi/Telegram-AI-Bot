import os
import tempfile
import threading
import cv2
import telebot
from flask import Flask
from huggingface_hub import InferenceClient

# Tokens - Make sure to set these in your Render Environment Variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize the Telegram Bot and Hugging Face Client
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="Markdown")
hf_client = InferenceClient(token=HF_TOKEN)

# --- Helper Functions ---
def save_telegram_file(file_id, suffix):
    """Downloads a file from Telegram servers to a local temporary file."""
    file_info = bot.get_file(file_id)
    file_bytes = bot.download_file(file_info.file_path)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(temp_file.name, "wb") as f:
        f.write(file_bytes)
    return temp_file.name

def extract_keyframe(video_path):
    """Extracts a middle frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
    success, frame = cap.read()
    cap.release()
    if not success: return None
    temp_frame = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_frame.name, frame)
    return temp_frame.name

# --- Bot Handlers ---
@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(message, "Bot is online 24/7! Send me Text, Photo, Audio, or Video.")

@bot.message_handler(func=lambda msg: msg.content_type == "text")
def handle_text(message):
    bot.send_chat_action(message.chat.id, "typing")
    try:
        # Using Hugging Face chat_completion API
        response = hf_client.chat_completion(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            messages=[{"role": "user", "content": message.text}],
            max_tokens=200
        )
        # Extract the text from the API response
        bot.reply_to(message, response.choices[0].message.content)
    except Exception as e:
        bot.reply_to(message, f"Error: {e}")

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    bot.send_chat_action(message.chat.id, "typing")
    try:
        temp_path = save_telegram_file(message.photo[-1].file_id, ".jpg")
        
        # Using Hugging Face image_to_text API for image captioning
        result = hf_client.image_to_text(temp_path, model="Salesforce/blip-image-captioning-base")
        bot.reply_to(message, f"📸 **Image Description:**\n{result[0]['generated_text'].capitalize()}")
    except Exception as e:
        bot.reply_to(message, f"Error: {e}")
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path): os.remove(temp_path)

@bot.message_handler(content_types=["voice", "audio"])
def handle_audio(message):
    bot.send_chat_action(message.chat.id, "typing")
    try:
        file_id = message.voice.file_id if message.voice else message.audio.file_id
        temp_path = save_telegram_file(file_id, ".ogg")
        
        # Using Hugging Face automatic_speech_recognition API
        result = hf_client.automatic_speech_recognition(temp_path, model="openai/whisper-tiny")
        bot.reply_to(message, f"🎙️ **Transcription:**\n\"{result.text}\"")
    except Exception as e:
         bot.reply_to(message, f"Error: {e}")
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path): os.remove(temp_path)

@bot.message_handler(content_types=["video"])
def handle_video(message):
    bot.send_chat_action(message.chat.id, "typing")
    video_path, frame_path = None, None
    try:
        video_path = save_telegram_file(message.video.file_id, ".mp4")
        frame_path = extract_keyframe(video_path)
        if frame_path:
            # Re-using the image_to_text API on the extracted keyframe
            result = hf_client.image_to_text(frame_path, model="Salesforce/blip-image-captioning-base")
            bot.reply_to(message, f"🎬 **Video Snapshot:**\n{result[0]['generated_text'].capitalize()}")
        else:
             bot.reply_to(message, "Unable to extract a frame from this video.")
    except Exception as e:
         bot.reply_to(message, f"Error: {e}")
    finally:
        if video_path and os.path.exists(video_path): os.remove(video_path)
        if frame_path and os.path.exists(frame_path): os.remove(frame_path)

# --- Web Server (Required for Render to keep the service alive) ---
app = Flask(__name__)

@app.route('/')
def home():
    return "Telegram Bot is running!"

def run_bot():
    """Runs the Telegram bot polling loop."""
    try:
        bot.infinity_polling(timeout=20, long_polling_timeout=5)
    except Exception as e:
        print(f"Bot crashed: {e}")

# Start the Telegram bot loop in a background thread IMMEDIATELY.
# This ensures it starts even when Gunicorn imports the file.
bot_thread = threading.Thread(target=run_bot, daemon=True)
bot_thread.start()

# Start the Flask app for local testing. 
# Gunicorn bypasses this block when deployed on Render.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
        
