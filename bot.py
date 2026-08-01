import os
import tempfile
import threading
import cv2
import telebot
from flask import Flask
from huggingface_hub import InferenceClient

# Tokens
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="Markdown")
hf_client = InferenceClient(token=HF_TOKEN)

# --- Helper Functions ---
def save_telegram_file(file_id, suffix):
    file_info = bot.get_file(file_id)
    file_bytes = bot.download_file(file_info.file_path)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(temp_file.name, "wb") as f:
        f.write(file_bytes)
    return temp_file.name

def extract_keyframe(video_path):
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
@bot.message_handler(commands=["start"])
def handle_start(message):
    bot.reply_to(message, "Bot is online 24/7! Send me Text, Photo, Audio, or Video.")

@bot.message_handler(func=lambda msg: msg.content_type == "text")
def handle_text(message):
    bot.send_chat_action(message.chat.id, "typing")
    try:
        # Using Hugging Face API instead of local pipeline
        response = hf_client.chat_completion(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            messages=[{"role": "user", "content": message.text}],
            max_tokens=200
        )
        bot.reply_to(message, response.choices[0].message.content)
    except Exception as e:
        bot.reply_to(message, f"Error: {e}")

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    bot.send_chat_action(message.chat.id, "typing")
    try:
        temp_path = save_telegram_file(message.photo[-1].file_id, ".jpg")
        # Image-to-Text API
        result = hf_client.image_to_text(temp_path, model="Salesforce/blip-image-captioning-base")
        bot.reply_to(message, f"📸 **Image Description:**\n{result[0]['generated_text'].capitalize()}")
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@bot.message_handler(content_types=["voice", "audio"])
def handle_audio(message):
    bot.send_chat_action(message.chat.id, "typing")
    try:
        file_id = message.voice.file_id if message.voice else message.audio.file_id
        temp_path = save_telegram_file(file_id, ".ogg")
        # Speech-to-Text API
        result = hf_client.automatic_speech_recognition(temp_path, model="openai/whisper-tiny")
        bot.reply_to(message, f"🎙️ **Transcription:**\n\"{result.text}\"")
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@bot.message_handler(content_types=["video"])
def handle_video(message):
    bot.send_chat_action(message.chat.id, "typing")
    try:
        video_path = save_telegram_file(message.video.file_id, ".mp4")
        frame_path = extract_keyframe(video_path)
        if frame_path:
            result = hf_client.image_to_text(frame_path, model="Salesforce/blip-image-captioning-base")
            bot.reply_to(message, f"🎬 **Video Snapshot:**\n{result[0]['generated_text'].capitalize()}")
    finally:
        if os.path.exists(video_path): os.remove(video_path)
        if frame_path and os.path.exists(frame_path): os.remove(frame_path)

# --- Web Server (Required for Render) ---
app = Flask(__name__)

@app.route('/')
def home():
    return "Telegram Bot is running!"

def run_bot():
    bot.infinity_polling()

if __name__ == "__main__":
    # Start the bot in a background thread
    threading.Thread(target=run_bot).start()
    # Start the web server on the port Render assigns
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
