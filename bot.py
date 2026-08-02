import os
import base64
import html
import re
import tempfile
import threading
import cv2
import telebot
from telebot import types
from flask import Flask
from huggingface_hub import InferenceClient

# --- Tokens --- Read from Render Environment Variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Initialization ---
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode=None)

# provider="auto" is required here so Hugging Face can find a provider for Audio tasks, 
# while we force Groq for text and vision using the :groq suffix below.
hf_client = InferenceClient(token=HF_TOKEN, provider="auto", timeout=120)

# --- Verified Models ---
CHAT_MODEL = "openai/gpt-oss-120b:groq" 
VISION_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct:groq"
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"

# --- UI Keyboards ---
def get_main_menu_keyboard():
    """Generates the interactive main menu inline keyboard."""
    markup = types.InlineKeyboardMarkup(row_width=2)
    btn_chat = types.InlineKeyboardButton("💬 Ask AI", callback_data="action_chat_help")
    btn_vision = types.InlineKeyboardButton("📸 Vision & Image", callback_data="action_vision_help")
    btn_audio = types.InlineKeyboardButton("🎙️ Audio Transcribe", callback_data="action_audio_help")
    btn_video = types.InlineKeyboardButton("🎬 Video Keyframe", callback_data="action_video_help")
    markup.add(btn_chat, btn_vision, btn_audio, btn_video)
    return markup

# --- Helper Functions ---
def save_telegram_file(file_id, suffix):
    """Downloads a file from Telegram servers to a local temporary file."""
    file_info = bot.get_file(file_id)
    file_bytes = bot.download_file(file_info.file_path)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(temp_file.name, "wb") as f:
        f.write(file_bytes)
    return temp_file.name

def markdown_to_safe_html(text):
    """Safely escapes text and converts light bolding for Telegram HTML mode."""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<b>\1</b>", escaped)
    return escaped

def safe_reply(message, text, reply_markup=None):
    """Replies safely using HTML parsing with optional inline keyboards."""
    try:
        return bot.reply_to(message, markdown_to_safe_html(text), parse_mode="HTML", reply_markup=reply_markup)
    except telebot.apihelper.ApiTelegramException:
        return bot.reply_to(message, text, parse_mode=None, reply_markup=reply_markup)

def extract_keyframe(video_path):
    """Extracts a middle frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    temp_frame = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_frame.name, frame)
    return temp_frame.name

def caption_image(image_path):
    """Describes an image using a robust vision model on Groq."""
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    response = hf_client.chat_completion(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one short sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            }
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

# --- Bot Command Handlers ---
@bot.message_handler(commands=["start", "help", "menu"])
def handle_start(message):
    welcome_text = (
        "🤖 <b>Welcome to FlowMind AI!</b>\n\n"
        "I can process text, analyze photos, transcribe audio, and inspect video keyframes.\n\n"
        "Tap an option below to get started, or just send me a message directly!"
    )
    safe_reply(message, welcome_text, reply_markup=get_main_menu_keyboard())

# --- Callback Query Handler (Interactive Button Presses) ---
@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(call):
    # Acknowledge the callback to remove the loading spinner
    bot.answer_callback_query(call.id)

    if call.data == "action_chat_help":
        safe_reply(call.message, "💬 <b>AI Chat:</b> Type any prompt or question in the chat, and I'll respond instantly!")
    elif call.data == "action_vision_help":
        safe_reply(call.message, "📸 <b>Vision Mode:</b> Send or forward any photo, and I'll generate a description for you.")
    elif call.data == "action_audio_help":
        safe_reply(call.message, "🎙️ <b>Speech-to-Text:</b> Send a voice note or audio file, and I will transcribe it.")
    elif call.data == "action_video_help":
        safe_reply(call.message, "🎬 <b>Video Analysis:</b> Send a short video clip, and I will extract and describe a keyframe.")

# --- Content Handlers ---
@bot.message_handler(func=lambda msg: msg.content_type == "text")
def handle_text(message):
    bot.send_chat_action(message.chat.id, "typing")
    try:
        response = hf_client.chat_completion(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant in a Telegram chat. "
                        "Telegram does not support Markdown tables or '###' headers. "
                        "Never use tables or '###' headers in your response. "
                        "Instead, format your response using bullet points, bold text (*word*), and emojis to keep it clean and scannable."
                    )
                },
                {"role": "user", "content": message.text}
            ],
            max_tokens=1000
        )
        raw_reply = response.choices[0].message.content

        if "</think>" in raw_reply:
            reply_text = raw_reply.split("</think>")[-1].strip()
        else:
            reply_text = raw_reply.strip()

        # Reply with pure text, no extra buttons underneath
        safe_reply(message, reply_text)
    except Exception as e:
        print(f"Text Processing Error: {e}")
        safe_reply(message, f"⚠️ Error processing text: {e}")

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    bot.send_chat_action(message.chat.id, "typing")
    temp_path = None
    try:
        temp_path = save_telegram_file(message.photo[-1].file_id, ".jpg")
        caption = caption_image(temp_path).capitalize()
        safe_reply(message, f"📸 <b>Image Description:</b>\n\n{caption}")
    except Exception as e:
        print(f"Photo Processing Error: {e}")
        safe_reply(message, f"⚠️ Error processing image: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@bot.message_handler(content_types=["voice", "audio"])
def handle_audio(message):
    bot.send_chat_action(message.chat.id, "typing")
    temp_path = None
    try:
        file_id = message.voice.file_id if message.voice else message.audio.file_id
        temp_path = save_telegram_file(file_id, ".ogg")
        result = hf_client.automatic_speech_recognition(temp_path, model=ASR_MODEL)
        safe_reply(message, f"🎙️ <b>Transcription:</b>\n\n\"{result.text}\"")
    except Exception as e:
        print(f"Audio Processing Error: {e}")
        safe_reply(message, f"⚠️ Error transcribing audio: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@bot.message_handler(content_types=["video"])
def handle_video(message):
    bot.send_chat_action(message.chat.id, "typing")
    video_path, frame_path = None, None
    try:
        video_path = save_telegram_file(message.video.file_id, ".mp4")
        frame_path = extract_keyframe(video_path)
        if frame_path:
            caption = caption_image(frame_path).capitalize()
            safe_reply(message, f"🎬 <b>Video Snapshot:</b>\n\n{caption}")
        else:
            safe_reply(message, "Unable to extract a keyframe from this video.")
    except Exception as e:
        print(f"Video Processing Error: {e}")
        safe_reply(message, f"⚠️ Error processing video: {e}")
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if frame_path and os.path.exists(frame_path):
            os.remove(frame_path)

# --- Web Server ---
app = Flask(__name__)

@app.route('/')
def home():
    return "Telegram Bot is running!"

def run_bot():
    print("⚡ Starting Telegram Bot polling loop...")
    bot.infinity_polling(timeout=20, long_polling_timeout=5)

if not hasattr(app, 'bot_started'):
    app.bot_started = True
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
