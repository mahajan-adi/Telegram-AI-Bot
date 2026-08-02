import os
import base64
import html
import re
import tempfile
import threading
import cv2
import telebot
from flask import Flask
from huggingface_hub import InferenceClient

# Tokens - Read from Render Environment Variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize Telegram Bot and Hugging Face Client.
# NOTE: no default parse_mode here. Telegram's legacy "Markdown" mode is very
# strict (every *, _, [, ` must be perfectly paired) and AI-generated text
# frequently breaks it, causing "can't parse entities" errors. We use HTML
# mode instead, applied per-message via safe_reply(), which escapes text
# first so it can never accidentally form invalid markup.
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode=None)

# provider="auto" lets HF pick whichever enabled provider serves the model.
# Make sure providers are enabled at https://huggingface.co/settings/inference-providers
hf_client = InferenceClient(token=HF_TOKEN, provider="auto", timeout=120)

# --- Model choices ---
# These are models that are actually live on Inference Providers as of this
# writing. If one of them stops working, check the model's page on
# huggingface.co -> "Inference Providers" tab to see what's currently hosting
# it, and update the provider/model below to match.
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"                # served by Together AI
VISION_MODEL = "moonshotai/Kimi-K3"                    # served by Together AI (note: very large model, expect slower responses)
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"                      # served by DeepInfra
# Note: provider is set once on the InferenceClient above (provider="auto").
# Older huggingface_hub versions don't accept a per-call `provider=` kwarg on
# chat_completion()/automatic_speech_recognition(), so we rely on "auto" to
# pick the right one of your enabled providers for each model.

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
    """
    Converts light **bold**/*bold* markdown into real <b> tags, and
    HTML-escapes everything else first so the AI's text can never
    accidentally produce broken or malicious markup.
    """
    escaped = html.escape(text)
    # Double-asterisk bold (most common LLM style)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    # Leftover single-asterisk bold
    escaped = re.sub(r"\*(.+?)\*", r"<b>\1</b>", escaped)
    return escaped

def safe_reply(message, text):
    """
    Replies with lightweight bold formatting via HTML (which Telegram
    parses far more forgivingly than Markdown). Falls back to plain,
    unformatted text if anything still goes wrong, so a reply is never
    silently lost.
    """
    try:
        bot.reply_to(message, markdown_to_safe_html(text), parse_mode="HTML")
    except telebot.apihelper.ApiTelegramException:
        bot.reply_to(message, text, parse_mode=None)

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
    """Describes an image using a vision-capable chat model (replaces BLIP)."""
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

# --- Bot Handlers ---
@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    safe_reply(message, "👋 Bot is online 24/7! Send me Text, Photo, Audio, or Video.")

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

        # Remove internal thinking block if present
        if "</think>" in raw_reply:
            reply_text = raw_reply.split("</think>")[-1].strip()
        else:
            reply_text = raw_reply.strip()

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
        safe_reply(message, f"📸 **Image Description:**\n{caption}")
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
        safe_reply(message, f"🎙️ **Transcription:**\n\"{result.text}\"")
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
            safe_reply(message, f"🎬 **Video Snapshot:**\n{caption}")
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

# Guard against duplicate polling threads started by Gunicorn workers
if not hasattr(app, 'bot_started'):
    app.bot_started = True
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
