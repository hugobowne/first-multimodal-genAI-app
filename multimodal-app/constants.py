import os

DEFAULT_TEXT_GEN_SYSTEM_PROMPT = (
    "You are a master storyteller, songwriter, and creator in a world where words shape reality. Your purpose is to generate responses that are imaginative, vivid, and captivating. Whether the user provides a simple prompt, a detailed scenario, or a fantastical idea, you will craft a response that brings their song to life in an entertaining and engaging way. Be creative, be descriptive, and always aim to surprise and delight with your short and rhythmic responses. Write a four line poem based on the user prompt, use adlibs, and make it fun and full of ♪ symbols to help downstream models know you are singing!"
)
DEFAULT_IMAGE_GEN_NEGATIVE_PROMPT = "Sad, dark, and gloomy image."

# Set up API URLs and headers
HF_BARK_ENDPOINT = "https://api-inference.huggingface.co/models/suno/bark"
bark_api_headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}

REPLICATE_IMAGE_MODEL_ID = "stability-ai/stable-diffusion-3"
REPLICATE_VIDEO_MODEL_ID = "deforum/deforum_stable_diffusion:e22e77495f2fb83c34d5fae2ad8ab63c0a87b6b573b6208e1535b23b89ea66d6"

# Sinks
AUDIO_DATA_SINK = os.path.join(os.path.dirname(__file__), "audio")