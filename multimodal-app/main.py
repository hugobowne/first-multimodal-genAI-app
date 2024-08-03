import os
import io
import time
import requests
import tempfile
import streamlit as st
import pandas as pd
from functools import partial
from openai import OpenAI
import replicate
import uuid
from audio_recorder_streamlit import audio_recorder
import pydub
import numpy as np
import torch

st.set_page_config(page_title="Audio Chat", page_icon="🎤", layout="wide")

# TODO: move to a config file
DEFAULT_TEXT_GEN_SYSTEM_PROMPT = (
    "You are a master storyteller, songwriter, and creator in a world where words shape reality. Your purpose is to generate responses that are imaginative, vivid, and captivating. Whether the user provides a simple prompt, a detailed scenario, or a fantastical idea, you will craft a response that brings their song to life in an entertaining and engaging way. Be creative, be descriptive, and always aim to surprise and delight with your short and rhythmic responses. Write a four line poem based on the user prompt, use adlibs, and make it fun and full of ♪ symbols to help downstream models know you are singing!"
)
DEFAULT_IMAGE_GEN_NEGATIVE_PROMPT = "Sad, dark, and gloomy image."

# Set up API URLs and headers
# Set up API URLs and headers
HF_BARK_ENDPOINT = "https://api-inference.huggingface.co/models/suno/bark"
bark_api_headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}

HF_BARK_ENDPOINT = "https://api-inference.huggingface.co/models/suno/bark"
REPLICATE_IMAGE_MODEL_ID = "stability-ai/stable-diffusion-3"
REPLICATE_VIDEO_MODEL_ID = "deforum/deforum_stable_diffusion:e22e77495f2fb83c34d5fae2ad8ab63c0a87b6b573b6208e1535b23b89ea66d6"

# Sinks
AUDIO_DATA_SINK = os.path.join(os.path.dirname(__file__), "audio")

# Layout
left_column, right_column = st.columns(2)
left_column.title("User input")
right_column.title("AI generations")

# Session storage
if "text" not in st.session_state:
    st.session_state["text"] = None
if "text_gen_stream_resp" not in st.session_state:
    st.session_state["text_gen_stream_resp"] = None
if "text_gen_evals_df" not in st.session_state:
    st.session_state["text_gen_evals_df"] = pd.DataFrame()
if "user_audio_bytes" not in st.session_state:
    st.session_state["user_audio_bytes"] = None
if "llm_audio_bytes" not in st.session_state:
    st.session_state["llm_audio_bytes"] = None
if "audio_gen_evals_df" not in st.session_state:
    st.session_state["audio_gen_evals_df"] = pd.DataFrame()
if "user_image_url" not in st.session_state:
    st.session_state["user_image_url"] = None
if "llm_image_url" not in st.session_state:
    st.session_state["llm_image_url"] = None
if "image_gen_evals_df" not in st.session_state:
    st.session_state["image_gen_evals_df"] = pd.DataFrame()
if 'negative_prompt' not in st.session_state:
    st.session_state['negative_prompt'] = DEFAULT_IMAGE_GEN_NEGATIVE_PROMPT
if "user_video_url" not in st.session_state:
    st.session_state["user_video_url"] = None
if "llm_video_url" not in st.session_state:
    st.session_state["llm_video_url"] = None
if "video_gen_evals_df" not in st.session_state:
    st.session_state["video_gen_evals_df"] = pd.DataFrame()
if "audio_gen_evals_df" not in st.session_state:
    st.session_state["audio_gen_evals_df"] = pd.DataFrame()
if "transcription_evals_df" not in st.session_state:
    st.session_state["transcription_evals_df"] = pd.DataFrame()

def generate_text(text: str, model: str) -> str:
    text_gen_response = ""
    client = OpenAI()
    t0 = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": text_gen_sys_prompt},
            {"role": "user", "content": text},
        ],
        stream=True,
        stream_options={"include_usage": True}
    )
    for chunk in completion:
        if chunk.usage is None and chunk.choices[0].delta.content is not None:
            print(f"[DEBUG] Regular chunk: {chunk}")
            text_gen_response += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content
        elif chunk.usage is not None:
            print(f"[DEBUG] Final chunk: {chunk}")
            data = {
                "prompt": text,
                "system_prompt": text_gen_sys_prompt,
                "response": text_gen_response,
                "model": model,
                "client_time": time.time() - t0,
                "date": pd.Timestamp.now(),
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens
            }
            df = pd.DataFrame(data, index=[0])
            st.session_state["text_gen_evals_df"] = pd.concat(
                [st.session_state["text_gen_evals_df"], df], ignore_index=True
            )
        else:
            print(f"[DEBUG] Empty chunk: {chunk}")
            yield ""    
        


def text_to_audio(text: str, src: str) -> bytes:
    st.session_state[f"{src}_audio_bytes"] = None
    t0 = time.time()
    response = requests.post(
        HF_BARK_ENDPOINT, headers=bark_api_headers, json={"inputs": text}
    )
    tf = time.time()
    print(f"[DEBUG] text_to_audio request took {tf - t0:.2f} seconds")
    if response.status_code == 200:
        out_dir = os.path.join(AUDIO_DATA_SINK, src)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}_audio.wav")
        with open(out_path, "wb") as f:
            f.write(response.content)
        data = {
            "text": text,
            "date": pd.Timestamp.now(),
            "model": "suno/bark",
            "provider": "Hugging Face",
            "client_time": tf - t0,
        }
        df = pd.DataFrame(data, index=[0])
        st.session_state["audio_gen_evals_df"] = pd.concat(
            [st.session_state["audio_gen_evals_df"], df], ignore_index=True
        )
        return response.content
    else:
        raise Exception(
            f"Request failed with status code {response.status_code}: {response.text}"
        )


def text_to_image(text: str, negative_prompt: str) -> str:

    input = {
        "seed": 42,
        "prompt": text,
        "aspect_ratio": "3:2",
        "output_quality": 79,
        "negative_prompt": negative_prompt,
    }
    t0 = time.time()
    output = replicate.run(REPLICATE_IMAGE_MODEL_ID, input)
    tf = time.time()

    if output and isinstance(output, list) and len(output) > 0:
        image_url = output[0]
        data = {
            "text": text,
            "negative_prompt": negative_prompt,
            "image_url": image_url,
            "date": pd.Timestamp.now(),
            "model": REPLICATE_IMAGE_MODEL_ID,
            "provider": "Replicate",
            "client_time": tf - t0,
        }
        df = pd.DataFrame(data, index=[0])
        st.session_state["image_gen_evals_df"] = pd.concat(
            [st.session_state["image_gen_evals_df"], df], ignore_index=True
        )
        return image_url
    else:
        raise Exception("Text-to-image model did not return a valid URL.")


def text_to_video(text: str, max_frames: int = 100, sampler: str = "klms") -> str:
    input = {
        "sampler": sampler,
        "max_frames": max_frames,
        "animation_prompts": text
    }
    t0 = time.time()
    print("[DEBUG] Generating video...")
    output = replicate.run(REPLICATE_VIDEO_MODEL_ID, input)
    tf = time.time()
    print("[DEBUG] Video generation complete. %s" % output)

    if output and isinstance(output, list) and len(output) > 0:
        video_url = output[0]
    elif output and isinstance(output, str):
        video_url = output
    data = {
        "text": text,
        "video_url": video_url,
        "date": pd.Timestamp.now(),
        "model": REPLICATE_VIDEO_MODEL_ID,
        "provider": "Replicate",
        "client_time": tf - t0,
    }
    df = pd.DataFrame(data, index=[0])
    st.session_state["video_gen_evals_df"] = pd.concat(
        [st.session_state["video_gen_evals_df"], df], ignore_index=True
    )
    print("[DEBUG] Video URL:", video_url)
    return video_url

def transcribe_audio(audio_data):
    try:
        client = OpenAI()
        print("[DEBUG] Transcribing audio with openai/whisper-1...")
        file_like = io.BytesIO(audio_data)
        file_like.name = "audio.wav"  
        file_like.seek(0)

        t0 = time.time()
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=file_like, 
            response_format="verbose_json"
        )
        tf = time.time()
        print(f"[DEBUG] Transcription took {tf - t0:.2f} seconds")
        print("Transcription", transcription)
        data = {
            "transcription": transcription.text,
            "duration": transcription.duration,
            "date": pd.Timestamp.now(),
            "model": "openai/whisper-1",
            "provider": "OpenAI",
            "client_time": tf - t0,
            "language": transcription.language,
        }
        df = pd.DataFrame(data, index=[0])
        st.session_state["transcription_evals_df"] = pd.concat(
            [st.session_state["transcription_evals_df"], df], ignore_index=True
        )

        print("[DEBUG] Transcription type:", type(transcription))
        print("[DEBUG] Transcription:", transcription)
        return transcription

    except Exception as e:
        print(f"[ERROR] An error occurred during transcription: {e}")
        return None


left_column.subheader("Text generation")

if st.session_state["text"] is not None:
    if reset := left_column.button("Reset session data"):
        st.session_state["text"] = None
        st.session_state["text_gen_stream_resp"] = None
        st.session_state["user_audio_bytes"] = None
        st.session_state["llm_audio_bytes"] = None
        st.session_state["user_image_url"] = None
        st.session_state["llm_image_url"] = None
        st.session_state['negative_prompt'] = DEFAULT_IMAGE_GEN_NEGATIVE_PROMPT
        st.session_state["user_video_url"] = None
        st.session_state["llm_video_url"] = None
        # st.session_state["text_gen_evals_df"] = pd.DataFrame()
        # st.session_state["audio_gen_evals_df"] = pd.DataFrame()
        # st.session_state["image_gen_evals_df"] = pd.DataFrame()
        # st.session_state["video_gen_evals_df"] = pd.DataFrame()
        st.rerun()

text_gen_sys_prompt = left_column.text_area(
    "System prompt", DEFAULT_TEXT_GEN_SYSTEM_PROMPT
)
model = left_column.selectbox("Model", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"])

if st.session_state["text"] is None:
    text = left_column.chat_input("Say something")
    st.session_state["text"] = text
    st.session_state["user_init_audio_bytes"] = audio_recorder(
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
    )
    if "user_init_audio_bytes" in st.session_state:
        left_column.audio(data=st.session_state["user_init_audio_bytes"], format="audio/wav")
        if left_column.button("Transcribe"):
            transcription = transcribe_audio(st.session_state["user_init_audio_bytes"])
            st.session_state["text"] = transcription.text

_generate_text = partial(generate_text, st.session_state["text"], model)

if st.session_state["text"]:
    with right_column.chat_message("user", avatar="👤"):
        right_column.write(st.session_state["text"])
        if st.session_state["user_audio_bytes"] is not None:
            right_column.audio(st.session_state["user_audio_bytes"])
        if st.session_state["user_image_url"] is not None:
            right_column.image(st.session_state["user_image_url"])
        if st.session_state["user_video_url"] is not None:
            right_column.video(st.session_state["user_video_url"])
    with right_column.chat_message("ai", avatar="🤖"):
        if st.session_state["text_gen_stream_resp"] is None:
            st.session_state["text_gen_stream_resp"] = right_column.write_stream(
                _generate_text
            )
            st.rerun()
        else:
            right_column.write(st.session_state["text_gen_stream_resp"])
        if st.session_state["llm_audio_bytes"] is not None:
            right_column.audio(st.session_state["llm_audio_bytes"])
        if st.session_state["llm_image_url"] is not None:
            right_column.image(st.session_state["llm_image_url"])
        if st.session_state["llm_video_url"] is not None:
            right_column.video(st.session_state["llm_video_url"])

    left_column.subheader("Text-to-audio generation")
    left_column.write(f"Click the buttons to generate audio using {HF_BARK_ENDPOINT}")

    if left_column.button("Generate audio for user prompt"):
        with st.spinner("Generating audio for user prompt..."):
            st.session_state["user_audio_bytes"] = text_to_audio(
                st.session_state["text"], "user"
            )
            st.rerun()

    if left_column.button("Generate audio for AI prompt"):
        with st.spinner("Generating audio for AI prompt..."):
            st.session_state["llm_audio_bytes"] = text_to_audio(
                st.session_state["text_gen_stream_resp"], "llm"
            )
            st.rerun()

    left_column.subheader("Text-to-image generation")
    left_column.write(
        "Click the buttons to generate an image using {}".format(REPLICATE_IMAGE_MODEL_ID)
    )

    negative_prompt = left_column.text_area("Negative prompt", st.session_state['negative_prompt'])
    st.session_state['negative_prompt'] = negative_prompt

    if left_column.button("Generate image for user prompt"):
        with st.spinner("Generating image for user prompt..."):
            st.session_state["user_image_url"] = text_to_image(
                st.session_state["text"],
                st.session_state['negative_prompt']
            )
            st.rerun()

    if left_column.button("Generate image for AI prompt"):
        with st.spinner("Generating image for AI prompt..."):
            st.session_state["llm_image_url"] = text_to_image(
                st.session_state["text_gen_stream_resp"],
                st.session_state['negative_prompt']
            )
            st.rerun()

    left_column.subheader("Text-to-video generation")
    left_column.write(
        "Click the buttons to generate a video using {}".format(REPLICATE_VIDEO_MODEL_ID)
    )

    if left_column.button("Generate video for user prompt"):
        with st.spinner("Generating video for user prompt..."):
            st.session_state["user_video_url"] = text_to_video(
                st.session_state["text"]
            )
            st.rerun()

    if left_column.button("Generate video for AI prompt"):
        with st.spinner("Generating video for AI prompt..."):
            st.session_state["llm_video_url"] = text_to_video(
                st.session_state["text_gen_stream_resp"]
            )
            st.rerun()

