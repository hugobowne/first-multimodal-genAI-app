import io
import time
import uuid

import asyncio
import aiohttp
import replicate
import pandas as pd
import streamlit as st
from openai import OpenAI
from groq import Groq

from constants import *

def generate_text(text: str, model: str) -> str:
    text_gen_response = ""

    if st.session_state.init_model_provider == "groq":
        client = Groq()
    elif st.session_state.init_model_provider == 'openai':
        client = OpenAI()
    
    t0 = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": st.session_state.text_gen_sys_prompt},
            {"role": "user", "content": text},
        ],
        stream=True
    )
    for chunk in completion:
        if chunk.usage is None and chunk.choices[0].delta.content is not None:
            print(f"[DEBUG] Regular chunk: {chunk}")
            text_gen_response += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content
        else:
            print(f"[DEBUG] Final chunk: {chunk}")
            data = {
                "prompt": text,
                "system_prompt": st.session_state.text_gen_sys_prompt,
                "response": text_gen_response,
                "model": model,
                "client_time": time.time() - t0,
                "date": pd.Timestamp.now()
            }
            df = pd.DataFrame(data, index=[0])
            st.session_state["text_gen_evals_df"] = pd.concat(
                [st.session_state["text_gen_evals_df"], df], ignore_index=True
            )  


# Access the API key from secrets
REPLICATE_API_KEY = st.secrets["REPLICATE_API_TOKEN"]
client = replicate.Client(api_token=REPLICATE_API_KEY)


async def text_to_audio(text: str, src: str) -> bytes:
    st.session_state["running_audio_job"] = True
    st.session_state[f"{src}_audio_bytes"] = None

    print(f"[DEBUG] Generating audio...")
    t0 = time.time()

    # Define the input parameters for the model
    input_params = {
        "prompt": text,
        "text_temp": 0.7,
        "output_full": False,
        "waveform_temp": 0.7,
        "history_prompt": "announcer",
        # "duration": 30  # Uncomment if you want to set a specific duration
    }

    try:
        # Run the model using Replicate API
        output = client.run(
            "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
            input=input_params
        )
        tf = time.time()
        print(f"[DEBUG] text_to_audio request took {tf - t0:.2f} seconds")
        print(f"[DEBUG] Replicate API output: {output}")

        # Fetch the audio from the returned URL
        audio_url = output['audio_out']
        async with aiohttp.ClientSession() as session:
            async with session.get(audio_url) as audio_response:
                audio_bytes = await audio_response.read()

        out_dir = os.path.join(AUDIO_DATA_SINK, src)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}_audio.wav")

        with open(out_path, "wb") as f:
            f.write(audio_bytes)

        data = {
            "text": text,
            "date": pd.Timestamp.now(),
            "model": "suno-ai/bark",
            "provider": "Replicate",
            "client_time": tf - t0,
        }
        df = pd.DataFrame(data, index=[0])
        st.session_state["audio_gen_evals_df"] = pd.concat(
            [st.session_state["audio_gen_evals_df"], df], ignore_index=True
        )

        if src == 'user':
            st.session_state["user_audio_bytes"] = audio_bytes
        elif src == 'llm':
            st.session_state["llm_audio_bytes"] = audio_bytes
        
        st.session_state["running_audio_job"] = False

    except Exception as e:
        st.session_state["running_audio_job"] = False
        raise Exception(f"Request failed: {e}")

async def text_to_image(text: str, negative_prompt: str, src: str = "human") -> str:
    st.session_state["running_image_job"] = True
    if st.session_state.image_model.startswith('stability-ai/stable-diffusion-3'):
        input = {
            "seed": 42,
            "prompt": text,
            "aspect_ratio": "3:2",
            "output_quality": 79,
            "negative_prompt": negative_prompt,
        }
    elif st.session_state.image_model.startswith('black-forest-labs/flux-dev'):
        input = {
            "prompt": text,
            "guidance": 3.5,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
            "prompt_strength": 0.8
        }
    else:
        raise ValueError(f'Unsupported video model/version type {st.session_state.image_model}.')
    
    print(f"[DEBUG] Generating image...")
    t0 = time.time()
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, replicate.run, st.session_state.image_model, input)
    tf = time.time()
    print(f"[DEBUG] text_to_image request took {tf - t0:.2f} seconds")

    if output and isinstance(output, list) and len(output) > 0:
        image_url = output[0]
        data = {
            "text": text,
            "negative_prompt": negative_prompt,
            "image_url": image_url,
            "date": pd.Timestamp.now(),
            "model": st.session_state.image_model,
            "provider": "Replicate",
            "client_time": tf - t0
        }
        df = pd.DataFrame(data, index=[0])
        st.session_state["image_gen_evals_df"] = pd.concat(
            [st.session_state["image_gen_evals_df"], df], ignore_index=True
        )
        if src == 'user':
            st.session_state["user_image_url"] = image_url
        elif src == 'llm':
            st.session_state["llm_image_url"] = image_url
        st.session_state["running_image_job"] = False
    else:
        st.session_state["running_image_job"] = False
        raise Exception("Text-to-image model did not return a valid URL.")

async def text_to_video(text: str, src: str = "user") -> str:
    st.session_state.running_video_job = True
    if st.session_state.video_model.startswith('lucataco/hotshot-xl'):
        input = {
            "prompt": text,
            "mp4": True
        }
    elif st.session_state.video_model.startswith('deforum/deforum_stable_diffusion'):
        input = {
            "animation_prompts": text,
            "sampler": "klms",
            "max_frames": 100,
        }
    else:
        raise ValueError(f'Unsupported video model/version type {st.session_state.video_model}.')
    t0 = time.time()
    print("[DEBUG] Generating video...")
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, replicate.run, st.session_state.video_model, input)
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
        "model": st.session_state.video_model,
        "provider": "Replicate",
        "client_time": tf - t0,
    }
    df = pd.DataFrame(data, index=[0])
    st.session_state["video_gen_evals_df"] = pd.concat(
        [st.session_state["video_gen_evals_df"], df], ignore_index=True
    )
    print("[DEBUG] Video URL:", video_url)
    if src == 'user':
            st.session_state["user_video_url"] = video_url
    elif src == 'llm':
        st.session_state["llm_video_url"] = video_url
    st.session_state["running_video_job"] = False

def transcribe_audio(audio_data):
    try:
        if st.session_state.init_model_provider == "groq":
            client = Groq()
            model_wh = "whisper-large-v3"
            print("[DEBUG] Transcribing audio with groq/whisper-1...")
        elif st.session_state.init_model_provider == 'openai':
            client = OpenAI()
            model_wh = "whisper-1"
            print("[DEBUG] Transcribing audio with openai/whisper-1...")
        file_like = io.BytesIO(audio_data)
        file_like.name = "audio.wav"  
        file_like.seek(0)

        t0 = time.time()
        transcription = client.audio.transcriptions.create(
            model=model_wh, 
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