import io
import time
import uuid

import asyncio
import aiohttp
import replicate
import pandas as pd
import streamlit as st
from openai import OpenAI

from constants import *

def generate_text(text: str, model: str) -> str:
    text_gen_response = ""
    client = OpenAI()
    t0 = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": st.session_state.text_gen_sys_prompt},
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
                "system_prompt": st.session_state.text_gen_sys_prompt,
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

async def text_to_audio(text: str, src: str) -> bytes:
    st.session_state["running_audio_job"] = True
    st.session_state[f"{src}_audio_bytes"] = None

    print(f"[DEBUG] Generating audio...")
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            HF_BARK_ENDPOINT, headers=bark_api_headers, json={"inputs": text}
        ) as response:
            tf = time.time()
            print(f"[DEBUG] text_to_audio request took {tf - t0:.2f} seconds")
            if response.status == 200:
                out_dir = os.path.join(AUDIO_DATA_SINK, src)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}_audio.wav")
                with open(out_path, "wb") as f:
                    f.write(await response.read())
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
                if src == 'user':
                    st.session_state["user_audio_bytes"] = await response.read()
                elif src == 'llm':
                    st.session_state["llm_audio_bytes"] = await response.read()
                st.session_state["running_audio_job"] = False
            else:
                st.session_state["running_audio_job"] = False
                raise Exception(
                    f"Request failed with status code {response.status}: {await response.text()}"
                )

async def text_to_image(text: str, negative_prompt: str, src: str = "human") -> str:
    st.session_state["running_image_job"] = True
    input = {
        "seed": 42,
        "prompt": text,
        "aspect_ratio": "3:2",
        "output_quality": 79,
        "negative_prompt": negative_prompt,
    }
    print(f"[DEBUG] Generating image...")
    t0 = time.time()
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, replicate.run, REPLICATE_IMAGE_MODEL_ID, input)
    tf = time.time()
    print(f"[DEBUG] text_to_image request took {tf - t0:.2f} seconds")

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
        if src == 'user':
            st.session_state["user_image_url"] = image_url
        elif src == 'llm':
            st.session_state["llm_image_url"] = image_url
        st.session_state["running_image_job"] = False
    else:
        st.session_state["running_image_job"] = False
        raise Exception("Text-to-image model did not return a valid URL.")

async def text_to_video(text: str, max_frames: int = 100, sampler: str = "klms", src: str = "user") -> str:
    st.session_state["running_video_job"] = True
    input = {
        "sampler": sampler,
        "max_frames": max_frames,
        "animation_prompts": text
    }
    t0 = time.time()
    print("[DEBUG] Generating video...")
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, replicate.run, REPLICATE_VIDEO_MODEL_ID, input)
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
    if src == 'user':
            st.session_state["user_video_url"] = video_url
    elif src == 'llm':
        st.session_state["llm_video_url"] = video_url
    st.session_state["running_video_job"] = False

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