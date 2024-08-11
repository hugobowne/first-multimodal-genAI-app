import asyncio
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from api import transcribe_audio, text_to_audio, text_to_image, text_to_video
from constants import *

def set_users_initial_prompt(col_handler):
    text = col_handler.chat_input("Say something")
    st.session_state["text"] = text
    st.session_state.user_init_audio_bytes = audio_recorder(
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
    )
    if "user_init_audio_bytes" in st.session_state and st.session_state.user_init_audio_bytes is not None:
        col_handler.audio(data=st.session_state["user_init_audio_bytes"], format="audio/wav")
        if col_handler.button("Transcribe"):
            transcription = transcribe_audio(st.session_state["user_init_audio_bytes"])
            st.session_state["text"] = transcription.text

def display_user_info(col_handler):
    with col_handler.chat_message("user", avatar="👤"):
        col_handler.write(st.session_state["text"])
        if st.session_state["user_audio_bytes"] is not None:
            col_handler.audio(st.session_state["user_audio_bytes"])
        if st.session_state["user_image_url"] is not None:
            col_handler.image(st.session_state["user_image_url"])
        if st.session_state["user_video_url"] is not None:
            col_handler.video(st.session_state["user_video_url"])

def display_llm_info(col_handler, f):
    with col_handler.chat_message("ai", avatar="🤖"):
        if st.session_state["text_gen_stream_resp"] is None:
            st.session_state["text_gen_stream_resp"] = col_handler.write_stream(
                f
            )
            st.rerun()
        else:
            col_handler.write(st.session_state["text_gen_stream_resp"])
        if st.session_state["llm_audio_bytes"] is not None:
            col_handler.audio(st.session_state["llm_audio_bytes"])
        if st.session_state["llm_image_url"] is not None:
            col_handler.image(st.session_state["llm_image_url"])
        if st.session_state["llm_video_url"] is not None:
            col_handler.video(st.session_state["llm_video_url"])

def display_audio_section(col_handler):
    col_handler.subheader("Text-to-audio generation")
    col_handler.write(f"Click the buttons to generate audio using Suno Bark")
    if not st.session_state['running_audio_job'] or st.session_state['running_image_job'] or st.session_state['running_video_job']:
        if col_handler.button("Generate audio for user prompt"):
            return asyncio.create_task(text_to_audio(st.session_state["text"], "user"))
        if col_handler.button("Generate audio for AI prompt"):
            return asyncio.create_task(text_to_audio(st.session_state["text_gen_stream_resp"], "llm"))
    else:
        col_handler.write('Please wait for current job to complete.')
    return None 

def display_image_section(col_handler):
    col_handler.subheader("Text-to-image generation")
    col_handler.write(
        "Click the buttons to generate an image using {}".format(st.session_state.image_model)
    )
    negative_prompt = col_handler.text_area("Negative prompt", st.session_state['negative_prompt'])
    st.session_state['negative_prompt'] = negative_prompt
    if not st.session_state['running_audio_job'] or st.session_state['running_image_job'] or st.session_state['running_video_job']:
        if col_handler.button("Generate image for user prompt"):
            return asyncio.create_task(text_to_image(st.session_state["text"], st.session_state['negative_prompt'], src="user"))
        if col_handler.button("Generate image for AI prompt"):
            return asyncio.create_task(text_to_image(st.session_state["text_gen_stream_resp"],st.session_state['negative_prompt'], src="llm"))
    else:
        col_handler.write('Please wait for current job to complete.')
    return None 

def display_video_section(col_handler):
    col_handler.subheader("Text-to-video generation")
    col_handler.write(
        "Click the buttons to generate a video using {}".format(st.session_state.video_model)
    )
    if not st.session_state['running_audio_job'] or st.session_state['running_image_job'] or st.session_state['running_video_job']:
        if col_handler.button("Generate video for user prompt"):
            return asyncio.create_task(text_to_video(st.session_state["text"], src="user"))
        if col_handler.button("Generate video for AI prompt"):
            return asyncio.create_task(text_to_video(st.session_state["text_gen_stream_resp"], src="llm"))
    else:
        col_handler.write('Please wait for current job to complete.')
    return None 