from functools import partial
import os
import asyncio
import streamlit as st

from constants import *
from ui import (
    set_users_initial_prompt,
    display_user_info,
    display_llm_info,
    display_audio_section,
    display_image_section,
    display_video_section,
)
from api import (
    generate_text, 
    text_to_audio,
    text_to_image,
    text_to_video
)
from utils import init_session_state, show_quick_reset_option

st.set_page_config(page_title="Audio Chat", page_icon="🎤", layout="wide")

async def background_tasks(placeholder):
    while True:
        _n_complete = sum([t.done() for t in st.session_state.tasks])
        with placeholder:
            st.write(f'Completed `{_n_complete}` of `{len(st.session_state.tasks)}` generations. 🚨 Starting other tasks will erase this queue 🚨')
        if _n_complete != len(st.session_state.tasks):
            await asyncio.sleep(1.2)
        else:
            st.rerun()
            break

async def main():

    if 'text' not in st.session_state:
        init_session_state()

    left_column, right_column = st.columns(2)
    left_column.title("User input")
    right_column.title("AI generations")
    placeholder = right_column.empty()

    if st.session_state["text"] is None:
        left_column.subheader("Text generation")
        openai_model_ls = ["openai:gpt-3.5-turbo", "openai:gpt-4o-mini", "openai:gpt-4o"]
        groq_model_ls = ["groq:llama-3.1-70b-versatile", "groq:mixtral-8x7b-32768", "groq:gemma2-9b-it"]
        model_choice = left_column.selectbox("Initial text reply model", groq_model_ls + openai_model_ls)
        model_choice_ls = model_choice.split(':')
        st.session_state.init_model_provider, st.session_state.init_model = model_choice_ls[0], model_choice_ls[1]
        st.session_state.text_gen_sys_prompt = left_column.text_area("System prompt", DEFAULT_TEXT_GEN_SYSTEM_PROMPT)
        set_users_initial_prompt(left_column)
    else:
        show_quick_reset_option(left_column)
        left_column.subheader("Pick your poison")
        st.session_state.image_model = left_column.selectbox("Image model", REPLICATE_IMAGE_MODEL_ID_LS)
        st.session_state.video_model = left_column.selectbox("Video model", REPLICATE_VIDEO_MODEL_ID_LS)

    _generate_text = partial(generate_text, st.session_state["text"], st.session_state.init_model)

    if st.session_state["text"]:
        display_user_info(right_column)
        display_llm_info(right_column, _generate_text)
        if left_column.button('Run all tasks concurrently'):
            st.session_state.tasks = [
                asyncio.create_task(text_to_image(st.session_state["text"], st.session_state['negative_prompt'], src="user")),
                asyncio.create_task(text_to_image(st.session_state["text_gen_stream_resp"],st.session_state['negative_prompt'], src="llm")),
                asyncio.create_task(text_to_video(st.session_state["text"], src="user")),
                asyncio.create_task(text_to_video(st.session_state["text_gen_stream_resp"], src="llm")),
                asyncio.create_task(text_to_audio(st.session_state["text"], "user")),
                asyncio.create_task(text_to_audio(st.session_state["text_gen_stream_resp"], "llm")),
            ]
            await background_tasks(placeholder)
            st.rerun()
        else:
            audio_task = display_audio_section(left_column)
            image_task = display_image_section(left_column)
            video_task = display_video_section(left_column)
            st.session_state.tasks = list(filter(lambda x: x is not None, [audio_task, image_task, video_task]))
            if st.session_state.tasks:
                await background_tasks(placeholder)
                st.rerun()

asyncio.run(main())