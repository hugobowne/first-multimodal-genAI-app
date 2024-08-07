import streamlit as st
import pandas as pd
from constants import DEFAULT_IMAGE_GEN_NEGATIVE_PROMPT

def init_session_state():
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
    if "transcription_evals_df" not in st.session_state:
        st.session_state["transcription_evals_df"] = pd.DataFrame()
    if "tasks" not in st.session_state:
        st.session_state['tasks'] = []
    st.session_state["running_text_job"] = False
    st.session_state["running_audio_job"] = False
    st.session_state["running_image_job"] = False
    st.session_state["running_video_job"] = False

def show_quick_reset_option(col_handler):
    if col_handler.button("Reset session data"):
        st.session_state["text"] = None
        st.session_state["text_gen_stream_resp"] = None
        st.session_state["user_audio_bytes"] = None
        st.session_state["llm_audio_bytes"] = None
        st.session_state["user_image_url"] = None
        st.session_state["llm_image_url"] = None
        st.session_state['negative_prompt'] = DEFAULT_IMAGE_GEN_NEGATIVE_PROMPT
        st.session_state["user_video_url"] = None
        st.session_state["llm_video_url"] = None
        st.session_state["running_text_job"] = False
        st.session_state["running_audio_job"] = False
        st.session_state["running_image_job"] = False
        st.session_state["running_video_job"] = False
        st.rerun()

