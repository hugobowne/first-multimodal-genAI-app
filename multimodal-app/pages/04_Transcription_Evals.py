import streamlit as st

st.markdown("# Transcription Evals")
st.dataframe(st.session_state["transcription_evals_df"])
