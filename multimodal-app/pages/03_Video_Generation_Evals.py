import streamlit as st

st.markdown("# Video Generation Evals")
st.dataframe(st.session_state["video_gen_evals_df"])
