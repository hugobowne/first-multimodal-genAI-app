import streamlit as st

st.markdown("# Image Generation Evals")
st.dataframe(st.session_state["image_gen_evals_df"])
