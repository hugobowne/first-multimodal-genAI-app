import streamlit as st

st.markdown("# Text Generation Evals")
st.dataframe(st.session_state["text_gen_evals_df"])
