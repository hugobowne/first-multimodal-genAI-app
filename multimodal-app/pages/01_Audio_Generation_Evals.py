import streamlit as st

st.markdown("# Audio Generation Evals")
st.dataframe(st.session_state["audio_gen_evals_df"])
