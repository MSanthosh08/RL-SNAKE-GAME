
import streamlit as st
import pandas as pd
import os

st.title("RL Snake Training Dashboard")

tab1, tab2 = st.tabs(["MLP DQN (State-Based)", "CNN DQN (Pixel-Based)"])

with tab1:
    st.subheader("State-Based Agent")
    log_file = "training_log.csv"
    if not os.path.exists(log_file):
        st.info("training_log.csv not found yet. Start training with train.py.")
    else:
        df = pd.read_csv(log_file)
        st.line_chart(df.set_index("game")[["score", "mean_score", "record"]])
        st.dataframe(df.tail(20))

with tab2:
    st.subheader("CNN Agent")
    log_file_cnn = "training_log_cnn.csv"
    if not os.path.exists(log_file_cnn):
        st.info("training_log_cnn.csv not found yet. Start training with train_cnn.py.")
    else:
        dfc = pd.read_csv(log_file_cnn)
        st.line_chart(dfc.set_index("game")[["score", "mean_score", "record"]])
        st.dataframe(dfc.tail(20))

st.write("Tip: Reload the page periodically to see live updates while training is running.")
