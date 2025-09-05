import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import zipfile
import os
import re

# ================================
# Streamlit Config
# ================================
st.set_page_config(page_title="Analisis Sentimen Teks", page_icon="💬")
st.title("💬 Analisis Sentimen Komentar Sosial Media")

# ================================
# Repo HuggingFace
# ================================
REPO_ID = "zahratalitha/sentimenteks"
MODEL_ZIP = "sentiment_model_tf.zip"
TOKENIZER_ZIP = "tokenizer.zip"

MODEL_DIR = "sentiment_model_tf"
TOKENIZER_DIR = "tokenizer"

# ================================
# Download & Extract
# ================================
if not os.path.exists(MODEL_DIR):
    model_zip = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=MODEL_ZIP)
    with zipfile.ZipFile(model_zip, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

if not os.path.exists(TOKENIZER_DIR):
    tok_zip = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=TOKENIZER_ZIP)
    with zipfile.ZipFile(tok_zip, "r") as zip_ref:
        zip_ref.extractall(TOKENIZER_DIR)

from tensorflow import keras
custom_objects = {"TFOpLambda": lambda x: x}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_DIR, custom_objects=custom_objects)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return model, tokenizer

model, tokenizer = load_model()

