# Article Structuring Assistant
# Imports
# ============

import streamlit as st
from pydantic import BaseModel
from typing import List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json


# ==========================================
# Pydantic Schema
# Defines the structure of extracted data
# ==========================================

class Article(BaseModel):
    title: str
    author: str
    published_date: str
    summary: str
    tags: List[str]


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()


# ==========================================
# Initialize Groq LLM
# ==========================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# ==========================================
# Convert LLM Outputs into Structured Objects
# ==========================================

structured_llm = llm.with_structured_output(Article)


# ==========================================
# Extraction Function
# ==========================================

def extract_article(text):
    """
    Extract structured information from raw article text.
    """

    try:
        return structured_llm.invoke(
            f"""
            You are an information extraction assistant.

            Extract the following fields from the article.

            Rules:
            - Use only information explicitly stated in the article.
            - Do not guess or infer missing information.
            - If a field is not specified, return "Not specified".
            - Keep the summary concise (2–3 sentences).
            - Return 3–5 relevant tags.

            Extract:
            - Title
            - Author
            - Published Date
            - Summary
            - Tags

            Article:
            {text}
            """
        )

    except Exception as e:
        st.error(f"Extraction failed: {e}")
        return None


# ==========================================
# Streamlit Page Configuration
# ==========================================

st.set_page_config(
    page_title="Article Structuring Assistant",
    page_icon="📰",
    layout="centered"
)


# ==========================================
# Session State
# ==========================================

if "article" not in st.session_state:
    st.session_state.article = None

if "article_input" not in st.session_state:
    st.session_state.article_input = ""


# ==========================================
# App Header
# ==========================================

st.title("📰 Article Structuring Assistant")

st.markdown("""
Convert unstructured article text into validated structured JSON using
**Groq + LangChain + Pydantic**.
""")

st.divider()


# ==========================================
# Input Method Selection
# ==========================================

input_method = st.radio(
    "Choose how you want to provide the article:",
    ["Paste Article", "Upload TXT File"]
)


# ==========================================
# Collect User Input
# ==========================================

article_text = ""

if input_method == "Paste Article":

    article_text = st.text_area(
        "Paste your article here:",
        height=250,
        placeholder="Enter article text...",
        key="article_input"
    )

else:

    uploaded_file = st.file_uploader(
        "Upload a TXT file",
        type=["txt"]
    )

    if uploaded_file is not None:

        article_text = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

        st.success("File uploaded successfully!")


# ==========================================
# Word Count Validation
# ==========================================

word_count = 0

if article_text:

    word_count = len(article_text.split())

    st.caption(f"Word Count: {word_count}/1000")

    if word_count > 1000:

        st.warning(
            "Article exceeds the maximum limit of 1000 words."
        )


# ==========================================
# Extract Button
# ==========================================

extract_button = st.button(
    "Extract Information",
    disabled=(
        not article_text or
        word_count > 1000
    )
)


# ==========================================
# Perform Extraction
# ==========================================

if extract_button:

    with st.spinner("Extracting information..."):

        article = extract_article(article_text)

        st.session_state.article = article


# ==========================================
# Display Results
# ==========================================

if st.session_state.article:

    article = st.session_state.article

    st.success("Information extracted successfully!")

    st.subheader("Extracted Information")

    st.write("**Title:**", article.title)

    st.write("**Author:**", article.author)

    st.write("**Published Date:**", article.published_date)

    st.write("**Summary:**", article.summary)

    st.write("**Tags:**", ", ".join(article.tags))


    # ======================================
    # JSON Preview
    # ======================================

    st.subheader("JSON Preview")

    st.json(article.model_dump())


    # ======================================
    # Download JSON
    # ======================================

    article_json = json.dumps(
        article.model_dump(),
        indent=4
    )

    st.download_button(
        label="📥 Download JSON",
        data=article_json,
        file_name="structured_article.json",
        mime="application/json"
    )


    # ======================================
    # Analyze Another Article
    # ======================================

    if st.button("🔄 Analyze Another Article"):

        st.session_state.clear()
        st.rerun()

        