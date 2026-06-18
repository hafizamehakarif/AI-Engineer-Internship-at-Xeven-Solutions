import os
import tempfile
import json

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    JsonOutputParser,
    StrOutputParser
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pydantic import BaseModel, Field


# =====================================================
# ENVIRONMENT
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()


# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📰",
    layout="centered"
)

st.title("AI Document Analyzer")

st.write(
    "Upload a PDF or TXT file to generate summaries, "
    "extract entities, and perform semantic search."
)


# =====================================================
# PYDANTIC MODEL
# =====================================================

class Entities(BaseModel):
    people: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0
    )


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


llm = load_llm()
embedding_model = load_embedding_model()


# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"]
)

if uploaded_file:

    # =================================================
    # SAVE FILE
    # =================================================

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{uploaded_file.name.split('.')[-1]}"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # =================================================
    # LOAD DOCUMENT
    # =================================================

    try:

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)

        docs = loader.load()

        full_text = "\n".join(
            doc.page_content
            for doc in docs
        )

        st.success("Document loaded successfully.")

    except Exception as e:

        st.error(
            f"Failed to load document: {e}"
        )
        st.stop()

    # =================================================
    # CHUNKING
    # =================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(full_text)

    # =================================================
    # DOCUMENT STATS
    # =================================================

    st.subheader("Document Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Characters",
        len(full_text)
    )

    col2.metric(
        "Chunks",
        len(chunks)
    )

    col3.metric(
        "Chunk Size",
        "500"
    )

    # =================================================
    # EMBEDDINGS
    # =================================================

    with st.spinner(
        "Generating embeddings..."
    ):

        embeddings = embedding_model.encode(
            chunks
        )

    # =================================================
    # SUMMARY
    # =================================================

    st.subheader("Document Summary")

    summary_prompt = (
        PromptTemplate.from_template(
            """
            Summarize the following document
            in concise bullet points.

            Document:
            {document}
            """
        )
    )

    summary_chain = (
        summary_prompt
        | llm
        | StrOutputParser()
    )

    try:

        summary = summary_chain.invoke(
            {
                "document": full_text[:6000]
            }
        )

        st.write(summary)

    except Exception as e:

        st.error(
            f"Summary generation failed: {e}"
        )

        summary = "Summary unavailable."

    # =================================================
    # ENTITY EXTRACTION
    # =================================================

    st.subheader(
        "Extracted Entities"
    )

    parser = JsonOutputParser()

    entity_prompt = (
        PromptTemplate.from_template(
            """
            Extract entities from the
            document.

            Return ONLY valid JSON.

            {{
                "people": [],
                "organizations": [],
                "technologies": []
            }}

            Document:
            {document}
            """
        )
    )

    entity_chain = (
        entity_prompt
        | llm
        | parser
    )

    entities = Entities()

    try:

        entity_result = (
            entity_chain.invoke(
                {
                    "document": full_text[:6000]
                }
            )
        )

        entities = Entities(
            **entity_result
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.write(
                "### People"
            )

            st.write(
                entities.people
                if entities.people
                else "None"
            )

        with col2:

            st.write(
                "### Organizations"
            )

            st.write(
                entities.organizations
                if entities.organizations
                else "None"
            )

        with col3:

            st.write(
                "### Technologies"
            )

            st.write(
                entities.technologies
                if entities.technologies
                else "None"
            )

    except Exception as e:

        st.warning(
            f"Entity extraction failed: {e}"
        )

    # =================================================
    # SEMANTIC SEARCH
    # =================================================

    st.subheader(
        "Semantic Search"
    )

    query = st.text_input(
        "Ask a question about the document:"
    )

    if query:

        query_embedding = (
            embedding_model.encode(
                query
            )
        )

        scores = (
            cosine_similarity(
                [query_embedding],
                embeddings
            )[0]
        )

        top_indices = (
            scores.argsort()
            [-3:]
            [::-1]
        )

        st.write(
            "### Top Relevant Chunks"
        )

        for rank, idx in enumerate(
            top_indices,
            1
        ):

            st.write(
                f"#### Result {rank}"
            )

            st.write(
                f"Relevance Score: "
                f"{scores[idx]:.3f}"
            )

            st.write(
                chunks[idx]
            )

            st.divider()

    # =================================================
    # DOWNLOAD REPORT
    # =================================================

    report = {
        "summary": summary,
        "people": entities.people,
        "organizations":
            entities.organizations,
        "technologies":
            entities.technologies,
        "total_chunks":
            len(chunks),
        "total_characters":
            len(full_text)
    }

    st.download_button(
        label="Download Analysis Report",
        data=json.dumps(
            report,
            indent=4
        ),
        file_name="analysis_report.json",
        mime="application/json"
    )
 # ======================================
    # Analyze Another Article
    # ======================================

    if st.button("🔄 Analyze Another Article"):

        st.session_state.clear()
        st.rerun()

      