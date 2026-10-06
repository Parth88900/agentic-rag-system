```python
import os
from pathlib import Path

import streamlit as st

from router import QueryRouter
from retriever import Retriever
from generator import AnswerGenerator
from utils import logger


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Make sure Python can find project files correctly
os.chdir(BASE_DIR)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Agentic RAG System")

st.write(
    "Ask questions from your knowledge base using "
    "Retrieval Augmented Generation."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Status")

    st.write("Agentic RAG Pipeline")

    st.caption(
        "Router → Retriever → Generator"
    )

    st.divider()

    # Check important files
    index_file = BASE_DIR / "vector_store.index"
    metadata_file = BASE_DIR / "chunk_metadata.pkl"

    if index_file.exists():
        st.success("✅ Vector index found")
    else:
        st.error("❌ Vector index missing")

    if metadata_file.exists():
        st.success("✅ Chunk metadata found")
    else:
        st.error("❌ Chunk metadata missing")

    st.divider()

    if os.getenv("GEMINI_API_KEY"):
        st.success("🔑 Gemini API configured")
    else:
        st.info(
            "ℹ️ Gemini API key not configured.\n\n"
            "The system can use its fallback generator."
        )


# ============================================================
# INITIALIZE RAG SYSTEM
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag():

    logger.info(
        "Initializing RAG components..."
    )

    # --------------------------------------------------------
    # Check required files before loading
    # --------------------------------------------------------

    index_file = BASE_DIR / "vector_store.index"
    metadata_file = BASE_DIR / "chunk_metadata.pkl"

    if not index_file.exists():

        raise FileNotFoundError(
            "vector_store.index was not found. "
            "Run the ingestion process first."
        )

    if not metadata_file.exists():

        raise FileNotFoundError(
            "chunk_metadata.pkl was not found. "
            "Run the ingestion process first."
        )

    # --------------------------------------------------------
    # Initialize Retriever
    # --------------------------------------------------------

    retriever = Retriever()

    # --------------------------------------------------------
    # Initialize Router
    # --------------------------------------------------------

    router = QueryRouter(
        corpus_texts=retriever.get_all_chunk_texts()
    )

    # --------------------------------------------------------
    # Initialize Generator
    # --------------------------------------------------------

    generator = AnswerGenerator()

    return retriever, router, generator


# ============================================================
# LOAD SYSTEM
# ============================================================

try:

    with st.spinner("Loading RAG system..."):

        retriever, router, generator = load_rag()

    st.success(
        "🚀 RAG system loaded successfully!"
    )

except Exception as e:

    st.error(
        "❌ Failed to load the RAG system."
    )

    st.exception(e)

    st.info(
        "Make sure vector_store.index and "
        "chunk_metadata.pkl are available in the project."
    )

    st.stop()


# ============================================================
# QUERY SECTION
# ============================================================

st.subheader("💬 Ask a Question")

query = st.text_input(
    "Enter your question:",
    placeholder="Example: What is the main topic discussed in the documents?"
)


# ============================================================
# GENERATE ANSWER
# ============================================================

if st.button(
    "🚀 Generate Answer",
    type="primary",
    use_container_width=True
):

    if not query.strip():

        st.warning(
            "⚠️ Please enter a question first."
        )

    else:

        try:

            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            with st.spinner(
                "🔎 Searching knowledge base..."
            ):

                retrieval_result = retriever.retrieve(
                    query.strip()
                )

            # ------------------------------------------------
            # Routing
            # ------------------------------------------------

            with st.spinner(
                "🧠 Routing query..."
            ):

                routing_result = router.route(
                    query.strip(),

                    retrieval_top_score=
                    retrieval_result.top_score,

                    retrieval_avg_score=
                    retrieval_result.avg_score
                )

            # ------------------------------------------------
            # Generation
            # ------------------------------------------------

            with st.spinner(
                "✍️ Generating answer..."
            ):

                answer = generator.generate(
                    query.strip(),
                    routing_result,
                    retrieval_result
                )


            # =================================================
            # ANSWER
            # =================================================

            st.divider()

            st.subheader("💡 Answer")

            st.write(
                answer.answer
            )


            # =================================================
            # METRICS
            # =================================================

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🎯 Confidence",
                    f"{answer.confidence:.2f}"
                )

            with col2:

                st.metric(
                    "⚙️ Method",
                    answer.generation_method
                )


            # =================================================
            # SOURCES
            # =================================================

            st.divider()

            st.subheader("📚 Sources")

            if answer.sources:

                for i, source in enumerate(
                    answer.sources,
                    start=1
                ):

                    st.write(
                        f"**{i}.** {source}"
                    )

            else:

                st.info(
                    "No sources were returned."
                )


            # =================================================
            # REASONING
            # =================================================

            st.divider()

            st.subheader("🧠 Reasoning")

            if answer.reasoning:

                st.write(
                    answer.reasoning
                )

            else:

                st.info(
                    "No reasoning information available."
                )


        except Exception as e:

            st.error(
                "❌ An error occurred while processing your question."
            )

            st.exception(e)
```
