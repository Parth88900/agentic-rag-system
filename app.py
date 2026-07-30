import streamlit as st
import os

from router import QueryRouter
from retriever import Retriever
from generator import AnswerGenerator
from utils import logger


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Agentic RAG System")
st.write(
    "Ask questions from your knowledge base using Retrieval Augmented Generation."
)


# ==========================
# INITIALIZE RAG
# ==========================

@st.cache_resource
def load_rag():

    logger.info(
        "Initializing RAG components..."
    )

    retriever = Retriever()

    router = QueryRouter(
        corpus_texts=retriever.get_all_chunk_texts()
    )

    generator = AnswerGenerator()

    return retriever, router, generator



try:

    retriever, router, generator = load_rag()

    st.success(
        "RAG system loaded successfully 🚀"
    )


except Exception as e:

    st.error(
        f"Failed loading RAG system: {e}"
    )

    st.stop()



# ==========================
# QUERY BOX
# ==========================


query = st.text_input(
    "Ask your question:"
)


if st.button("Generate Answer"):

    if query.strip():

        with st.spinner(
            "Searching knowledge base..."
        ):

            try:

                # Retrieve
                retrieval_result = retriever.retrieve(
                    query
                )


                # Route
                routing_result = router.route(
                    query,
                    retrieval_top_score=
                    retrieval_result.top_score,

                    retrieval_avg_score=
                    retrieval_result.avg_score
                )


                # Generate
                answer = generator.generate(
                    query,
                    routing_result,
                    retrieval_result
                )


                st.subheader(
                    "Answer"
                )

                st.write(
                    answer.answer
                )


                st.divider()


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Confidence",
                        f"{answer.confidence:.2f}"
                    )


                with col2:

                    st.metric(
                        "Method",
                        answer.generation_method
                    )


                st.subheader(
                    "Sources"
                )

                for source in answer.sources:

                    st.write(
                        source
                    )


                st.subheader(
                    "Reasoning"
                )

                st.write(
                    answer.reasoning
                )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )

    else:

        st.warning(
            "Enter a question first."
        )
