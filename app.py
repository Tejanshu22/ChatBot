import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

# ---------------- CONFIG ---------------- #

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered"
)

# ---------------- UI ---------------- #

st.title("🎥 YouTube Video Chatbot")
st.write("Ask questions about the video transcript using AI")

VIDEO_ID = "Gfr50f6ZBvo"


# ---------------- CACHE FUNCTIONS ---------------- #

@st.cache_resource
def load_transcript(video_id):

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)

        transcript = " ".join(
            snippet.text for snippet in transcript_list
        )

        return transcript

    except TranscriptsDisabled:
        st.error("Transcript is disabled for this video.")
        return None

    except Exception as e:
        st.error(f"Error loading transcript: {e}")
        return None


@st.cache_resource
def build_vectorstore(transcript):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


@st.cache_resource
def build_chain(_vector_store):

    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = OllamaLLM(model="phi")

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the transcript context.
If you don't know, say "I don't know."

Context:
{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser

    return main_chain


# ---------------- LOAD DATA ---------------- #

with st.spinner("📥 Loading transcript and building AI..."):

    transcript = load_transcript(VIDEO_ID)
    if transcript is None:
        st.stop()

    vector_store = build_vectorstore(transcript)

    chain = build_chain(vector_store)


# ---------------- CHAT SYSTEM ---------------- #

st.divider()

if "history" not in st.session_state:
    st.session_state.history = []


# Show chat history
for q, a in st.session_state.history:

    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        st.write(a)


# Chat input
user_input = st.chat_input("Ask something about this video...")


# Handle input
if user_input:

    # Show user msg
    with st.chat_message("user"):
        st.write(user_input)

    # Generate answer
    with st.spinner("🤔 Thinking..."):

        answer = chain.invoke(user_input)

    # Show assistant msg
    with st.chat_message("assistant"):
        st.write(answer)

    # Save history
    st.session_state.history.append(
        (user_input, answer)
    )
