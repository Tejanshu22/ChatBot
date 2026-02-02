from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import streamlit as st
import os
from dotenv import load_dotenv



load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
## Indexing (Document Ingesion)
video_id = "Gfr50f6ZBvo"  # only the video ID
try:
    api = YouTubeTranscriptApi()
    # Fetch transcript (instance-based API)
    transcript_list = api.fetch(video_id)
    # Convert to plain text (IMPORTANT FIX HERE)
    transcript = " ".join(snippet.text for snippet in transcript_list)
    #print(transcript)
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
except Exception as e:
    print("Failed to fetch transcript:", e)
## Indexing (Text Spiltting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript]) # list of chunks of splitted text
## Indexing (Embedding Generation and Storing in Vector Store)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(chunks, embeddings)
## Retrival
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
## Augmentation
llm=OllamaLLM(model="phi")
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.
      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})
## Generation
answer = llm.invoke(final_prompt)
print(answer)
## Building a chain
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
main_chain.invoke('Can you summarize the video')