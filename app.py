import os
import validators
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Load environment variables
load_dotenv(r'F:\Udemy\langchain\.env')
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit app configuration
st.set_page_config(page_title="Langchain: Summarize text from website")
st.title("Summarize text from website")
st.subheader("Summarize URL")
st.markdown("Sample URL: https://medium.com/geekculture/git-and-github-for-beginner-data-scientists-602a3a5574d8")

# Sidebar for model selection
model_options = ["Gemma2-9b-It","mixtral-8x7b-32768",
                "llama3-70b-8192"]
selected_model = st.sidebar.selectbox("Select the Groq Model", model_options)
language = st.sidebar.text_input("Preferred Language", placeholder="e.g., English, Spanish, French") or "English"

# LLM setup
groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model=selected_model)

session_id=st.text_input("Session ID",value="default_session")

## statefully manage chat history
if 'store' not in st.session_state:
    st.session_state.store={}

# Helper Functions
def fetch_url_content(url):
    """Fetch and process content from a URL."""
    try:
        loader = UnstructuredURLLoader(
            urls=[url],
            ssl_verify=False,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
            }
        )
        return loader.load()
    except Exception as e:
        st.error(f"Failed to load content from the URL: {e}")
        return None

prompt_template = ("Summarize the following content, ensuring "
                "the summary is clear, concise, and retains the most important information. "
                "Please don't generate any text outside of the summary. "
                "The summary should be written in the specified language: {language}. "
                "Content: {text}")
prompt = PromptTemplate(template=prompt_template, input_variables=['text', 'language'])

# Input for URL summarization
url_to_summarize = st.text_input("URL", label_visibility='collapsed', placeholder="Enter the URL to Summarize")

if st.button("Summarize the content from website"):
    if not groq_api_key.strip() or not url_to_summarize.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(url_to_summarize) or 'http' not in url_to_summarize:
        st.error("Please enter a valid website URL")
    elif "youtube.com" in url_to_summarize or "youtu.be" in url_to_summarize:
        st.error("YouTube URLs are not supported. Please enter a regular website URL.")
    else:
        with st.spinner("Processing..."):
            docs = fetch_url_content(url_to_summarize)

            if docs and docs[0].page_content:
                text_content = docs[0].page_content
                chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt)
                output_summary = chain.invoke({"input_documents": docs, "language": language})
                summary_text = output_summary.get('output_text', '')

                if summary_text:
                    st.session_state['summary'] = summary_text
                else:
                    st.error("Failed to generate the summary.")
            else:
                st.error("Failed to extract content from the provided URL.")

# Question-Answering Based on Summary
if 'summary' in st.session_state:
    # Always display the summary
    st.subheader("Summary")
    st.write(st.session_state['summary'])

    st.subheader("Ask questions about the summary")
    user_question = st.text_input("Your question", key="user_question")

    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    summary_doc = Document(page_content=st.session_state['summary'])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents([summary_doc])
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Define a custom question-answering chain
    system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    if st.button("Ask"):
        if user_question.strip():
            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_question},
                    config={"configurable": {"session_id": session_id}}
                )
                answer = response['answer']

                # Save the conversation
                st.session_state['conversation'].append({"question": user_question, "answer": answer})
                st.write(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Error processing the question: {e}")
        else:
            st.error("Please enter a question.")

    # Display conversation history
    if st.session_state['conversation']:
        st.subheader("Conversation History")
        for idx, qa in enumerate(st.session_state['conversation']):
            st.write(f"**Q{idx + 1}:** {qa['question']}")
            st.write(f"**A{idx + 1}:** {qa['answer']}")