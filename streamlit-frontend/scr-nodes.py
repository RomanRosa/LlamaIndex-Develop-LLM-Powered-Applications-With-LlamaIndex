from dotenv import load_dotenv
import os

import streamlit as st
import pinecone
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.chat_engine.types import ChatMode
from llama_index.llms import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Chat with LlamaIndex docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

print("***Streamlit LlamaIndex Documentation Helper***")

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_index() -> VectorStoreIndex:
    pinecone_client = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )
    index_name = "llamaindex-documentation-helper"
    pinecone_index = pinecone_client.Index(index_name=index_name, host=os.environ["PINECONE_HOST"])
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

index = get_index()

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, verbose=True)


st.title("Chat with LlamaIndex docs 💬🦙")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex's open source python library?",
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)
            nodes= [node for node in response.source_nodes] #Citations & Listening Source Nodes
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))): #Citations & Listening Source Nodes
                with col: #Citations & Listening Source Nodes
                    st.header(f"Source Node: score={node.score}") #Citations & Listening Source Nodes
                    st.write(node.text) #Citations & Listening Source Nodes
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
