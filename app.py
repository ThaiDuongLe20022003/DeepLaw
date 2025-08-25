import os
import chainlit as cl
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Thêm vào sau khi import các thư viện khác
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize Ollama LLM with DeepSeek-R1 model
llm = Ollama(model="deepseek-r1:1.5b", request_timeout=60.0)

# Configuration paths
STORAGE_DIR = "./storage"  # Vector index storage
CHAT_STORE_PATH = "./chat/chat_store.json"  # Global chat history storage

def load_index():
    """Loads the pre-built vector index from storage"""
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    return load_index_from_storage(storage_context)

def create_query_engine(index):
    """
    Creates an optimized query engine with:
    - AutoMergingRetriever for context reconstruction
    - Cross-encoder reranker for precision
    """
    # Base retriever with increased candidate count
    base_retriever = index.as_retriever(similarity_top_k=6)
    
    # Automatically merge small chunks into coherent context
    retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context=index.storage_context, 
        verbose=False
    )
    
    # Re-rank results using cross-encoder model
    reranker = SentenceTransformerRerank(
        top_n=3, 
        model="BAAI/bge-reranker-base"
    )
    
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        streaming=True,
        llm=llm
    )

@cl.on_chat_start
async def init_chat():
    """Initializes chat session with persistent memory"""
    # Load pre-built vector index
    index = load_index()
    
    # Create optimized query engine
    query_engine = create_query_engine(index)
    
    # Load or initialize chat history store
    if os.path.exists(CHAT_STORE_PATH) and os.path.getsize(CHAT_STORE_PATH) > 0:
        chat_store = SimpleChatStore.from_persist_path(CHAT_STORE_PATH)
    else:
        chat_store = SimpleChatStore()
    
    # Initialize chat memory with token limit
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=1500, 
        chat_store=chat_store,
        chat_store_key="user_session",
        llm=llm
    )
    
    # Store components in user session
    cl.user_session.set("query_engine", query_engine)
    cl.user_session.set("chat_memory", chat_memory)
    cl.user_session.set("chat_store", chat_store)

@cl.on_chat_resume
async def resume_chat():
    """Handles chat session resumption"""
    await init_chat()

@cl.password_auth_callback
def authenticate(username: str, password: str):
    """Simple password-based authentication"""
    # Replace with your credentials
    valid_users = {"admin": "secret"}
    if username in valid_users and valid_users[username] == password:
        return cl.User(identifier=username)
    return None

@cl.on_message
async def handle_message(message: cl.Message):
    """Processes incoming messages with streaming response"""
    # Retrieve session components
    query_engine = cl.user_session.get("query_engine")
    chat_memory = cl.user_session.get("chat_memory")
    chat_store = cl.user_session.get("chat_store")
    
    # Build context-aware prompt with chat history
    history = chat_memory.get()
    full_prompt = f"{history}\nUser: {message.content}"
    
    # Initialize empty message for streaming
    reply = cl.Message(content="")
    await reply.send()
    
    # Execute RAG query asynchronously
    response = await cl.make_async(query_engine.query)(full_prompt)
    
    # Stream response tokens
    for token in response.response_gen:
        await reply.stream_token(token)
    await reply.update()
    
    # Update chat memory with new exchange
    chat_memory.put(f"User: {message.content}")
    chat_memory.put(f"Assistant: {response.response}")
    
    # Persist updated chat history
    chat_store.persist(persist_path=CHAT_STORE_PATH)