from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import nest_asyncio

# Apply nest_asyncio for async operations
nest_asyncio.apply()

# Configure embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"
)
Settings.chunk_size = 512
Settings.chunk_overlap = 50

def create_index_batch(documents, batch_size=1000):
    """Process documents in batches for memory efficiency"""
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[256, 512])
    
    all_nodes = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing documents"):
        batch = documents[i:i + batch_size]
        nodes = node_parser.get_nodes_from_documents(batch)
        all_nodes.extend(nodes)
    
    return all_nodes

def train_faiss_index(nodes, dimension=384, nlist=100):
    """Train FAISS index with sample embeddings"""
    print("Training FAISS index...")
    
    # Collect embeddings for training
    train_embeddings = []
    
    # First try to use existing embeddings
    for node in nodes[:min(20000, len(nodes))]:
        if hasattr(node, 'embedding') and node.embedding is not None:
            train_embeddings.append(node.embedding)
    
    # If not enough embeddings, generate some
    if len(train_embeddings) < 10000:
        print("Generating additional embeddings for training...")
        sample_nodes = nodes[:min(10000, len(nodes))]
        for i, node in enumerate(tqdm(sample_nodes, desc="Generating training embeddings")):
            if i % 10 == 0:  # Sample every 10th node
                try:
                    embedding = Settings.embed_model.get_text_embedding(node.text)
                    train_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error generating embedding: {e}")
                    continue
    
    if not train_embeddings:
        raise ValueError("No embeddings available for training")
    
    # Convert to numpy array
    train_embeddings = np.array(train_embeddings).astype('float32')
    print(f"Training with {len(train_embeddings)} embeddings")
    
    # Create and train FAISS index
    quantizer = faiss.IndexFlatL2(dimension)
    faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    
    # Train the index
    faiss_index.train(train_embeddings)
    
    return faiss_index

def create_index():
    """
    Creates optimized FAISS index for large datasets
    while maintaining metadata in ./storage folder
    """
    print("Loading documents...")
    documents = SimpleDirectoryReader(
        "./data",
        recursive=True,
        exclude_hidden=True,
        required_exts=[".txt", ".pdf", ".docx", ".pptx", ".md", ".html"]
    ).load_data()
    
    print(f"Loaded {len(documents)} documents")
    
    # Process in batches to avoid memory issues
    print("Processing documents in batches...")
    nodes = create_index_batch(documents, batch_size=500)
    
    print(f"Created {len(nodes)} nodes")
    
    # Train FAISS index first
    dimension = 384
    nlist = min(100, len(nodes) // 100)  # Adjust nlist based on dataset size
    if nlist < 10:
        nlist = 10
    
    faiss_index = train_faiss_index(nodes, dimension, nlist)
    
    # Create vector store with trained index
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Build index with progress tracking (disable async to avoid nesting issues)
    print("Building FAISS index...")
    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context,
        show_progress=True,
        use_async=False  # Disable async to avoid nesting issues
    )
    
    # Ensure storage directory exists
    os.makedirs("./storage", exist_ok=True)
    
    # Persist FAISS index
    faiss.write_index(faiss_index, "./storage/faiss_index.bin")
    
    # Persist metadata to storage folder
    storage_context.persist(persist_dir="./storage")
    
    # Save additional metadata
    metadata = {
        "total_documents": len(documents),
        "total_nodes": len(nodes),
        "embedding_dimension": dimension,
        "index_type": "IVFFlat",
        "nlist": nlist,
        "faiss_index_path": "./storage/faiss_index.bin",
        "created_at": datetime.now().isoformat(),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "chunk_size": 512,
        "chunk_overlap": 50
    }
    
    with open("./storage/index_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"FAISS index created successfully with {len(nodes)} nodes")
    print(f"Metadata persisted to ./storage folder")
    return index

if __name__ == "__main__":
    create_index()