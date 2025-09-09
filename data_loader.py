from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser

# Configure embedding model for document processing
Settings.embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512  # Optimal chunk size for retrieval
Settings.chunk_overlap = 50  # Context overlap between chunks

def create_index():
    """
    Creates and persists a vector index from documents in the data directory
    using hierarchical chunking for improved context retrieval.
    """
    # Load all documents from the data directory
    documents = SimpleDirectoryReader("./data").load_data()
    
    # Create hierarchical chunks (small chunks nested in larger parent chunks)
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes = [128, 512])
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # Initialize storage and add nodes
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    
    # Build vector index
    index = VectorStoreIndex(nodes, storage_context = storage_context)
    
    # Persist index to disk for future use
    index.storage_context.persist(persist_dir = "./storage")
    
    return index

if __name__ == "__main__":
    create_index()
    print("Vector index created successfully at ./storage")