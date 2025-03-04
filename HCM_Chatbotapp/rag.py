import pdfplumber
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Pinecone as PineconeStore
import pinecone.data.index
pinecone.Index = pinecone.data.index.Index

# 1. Globally configure Pinecone (v2)
pc = Pinecone(api_key="pcsk_4rDQwb_KmGzMCsQJbwq1srAUxYZphN6Pscy9RoHVu4MLMdEiKzmsAbXxmUBcCuTwiDLAhE")

Index_Name = "chatbot"

# 2. Check if the index exists; if not, create it
def ensure_index_exists():
    # This function will be called only when needed
    if Index_Name not in pc.list_indexes().names():
        # Create the index or handle the condition
        pc.create_index(
        name=Index_Name,
        dimension=1024,          
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )  
    #my_index = pc.Index(Index_Name)


 
# 3. Create embeddings
embeddings_instance = None

def get_embeddings():
    global embeddings_instance
    if embeddings_instance is None:
        embeddings_instance = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    return embeddings_instance
# 4. Define PDF text extraction
def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            all_text += page_text + "\n"
    return all_text

# 5. Define chunking
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

# 6. Store in Pinecone via langchain_community
def store_in_pinecone(doc_id, text_content):
    """
    Chunk text, embed, and store in Pinecone as separate 'Documents'.
    doc_id is metadata referencing the HRDocument primary key.
    """
    # Ensure text_content is a single string
    if isinstance(text_content, list):
        text_content = "\n".join(text_content)

    # Step 1: Chunk the text
    chunks = chunk_text(text_content, chunk_size=500)

    # Step 2: Each chunk becomes a "Document"
    docs = []
    for i, chunk in enumerate(chunks):
        metadata = {"source_doc_id": doc_id, "chunk_index": i}
        docs.append(Document(page_content=chunk, metadata=metadata))

    # Step 3: Convert to Pinecone
    pinecone_store = PineconeStore.from_documents(
        docs,
        embedding=get_embeddings(),
        index_name=Index_Name
    )
