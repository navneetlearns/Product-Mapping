import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF
import zipfile
import tempfile
import uuid
from sentence_transformers import SentenceTransformer
import chromadb

# --- CONFIGURATION ---
st.set_page_config(page_title="Invoice Semantic Search", layout="wide")
st.title("📄 Invoice Semantic Search Engine")
st.markdown("Upload your invoices (PDF, ZIP of PDFs, or Excel), and ask questions about them!")

# --- MODEL & DB INITIALIZATION ---

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    """Loads the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Initialize ChromaDB client. It's lightweight and runs in memory.
# We store the collection in Streamlit's session state to persist it across reruns.
client = chromadb.Client()

if 'collection' not in st.session_state:
    # Use a unique name for the collection, e.g., based on a session ID
    # For simplicity, we'll use a fixed name but in a multi-user env, this should be dynamic
    collection_name = f"invoices_{str(uuid.uuid4())}"
    st.session_state.collection = client.create_collection(name=collection_name)
    print(f"Created new ChromaDB collection: {collection_name}")

# --- DATA PROCESSING LOGIC ---

def process_pdf(file_content):
    """Extracts text from a single PDF file's content, page by page."""
    chunks, metadatas = [], []
    with fitz.open(stream=file_content, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                chunks.append(text.strip())
                metadatas.append({"page_number": page_num + 1})
    return chunks, metadatas

def process_zip(file_content):
    """Extracts text from all PDFs within a ZIP file."""
    chunks, metadatas = [], []
    # Use a temporary directory to safely extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(file_content) as z:
            z.extractall(temp_dir)
        
        for filename in os.listdir(temp_dir):
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(temp_dir, filename)
                with fitz.open(filepath) as doc:
                    for page_num, page in enumerate(doc):
                        text = page.get_text("text")
                        if text.strip():
                            chunks.append(text.strip())
                            metadatas.append({"source_file": filename, "page_number": page_num + 1})
    return chunks, metadatas

def process_excel(file_content):
    """Extracts data from an Excel file, row by row."""
    chunks, metadatas = [], []
    df = pd.read_excel(file_content)
    for index, row in df.iterrows():
        # Create a descriptive text chunk from the row's data
        chunk_text = ". ".join([f"{col}: {val}" for col, val in row.astype(str).items() if pd.notna(val)])
        if chunk_text:
            chunks.append(chunk_text)
            metadatas.append({"source_row": index + 2}) # +2 for 1-based index and header
    return chunks, metadatas

# --- STREAMLIT UI ---

# File uploader
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['pdf', 'zip', 'xlsx', 'xls'],
    help="Upload a single PDF, a ZIP containing multiple PDFs, or an Excel file."
)

if uploaded_file is not None:
    # Button to trigger the processing
    if st.button("Process File"):
        # Reset the collection if a new file is processed
        st.session_state.collection.delete(ids=st.session_state.collection.get()['ids'])
        
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_content = uploaded_file.getvalue()
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            chunks, metadatas = [], []
            if file_extension == '.pdf':
                chunks, page_metadatas = process_pdf(file_content)
                # Add source file name to all metadata entries
                for meta in page_metadatas:
                    meta['source_file'] = uploaded_file.name
                metadatas.extend(page_metadatas)

            elif file_extension == '.zip':
                chunks, metadatas = process_zip(file_content)
            
            elif file_extension in ['.xlsx', '.xls']:
                chunks, row_metadatas = process_excel(file_content)
                # Add source file name to all metadata entries
                for meta in row_metadatas:
                    meta['source_file'] = uploaded_file.name
                metadatas.extend(row_metadatas)
                
            else:
                st.error("Unsupported file type.")
                st.stop()
        
            if not chunks:
                st.warning("Could not extract any text from the document. Please check the file.")
            else:
                st.write(f"Extracted {len(chunks)} text chunks from the document.")
                with st.spinner("Generating embeddings and indexing... This may take a moment."):
                    embeddings = model.encode(chunks, show_progress_bar=True).tolist()
                    ids = [str(uuid.uuid4()) for _ in chunks]
                    
                    st.session_state.collection.add(
                        embeddings=embeddings,
                        documents=chunks,
                        metadatas=metadatas,
                        ids=ids
                    )
                st.success(f"✅ Successfully indexed {st.session_state.collection.count()} document chunks!")

# --- SEARCH INTERFACE ---
st.write("---")

# Only show the search box if documents have been indexed
if st.session_state.collection.count() > 0:
    st.header("Ask a Question")
    query_text = st.text_input("Search for information in the uploaded documents:", placeholder="e.g., 'What was the total for invoice INV001?'")

    if query_text:
        with st.spinner("Searching..."):
            query_embedding = model.encode(query_text).tolist()
            results = st.session_state.collection.query(
                query_embeddings=[query_embedding],
                n_results=5  # Retrieve top 5 results
            )
            
            st.subheader("Search Results")
            if not results['documents'][0]:
                st.info("No relevant information found.")
            else:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    with st.expander(f"**Result {i+1}** - Source: `{metadata.get('source_file', 'N/A')}` Page: `{metadata.get('page_number', 'N/A')}` Row: `{metadata.get('source_row', 'N/A')}`"):
                        st.markdown(doc)
                        st.caption(f"Similarity Score (Distance): {results['distances'][0][i]:.4f}")

else:
    st.info("Please upload and process a file to begin searching.")
