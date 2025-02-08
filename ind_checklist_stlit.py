import os
import json
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import nest_asyncio
from langchain.schema import Document

# Apply nest_asyncio for async operations
nest_asyncio.apply()

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # OpenAI API Key
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")  # Llama Cloud API Key

# File paths
PDF_FILE = "IND-312.pdf"
PREPROCESSED_FILE = "preprocessed_docs.json"

# Load and parse PDF (only for preprocessing)
def load_pdf(pdf_path: str) -> List[Document]:
    """Loads a PDF, processes it with LlamaParse, and splits it into LangChain documents."""
    from llama_parse import LlamaParse  # Import only if needed

    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
    workers = 2 if file_size > 2 else 1  # Use 2 workers for PDFs >2MB

    parser = LlamaParse(
        api_key=os.environ["LLAMA_CLOUD_API_KEY"],
        result_type="markdown",
        num_workers=workers,
        verbose=True
    )

    # Parse PDF to documents
    llama_documents = parser.load_data(pdf_path)

    # Convert to LangChain documents
    documents = [
        Document(
            page_content=doc.text,
            metadata={"source": pdf_path, "page": doc.metadata.get("page_number", 0)}
        ) for doc in llama_documents
    ]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    return text_splitter.split_documents(documents)

# Preprocess the PDF and save to JSON (Only if it doesn't exist)
def preprocess_pdf(pdf_path: str, output_path: str = PREPROCESSED_FILE):
    """Preprocess PDF only if the output file does not exist."""
    if os.path.exists(output_path):
        print(f"Preprocessed data already exists at {output_path}. Skipping PDF processing.")
        return  # Skip processing if file already exists

    print("Processing PDF for the first time...")

    documents = load_pdf(pdf_path)  # Load and process the PDF

    # Convert documents to JSON format
    json_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    print(f"Preprocessed PDF saved to {output_path}")

# Load preprocessed data instead of parsing PDF
def load_preprocessed_data(json_path: str) -> List[Document]:
    """Load preprocessed data from JSON."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Preprocessed file {json_path} not found. Run preprocessing first.")

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    return [Document(page_content=d["content"], metadata=d["metadata"]) for d in json_data]

# Initialize vector store from preprocessed data
def init_vector_store(documents: List[Document]):
    """Initialize a vector store using HuggingFace embeddings and Qdrant."""
    if not documents or not all(doc.page_content.strip() for doc in documents):
        raise ValueError("No valid documents found for vector storage")

    # Initialize embedding model
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )

    return Qdrant.from_documents(
        documents=documents,
        embedding=embedding_model,
        location=":memory:",
        collection_name="ind312_docs",
        force_recreate=False
    )

# Create RAG chain for retrieval-based Q&A
def create_rag_chain(retriever):
    """Create a retrieval-augmented generation (RAG) chain for answering questions."""
    # Load prompt template
    with open("template.md") as f:
        template_content = f.read()

    prompt = ChatPromptTemplate.from_template("""
    You are an FDA regulatory expert. Use this structure for checklists:
    {template}

    Context from IND-312:
    {context}

    Question: {question}

    Answer in Markdown with checkboxes (- [ ]). If unsure, say "I don't know".
    """)

    return (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "template": lambda _: template_content  # Inject template content
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | ChatOpenAI(model="gpt-4") | StrOutputParser()}
    )

# Run preprocessing only if executed directly (NOT when imported)
if __name__ == "__main__":
    preprocess_pdf(PDF_FILE)

