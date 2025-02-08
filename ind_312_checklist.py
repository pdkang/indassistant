import os
from typing import List
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv
import nest_asyncio
from langchain.schema import Document

# Apply nest_asyncio for async operations
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Load and parse PDF
def load_pdf(pdf_path: str) -> List[Document]:
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Get size in MB
    workers = 2 if file_size > 2 else 1  # Use 2 workers for PDFs >2MB
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
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

# Initialize vector store
def init_vector_store(documents):
    # Ensure documents have content
    if not documents or not all(doc.page_content.strip() for doc in documents):
        raise ValueError("No valid documents found for vector storage")
    
    # Initialize embedding model
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Verify embeddings work
    try:
        test_embedding = embedding_model.embed_query("test")
        if len(test_embedding) != 768:
            raise ValueError("Incorrect embedding dimensions")
    except Exception as e:
        raise ConnectionError(f"Embedding model failed: {str(e)}")
    
    return Qdrant.from_documents(
        documents=documents,
        embedding=embedding_model,
        location=":memory:",
        collection_name="ind312_docs",
        force_recreate=False
    )

# Create RAG chain
def create_rag_chain(retriever):
    # Load template from file
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

# Nothing else - no Chainlit code at the bottom 