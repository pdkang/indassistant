"""
Merged Streamlit App: IND Assistant and Submission Assessment

This app combines the functionality of the IND Assistant (chat-based Q&A)
and the Submission Assessment (checklist-based analysis) into a single
Streamlit interface.
"""

import os
import json
import tempfile
from zipfile import ZipFile
import streamlit as st
from llama_parse import LlamaParse
import pickle
import hashlib
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
import boto3  # Import boto3 for S3 interaction
import requests
from io import BytesIO
import pathlib  # Import pathlib for path operations
from qdrant_client import QdrantClient  # Import QdrantClient directly
import sys
import types
import re

# Set page config at the very beginning
st.set_page_config(page_title="IND Assistant & Submission Assessment", layout="wide")

# Constants
PREPROCESSED_FILE = "preprocessed_data.json"
QDRANT_STORE_DIR = "qdrant_store"
VECTOR_STORE_PATH = os.path.join(QDRANT_STORE_DIR, "vector_store.qdrant")

# Prevent Streamlit from auto-reloading on file changes
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Apply nest_asyncio for async operations
nest_asyncio.apply()

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # OpenAI API Key
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")  # Llama Cloud API Key
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_REGION"] = os.getenv("AWS_REGION")

# File paths
PDF_FILE = "IND-312.pdf"

# --- IND Assistant Functions ---

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

    try:
        # Create qdrant-store directory if it doesn't exist
        pathlib.Path(QDRANT_STORE_DIR).mkdir(exist_ok=True)
        st.info(f"Created or verified directory: {QDRANT_STORE_DIR}")
        
        # Check if we have a saved vector store
        if os.path.exists(VECTOR_STORE_PATH):
            st.info(f"Found saved vector store at {VECTOR_STORE_PATH}")
            try:
                # Load the vector store from disk
                with open(VECTOR_STORE_PATH, "rb") as f:
                    vector_store = pickle.load(f)
                st.success("Successfully loaded vector store from disk!")
                return vector_store
            except Exception as load_error:
                st.warning(f"Could not load vector store from disk: {str(load_error)}")
                st.info("Creating new vector store...")
                # Continue with creating a new vector store

        # Initialize embedding model
        st.info("Initializing embedding model...")
        try:
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Error initializing embedding model: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            # Use a simpler embedding model as fallback
            from langchain_community.embeddings import OpenAIEmbeddings
            st.warning("Falling back to OpenAI embeddings")
            embedding_model = OpenAIEmbeddings()

        # Use in-memory storage with persistence to avoid network issues
        st.info("Using in-memory storage with persistence...")
        
        # Create vector store with documents
        st.info(f"Creating vector store with {len(documents)} documents...")
        vector_store = Qdrant.from_documents(
            documents=documents,
            embedding=embedding_model,
            location=":memory:",  # Use in-memory storage
            collection_name="ind312_docs",
            force_recreate=True
        )
        
        st.success("Successfully created vector store in memory!")
        
        # Save the vector store to disk for future use
        try:
            st.info(f"Saving vector store to disk at {VECTOR_STORE_PATH}...")
            # Create directory if it doesn't exist
            os.makedirs(QDRANT_STORE_DIR, exist_ok=True)
            
            # Serialize and save the vector store
            with open(VECTOR_STORE_PATH, "wb") as f:
                pickle.dump(vector_store, f)
            st.success("Successfully saved vector store to disk!")
        except Exception as save_error:
            st.warning(f"Could not save vector store to disk: {str(save_error)}")
            
        return vector_store
            
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        # Fallback to in-memory storage if local storage fails
        st.warning("Falling back to basic in-memory vector store")
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )
        return Qdrant.from_documents(
            documents=documents,
            embedding=embedding_model,
            location=":memory:",
            collection_name="ind312_docs",
            force_recreate=True
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

    Answer in Markdown with checkboxes (- [ ]). If unsure, say "I can only answer IND related questions.".
    """)

    return (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "template": lambda _: template_content  # Inject template content
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | prompt
        | ChatOpenAI(model="gpt-4", temperature=0)
        | StrOutputParser()
    )

# Function to load the checklist
def load_checklist():
    """Load the IND checklist for submission assessment."""
    return IND_CHECKLIST

# Caching function to prevent redundant RAG processing
@st.cache_data
def cached_response(question: str):
    """Retrieve cached response if available, otherwise compute response."""
    if "rag_chain" in st.session_state:
        # The chain now returns a string directly, not a dictionary
        return st.session_state.rag_chain.invoke({"question": question})
    else:
        st.error("RAG chain not initialized. Please initialize the IND Assistant first.")
        return "Error: RAG chain not initialized. Please initialize the IND Assistant first."

# --- Submission Assessment Functions ---

# Access API key from environment variable
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")

# Check if the API key is available
if not LLAMA_CLOUD_API_KEY:
    st.error("LLAMA_CLOUD_API_KEY not found in environment variables. Please set it in your Hugging Face Space secrets.")
    st.stop()

# Sample Checklist Configuration (this should be adjusted to your actual IND requirements)
IND_CHECKLIST = {
    "Form FDA-1571": {
        "file_patterns": ["1571", "fda-1571"],
        "required_keywords": [
            # Sponsor Information
            "Name of Sponsor",
            "Date of Submission",
            "Address 1",
            "Sponsor Telephone Number",
            # Drug Information
            "Name of Drug",
            "IND Type",
            "Proposed Indication for Use",
            # Regulatory Information
            "Phase of Clinical Investigation",
            "Serial Number",
            # Application Contents
            "Table of Contents",
            "Investigator's Brochure",
            "Study protocol",
            "Investigator data",
            "Facilities data",
            "Institutional Review Board data",
            "Environmental assessment",
            "Pharmacology and Toxicology",
            # Signatures and Certifications
            #"Person Responsible for Clinical Investigation Monitoring",
            #"Person Responsible for Reviewing Safety Information",
            "Sponsor or Sponsor's Authorized Representative First Name",
            "Sponsor or Sponsor's Authorized Representative Last Name",
            "Sponsor or Sponsor's Authorized Representative Title",
            "Sponsor or Sponsor's Authorized Representative Telephone Number",
            "Date of Sponsor's Signature"
        ]
    },
    "Table of Contents": {
        "file_patterns": ["toc", "table of contents"],
        "required_keywords": ["table of contents", "sections", "appendices"]
    },
    "Introductory Statement": {
        "file_patterns": ["intro", "introductory", "general plan"],
        "required_keywords": ["introduction", "investigational plan", "objectives"]
    },
    "Investigator Brochure": {
        "file_patterns": ["brochure", "ib"],
        "required_keywords": ["pharmacology", "toxicology", "clinical data"]
    },
    "Clinical Protocol": {
        "file_patterns": ["clinical", "protocol"],
        "required_keywords": ["study design", "objectives", "patient population", "dosing regimen", "endpoints"]
    },
    "CMC Information": {
        "file_patterns": ["cmc", "chemistry", "manufacturing"],
        "required_keywords": ["manufacturing", "controls", "specifications", "stability"]
    },
    "Pharmacology and Toxicology": {
        "file_patterns": ["pharm", "tox", "pharmacology", "toxicology"],
        "required_keywords": ["pharmacology studies", "toxicology studies", "animal studies"]
    },
    "Previous Human Experience": {
        "file_patterns": ["human", "experience", "previous"],
        "required_keywords": ["previous studies", "human subjects", "clinical experience"]
    },
    "Additional Information": {
        "file_patterns": ["additional", "other", "supplemental"],
        "required_keywords": ["additional data", "supplementary information"]
    }
}


class ChecklistCrossReferenceAgent:
    """
    Agent that cross-references the pre-parsed submission package data
    against a predefined IND checklist.

    Input:
        submission_data: list of dicts representing each file with keys:
            - "filename": Filename of the document.
            - "file_type": e.g., "pdf" or "txt"
            - "content": Extracted text from the document.
            - "metadata": (Optional) Additional metadata.
        checklist: dict representing the IND checklist.
    Output:
        A mapping of checklist items to their verification status.
    """
    def __init__(self, checklist):
        self.checklist = checklist

    def run(self, submission_data):
        """
        Cross-reference the submission data against the checklist.
        
        Args:
            submission_data: A list of dictionaries containing file information.
            
        Returns:
            A dictionary with the cross-reference results.
        """
        try:
            results = {}
            
            # Process each checklist item
            for document_name, document_info in self.checklist.items():
                # Initialize result for this document
                results[document_name] = {
                    "found": False,
                    "filename": None,
                    "file_type": None,
                    "complete": False,
                    "missing_fields": []
                }
                
                # Check if any of the submission files match this document
                for file_data in submission_data:
                    filename = file_data.get("filename", "")
                    file_type = file_data.get("file_type", "")
                    
                    # Check if the filename matches any of the patterns for this document
                    patterns = document_info.get("file_patterns", [])
                    for pattern in patterns:
                        if re.search(pattern, filename, re.IGNORECASE):
                            # Found a match
                            results[document_name]["found"] = True
                            results[document_name]["filename"] = filename
                            results[document_name]["file_type"] = file_type
                            
                            # Check if the document is complete (has all required fields)
                            required_fields = document_info.get("required_keywords", [])
                            if required_fields:
                                # Get the content of the file
                                content = file_data.get("content", "")
                                
                                # Check each required field
                                missing_fields = []
                                for field in required_fields:
                                    if not re.search(field, content, re.IGNORECASE):
                                        missing_fields.append(field)
                                
                                if missing_fields:
                                    results[document_name]["missing_fields"] = missing_fields
                                else:
                                    results[document_name]["complete"] = True
                            else:
                                # If no required fields are specified, consider it complete
                                results[document_name]["complete"] = True
                            
                            # Break out of the pattern loop once a match is found
                            break
                    
                    # Break out of the file loop once a match is found
                    if results[document_name]["found"]:
                        break
            
            return results
            
        except RuntimeError as e:
            # Handle PyTorch-specific errors
            if "__path__._path" in str(e) and "torch" in str(e):
                # Apply PyTorch-specific workaround
                import torch
                import types
                # Create a dummy module to handle the __path__._path attribute
                if not hasattr(torch, "__path__"):
                    torch.__path__ = types.SimpleNamespace()
                if not hasattr(torch.__path__, "_path"):
                    torch.__path__._path = []
                
                # Try again with the workaround
                return self.run(submission_data)
            else:
                # Re-raise non-PyTorch errors
                raise e
        except Exception as e:
            # Re-raise other exceptions
            raise e


class AssessmentRecommendationAgent:
    """
    Agent that analyzes cross-reference results and produces assessment and recommendations.
    """
    
    def run(self, cross_reference_result):
        """
        Analyze the cross-reference results to produce assessment and recommendations.
        
        Args:
            cross_reference_result: The results from the ChecklistCrossReferenceAgent.
            
        Returns:
            A dictionary with the assessment and recommendations.
        """
        assessment = {
            "missing_documents": [],
            "incomplete_documents": [],
            "complete_documents": [],
            "recommendations": []
        }
        
        # Process each document in the cross-reference result
        for document_name, result in cross_reference_result.items():
            if not result["found"]:
                # Document is missing
                assessment["missing_documents"].append(document_name)
                assessment["recommendations"].append(f"Add the required document: {document_name}")
            elif result["missing_fields"]:
                # Document is incomplete
                assessment["incomplete_documents"].append({
                    "name": document_name,
                    "filename": result["filename"],
                    "missing_fields": result["missing_fields"]
                })
                
                # Create a more detailed recommendation
                recommendation = f"Complete the document '{document_name}' ({result['filename']}) by adding the following missing fields:"
                assessment["recommendations"].append(recommendation)
                
                # Add each missing field as a separate recommendation
                for field in result["missing_fields"]:
                    assessment["recommendations"].append(f"  - Add {field} to {document_name}")
            else:
                # Document is complete
                assessment["complete_documents"].append({
                    "name": document_name,
                    "filename": result["filename"]
                })
        
        return assessment


class OutputFormatterAgent:
    """
    Agent that formats the assessment results for display.
    """
    
    def run(self, assessment):
        """
        Format the assessment results for display.
        
        Args:
            assessment: The assessment results from the AssessmentRecommendationAgent.
            
        Returns:
            A formatted string with the assessment results.
        """
        # Format the assessment results as markdown
        formatted_report = "## Assessment Report\n\n"
        
        # Add complete documents section
        formatted_report += "### ✅ Complete Documents\n"
        if assessment["complete_documents"]:
            for doc in assessment["complete_documents"]:
                formatted_report += f"- **{doc['name']}** ({doc['filename']})\n"
        else:
            formatted_report += "- No complete documents found\n"
        
        # Add incomplete documents section
        formatted_report += "\n### ⚠️ Incomplete Documents\n"
        if assessment["incomplete_documents"]:
            for doc in assessment["incomplete_documents"]:
                formatted_report += f"- **{doc['name']}** ({doc['filename']})\n"
                formatted_report += "  - Missing fields:\n"
                for field in doc["missing_fields"]:
                    formatted_report += f"    - {field}\n"
        else:
            formatted_report += "- No incomplete documents found\n"
        
        # Add missing documents section
        formatted_report += "\n### ❌ Missing Documents\n"
        if assessment["missing_documents"]:
            for doc_name in assessment["missing_documents"]:
                formatted_report += f"- **{doc_name}**\n"
        else:
            formatted_report += "- No missing documents\n"
        
        # Add recommendations section
        formatted_report += "\n### 📋 Recommendations\n"
        if assessment["recommendations"]:
            for recommendation in assessment["recommendations"]:
                formatted_report += f"- {recommendation}\n"
        else:
            formatted_report += "- No recommendations\n"
        
        return formatted_report


class SupervisorAgent:
    """
    Supervisor Agent to orchestrate the agent pipeline in a serial, chained flow:
    
      1. ChecklistCrossReferenceAgent
      2. AssessmentRecommendationAgent
      3. OutputFormatterAgent

    Input:
        submission_data: Pre-processed submission package data.
    Output:
        A final formatted report and completeness percentage.
    """
    def __init__(self, checklist):
        self.checklist_agent = ChecklistCrossReferenceAgent(checklist)
        self.assessment_agent = AssessmentRecommendationAgent()
        self.formatter_agent = OutputFormatterAgent()
        self.total_required_files = 9  # Total number of required files

    def run(self, submission_data):
        try:
            # Step 1: Cross-reference the submission data against the checklist
            cross_ref_result = self.checklist_agent.run(submission_data)
            
            # Step 2: Analyze the cross-reference result
            assessment_report = self.assessment_agent.run(cross_ref_result)
            
            # Step 3: Calculate completeness percentage
            completeness_percentage = self.calculate_completeness(cross_ref_result)
            
            # Step 4: Format the assessment report
            formatted_report = self.formatter_agent.run(assessment_report)
            
            return formatted_report, completeness_percentage
            
        except Exception as e:
            raise e

    def calculate_completeness(self, cross_ref_result):
        """Calculate the completeness percentage of the submission package."""
        completed_files = 0
        for result in cross_ref_result.values():
            if result["found"] and result["complete"]:
                completed_files += 1
            elif result["found"] and not result["complete"]:
                completed_files += 0.5  # Consider incomplete files as half finished
        return (completed_files / self.total_required_files) * 100


# --- Helper Functions for ZIP Processing ---

def download_zip_from_s3(s3_url: str) -> BytesIO:
    """Downloads a ZIP file from S3 and returns it as a BytesIO object."""
    try:
        # First try to use boto3 if AWS credentials are available
        if all(key in os.environ for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]):
            try:
                s3 = boto3.client(
                    's3',
                    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                    region_name=os.environ["AWS_REGION"]
                )

                # Parse S3 URL
                bucket_name = s3_url.split('/')[2]
                key = '/'.join(s3_url.split('/')[3:])

                # Download the file
                response = s3.get_object(Bucket=bucket_name, Key=key)
                zip_bytes = response['Body'].read()
                return BytesIO(zip_bytes)
            except Exception as e:
                st.warning(f"Direct S3 access failed: {str(e)}. Trying HTTP fallback...")
                # Fall back to HTTP request
        
        # If AWS credentials are not available or direct S3 access failed, try HTTP request
        if s3_url.startswith('s3://'):
            # Convert s3:// URL to https:// URL for public buckets
            bucket_name = s3_url.split('/')[2]
            key = '/'.join(s3_url.split('/')[3:])
            http_url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
        else:
            http_url = s3_url
            
        st.info(f"Attempting to download from: {http_url}")
        return download_zip_from_url(http_url)
        
    except Exception as e:
        st.error(f"Error downloading ZIP file from S3: {str(e)}")
        return None

def download_zip_from_url(url: str) -> BytesIO:
    """Downloads a ZIP file from a URL and returns it as a BytesIO object."""
    try:
        st.info(f"Downloading from URL: {url}")
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/zip, application/octet-stream, */*'
        }
        
        # Make the request with a timeout
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        
        # Check if the request was successful
        if response.status_code != 200:
            st.error(f"Failed to download: HTTP status code {response.status_code}")
            return None
            
        # Check if the content type is a ZIP file
        content_type = response.headers.get('Content-Type', '')
        if 'application/zip' not in content_type and 'application/octet-stream' not in content_type:
            st.warning(f"Warning: Content type '{content_type}' may not be a ZIP file. Attempting to process anyway.")
        
        # Get the content and return it as BytesIO
        content = response.content
        if not content:
            st.error("Downloaded file is empty")
            return None
            
        st.success(f"Successfully downloaded {len(content) / 1024:.1f} KB from URL")
        return BytesIO(content)
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading ZIP file from URL: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error downloading ZIP file: {str(e)}")
        return None

def process_uploaded_zip(zip_file: BytesIO) -> list:
    """
    Processes a ZIP file (from BytesIO), caches embeddings, and returns a list of file dictionaries.
    """
    submission_data = []

    with ZipFile(zip_file) as zip_ref:
        for filename in zip_ref.namelist():
            file_ext = os.path.splitext(filename)[1].lower()
            file_bytes = zip_ref.read(filename)
            content = ""

            # Generate a unique cache key based on the file content
            file_hash = hashlib.md5(file_bytes).hexdigest()
            cache_key = f"{filename}_{file_hash}"
            cache_file = f".cache/{cache_key}.pkl"  # Cache file path

            # Create the cache directory if it doesn't exist
            os.makedirs(".cache", exist_ok=True)

            if os.path.exists(cache_file):
                # Load from cache
                print(f"Loading {filename} from cache")
                try:
                    with open(cache_file, "rb") as f:
                        content = pickle.load(f)
                except Exception as e:
                    st.error(f"Error loading {filename} from cache: {str(e)}")
                    content = ""  # Or handle the error as appropriate
            else:
                # Process and cache
                print(f"Processing {filename} and caching")
                if file_ext == ".pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp.flush()
                        tmp_path = tmp.name
                    file_size = os.path.getsize(tmp_path) / (1024 * 1024)
                    workers = 2 if file_size > 2 else 1
                    try:
                        parser = LlamaParse(
                            api_key=LLAMA_CLOUD_API_KEY,
                            result_type="markdown",
                            num_workers=workers,
                            verbose=True
                        )
                        llama_documents = parser.load_data(tmp_path)
                        content = "\n".join([doc.text for doc in llama_documents])
                    except Exception as e:
                        content = f"Error parsing PDF: {str(e)}"
                        st.error(f"Error parsing PDF {filename}: {str(e)}")
                    finally:
                        os.remove(tmp_path)
                elif file_ext == ".txt":
                    try:
                        content = file_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        content = file_bytes.decode("latin1")
                    except Exception as e:
                        content = f"Error decoding text file {filename}: {str(e)}"
                        st.error(f"Error decoding text file {filename}: {str(e)}")
                else:
                    continue  # Skip unsupported file types

                # Save to cache
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(content, f)
                except Exception as e:
                    st.error(f"Error saving {filename} to cache: {str(e)}")

            submission_data.append({
                "filename": filename,
                "file_type": file_ext.replace(".", ""),
                "content": content,
                "metadata": {}
            })
    return submission_data

# --- Main Streamlit App ---

def main():
    """Main function to run the Streamlit app."""
    # Set page config moved to the top of the file
    
    # Add custom CSS
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.25rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📚 IND Assistant", "📋 Submission Assessment"])
    
    # Tab 1: IND Assistant (Chat Interface)
    with tab1:
        st.title("📚 IND Assistant")
        st.markdown("Ask questions about IND submissionrequirements and get detailed answers.")
        
        # Add "Clear Chat History" button
        if st.button("Clear Chat History"):
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()
        
        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Check if vector store exists first
        if "rag_chain" not in st.session_state or "vectorstore" not in st.session_state:
            # First try to load from the serialized vector store
            if os.path.exists(VECTOR_STORE_PATH):
                with st.spinner("🔄 Loading knowledge base from disk..."):
                    try:
                        with open(VECTOR_STORE_PATH, "rb") as f:
                            vectorstore = pickle.load(f)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.rag_chain = create_rag_chain(vectorstore.as_retriever())
                        st.success("✅ Knowledge base loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading vector store: {str(e)}")
                        # Fall back to loading from preprocessed data or PDF
            
            # If vector store couldn't be loaded, try preprocessed data
            if "vectorstore" not in st.session_state:
                if os.path.exists(PREPROCESSED_FILE):
                    with st.spinner("🔄 Initializing knowledge base from preprocessed data..."):
                        documents = load_preprocessed_data(PREPROCESSED_FILE)
                        vectorstore = init_vector_store(documents)
                        st.session_state.rag_chain = create_rag_chain(vectorstore.as_retriever())
                        st.session_state.vectorstore = vectorstore
                # If no preprocessed data, try processing the PDF directly
                elif os.path.exists(PDF_FILE):
                    with st.spinner("🔄 Processing PDF and initializing knowledge base..."):
                        documents = load_pdf(PDF_FILE)
                        # Optionally save preprocessed data for backward compatibility
                        json_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
                        with open(PREPROCESSED_FILE, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, indent=4)
                        
                        vectorstore = init_vector_store(documents)
                        st.session_state.rag_chain = create_rag_chain(vectorstore.as_retriever())
                        st.session_state.vectorstore = vectorstore
                else:
                    st.error(f"❌ Neither vector store, preprocessed data, nor PDF file found. Please upload the PDF.")
                    return

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input and response handling
        if question := st.chat_input("Ask a question about IND requirements"):
            # Display user question
            with st.chat_message("user"):
                st.markdown(question)
            
            # Add to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = cached_response(question)
                st.markdown(response)

            # Store bot response in chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Tab 2: Submission Assessment
    with tab2:
        st.title("📋 Submission Assessment")
        st.write(
            """
            Upload a ZIP file containing your submission package or enter an S3 URL.
            The ZIP file can include PDF and text files.
            """
        )

        # Input method selection
        input_method = st.radio(
            "Choose input method",
            ["Upload ZIP file", "Enter S3 URL"],
            key="input_method"
        )
        
        # Load checklist
        try:
            checklist = load_checklist()
        except Exception as e:
            st.error(f"Error loading checklist: {str(e)}")
            checklist = {}
        
        submission_data = None
        
        # Initialize session state for tracking S3 processing
        if 's3_processed' not in st.session_state:
            st.session_state.s3_processed = False
        if 'previous_input_method' not in st.session_state:
            st.session_state.previous_input_method = input_method
            
        # Reset processing state if input method changes
        if st.session_state.previous_input_method != input_method:
            st.session_state.s3_processed = False
            st.session_state.previous_input_method = input_method
        
        if input_method == "Upload ZIP file":
            uploaded_file = st.file_uploader("Upload ZIP file", type="zip")
            if uploaded_file is not None:
                with st.spinner("Processing uploaded ZIP file..."):
                    zip_bytes = BytesIO(uploaded_file.getvalue())
                    submission_data = process_uploaded_zip(zip_bytes)
        
        elif input_method == "Enter S3 URL":
            s3_url = st.text_input("Enter S3 URL")
            process_button = st.button("Process S3 URL")
            
            # Process the S3 URL when the button is clicked
            if process_button or st.session_state.s3_processed:
                if not st.session_state.s3_processed:  # Only download if not already processed
                    if not s3_url:
                        st.error("Please enter an S3 URL")
                    else:
                        with st.spinner("Downloading and processing ZIP from S3..."):
                            try:
                                zip_bytes = download_zip_from_s3(s3_url)
                                if zip_bytes is not None:
                                    submission_data = process_uploaded_zip(zip_bytes)
                                    st.session_state.s3_processed = True
                                    st.session_state.submission_data = submission_data  # Store in session state
                                else:
                                    st.error("Failed to download ZIP file from S3 URL.")
                            except Exception as e:
                                st.error(f"Error processing S3 URL: {str(e)}")
                else:
                    # Use the stored submission data
                    submission_data = st.session_state.submission_data
        
        # Process submission data if available
        if submission_data:
            st.success(f"✅ Successfully processed {len(submission_data)} files")
            
            # Display file list
            with st.expander("View processed files"):
                for i, doc in enumerate(submission_data):
                    # Handle the submission data as a dictionary, not a Document object
                    st.write(f"{i+1}. {doc.get('filename', 'Unknown file')}")
            
            # Run assessment with a unique key for the button
            run_assessment = st.button("Run Assessment", key="run_assessment_button")
            if run_assessment:
                # Initialize the supervisor agent without a debug expander
                supervisor = SupervisorAgent(checklist)
                
                with st.spinner("Running assessment..."):
                    try:
                        # Run the assessment without any debug messages
                        formatted_report, completeness_percentage = supervisor.run(submission_data)
                    except Exception as e:
                        # Simple error message
                        st.error("An error occurred during assessment.")
                        formatted_report = "Assessment failed due to an error."
                        completeness_percentage = 0
                
                # Display assessment results
                st.subheader("Assessment Results")
                
                # Display Completeness Percentage
                st.subheader("Submission Package Completeness")
                st.progress(completeness_percentage / 100)
                st.write(f"Overall Completeness: {completeness_percentage:.1f}%")
                
                # Display the formatted report
                st.markdown(formatted_report)
                
                # Download button for results
                result_bytes = formatted_report.encode()
                st.download_button(
                    label="Download Assessment Results",
                    data=result_bytes,
                    file_name="assessment_results.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main() 