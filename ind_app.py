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


# File paths for IND Assistant
PDF_FILE = "IND-312.pdf"
PREPROCESSED_FILE = "preprocessed_docs.json"

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

    Answer in Markdown with checkboxes (- [ ]). If unsure, say "I can only answer IND related questions.".
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

# Caching function to prevent redundant RAG processing
@st.cache_data
def cached_response(question: str):
    """Retrieve cached response if available, otherwise compute response."""
    if "rag_chain" in st.session_state:
        return st.session_state.rag_chain.invoke({"question": question})["response"]
    else:
        st.error("RAG chain not initialized. Please initialize the IND Assistant first.")
        return ""

# --- Submission Assessment Functions ---

# Access API key from environment variable
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")

# Check if the API key is available
if not LLAMA_CLOUD_API_KEY:
    st.error("LLAMA_CLOUD_API_KEY not found in environment variables. Please set it in your Hugging Face Space secrets.")
    st.stop()

# Sample Checklist Configuration (this should be adjusted to your actual IND requirements)
IND_CHECKLIST = {
    "Investigator Brochure": {
        "file_patterns": ["brochure", "ib"],
        "required_keywords": ["pharmacology", "toxicology", "clinical data"]
    },
    "Clinical Protocol": {
        "file_patterns": ["clinical", "protocol"],
        "required_keywords": ["study design", "objectives", "patient population", "dosing regimen", "endpoints"]
    },
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
        cross_reference_result = {}
        for document_name, config in self.checklist.items():
            file_patterns = config.get("file_patterns", [])
            required_keywords = config.get("required_keywords", [])
            matched_file = None
            
            # Attempt to find a matching file based on filename patterns.
            for file_info in submission_data:
                filename = file_info.get("filename", "").lower()
                if any(pattern.lower() in filename for pattern in file_patterns):
                    matched_file = file_info
                    break
            
            # Build the result per checklist item.
            if not matched_file:
                # File is completely missing.
                cross_reference_result[document_name] = {
                    "status": "missing",
                    "missing_fields": required_keywords
                }
            else:
                # File found, check if its content includes the required keywords.
                content = matched_file.get("content", "").lower()
                missing_fields = []
                for keyword in required_keywords:
                    if keyword.lower() not in content:
                        missing_fields.append(keyword)
                if missing_fields:
                    cross_reference_result[document_name] = {
                        "status": "incomplete",
                        "missing_fields": missing_fields
                    }
                else:
                    cross_reference_result[document_name] = {
                        "status": "present",
                        "missing_fields": []
                    }
        return cross_reference_result


class AssessmentRecommendationAgent:
    """
    Agent that analyzes the cross-reference data and produces an
    assessment report with recommendations.

    Input:
        cross_reference_result: dict mapping checklist items to their status.
    Output:
        A dict containing an overall compliance flag and detailed recommendations.
    """
    def run(self, cross_reference_result):
        recommendations = {}
        overall_compliant = True

        for doc, result in cross_reference_result.items():
            status = result.get("status")
            if status == "missing":
                recommendations[doc] = f"{doc} is missing. Please include the document."
                overall_compliant = False
            elif status == "incomplete":
                missing = ", ".join(result.get("missing_fields", []))
                recommendations[doc] = (f"{doc} is incomplete. Missing required fields: {missing}. "
                                        "Please update accordingly.")
                overall_compliant = False
            else:
                recommendations[doc] = f"{doc} is complete."
        assessment = {
            "overall_compliant": overall_compliant,
            "recommendations": recommendations
        }
        return assessment


class OutputFormatterAgent:
    """
    Agent that formats the assessment report into a user-friendly format.
    This example formats the output as Markdown.
    
    Input:
        assessment: dict output from AssessmentRecommendationAgent.
    Output:
        A formatted string report.
    """
    def run(self, assessment):
        overall = "Compliant" if assessment.get("overall_compliant") else "Non-Compliant"
        lines = []
        lines.append("# Submission Package Assessment Report")
        lines.append(f"**Overall Compliance:** {overall}\n")
        recommendations = assessment.get("recommendations", {})
        for doc, rec in recommendations.items():
            lines.append(f"### {doc}")
            # Format recommendations as bullet points
            if "incomplete" in rec.lower():
                missing_fields = rec.split("Missing required fields: ")[1].split(".")[0].split(", ")
                lines.append("- Status: Incomplete")
                lines.append("  - Missing Fields:")
                for field in missing_fields:
                    lines.append(f"    - {field}")
            else:
                lines.append(f"- Status: {rec}")
        return "\n".join(lines)


class SupervisorAgent:
    """
    Supervisor Agent to orchestrate the agent pipeline in a serial, chained flow:
    
      1. ChecklistCrossReferenceAgent
      2. AssessmentRecommendationAgent
      3. OutputFormatterAgent

    Input:
        submission_data: Pre-processed submission package data.
    Output:
        A final formatted report.
    """
    def __init__(self, checklist):
        self.checklist_agent = ChecklistCrossReferenceAgent(checklist)
        self.assessment_agent = AssessmentRecommendationAgent()
        self.formatter_agent = OutputFormatterAgent()

    def run(self, submission_data):
        # Step 1: Cross-reference the submission data against the checklist.
        cross_ref_result = self.checklist_agent.run(submission_data)
        # Step 2: Analyze the cross-reference result to produce assessment and recommendations.
        assessment_report = self.assessment_agent.run(cross_ref_result)
        # Step 3: Format the assessment report for display.
        formatted_report = self.formatter_agent.run(assessment_report)
        return formatted_report


# --- Helper Functions for ZIP Processing ---

def download_zip_from_s3(s3_url: str) -> BytesIO:
    """Downloads a ZIP file from S3 and returns it as a BytesIO object."""
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
        st.error(f"Error downloading ZIP file from S3: {str(e)}")
        return None

def download_zip_from_url(url: str) -> BytesIO:
    """Downloads a ZIP file from a URL and returns it as a BytesIO object."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading ZIP file from URL: {str(e)}")
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
    st.title("IND Assistant and Submission Assessment")

    # Sidebar for app selection
    app_mode = st.sidebar.selectbox(
        "Choose an app mode",
        ["IND Assistant", "Submission Assessment"]
    )

    if app_mode == "IND Assistant":
        st.header("IND Assistant")
        st.markdown("Chat about Investigational New Drug Applications")

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Load preprocessed data and initialize the RAG chain
        if "rag_chain" not in st.session_state:
            if not os.path.exists(PREPROCESSED_FILE):
                st.error(f"❌ Preprocessed file '{PREPROCESSED_FILE}' not found. Please run preprocessing first.")
                return  # Stop execution if preprocessed data is missing

            with st.spinner("🔄 Initializing knowledge base..."):
                documents = load_preprocessed_data(PREPROCESSED_FILE)
                vectorstore = init_vector_store(documents)
                st.session_state.rag_chain = create_rag_chain(vectorstore.as_retriever())

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input and response handling
        if prompt := st.chat_input("Ask about IND requirements"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response (cached if already asked before)
            with st.chat_message("assistant"):
                response = cached_response(prompt)
                st.markdown(response)

            # Store bot response in chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    elif app_mode == "Submission Assessment":
        st.header("Submission Package Assessment")
        st.write(
            """
            Upload a ZIP file containing your submission package, or enter the S3 URL of the ZIP file.
            The ZIP file can include PDF and text files.
            """
        )

        # Option 1: Upload ZIP file
        uploaded_file = st.file_uploader("Choose a ZIP file", type=["zip"])

        # Option 2: Enter S3 URL
        s3_url = st.text_input("Or enter S3 URL of the ZIP file:")

        zip_file = None  # Initialize zip_file

        if uploaded_file is not None:
            zip_file = BytesIO(uploaded_file.read())
        elif s3_url:
            zip_file = download_zip_from_s3(s3_url)
        
        if zip_file:
            try:
                # Process the ZIP file
                submission_data = process_uploaded_zip(zip_file)
                st.success("File processed successfully!")

                # Display a summary of the extracted files
                st.subheader("Extracted Files")
                for file_info in submission_data:
                    st.write(f"**{file_info['filename']}** - ({file_info['file_type'].upper()})")

                # Instantiate and run the SupervisorAgent
                supervisor = SupervisorAgent(IND_CHECKLIST)
                assessment_report = supervisor.run(submission_data)

                st.subheader("Assessment Report")
                st.markdown(assessment_report)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    # Preprocess PDF if it doesn't exist
    if not os.path.exists(PREPROCESSED_FILE):
        preprocess_pdf(PDF_FILE)
    main() 