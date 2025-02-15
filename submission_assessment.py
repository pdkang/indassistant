"""
Submission Assessment Module

This module implements a LangGraph agentic pipeline to perform
cross-reference of an uploaded submission package (ZIP file) against a predefined
IND checklist. It supports processing of both PDF (using LlamaParse in the
pre-agent phase) and text files.

A Streamlit interface is provided to allow users to upload a ZIP file and view the assessment report.
"""

import os
import io
import tempfile
from zipfile import ZipFile
import streamlit as st
from llama_parse import LlamaParse

import pickle
import hashlib


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

def process_uploaded_zip(uploaded_zip) -> list:
    """
    Processes an uploaded ZIP file, caches embeddings, and returns a list of file dictionaries.
    """
    submission_data = []

    with ZipFile(uploaded_zip) as zip_ref:
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


# --- Streamlit Interface ---

def main():
    st.title("Submission Package Assessment")
    st.write(
        """
        Upload a ZIP file containing your submission package.
        The ZIP file can include PDF and text files.
        """
    )
    
    uploaded_file = st.file_uploader("Choose a ZIP file", type=["zip"])
    
    if uploaded_file is not None:
        try:
            # Process the uploaded ZIP file to extract submission data
            submission_data = process_uploaded_zip(uploaded_file)
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
    # To run with Streamlit, use: streamlit run submission_assessment.py
    main() 