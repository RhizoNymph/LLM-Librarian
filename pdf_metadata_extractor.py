from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Union
import json
import hashlib

import fitz  
import pytesseract
from PIL import Image
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

class DocumentType(Enum):
    BOOK = "book"
    PAPER = "paper"
    BLOG_ARTICLE = "blog_article"
    TECHNICAL_REPORT = "technical_report"
    THESIS = "thesis"
    PRESENTATION = "presentation"
    DOCUMENTATION = "documentation"
    PATENT = "patent"

class BookMetadata(BaseModel):
    """Metadata schema for books"""
    title: str = Field(description="The full title of the book")
    authors: List[str] = Field(description="List of author names")
    publisher: Optional[str] = Field(description="Name of the publishing company")
    publication_year: Optional[int] = Field(description="Year the book was published")
    isbn: Optional[str] = Field(description="ISBN number if available")
    edition: Optional[str] = Field(description="Edition information if available")
    language: Optional[str] = Field(description="Primary language of the book")
    subject_areas: List[str] = Field(description="Main subject areas or categories")
    table_of_contents: Optional[List[str]] = Field(description="Main chapter titles")

class PaperMetadata(BaseModel):
    """Metadata schema for academic papers"""
    title: str = Field(description="The full title of the paper")
    authors: List[str] = Field(description="List of author names")
    abstract: str = Field(description="Paper abstract")
    keywords: List[str] = Field(description="Keywords or subject terms")
    doi: Optional[str] = Field(description="Digital Object Identifier")
    journal: Optional[str] = Field(description="Journal name if published")
    conference: Optional[str] = Field(description="Conference name if presented")
    publication_year: Optional[int] = Field(description="Year published/presented")
    institution: Optional[str] = Field(description="Research institution(s)")
    citations: Optional[List[str]] = Field(description="Key citations from first page")

class BlogArticleMetadata(BaseModel):
    """Metadata schema for blog articles"""
    title: str = Field(description="The full title of the article")
    authors: List[str] = Field(description="List of author names")
    publication_date: Optional[datetime] = Field(description="Publication date")
    blog_name: Optional[str] = Field(description="Name of the blog or platform")
    url: Optional[str] = Field(description="Original URL if available")
    tags: List[str] = Field(description="Article tags or categories")
    reading_time: Optional[int] = Field(description="Estimated reading time in minutes")
    summary: str = Field(description="Article summary or introduction")
    series: Optional[str] = Field(description="Blog post series name if part of one")

class TechnicalReportMetadata(BaseModel):
    """Metadata schema for technical reports"""
    title: str = Field(description="Report title")
    authors: List[str] = Field(description="List of authors")
    organization: str = Field(description="Organization that produced the report")
    report_number: Optional[str] = Field(description="Report identifier/number")
    date: Optional[datetime] = Field(description="Publication date")
    executive_summary: Optional[str] = Field(description="Executive summary")
    keywords: List[str] = Field(description="Key terms")
    classification: Optional[str] = Field(description="Report classification (e.g., Internal, Public)")

class ThesisMetadata(BaseModel):
    """Metadata schema for theses and dissertations"""
    title: str = Field(description="Thesis title")
    author: str = Field(description="Author name")
    degree: str = Field(description="Degree type (e.g., PhD, Masters)")
    institution: str = Field(description="Academic institution")
    department: Optional[str] = Field(description="Department or faculty")
    year: int = Field(description="Year of submission")
    advisors: List[str] = Field(description="Thesis advisors/supervisors")
    abstract: str = Field(description="Thesis abstract")
    keywords: List[str] = Field(description="Key terms")

class PatentMetadata(BaseModel):
    """Metadata schema for patents"""
    title: str = Field(description="Patent title")
    inventors: List[str] = Field(description="List of inventors")
    assignee: Optional[str] = Field(description="Patent assignee/owner")
    patent_number: Optional[str] = Field(description="Patent number")
    filing_date: Optional[datetime] = Field(description="Filing date")
    publication_date: Optional[datetime] = Field(description="Publication date")
    abstract: str = Field(description="Patent abstract")
    classification: Optional[str] = Field(description="Patent classification")
    claims: Optional[List[str]] = Field(description="Main patent claims")

@dataclass
class PDFContext:
    pdf_path: Path
    ocr_text: str
    document_type: DocumentType

class MetadataStore(BaseModel):
    """Container for all document metadata"""
    version: str = "1.0"
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    documents: Dict[str, Union[BookMetadata, PaperMetadata, BlogArticleMetadata, TechnicalReportMetadata, ThesisMetadata, PatentMetadata]] = Field(
        default_factory=dict,
        description="Dictionary of document metadata keyed by file hash"
    )

    def get_by_hash(self, file_hash: str):
        """Get metadata for a file by its hash"""
        return self.documents.get(file_hash)

class PDFProcessor:
    def __init__(self, pdf_dir: str = "./pdfs", metadata_file: str = "./pdf_metadata.json"):
        print("Initializing PDFProcessor...")
        self.pdf_dir = Path(pdf_dir)
        self.metadata_file = Path(metadata_file)

        if not self.pdf_dir.exists():
            self.pdf_dir.mkdir(parents=True)
        self.metadata_store = self._load_metadata_store()
    def _load_metadata_store(self) -> MetadataStore:
        """Load or create metadata store"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return MetadataStore.model_validate(data)
            except Exception as e:
                print(f"Error loading metadata file: {e}")
                print("Creating new metadata store...")
        return MetadataStore()

    def _save_metadata_store(self):
        """Save metadata store to file"""
        with open(self.metadata_file, 'w') as f:
            json_data = self.metadata_store.model_dump_json(indent=2)
            f.write(json_data)

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        print(f"Calculating hash for {file_path}")
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_processed(self, file_hash: str) -> bool:
        """Check if file has already been processed"""
        return file_hash in self.metadata_store.documents

    def extract_text_from_pdf(self, pdf_path: Path, all_pages: bool = True) -> str:
        """Extract text from PDF using PyMuPDF and OCR for first page"""
        print(f"\nStarting text extraction for: {pdf_path}")
        try:
            doc = fitz.open(str(pdf_path))
            print(f"PDF opened successfully. Total pages: {len(doc)}")
                        
            print("Processing first page with OCR...")
            first_page = doc[0]
            pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            first_page_text = self._process_image_ocr(img)
                        
            if all_pages:
                pages_to_process = range(1, len(doc))
                print("Processing remaining pages with PyMuPDF...")
            else:                
                main_pages = list(range(1, min(10, len(doc))))
                end_pages = list(range(max(1, len(doc)-5), len(doc)))
                middle_pages = list(range(10, min(20, len(doc))))
                pages_to_process = sorted(set(main_pages + middle_pages + end_pages))
                print(f"Processing partial pages with PyMuPDF: {pages_to_process}")
            
            remaining_text = []
            for page_num in pages_to_process:                
                page = doc[page_num]
                remaining_text.append(page.get_text())
                        
            full_text = f"{first_page_text}\n" + "\n".join(remaining_text)
            print(f"Successfully extracted text. Total length: {len(full_text)}")
            return full_text
            
        except Exception as e:
            print(f"Error in extract_text_from_pdf: {str(e)}")
            raise

    def _process_image_ocr(self, img: Image.Image) -> str:
        """Process a single image with OCR"""
        try:
            img_array = np.array(img)
            text = pytesseract.image_to_string(img_array, lang='eng')
            return text.strip()
        except Exception as e:
            print(f"Error in OCR processing: {str(e)}")
            return ""

    async def determine_document_type(self, ocr_text: str) -> DocumentType:
        """Determine document type using LLM"""
        print("Using LLM to determine document type...")
        try:
            # Get a representative sample (first ~2000 chars should be enough for classification)
            text_sample = ocr_text[:2000]
            
            result = await document_type_agent.run(text_sample)
            print(f"LLM detected document type: {result.data}")
            return result.data
            
        except Exception as e:
            print(f"Error in LLM document type detection: {str(e)}")
            print("Falling back to heuristic detection...")
            
            # Fallback to basic heuristics if LLM fails
            text_lower = ocr_text.lower()
            
            if any(x in text_lower for x in ["patent", "claims:", "inventors:", "assignee:"]):
                return DocumentType.PATENT
                    
            if any(x in text_lower for x in ["thesis", "dissertation", "submitted in partial fulfillment"]):
                return DocumentType.THESIS
                    
            if any(x in text_lower for x in ["technical report", "tr-", "executive summary"]):
                return DocumentType.TECHNICAL_REPORT
                    
            if any(x in text_lower for x in ["posted on", "reading time:", "originally published at"]):
                return DocumentType.BLOG_ARTICLE
                    
            if any(x in text_lower for x in ["abstract", "doi:", "journal", "conference"]):
                return DocumentType.PAPER
                    
            return DocumentType.BOOK

document_type_agent = Agent(
    'google-gla:gemini-2.0-flash',
    result_type=DocumentType,
    system_prompt="""
    You are a document classification specialist. Analyze the text from a document and determine its type.
    Choose from the following categories:
    - BOOK: Books, textbooks, manuals (typically have chapters, table of contents, publisher info)
    - PAPER: Academic papers, research articles, conference papers (have abstract, citations, academic formatting)
    - BLOG_ARTICLE: Blog posts, online articles (informal, web-focused)
    - TECHNICAL_REPORT: Technical reports, white papers (formal reports from organizations)
    - THESIS: PhD theses, Masters dissertations (long-form academic work)
    - PATENT: Patent documents (legal format, claims)
    - PRESENTATION: Slides, presentations (bullet points, visual focus)
    - DOCUMENTATION: Software documentation, API docs (technical reference material)
    
    Pay special attention to:
    - Document structure and formatting
    - Presence of abstract, citations, or references
    - Academic vs commercial tone
    - Length and depth of content
    - Section organization
    """
)

@document_type_agent.tool
async def get_document_sample(ctx: RunContext, text: str) -> str:
    """Get a representative sample of the document text for classification"""
    # Take first 2000 chars as sample
    return text[:2000]

metadata_agent = Agent(
    'google-gla:gemini-2.0-flash',
    deps_type=PDFContext,
    result_type=BookMetadata | PaperMetadata | BlogArticleMetadata | TechnicalReportMetadata | ThesisMetadata | PatentMetadata,
    system_prompt="""
    You are a metadata extraction specialist. Analyze the OCR text from the first 10 pages 
    of a PDF document and extract relevant metadata based on the document type (book or paper).
    Be thorough but do not make up information - if a field cannot be determined from the 
    provided text, leave it as None.
    """
)

@metadata_agent.tool
async def get_document_text(ctx: RunContext[PDFContext]) -> str:
    """Get the OCR text from the document"""
    return ctx.deps.ocr_text

async def process_pdf(pdf_path: Path) -> None:
    print(f"\n{'='*50}")
    print(f"Starting processing of: {pdf_path}")
    print(f"{'='*50}")
    
    try:
        processor = PDFProcessor()
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        file_hash = processor.calculate_file_hash(pdf_path)
        print(f"File hash: {file_hash}")
        
        if processor.is_processed(file_hash):
            print(f"File already processed: {pdf_path}")
            return
                    
        try:
            print(f"\nStep 1: Extracting text from PDF (full document)")
            text = processor.extract_text_from_pdf(pdf_path, all_pages=True)
            if not text.strip():
                raise ValueError("No text extracted from PDF")
        except Exception as e:
            print(f"Full document processing failed: {str(e)}")
            print("Falling back to partial document processing...")
            
            text = processor.extract_text_from_pdf(pdf_path, all_pages=False)
            if not text.strip():
                raise ValueError(f"No text extracted from PDF: {pdf_path}")
            
        print(f"\nStep 2: Determining document type")
        doc_type = await processor.determine_document_type(text)
        print(f"Detected document type: {doc_type}")
        
        print(f"\nStep 3: Creating context for PydanticAI")
        context = PDFContext(
            pdf_path=pdf_path,
            ocr_text=text,  
            document_type=doc_type
        )
        
        print(f"\nStep 4: Running metadata extraction")
        result = await metadata_agent.run(
            "Extract all available metadata from this document.",
            deps=context
        )
        
        print(f"\nStep 5: Saving results")        
        processor.metadata_store.documents[file_hash] = result.data
        processor.metadata_store.last_updated = datetime.utcnow()
        processor._save_metadata_store()
        print(f"Updated metadata store: {processor.metadata_file}")
            
    except Exception as e:
        print(f"Error in process_pdf: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

print("pdf_metadata_extractor.py loaded") 