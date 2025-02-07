from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Union
import json
import hashlib

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class DocumentType(Enum):
    BOOK = "book"
    PAPER = "paper"

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

@dataclass
class PDFContext:
    pdf_path: Path
    ocr_text: str
    document_type: DocumentType

class MetadataStore(BaseModel):
    """Container for all document metadata"""
    version: str = "1.0"
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    documents: Dict[str, Union[BookMetadata, PaperMetadata]] = Field(
        default_factory=dict,
        description="Dictionary of document metadata keyed by file hash"
    )

class PDFProcessor:
    def __init__(self, pdf_dir: str = "./pdfs", metadata_file: str = "./pdf_metadata.json"):
        print("Initializing PDFProcessor...")
        self.pdf_dir = Path(pdf_dir)
        self.metadata_file = Path(metadata_file)
        self.max_workers = max(1, multiprocessing.cpu_count() - 2)
        print(f"Using {self.max_workers} workers for OCR")
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

    def convert_pdf_to_images(self, pdf_path: Path, max_pages: int = 10, extended_search: bool = False) -> List[Image.Image]:
        """Convert PDF pages to PIL Images"""
        print(f"\nStarting PDF conversion for: {pdf_path}")
        try:
            doc = fitz.open(str(pdf_path))
            print(f"PDF opened successfully. Total pages: {len(doc)}")
            images = []
            
            # If extended_search is True, look at more pages for specific content
            if extended_search:
                # Look at first 10 pages for main metadata
                main_pages = list(range(min(10, len(doc))))
                # Look at references/bibliography pages near the end
                end_pages = list(range(max(0, len(doc)-5), len(doc)))
                # Look at some middle pages for content analysis
                middle_pages = list(range(10, min(20, len(doc))))
                pages_to_process = sorted(set(main_pages + middle_pages + end_pages))
            else:
                pages_to_process = range(min(max_pages, len(doc)))
            
            for page_num in pages_to_process:
                print(f"Converting page {page_num + 1}")
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            
            print(f"Successfully converted {len(images)} pages")
            return images
        except Exception as e:
            print(f"Error in convert_pdf_to_images: {str(e)}")
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

    def perform_ocr(self, images: List[Image.Image]) -> str:
        """Perform parallel OCR on list of images and return combined text"""
        print("\nStarting parallel OCR process...")
        text_chunks = []
        
        try:
            # Create a process pool for parallel OCR
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Process images in parallel
                print(f"Processing {len(images)} images with {self.max_workers} workers")
                results = list(executor.map(self._process_image_ocr, images))
                
                # Filter and collect results
                text_chunks = [text for text in results if text]
                print(f"Successfully processed {len(text_chunks)} images")
                
            result = "\n".join(text_chunks)
            print(f"OCR complete. Total text length: {len(result)}")
            return result
            
        except Exception as e:
            print(f"Error in parallel OCR: {str(e)}")
            raise

    def determine_document_type(self, ocr_text: str) -> DocumentType:
        """Determine if document is a book or paper based on OCR text"""
        # Simple heuristic - can be improved
        paper_indicators = ["abstract", "doi:", "journal", "conference"]
        for indicator in paper_indicators:
            if indicator.lower() in ocr_text.lower():
                return DocumentType.PAPER
        return DocumentType.BOOK

metadata_agent = Agent(
    'openai:gpt-4',
    deps_type=PDFContext,
    result_type=BookMetadata | PaperMetadata,
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

        # Calculate file hash
        file_hash = processor.calculate_file_hash(pdf_path)
        print(f"File hash: {file_hash}")

        # Check if already processed
        if processor.is_processed(file_hash):
            print(f"File already processed: {pdf_path}")
            return
            
        print(f"\nStep 1: Converting PDF to images")
        images = processor.convert_pdf_to_images(pdf_path)
        if not images:
            raise ValueError(f"No images extracted from PDF: {pdf_path}")
            
        print(f"\nStep 2: Performing OCR")
        ocr_text = processor.perform_ocr(images)
        if not ocr_text.strip():
            raise ValueError(f"No text extracted from images: {pdf_path}")
            
        print(f"\nStep 3: Determining document type")
        doc_type = processor.determine_document_type(ocr_text)
        print(f"Detected document type: {doc_type}")
        
        print(f"\nStep 4: Creating context for PydanticAI")
        context = PDFContext(
            pdf_path=pdf_path,
            ocr_text=ocr_text,
            document_type=doc_type
        )
        
        print(f"\nStep 5: Running metadata extraction")
        result = await metadata_agent.run(
            "Extract all available metadata from this document.",
            deps=context
        )
        
        print(f"\nStep 6: Saving results")
        # Add to metadata store
        processor.metadata_store.documents[file_hash] = result.data
        processor.metadata_store.last_updated = datetime.utcnow()
        processor._save_metadata_store()
        print(f"Updated metadata store: {processor.metadata_file}")
            
    except Exception as e:
        print(f"Error in process_pdf: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Add debug print at module level
print("pdf_metadata_extractor.py loaded") 