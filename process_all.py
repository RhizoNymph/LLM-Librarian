import asyncio
import os
from pathlib import Path
from pdf_metadata_extractor import process_pdf, PDFProcessor
from rich.console import Console
from rich.table import Table
from opensearchpy import OpenSearch, OpenSearchException
from datetime import datetime
from typing import Optional, List, Union
from dotenv import load_dotenv
import hashlib
import base64
import aiohttp
import fitz  

load_dotenv()


OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT', '9200'))
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER', 'admin')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', 'admin')
OPENSEARCH_USE_SSL = os.getenv('OPENSEARCH_USE_SSL', 'true').lower() == 'true'
OPENSEARCH_VERIFY_CERTS = os.getenv('OPENSEARCH_VERIFY_CERTS', 'false').lower() == 'true'
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'pdf_documents')

class OpenSearchUploader:
    def __init__(self):
        self.client = None
        self.console = Console()
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.index_prefix = OPENSEARCH_INDEX
                
        self.mappings = {
            'book': {
                "properties": {
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "publisher": {"type": "keyword"},
                    "publication_year": {"type": "integer"},
                    "isbn": {"type": "keyword"},
                    "edition": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "subject_areas": {"type": "keyword"},
                    "table_of_contents": {"type": "text"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'paper': {
                "properties": {
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "abstract": {"type": "text"},
                    "keywords": {"type": "keyword"},
                    "doi": {"type": "keyword"},
                    "journal": {"type": "keyword"},
                    "conference": {"type": "keyword"},
                    "publication_year": {"type": "integer"},
                    "institution": {"type": "keyword"},
                    "citations": {"type": "text"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'blog_article': {
                "properties": {
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "publication_date": {"type": "date"},
                    "blog_name": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "reading_time": {"type": "integer"},
                    "summary": {"type": "text"},
                    "series": {"type": "keyword"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'technical_report': {
                "properties": {
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "organization": {"type": "keyword"},
                    "report_number": {"type": "keyword"},
                    "date": {"type": "date"},
                    "executive_summary": {"type": "text"},
                    "keywords": {"type": "keyword"},
                    "classification": {"type": "keyword"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'thesis': {
                "properties": {
                    "title": {"type": "text"},
                    "author": {"type": "keyword"},
                    "degree": {"type": "keyword"},
                    "institution": {"type": "keyword"},
                    "department": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "advisors": {"type": "keyword"},
                    "abstract": {"type": "text"},
                    "keywords": {"type": "keyword"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'patent': {
                "properties": {
                    "title": {"type": "text"},
                    "inventors": {"type": "keyword"},
                    "assignee": {"type": "keyword"},
                    "patent_number": {"type": "keyword"},
                    "filing_date": {"type": "date"},
                    "publication_date": {"type": "date"},
                    "abstract": {"type": "text"},
                    "classification": {"type": "keyword"},
                    "claims": {"type": "text"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            }
        }

    def connect(self) -> bool:
        """Establish connection to OpenSearch"""
        try:
            print(f"Connecting to OpenSearch on {OPENSEARCH_HOST}")
            self.client = OpenSearch(
                hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
                http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
                use_ssl=OPENSEARCH_USE_SSL,
                verify_certs=OPENSEARCH_VERIFY_CERTS,
                ssl_show_warn=False
            )
            
            self.client.info()
            return True
        except Exception as e:
            self.console.print(f"[red]Failed to connect to OpenSearch: {str(e)}[/red]")
            return False

    def get_index_name(self, doc_type: str) -> str:
        """Get the index name for a document type"""
        return f"{self.index_prefix}_{doc_type.lower()}"

    def create_index_if_not_exists(self, doc_type: str):
        """Create index with appropriate mappings for document type"""
        index_name = self.get_index_name(doc_type)
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(
                index=index_name,
                body={"mappings": self.mappings[doc_type.lower()]}
            )

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):            
            end = start + self.chunk_size
                        
            if end < len(text):                
                last_period = text[end-100:end].rfind('.')
                if last_period != -1:
                    end = end - (100 - last_period)
                else:                    
                    last_space = text[end-50:end].rfind(' ')
                    if last_space != -1:
                        end = end - (50 - last_space)
                        
            chunks.append(text[start:end].strip())
                        
            start = end - self.chunk_overlap
        
        return chunks

    async def upload_document(self, metadata, file_hash: str, ocr_text: str) -> bool:
        """Upload metadata and chunked text content to OpenSearch"""
        if not self.client:
            if not self.connect():
                return False

        try:            
            doc_type = metadata.__class__.__name__.lower().replace('metadata', '')
                        
            self.create_index_if_not_exists(doc_type)
                        
            chunks = self.chunk_text(ocr_text)
            total_chunks = len(chunks)
            
            base_document = metadata.model_dump(exclude_none=True)
            base_document.update({
                'file_hash': file_hash,
                'timestamp': datetime.now().isoformat(),
            })
            
            index_name = self.get_index_name(doc_type)
            for i, chunk in enumerate(chunks):                
                document = base_document.copy()
                document.update({
                    'content': chunk,
                    'chunk_index': i,
                    'total_chunks': total_chunks
                })

                response = self.client.index(
                    index=index_name,
                    body=document,
                    id=f"{file_hash}_{i}",
                    refresh=True
                )

            self.console.print(f"[green]Document indexed successfully in {total_chunks} chunks to {index_name}[/green]")
            return True
        except OpenSearchException as e:
            self.console.print(f"[red]OpenSearch indexing error: {str(e)}[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]Unexpected error during indexing: {str(e)}[/red]")
            return False

class DwarfUploader:
    """Uploader class for the Dwarf In The Flask server"""
    def __init__(self):
        self.host = os.getenv('DWARFINTHEFLASK_HOST', 'http://localhost:5000')
        self.console = Console()
        
        self.timeout = aiohttp.ClientTimeout(total=1800)

    async def upload_document(self, pdf_path: Path) -> bool:
        """Upload a PDF document to the Flask server"""
        try:
            
            with open(pdf_path, 'rb') as f:
                pdf_content = base64.b64encode(f.read()).decode('utf-8')

            
            data = {
                'pdf_content': pdf_content,
                'filename': pdf_path.name,
                'index_name': 'universal'  
            }

            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(f"{self.host}/indexPDF", json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.console.print(f"[green]Successfully uploaded {pdf_path.name} to DwarfInTheFlask[/green]")
                        return True
                    else:
                        error_text = await response.text()
                        self.console.print(f"[red]Failed to upload {pdf_path.name} to DwarfInTheFlask: {error_text}[/red]")
                        return False

        except Exception as e:
            self.console.print(f"[red]Error uploading {pdf_path.name} to DwarfInTheFlask: {str(e)}[/red]")
            return False

async def process_single_pdf(pdf_path: Path, uploaders: List[Union[OpenSearchUploader, DwarfUploader]]) -> bool:
    """Process a single PDF file and upload its metadata to all configured uploaders"""
    successes = []
    
    for uploader in uploaders:
        try:
            if isinstance(uploader, DwarfUploader):
                success = await uploader.upload_document(pdf_path)
            else:
                
                with open(pdf_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                
                metadata = await process_pdf(pdf_path)
                if metadata is None:
                    processor = PDFProcessor()
                    metadata = processor.metadata_store.get_by_hash(file_hash)
                    if metadata is None:
                        print(f"[red]Could not get metadata for {pdf_path}[/red]")
                        success = False
                        continue
                
                
                print("\nExtracting full document text...")
                full_text = ""
                with fitz.open(pdf_path) as doc:
                    total_pages = doc.page_count
                    for page_num in range(total_pages):
                        page = doc[page_num]                        
                        full_text += page.get_text()
                    print(f"processed {total_pages} pages")
                
                print(f"Extracted {len(full_text)} characters of text")
                
                success = await uploader.upload_document(metadata, file_hash, full_text)
            
            successes.append(success)
            
        except Exception as e:
            print(f"[red]Failed to process {pdf_path}: {str(e)}[/red]")
            import traceback
            traceback.print_exc()
            successes.append(False)
        
    return all(successes)

async def main():
    console = Console()
    console.print("[blue]Script started![/blue]")
    
    pdf_dir = Path("./pdfs")
    console.print(f"Looking for PDFs in: {pdf_dir.absolute()}")
    
    if not pdf_dir.exists():
        console.print(f"[yellow]Creating directory: {pdf_dir}[/yellow]")
        pdf_dir.mkdir(parents=True)
    
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    if not pdfs:
        console.print(f"[yellow]No PDF files found in {pdf_dir}[/yellow]")
        return
        
    console.print(f"[green]Found {len(pdfs)} PDF files[/green]")
        
    mode = os.getenv('INDEX_MODE', 'both').lower()
    uploaders = []
    
    if mode in ['opensearch', 'both']:
        opensearch = OpenSearchUploader()
        if not opensearch.connect():
            console.print("[red]Failed to establish OpenSearch connection.[/red]")
            if mode == 'opensearch':
                return
        else:
            uploaders.append(opensearch)
            console.print("[blue]Using OpenSearch uploader[/blue]")
    
    if mode in ['dwarf', 'both']:
        dwarf = DwarfUploader()
        uploaders.append(dwarf)
        console.print("[blue]Using DwarfInTheFlask uploader[/blue]")
    
    if not uploaders:
        console.print("[red]No valid uploaders configured. Please set INDEX_MODE to 'opensearch', 'dwarf', or 'both'[/red]")
        return

    results = []
    for pdf_path in pdfs:
        console.print(f"\n[blue]Processing {pdf_path}[/blue]")
        success = await process_single_pdf(pdf_path, uploaders)
        results.append((pdf_path, success))

    console.print("\n[bold]Processing Summary:[/bold]")
    for path, success in results:
        status = "[green]Success[/green]" if success else "[red]Failed[/red]"
        console.print(f"{path}: {status}")
    
    if any(isinstance(u, OpenSearchUploader) for u in uploaders):
        show_summary()

def show_summary():
    """Show summary of processed documents"""
    processor = PDFProcessor()
    console = Console()
    
    table = Table(title="PDF Metadata Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Authors", style="yellow")
    table.add_column("Year", style="blue")
    table.add_column("Hash", style="magenta")

    for file_hash, metadata in processor.metadata_store.documents.items():
        doc_type = "Book" if hasattr(metadata, "isbn") else "Paper"
        table.add_row(
            doc_type,
            metadata.title[:50] + "..." if len(metadata.title) > 50 else metadata.title,
            ", ".join(metadata.authors[:2]) + ("..." if len(metadata.authors) > 2 else ""),
            str(metadata.publication_year or "N/A"),
            file_hash[:8] + "..."
        )

    console.print(table)

if __name__ == "__main__":
    print("Starting main...")  
    asyncio.run(main()) 