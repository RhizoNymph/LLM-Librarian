import asyncio
from pathlib import Path
from pdf_metadata_extractor import process_pdf, PDFProcessor
from rich.console import Console
from rich.table import Table

async def main():
    print("Script started!")  # Debug line
    pdf_dir = Path("./pdfs")
    print(f"Looking for PDFs in: {pdf_dir.absolute()}")  # Debug line
    
    # Check if directory exists
    if not pdf_dir.exists():
        print(f"Creating directory: {pdf_dir}")
        pdf_dir.mkdir(parents=True)
    
    # Get list of PDFs
    pdfs = list(pdf_dir.glob("*.pdf"))
    print(f"PDF files found: {[str(p) for p in pdfs]}")  # Debug line
    
    if not pdfs:
        print(f"No PDF files found in {pdf_dir}")
        return
        
    print(f"Found {len(pdfs)} PDF files")
    
    # Process each PDF
    for pdf_path in pdfs:
        print(f"\nProcessing {pdf_path}")
        try:
            await process_pdf(pdf_path)
            print(f"Successfully processed {pdf_path}")
        except Exception as e:
            print(f"Failed to process {pdf_path}: {str(e)}")
            import traceback  # Debug line
            traceback.print_exc()  # Debug line

    # Show summary
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
    print("Starting main...")  # Debug line
    asyncio.run(main()) 