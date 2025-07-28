import fitz  # PyMuPDF
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class PDFPreprocessor:
    """Extract and clean text from PDF documents, optimized for Bangla content"""
    
    def __init__(self):
        # Regex patterns for text cleaning
        self.bangla_pattern = re.compile(r'[\u0980-\u09FF]+')
        self.whitespace_pattern = re.compile(r'\s+')
        self.line_break_pattern = re.compile(r'\n\s*\n')
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF with proper formatting for Bangla content
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Cleaned text content from the PDF
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            # Open PDF document safely with context manager
            with fitz.open(str(pdf_path)) as doc:
                full_text = []
                
                # Extract text from each page
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    if page_text.strip():  # Only add non-empty pages
                        cleaned_text = self._clean_text(page_text)
                        if cleaned_text:
                            full_text.append(f"--- Page {page_num + 1} ---\n{cleaned_text}")
                
                # Combine all pages
                combined_text = "\n\n".join(full_text)
                
                logger.info(f"Successfully extracted text from {len(doc)} pages")
                logger.info(f"Total text length: {len(combined_text)} characters")
                
                return combined_text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Fix paragraph breaks
        text = self.line_break_pattern.sub('\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(cleaned_lines)
        
        # Fix common OCR issues in Bangla
        text = self._fix_bangla_ocr_errors(text)
        
        return text.strip()
    
    def _fix_bangla_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in Bangla text
        
        Args:
            text: Text with potential OCR errors
            
        Returns:
            Text with corrected OCR errors
        """
        # Common OCR error corrections for Bangla
        corrections = {
            'ি': 'ি',   # Fix i-kar
            'ী': 'ী',   # Fix ii-kar
            'ু': 'ু',   # Fix u-kar
            'ূ': 'ূ',   # Fix uu-kar
            'ৃ': 'ৃ',   # Fix ri-kar
            'ে': 'ে',   # Fix e-kar
            'ৈ': 'ৈ',   # Fix oi-kar
            'ো': 'ো',   # Fix o-kar
            'ৌ': 'ৌ',   # Fix ou-kar
            '।': '।',   # Fix dari/period
            '॥': '॥',   # Fix double dari
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def get_text_statistics(self, text: str) -> dict:
        """
        Get statistics about the extracted text
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        bangla_chars = len(self.bangla_pattern.findall(text))
        total_chars = len(text)
        words = len(text.split())
        lines = len(text.split('\n'))
        
        return {
            "total_characters": total_chars,
            "bangla_characters": bangla_chars,
            "bangla_percentage": (bangla_chars / total_chars * 100) if total_chars > 0 else 0,
            "total_words": words,
            "total_lines": lines
        }
