from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

class IntelligentChunker:
    """
    Advanced text chunking optimized for multilingual content (Bangla + English)
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Define separators prioritized for Bangla and English
        self.separators = [
            "\n\n",      # Paragraph breaks (highest priority)
            "\n",        # Line breaks
            "।",         # Bangla sentence end (dari)
            "।।",        # Double dari
            ".",         # English period
            "?",         # Question mark
            "!",         # Exclamation
            ";",         # Semicolon
            ",",         # Comma
            " ",         # Space
            ""           # Character level (last resort)
        ]
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            keep_separator=True
        )
    
    def chunk_document(self, text: str) -> List[Document]:
        """
        Split document into intelligent chunks with metadata
        
        Args:
            text: Full document text
            
        Returns:
            List of Document objects with chunks and metadata
        """
        logger.info(f"Starting chunking process for text of length {len(text)}")
        
        # Pre-process text to identify sections
        sections = self._identify_sections(text)
        
        all_chunks = []
        chunk_id = 0
        
        for section_name, section_text in sections:
            logger.info(f"Processing section: {section_name}")
            
            # Create chunks for this section
            section_chunks = self.text_splitter.create_documents([section_text])
            
            # Add metadata to each chunk
            for i, chunk in enumerate(section_chunks):
                chunk_id += 1
                
                # Enhance metadata
                chunk.metadata = {
                    "chunk_id": chunk_id,
                    "section": section_name,
                    "section_chunk_index": i,
                    "total_section_chunks": len(section_chunks),
                    "chunk_length": len(chunk.page_content),
                    "language_profile": self._detect_language_profile(chunk.page_content)
                }
                
                all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks
    
    def _identify_sections(self, text: str) -> List[tuple]:
        """
        Identify major sections in the document
        
        Args:
            text: Full document text
            
        Returns:
            List of (section_name, section_text) tuples
        """
        # Patterns to identify sections/chapters
        patterns = [
            r'(অধ্যায়|Chapter|পাঠ|Lesson)\s*[০-৯\d]+',  # Chapter/Lesson numbers
            r'(প্রথম|দ্বিতীয়|তৃতীয়|চতুর্থ|পঞ্চম)\s*(অধ্যায়|পাঠ)',  # Ordinal chapters
            r'(ক|খ|গ|ঘ|ঙ)\s*\.',  # Section markers
            r'--- Page \d+ ---'  # Page markers
        ]
        
        sections = []
        lines = text.split('\n')
        current_section = "ভূমিকা (Introduction)"
        current_content = []
        
        section_pattern = '|'.join(f'({pattern})' for pattern in patterns)
        compiled_pattern = re.compile(section_pattern, re.IGNORECASE)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section header
            if compiled_pattern.search(line):
                # Save previous section if it has content
                if current_content:
                    section_text = '\n'.join(current_content)
                    if len(section_text.strip()) > 50:  # Only keep substantial sections
                        sections.append((current_section, section_text))
                
                # Start new section
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            section_text = '\n'.join(current_content)
            if len(section_text.strip()) > 50:
                sections.append((current_section, section_text))
        
        # If no sections were found, treat entire text as one section
        if not sections:
            sections = [("Main Content", text)]
        
        return sections
    
    def _detect_language_profile(self, text: str) -> Dict[str, float]:
        """
        Detect language distribution in a text chunk
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language percentages
        """
        bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        numbers = len(re.findall(r'[০-৯0-9]', text))
        
        total_chars = bangla_chars + english_chars + numbers
        
        if total_chars == 0:
            return {"bangla": 0, "english": 0, "numbers": 0}
        
        return {
            "bangla": round(bangla_chars / total_chars, 3),
            "english": round(english_chars / total_chars, 3),
            "numbers": round(numbers / total_chars, 3)
        }
    
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict:
        """
        Get statistics about the chunking results
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        sections = set(chunk.metadata.get("section", "Unknown") for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "unique_sections": len(sections),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_text_length": sum(chunk_lengths)
        }