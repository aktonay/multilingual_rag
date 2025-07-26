from openai import OpenAI
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class MultilingualRAGPipeline:
    """
    RAG pipeline using DeepSeek v3 via OpenRouter with direct OpenAI client
    """
    
    def __init__(self, vectorstore, openrouter_config: Dict[str, str]):
        self.vectorstore = vectorstore
        
        # Initialize DeepSeek v3 via OpenRouter using direct OpenAI client
        self.client = OpenAI(
            base_url=openrouter_config["base_url"],
            api_key=openrouter_config["api_key"]
        )
        
        self.model_name = openrouter_config["model"]
        self.site_url = openrouter_config.get("site_url", "http://localhost:8000")
        self.site_name = openrouter_config.get("site_name", "Multilingual RAG System")
        
        # Create system prompt for multilingual support
        self.system_prompt = self._create_system_prompt()
        
        logger.info(f"RAG pipeline initialized with DeepSeek v3 model: {self.model_name}")
    
    def _create_system_prompt(self) -> str:
        """Create a multilingual system prompt optimized for Bangla literature"""
        
        return """You are an expert assistant specializing in Bengali literature, particularly HSC-level Bangla texts. You can understand and respond fluently in both Bengali and English.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the provided context from the document
2. If the answer is not clearly stated in the context, say "আমি এই প্রশ্নের উত্তর প্রদত্ত তথ্যে পাইনি" (for Bangla) or "I couldn't find this answer in the provided information" (for English)
3. Always respond in the same language as the question
4. For Bengali questions, provide answers in proper Bengali script
5. Be precise and cite specific parts of the context when possible
6. For character names, dates, or specific facts, be exact
7. Keep answers concise and directly relevant to the question

Remember: Only use information from the context provided. Do not add external knowledge."""
    
    def query(
        self, 
        question: str, 
        top_k: int = 5, 
        score_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline using DeepSeek v3
        
        Args:
            question: User question in Bangla or English
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score for retrieval
            
        Returns:
            Dictionary containing answer and metadata
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vectorstore.similarity_search(
            query=question,
            k=top_k,
            score_threshold=score_threshold
        )
        
        if not retrieved_docs:
            # No relevant documents found
            is_bangla = self._is_bangla_question(question)
            no_info_message = (
                "দুঃখিত, আমি এই প্রশ্নের উত্তর প্রদত্ত তথ্যে পাইনি।" 
                if is_bangla 
                else "Sorry, I couldn't find relevant information to answer this question."
            )
            
            return {
                "answer": no_info_message,
                "confidence": 0.0,
                "sources": [],
                "retrieved_chunks": 0,
                "similarity_scores": []
            }
        
        # Step 2: Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        # Step 3: Generate answer using DeepSeek v3
        try:
            # Create the user message with context and question
            user_message = f"""Context from HSC Bangla Literature:
{context}

Question: {question}

Please provide a precise answer based only on the context provided above."""

            # Call DeepSeek v3 via OpenRouter using the exact format you provided
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,  # Optional for rankings
                    "X-Title": self.site_name,      # Optional for rankings
                },
                extra_body={},
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": self.system_prompt
                    },
                    {
                        "role": "user", 
                        "content": user_message
                    }
                ]
            )
            
            # Extract the response
            response = completion.choices[0].message.content
            
            # Step 4: Post-process the response
            processed_answer = self._post_process_answer(response)
            
            # Step 5: Calculate confidence based on retrieval scores
            confidence = self._calculate_confidence(retrieved_docs)
            
            result = {
                "answer": processed_answer,
                "confidence": confidence,
                "sources": [doc.metadata for doc, _ in retrieved_docs],
                "retrieved_chunks": len(retrieved_docs),
                "similarity_scores": [float(score) for _, score in retrieved_docs]
            }
            
            logger.info(f"Generated answer with confidence: {confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer with DeepSeek v3: {e}")
            is_bangla = self._is_bangla_question(question)
            error_message = (
                "দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে।" 
                if is_bangla 
                else "Sorry, there was an error generating the answer."
            )
            
            return {
                "answer": error_message,
                "confidence": 0.0,
                "sources": [],
                "retrieved_chunks": 0,
                "similarity_scores": [],
                "error": str(e)
            }
    
    def _prepare_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """
        Prepare context string from retrieved documents
        
        Args:
            retrieved_docs: List of (document, score) tuples
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, (doc, score) in enumerate(retrieved_docs):
            section = doc.metadata.get("section", "অজানা অংশ")
            chunk_id = doc.metadata.get("chunk_id", i+1)
            content = doc.page_content.strip()
            
            context_part = f"[Reference {i+1} - {section} (Chunk {chunk_id}, Score: {score:.3f})]:\n{content}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _post_process_answer(self, answer: str) -> str:
        """
        Post-process the generated answer
        
        Args:
            answer: Raw answer from DeepSeek v3
            
        Returns:
            Cleaned and formatted answer
        """
        if not answer:
            return "No answer generated."
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Ensure proper punctuation for Bangla
        if self._contains_bangla(answer):
            if not answer.endswith(('।', '.', '?', '!')):
                answer += '।'
        
        return answer
    
    def _calculate_confidence(self, retrieved_docs: List[Tuple[Document, float]]) -> float:
        """
        Calculate confidence score based on retrieval quality
        
        Args:
            retrieved_docs: Retrieved documents with scores
            
        Returns:
            Confidence score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0
        
        # Use average of retrieval scores as base confidence
        scores = [score for _, score in retrieved_docs]
        base_confidence = sum(scores) / len(scores)
        
        # Boost confidence if we have multiple high-scoring documents
        high_quality_docs = sum(1 for score in scores if score > 0.8)
        if high_quality_docs >= 2:
            base_confidence = min(base_confidence * 1.1, 1.0)
        
        return round(base_confidence, 3)
    
    def _is_bangla_question(self, text: str) -> bool:
        """
        Detect if the question is primarily in Bangla
        
        Args:
            text: Input text
            
        Returns:
            True if text is primarily Bangla
        """
        bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        total_chars = len(re.findall(r'[\u0980-\u09FF\w]', text))
        
        return bangla_chars / max(total_chars, 1) > 0.3
    
    def _contains_bangla(self, text: str) -> bool:
        """Check if text contains Bangla characters"""
        return bool(re.search(r'[\u0980-\u09FF]', text))
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to DeepSeek v3 via OpenRouter
        
        Returns:
            Dictionary with connection test results
        """
        try:
            # Test with a simple question
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                extra_body={},
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, can you understand Bengali?"
                    }
                ]
            )
            
            response = completion.choices[0].message.content
            
            return {
                "status": "success",
                "model": self.model_name,
                "response": response,
                "message": "DeepSeek v3 connection successful"
            }
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return {
                "status": "failed",
                "model": self.model_name,
                "error": str(e),
                "message": "DeepSeek v3 connection failed"
            }