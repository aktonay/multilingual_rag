from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """
    MongoDB-based chat history management for the RAG system
    """
    
    def __init__(self, mongo_uri: str, database_name: str, collection_name: str = "chat_history"):
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        
        # Initialize MongoDB connection
        try:
            self.client = MongoClient(
                mongo_uri, 
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000
            )
            
            # Test connection
            self.client.server_info()
            
            self.db = self.client[database_name]
            self.collection: Collection = self.db[collection_name]
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {database_name}.{collection_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"MongoDB connection failed: {e}")
            logger.warning("Chat history will not be saved")
            self.client = None
            self.collection = None
    
    def _create_indexes(self):
        """Create database indexes for optimal performance"""
        if self.collection is None:
            return
        
        try:
            # Compound index for session_id and timestamp
            self.collection.create_index([("session_id", 1), ("timestamp", -1)])
            
            # Single field indexes
            self.collection.create_index([("timestamp", -1)])
            self.collection.create_index([("language", 1)])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    def save_interaction(
        self,
        session_id: str,
        question: str,
        answer: str,
        confidence: float,
        sources: List[Dict] = None,
        metadata: Dict = None
    ) -> Optional[str]:
        """
        Save a chat interaction to MongoDB
        
        Args:
            session_id: Unique session identifier
            question: User question
            answer: System answer
            confidence: Confidence score
            sources: Retrieved source documents
            metadata: Additional metadata
            
        Returns:
            Document ID if saved successfully, None otherwise
        """
        if self.collection is None:
            logger.warning("MongoDB not connected. Interaction not saved.")
            return None
        
        interaction = {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": sources or [],
            "metadata": metadata or {},
            "timestamp": datetime.utcnow(),
            "language": self._detect_language(question)
        }
        
        try:
            result = self.collection.insert_one(interaction)
            logger.info(f"Saved interaction for session {session_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving interaction: {e}")
            return None
    
    def get_session_history(
        self, 
        session_id: str, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a specific session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of chat interactions
        """
        if self.collection is None:
            return []
        
        try:
            cursor = self.collection.find(
                {"session_id": session_id}
            ).sort("timestamp", 1).limit(limit)  # Ascending order (chronological)
            
            history = []
            for doc in cursor:
                # Convert ObjectId to string and format timestamp
                doc["_id"] = str(doc["_id"])
                doc["timestamp"] = doc["timestamp"].isoformat()
                history.append(doc)
            
            logger.info(f"Retrieved {len(history)} interactions for session {session_id}")
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving session history: {e}")
            return []
    
    def get_recent_interactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent interactions across all sessions
        
        Args:
            limit: Maximum number of interactions
            
        Returns:
            List of recent interactions
        """
        if self.collection is None:
            return []
        
        try:
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)
            
            interactions = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                doc["timestamp"] = doc["timestamp"].isoformat()
                interactions.append(doc)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error retrieving recent interactions: {e}")
            return []
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        if self.collection is None:
            return {"error": "Database not connected"}
        
        try:
            pipeline = [
                {"$match": {"session_id": session_id}},
                {
                    "$group": {
                        "_id": "$session_id",
                        "total_interactions": {"$sum": 1},
                        "avg_confidence": {"$avg": "$confidence"},
                        "languages_used": {"$addToSet": "$language"},
                        "first_interaction": {"$min": "$timestamp"},
                        "last_interaction": {"$max": "$timestamp"}
                    }
                }
            ]
            
            result = list(self.collection.aggregate(pipeline))
            
            if result:
                stats = result[0]
                stats["first_interaction"] = stats["first_interaction"].isoformat()
                stats["last_interaction"] = stats["last_interaction"].isoformat()
                return stats
            else:
                return {"error": "Session not found"}
                
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return {"error": str(e)}
    
    def cleanup_old_history(self, days_old: int = 30) -> int:
        """
        Clean up old chat history
        
        Args:
            days_old: Number of days after which to delete history
            
        Returns:
            Number of deleted documents
        """
        if self.collection is None:
            return 0
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            result = self.collection.delete_many({"timestamp": {"$lt": cutoff_date}})
            
            deleted_count = result.deleted_count
            logger.info(f"Cleaned up {deleted_count} old interactions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up history: {e}")
            return 0
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code ('bn' for Bangla, 'en' for English)
        """
        import re
        
        bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = bangla_chars + english_chars
        
        if total_chars == 0:
            return "unknown"
        
        return "bn" if bangla_chars > english_chars else "en"
