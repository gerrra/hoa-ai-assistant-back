#!/usr/bin/env python3
"""
Logging middleware for HOA AI Assistant
"""

import logging
import json
import time
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

class RequestLogger:
    """Logger for API requests and responses"""
    
    @staticmethod
    def log_request(request_data: Dict[str, Any]) -> str:
        """Log incoming request and return request ID"""
        request_id = f"req_{int(time.time() * 1000)}"
        logger.info(f"[{request_id}] Incoming request: {json.dumps(request_data, ensure_ascii=False)}")
        return request_id
    
    @staticmethod
    def log_database_connection(request_id: str, success: bool, error: str = None):
        """Log database connection status"""
        if success:
            logger.info(f"[{request_id}] Database connection: SUCCESS")
        else:
            logger.error(f"[{request_id}] Database connection: FAILED - {error}")
    
    @staticmethod
    def log_document_search(request_id: str, community_id: int, query: str, results_count: int, search_type: str = "chunks"):
        """Log document search results"""
        logger.info(f"[{request_id}] Document search ({search_type}): community_id={community_id}, query='{query}', results={results_count}")
    
    @staticmethod
    def log_openai_request(request_id: str, model: str, prompt_length: int, max_tokens: int):
        """Log OpenAI API request"""
        logger.info(f"[{request_id}] OpenAI request: model={model}, prompt_length={prompt_length}, max_tokens={max_tokens}")
    
    @staticmethod
    def log_openai_response(request_id: str, success: bool, response_length: int = 0, error: str = None):
        """Log OpenAI API response"""
        if success:
            logger.info(f"[{request_id}] OpenAI response: SUCCESS, length={response_length}")
        else:
            logger.error(f"[{request_id}] OpenAI response: FAILED - {error}")
    
    @staticmethod
    def log_final_response(request_id: str, answer_length: int, sources_count: int, confidence: float):
        """Log final response"""
        logger.info(f"[{request_id}] Final response: answer_length={answer_length}, sources={sources_count}, confidence={confidence:.3f}")
    
    @staticmethod
    def log_error(request_id: str, stage: str, error: str, details: Dict[str, Any] = None):
        """Log error with context"""
        error_data = {
            "stage": stage,
            "error": str(error),
            "details": details or {}
        }
        logger.error(f"[{request_id}] ERROR in {stage}: {json.dumps(error_data, ensure_ascii=False)}")
    
    @staticmethod
    def log_validation(request_id: str, field: str, value: Any, valid: bool, message: str = ""):
        """Log validation results"""
        status = "VALID" if valid else "INVALID"
        logger.info(f"[{request_id}] Validation {field}: {status} - {message}")

def log_function_call(func_name: str, request_id: str, **kwargs):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            logger.info(f"[{request_id}] Calling {func_name} with args: {kwargs}")
            try:
                result = func(*args, **func_kwargs)
                logger.info(f"[{request_id}] {func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"[{request_id}] {func_name} failed: {str(e)}")
                raise
        return wrapper
    return decorator
