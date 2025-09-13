"""
Shared test fixtures and configuration for the RAG system tests.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temporary paths."""
    config = Config()
    config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma")
    config.ANTHROPIC_API_KEY = "test_key_12345"
    config.MAX_RESULTS = 3
    config.CHUNK_SIZE = 500
    config.CHUNK_OVERLAP = 50
    return config


@pytest.fixture
def mock_ai_generator():
    """Create a mocked AI generator."""
    mock_ai = Mock()
    mock_ai.generate_response.return_value = "Mocked AI response"
    return mock_ai


@pytest.fixture
def sample_course():
    """Create sample course data for testing."""
    return Course(
        title="Python Basics",
        instructor="John Doe",
        course_link="https://example.com/python",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Introduction",
                lesson_link="https://example.com/lesson1"
            ),
            Lesson(
                lesson_number=2,
                title="Variables",
                lesson_link="https://example.com/lesson2"
            )
        ]
    )


@pytest.fixture
def sample_chunks():
    """Create sample course chunks for testing."""
    return [
        CourseChunk(
            content="Python is a high-level programming language",
            course_title="Python Basics",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables are used to store data values in Python",
            course_title="Python Basics",
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Python supports multiple data types like strings, integers, and floats",
            course_title="Python Basics",
            lesson_number=2,
            chunk_index=2
        )
    ]


@pytest.fixture
def rag_system_with_mock_ai(test_config, mock_ai_generator):
    """Create a RAG system with mocked AI generator."""
    with patch('rag_system.AIGenerator') as mock_ai_class:
        mock_ai_class.return_value = mock_ai_generator
        rag_system = RAGSystem(test_config)
        return rag_system, mock_ai_generator


@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounting issues."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    import logging
    import traceback

    # Create app without static files that cause test issues
    app = FastAPI(title="Test Course Materials RAG System")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Pydantic models
    class Source(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Create a mock RAG system for testing
    mock_rag_system = Mock()
    mock_rag_system.query.return_value = ("Test response", [{"text": "Test source", "link": None}])
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "Advanced Python"]
    }
    mock_rag_system.session_manager.create_session.return_value = "test_session_123"

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)

            formatted_sources = []
            for source in sources:
                if isinstance(source, dict):
                    formatted_sources.append(Source(text=source["text"], link=source.get("link")))
                else:
                    formatted_sources.append(Source(text=source, link=None))

            return QueryResponse(
                answer=answer,
                sources=formatted_sources,
                session_id=session_id
            )
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            logging.error(f"Error getting course stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}

    # Attach mock for test access
    app.mock_rag_system = mock_rag_system

    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for API testing."""
    return TestClient(test_app)


@pytest.fixture
def valid_query_request():
    """Create a valid query request for testing."""
    return {
        "query": "What is Python?",
        "session_id": "test_session_123"
    }


@pytest.fixture
def invalid_query_request():
    """Create an invalid query request for testing."""
    return {
        "query": "",  # Empty query should be invalid
        "session_id": "test_session_123"
    }


@pytest.fixture
def mock_course_analytics():
    """Mock course analytics data."""
    return {
        "total_courses": 3,
        "course_titles": ["Python Basics", "Advanced Python", "Data Science with Python"]
    }


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress warnings during tests."""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# Session-scoped fixtures for expensive operations
@pytest.fixture(scope="session")
def test_embeddings():
    """Mock embeddings for testing vector operations."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Mock embedding for chunk 1
        [0.2, 0.3, 0.4, 0.5, 0.6],  # Mock embedding for chunk 2
        [0.3, 0.4, 0.5, 0.6, 0.7],  # Mock embedding for chunk 3
    ]