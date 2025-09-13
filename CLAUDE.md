# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start using the provided script with cc_venv
source cc_venv/bin/activate
chmod +x run.sh
./run.sh

# Manual start from backend directory
source cc_venv/bin/activate
cd backend
python -m uvicorn app:app --reload --port 8000
```

### Environment Setup
```bash
# Activate the virtual environment
source cc_venv/bin/activate

# Install dependencies (if needed)
pip install chromadb==1.0.15 anthropic==0.58.2 sentence-transformers==5.0.0 fastapi==0.116.1 uvicorn==0.35.0 python-multipart==0.0.20 python-dotenv==1.1.1

# Set up environment variables
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

### Database Management
The system uses ChromaDB for vector storage. Database files are stored in `./backend/chroma_db/` and are created automatically on first run. To reset the knowledge base, delete this directory.

### Dependency Management
**IMPORTANT**: Before adding any new dependencies to this project, you must install them in the `cc_venv` virtual environment:

```bash
# Activate the virtual environment first
source cc_venv/bin/activate

# Then install the new dependency
pip install <new-dependency>

# Update pyproject.toml if needed
# Test that the application still works
```

This ensures compatibility and prevents import errors during development and testing.

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials. The architecture follows a modular design with clear separation of concerns:

### Core RAG Pipeline
- **Document Processing**: Structured parsing of course transcripts with lesson segmentation
- **Vector Storage**: ChromaDB with sentence-transformer embeddings for semantic search  
- **AI Generation**: Anthropic Claude with intelligent tool usage
- **Session Management**: Multi-turn conversation context preservation

### Key Components

**`rag_system.py`** - Main orchestrator that coordinates all system components. Handles document ingestion, query processing, and response assembly.

**`document_processor.py`** - Parses structured course documents (title/instructor metadata + lesson markers) into semantically coherent chunks with configurable overlap.

**`ai_generator.py`** - Manages Claude API interactions with a two-phase approach:
1. Initial call with available tools 
2. Tool execution + follow-up call for final response

**`search_tools.py`** - Implements the tool interface for Claude, providing semantic search with course/lesson filtering and source tracking.

**`vector_store.py`** - ChromaDB operations including embedding generation, similarity search, and metadata filtering.

**`session_manager.py`** - Conversation history management with configurable context window.

### Document Structure Expected
Course files should follow this format:
```
Course Title: [title]
Course Link: [optional url]  
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [optional url]
[lesson content]

Lesson 1: Next Topic
[lesson content]
```

### Configuration
All settings are centralized in `backend/config.py`:
- **Chunk size/overlap**: Controls document segmentation (800/100 chars)
- **Search results**: Maximum retrieved chunks (5)
- **Conversation history**: Context window (2 exchanges)
- **Models**: Claude Sonnet 4, sentence-transformers all-MiniLM-L6-v2

### Tool-Based Search Architecture
The system uses Anthropic's tool calling feature where Claude autonomously decides when to search course content versus using general knowledge. This prevents unnecessary searches for general questions while ensuring course-specific queries get accurate, sourced responses.

### Frontend Integration
The frontend (`frontend/`) is a pure HTML/CSS/JavaScript interface that communicates with the FastAPI backend via REST endpoints (`/api/query`, `/api/courses`). Session management is handled client-side with server-side conversation persistence.