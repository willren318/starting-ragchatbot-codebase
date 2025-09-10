# RAG System Query Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant Session as Session Manager<br/>(session_manager.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant Claude as Anthropic<br/>Claude API
    participant Tools as Tool Manager<br/>(search_tools.py)
    participant Vector as Vector Store<br/>(vector_store.py)
    participant ChromaDB as ChromaDB<br/>Database

    %% User initiates query
    User->>Frontend: Types query & clicks send
    Frontend->>Frontend: addMessage(query, 'user')
    Frontend->>Frontend: Show loading animation
    
    %% API call
    Frontend->>+API: POST /api/query<br/>{query, session_id}
    
    %% RAG orchestration
    API->>+RAG: query(query, session_id)
    RAG->>Session: get_conversation_history(session_id)
    Session-->>RAG: Previous Q&A context
    
    %% AI generation with tools
    RAG->>+AI: generate_response(query, history, tools, tool_manager)
    AI->>AI: Build system prompt + context
    
    %% First Claude call (with tools)
    AI->>+Claude: messages.create()<br/>system: prompt + history<br/>tools: [search_course_content]<br/>tool_choice: auto
    Claude-->>-AI: Response (may include tool_use)
    
    %% Tool execution (if Claude decides to search)
    alt Claude wants to search
        AI->>+Tools: execute_tool("search_course_content", query, course_name, lesson_number)
        Tools->>+Vector: search(query, course_name, lesson_number)
        Vector->>Vector: Convert query to embeddings
        Vector->>+ChromaDB: similarity_search_with_score()
        ChromaDB-->>-Vector: Relevant chunks + metadata
        Vector->>Vector: Apply filters (course/lesson)
        Vector-->>-Tools: SearchResults(documents, metadata)
        Tools->>Tools: Format results with context headers
        Tools->>Tools: Track sources for UI
        Tools-->>-AI: Formatted search results
        
        %% Second Claude call (with search results)
        AI->>+Claude: messages.create()<br/>Previous conversation + tool results
        Claude-->>-AI: Final contextual answer
    else Claude has enough context
        AI->>AI: Use existing knowledge
    end
    
    AI-->>-RAG: Generated response
    
    %% Extract sources and update session
    RAG->>Tools: get_last_sources()
    Tools-->>RAG: Sources list
    RAG->>Session: add_exchange(session_id, query, response)
    RAG-->>-API: (response, sources)
    
    %% Return to frontend
    API-->>-Frontend: JSON Response<br/>{answer, sources, session_id}
    
    %% UI updates
    Frontend->>Frontend: Remove loading animation
    Frontend->>Frontend: addMessage(answer, 'assistant', sources)
    Frontend->>Frontend: Update session_id if new
    Frontend->>Frontend: Re-enable input
    Frontend-->>User: Display answer with sources

    %% Background: Reset sources
    Note over RAG, Tools: tool_manager.reset_sources()
```

## Flow Components

### 🎨 **Frontend Layer**
- **User Interface**: HTML form with chat messages
- **JavaScript**: Event handling, API calls, UI updates
- **Session Tracking**: Maintains `currentSessionId` for context

### 🚀 **API Layer** 
- **FastAPI**: REST endpoint `/api/query`
- **Request Validation**: Pydantic models
- **Response Formatting**: JSON with answer + sources

### 🧠 **RAG Orchestration**
- **Query Processing**: Combines user input with conversation history
- **Component Coordination**: Manages AI, tools, and session state
- **Response Assembly**: Merges AI output with source attribution

### 🤖 **AI Generation**
- **System Prompt**: Instructions for educational assistance
- **Tool Integration**: Provides Claude with search capabilities  
- **Context Management**: Includes conversation history
- **Two-Phase Process**: Initial reasoning → Tool execution → Final answer

### 🔍 **Search Tools**
- **Semantic Search**: Vector similarity matching
- **Smart Filtering**: Course name and lesson number filters
- **Source Tracking**: Maintains attribution for UI
- **Result Formatting**: Contextual headers with course/lesson info

### 💾 **Data Layer**
- **Vector Store**: Sentence transformer embeddings
- **ChromaDB**: Vector database with metadata
- **Session Storage**: Conversation history persistence
- **Document Chunks**: Pre-processed course content

### 🔄 **Key Features**
- **Intelligent Tool Use**: Claude decides when to search
- **Conversation Context**: Multi-turn dialogue support  
- **Source Attribution**: Transparent content sourcing
- **Real-time UI**: Loading states and progressive enhancement