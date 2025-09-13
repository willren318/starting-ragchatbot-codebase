#!/usr/bin/env python3
"""
Debug utilities for analyzing RAG system health and identifying query failure points.
"""

import sys
import os
import traceback
import json
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import config
    from rag_system import RAGSystem
    from vector_store import VectorStore
    from search_tools import CourseSearchTool, CourseOutlineTool
    from ai_generator import AIGenerator
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class RAGSystemDebugger:
    """Debug utilities for RAG system analysis"""
    
    def __init__(self):
        self.config = config
        self.issues_found = []
        self.warnings = []
        
    def log_issue(self, component: str, issue: str, severity: str = "ERROR"):
        """Log an issue found during debugging"""
        self.issues_found.append({
            "component": component,
            "issue": issue,
            "severity": severity,
            "timestamp": self._get_timestamp()
        })
        
        severity_emoji = "❌" if severity == "ERROR" else "⚠️"
        print(f"{severity_emoji} [{component}] {issue}")
    
    def log_warning(self, component: str, warning: str):
        """Log a warning found during debugging"""
        self.log_issue(component, warning, "WARNING")
        
    def log_success(self, component: str, message: str):
        """Log a successful check"""
        print(f"✅ [{component}] {message}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def check_environment_variables(self):
        """Check environment variable configuration"""
        print("\n🔍 Checking Environment Variables...")
        print("-" * 40)
        
        # Check API key
        api_key = self.config.ANTHROPIC_API_KEY
        if not api_key:
            self.log_issue("CONFIG", "ANTHROPIC_API_KEY is not set")
        elif len(api_key) < 10:
            self.log_issue("CONFIG", "ANTHROPIC_API_KEY appears to be invalid (too short)")
        elif api_key == "test_key" or api_key.startswith("test"):
            self.log_warning("CONFIG", "ANTHROPIC_API_KEY appears to be a test key")
        else:
            self.log_success("CONFIG", f"ANTHROPIC_API_KEY is set (starts with '{api_key[:8]}...')")
        
        # Check other configuration
        configs_to_check = [
            ("ANTHROPIC_MODEL", self.config.ANTHROPIC_MODEL),
            ("EMBEDDING_MODEL", self.config.EMBEDDING_MODEL),
            ("CHROMA_PATH", self.config.CHROMA_PATH),
            ("CHUNK_SIZE", self.config.CHUNK_SIZE),
            ("MAX_RESULTS", self.config.MAX_RESULTS)
        ]
        
        for config_name, config_value in configs_to_check:
            if config_value:
                self.log_success("CONFIG", f"{config_name} = {config_value}")
            else:
                self.log_issue("CONFIG", f"{config_name} is not set")
    
    def check_database_connectivity(self):
        """Check ChromaDB database connectivity and content"""
        print("\n🔍 Checking Database Connectivity...")
        print("-" * 40)
        
        try:
            # Try to create vector store
            vector_store = VectorStore(
                chroma_path=self.config.CHROMA_PATH,
                embedding_model=self.config.EMBEDDING_MODEL,
                max_results=self.config.MAX_RESULTS
            )
            self.log_success("DATABASE", "Vector store initialized successfully")
            
            # Check if database files exist
            if os.path.exists(self.config.CHROMA_PATH):
                self.log_success("DATABASE", f"Database directory exists: {self.config.CHROMA_PATH}")
                
                # List database files
                db_files = os.listdir(self.config.CHROMA_PATH)
                if db_files:
                    self.log_success("DATABASE", f"Database files found: {db_files}")
                else:
                    self.log_warning("DATABASE", "Database directory is empty")
            else:
                self.log_warning("DATABASE", f"Database directory does not exist: {self.config.CHROMA_PATH}")
            
            # Check course count
            try:
                course_count = vector_store.get_course_count()
                if course_count > 0:
                    self.log_success("DATABASE", f"Found {course_count} courses in database")
                    
                    # List course titles
                    course_titles = vector_store.get_existing_course_titles()
                    print(f"📚 Course titles: {', '.join(course_titles)}")
                    
                    # Get detailed course metadata
                    courses_metadata = vector_store.get_all_courses_metadata()
                    for course_meta in courses_metadata:
                        title = course_meta.get('title', 'Unknown')
                        lesson_count = course_meta.get('lesson_count', 0)
                        instructor = course_meta.get('instructor', 'Unknown')
                        self.log_success("DATABASE", f"Course: {title} ({lesson_count} lessons, instructor: {instructor})")
                        
                else:
                    self.log_issue("DATABASE", "No courses found in database - this will cause query failures")
                    
            except Exception as e:
                self.log_issue("DATABASE", f"Error checking course count: {e}")
                
        except Exception as e:
            self.log_issue("DATABASE", f"Failed to initialize vector store: {e}")
            traceback.print_exc()
    
    def check_search_tools(self):
        """Check search tool functionality"""
        print("\n🔍 Checking Search Tools...")
        print("-" * 40)
        
        try:
            # Create vector store for tools
            vector_store = VectorStore(
                chroma_path=self.config.CHROMA_PATH,
                embedding_model=self.config.EMBEDDING_MODEL,
                max_results=self.config.MAX_RESULTS
            )
            
            # Test CourseSearchTool
            search_tool = CourseSearchTool(vector_store)
            tool_def = search_tool.get_tool_definition()
            
            if tool_def and "name" in tool_def:
                self.log_success("SEARCH_TOOL", f"CourseSearchTool definition valid: {tool_def['name']}")
            else:
                self.log_issue("SEARCH_TOOL", "CourseSearchTool definition is invalid")
            
            # Test tool execution with empty database
            try:
                result = search_tool.execute("test query")
                if "No relevant content found" in result:
                    self.log_success("SEARCH_TOOL", "CourseSearchTool handles empty results correctly")
                elif "error" in result.lower():
                    self.log_issue("SEARCH_TOOL", f"CourseSearchTool returned error: {result}")
                else:
                    self.log_success("SEARCH_TOOL", f"CourseSearchTool returned result: {result[:100]}...")
                    
            except Exception as e:
                self.log_issue("SEARCH_TOOL", f"CourseSearchTool execution failed: {e}")
            
            # Test CourseOutlineTool
            outline_tool = CourseOutlineTool(vector_store)
            outline_def = outline_tool.get_tool_definition()
            
            if outline_def and "name" in outline_def:
                self.log_success("OUTLINE_TOOL", f"CourseOutlineTool definition valid: {outline_def['name']}")
            else:
                self.log_issue("OUTLINE_TOOL", "CourseOutlineTool definition is invalid")
                
            # Test outline tool execution
            try:
                result = outline_tool.execute("test course")
                if "No course found" in result:
                    self.log_success("OUTLINE_TOOL", "CourseOutlineTool handles missing courses correctly")
                else:
                    self.log_success("OUTLINE_TOOL", f"CourseOutlineTool returned result: {result[:100]}...")
                    
            except Exception as e:
                self.log_issue("OUTLINE_TOOL", f"CourseOutlineTool execution failed: {e}")
                
        except Exception as e:
            self.log_issue("SEARCH_TOOLS", f"Failed to initialize search tools: {e}")
    
    def check_ai_generator(self):
        """Check AI generator functionality"""
        print("\n🔍 Checking AI Generator...")
        print("-" * 40)
        
        try:
            # Create AI generator
            ai_generator = AIGenerator(self.config.ANTHROPIC_API_KEY, self.config.ANTHROPIC_MODEL)
            self.log_success("AI_GENERATOR", "AI generator initialized successfully")
            
            # Check system prompt
            if hasattr(ai_generator, 'SYSTEM_PROMPT') and ai_generator.SYSTEM_PROMPT:
                self.log_success("AI_GENERATOR", "System prompt is configured")
                
                # Check for key phrases in system prompt
                prompt = ai_generator.SYSTEM_PROMPT
                if "Tool Usage" in prompt:
                    self.log_success("AI_GENERATOR", "System prompt includes tool usage guidance")
                else:
                    self.log_warning("AI_GENERATOR", "System prompt may be missing tool usage guidance")
                    
                if "Course Outline Tool" in prompt:
                    self.log_success("AI_GENERATOR", "System prompt includes course outline tool guidance")
                else:
                    self.log_warning("AI_GENERATOR", "System prompt may be missing course outline tool guidance")
            else:
                self.log_issue("AI_GENERATOR", "System prompt is not configured")
            
            # Test simple response (without actual API call to avoid costs)
            self.log_success("AI_GENERATOR", "AI generator configuration appears valid")
            
        except Exception as e:
            self.log_issue("AI_GENERATOR", f"Failed to initialize AI generator: {e}")
    
    def check_rag_system_integration(self):
        """Check complete RAG system integration"""
        print("\n🔍 Checking RAG System Integration...")
        print("-" * 40)
        
        try:
            # Initialize RAG system
            rag_system = RAGSystem(self.config)
            self.log_success("RAG_SYSTEM", "RAG system initialized successfully")
            
            # Check tool registration
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            tool_names = [tool["name"] for tool in tool_definitions]
            
            expected_tools = ["search_course_content", "get_course_outline"]
            for tool_name in expected_tools:
                if tool_name in tool_names:
                    self.log_success("RAG_SYSTEM", f"Tool '{tool_name}' is registered")
                else:
                    self.log_issue("RAG_SYSTEM", f"Tool '{tool_name}' is not registered")
            
            # Check analytics
            try:
                analytics = rag_system.get_course_analytics()
                total_courses = analytics.get("total_courses", 0)
                
                if total_courses > 0:
                    self.log_success("RAG_SYSTEM", f"System has {total_courses} courses loaded")
                else:
                    self.log_issue("RAG_SYSTEM", "No courses loaded - queries will fail with 'No relevant content found'")
                    
            except Exception as e:
                self.log_issue("RAG_SYSTEM", f"Failed to get analytics: {e}")
                
        except Exception as e:
            self.log_issue("RAG_SYSTEM", f"Failed to initialize RAG system: {e}")
            traceback.print_exc()
    
    def simulate_query_flow(self, test_query: str = "What is Python programming?"):
        """Simulate a complete query flow to identify failure points"""
        print(f"\n🔍 Simulating Query Flow: '{test_query}'")
        print("-" * 40)
        
        try:
            # Initialize RAG system
            rag_system = RAGSystem(self.config)
            
            # Check if we have any courses loaded
            analytics = rag_system.get_course_analytics()
            if analytics["total_courses"] == 0:
                self.log_warning("QUERY_SIMULATION", "No courses loaded - query will return 'No relevant content found'")
                
                # Try to execute query anyway to see what happens
                try:
                    # Mock the AI generator to avoid API costs
                    with self._mock_ai_generator():
                        response, sources = rag_system.query(test_query)
                        self.log_success("QUERY_SIMULATION", f"Query executed (mocked): '{response[:100]}...'")
                except Exception as e:
                    self.log_issue("QUERY_SIMULATION", f"Query failed even with mocked AI: {e}")
            else:
                self.log_success("QUERY_SIMULATION", f"Ready to process queries with {analytics['total_courses']} courses")
                
                # Note: We don't execute actual queries to avoid API costs
                self.log_success("QUERY_SIMULATION", "Query simulation would proceed normally")
                
        except Exception as e:
            self.log_issue("QUERY_SIMULATION", f"Query simulation failed: {e}")
            traceback.print_exc()
    
    def _mock_ai_generator(self):
        """Context manager to mock AI generator responses"""
        from unittest.mock import patch
        
        def mock_generate_response(*args, **kwargs):
            # Simulate tool usage
            if "tool_manager" in kwargs:
                tool_manager = kwargs["tool_manager"]
                # Try to execute search tool to see if it works
                try:
                    result = tool_manager.execute_tool("search_course_content", query="test")
                    return f"Based on the search results: {result[:50]}..."
                except Exception as e:
                    return f"Tool execution failed: {e}"
            return "Mocked AI response - tools not available"
        
        return patch.object(AIGenerator, 'generate_response', side_effect=mock_generate_response)
    
    def run_comprehensive_check(self):
        """Run all diagnostic checks"""
        print("🔍 RAG System Comprehensive Health Check")
        print("=" * 50)
        
        # Run all checks
        self.check_environment_variables()
        self.check_database_connectivity()
        self.check_search_tools()
        self.check_ai_generator()
        self.check_rag_system_integration()
        self.simulate_query_flow()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 50)
        
        errors = [issue for issue in self.issues_found if issue["severity"] == "ERROR"]
        warnings = [issue for issue in self.issues_found if issue["severity"] == "WARNING"]
        
        if not errors and not warnings:
            print("🎉 All checks passed! RAG system appears to be healthy.")
        else:
            if errors:
                print(f"❌ Found {len(errors)} error(s):")
                for error in errors:
                    print(f"   • [{error['component']}] {error['issue']}")
            
            if warnings:
                print(f"⚠️  Found {len(warnings)} warning(s):")
                for warning in warnings:
                    print(f"   • [{warning['component']}] {warning['issue']}")
        
        # Specific diagnosis for "query failed" issues
        print("\n🎯 QUERY FAILURE ANALYSIS")
        print("-" * 30)
        
        if any(issue["component"] == "DATABASE" and "No courses found" in issue["issue"] for issue in self.issues_found):
            print("💡 ROOT CAUSE IDENTIFIED: Empty database")
            print("   SOLUTION: Load course documents using the document ingestion process")
            print("   STEPS:")
            print("   1. Place course documents in the appropriate directory")
            print("   2. Run the document processing to populate the vector database")
            print("   3. Verify courses are loaded using get_course_analytics()")
        
        elif any(issue["component"] == "CONFIG" and "API_KEY" in issue["issue"] for issue in self.issues_found):
            print("💡 ROOT CAUSE IDENTIFIED: Invalid or missing API key")
            print("   SOLUTION: Set valid ANTHROPIC_API_KEY in environment")
            print("   STEPS:")
            print("   1. Get API key from Anthropic Console")
            print("   2. Set ANTHROPIC_API_KEY in .env file")
            print("   3. Restart the application")
        
        elif any(issue["component"] == "SEARCH_TOOL" and "execution failed" in issue["issue"] for issue in self.issues_found):
            print("💡 ROOT CAUSE IDENTIFIED: Search tool execution failure")
            print("   SOLUTION: Debug search tool implementation")
        
        else:
            print("💡 No clear root cause identified from diagnostics")
            print("   RECOMMENDATION: Run individual component tests for detailed analysis")
        
        return len(errors) == 0


def main():
    """Main diagnostic function"""
    debugger = RAGSystemDebugger()
    success = debugger.run_comprehensive_check()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())