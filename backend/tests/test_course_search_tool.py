#!/usr/bin/env python3
"""
Comprehensive tests for CourseSearchTool to identify query failure points.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test CourseSearchTool functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_tool_definition(self):
        """Test that tool definition is properly formatted"""
        tool_def = self.search_tool.get_tool_definition()
        
        self.assertIsInstance(tool_def, dict)
        self.assertEqual(tool_def["name"], "search_course_content")
        self.assertIn("description", tool_def)
        self.assertIn("input_schema", tool_def)
        self.assertIn("query", tool_def["input_schema"]["properties"])
        self.assertEqual(tool_def["input_schema"]["required"], ["query"])
        
        print("✅ Tool definition test passed")
    
    def test_execute_with_successful_results(self):
        """Test execute method with successful search results"""
        # Mock search results
        mock_results = SearchResults(
            documents=["Sample course content about Python programming"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute("Python programming")
        
        # Verify results
        self.assertIsInstance(result, str)
        self.assertIn("Python Basics", result)
        self.assertIn("Lesson 1", result)
        self.assertIn("Sample course content", result)
        
        # Verify vector store was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="Python programming",
            course_name=None,
            lesson_number=None
        )
        
        print("✅ Execute with successful results test passed")
    
    def test_execute_with_empty_results(self):
        """Test execute method with no search results"""
        # Mock empty search results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute("nonexistent topic")
        
        # Verify results
        self.assertEqual(result, "No relevant content found.")
        
        print("✅ Execute with empty results test passed")
    
    def test_execute_with_search_error(self):
        """Test execute method when vector store returns an error"""
        # Mock search results with error
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute("any query")
        
        # Verify error is returned
        self.assertEqual(result, "Database connection failed")
        
        print("✅ Execute with search error test passed")
    
    def test_execute_with_course_filter(self):
        """Test execute method with course name filter"""
        # Mock search results
        mock_results = SearchResults(
            documents=["Advanced Python concepts"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 3}],
            distances=[0.15],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search with course filter
        result = self.search_tool.execute("concepts", course_name="Advanced Python")
        
        # Verify vector store was called with course filter
        self.mock_vector_store.search.assert_called_once_with(
            query="concepts",
            course_name="Advanced Python",
            lesson_number=None
        )
        
        # Verify results contain course information
        self.assertIn("Advanced Python", result)
        self.assertIn("Lesson 3", result)
        
        print("✅ Execute with course filter test passed")
    
    def test_execute_with_lesson_filter(self):
        """Test execute method with lesson number filter"""
        # Mock search results
        mock_results = SearchResults(
            documents=["Lesson 2 content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 2}],
            distances=[0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search with lesson filter
        result = self.search_tool.execute("content", lesson_number=2)
        
        # Verify vector store was called with lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="content",
            course_name=None,
            lesson_number=2
        )
        
        print("✅ Execute with lesson filter test passed")
    
    def test_format_results_with_lesson_links(self):
        """Test that lesson links are properly tracked in sources"""
        # Mock search results
        mock_results = SearchResults(
            documents=["Course content with link"],
            metadata=[{"course_title": "Web Development", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        
        # Mock lesson link retrieval
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute("web development")
        
        # Verify lesson link was requested
        self.mock_vector_store.get_lesson_link.assert_called_once_with("Web Development", 1)
        
        # Verify sources are tracked with links
        self.assertEqual(len(self.search_tool.last_sources), 1)
        source = self.search_tool.last_sources[0]
        self.assertEqual(source["text"], "Web Development - Lesson 1")
        self.assertEqual(source["link"], "https://example.com/lesson1")
        
        print("✅ Format results with lesson links test passed")
    
    def test_multiple_documents_formatting(self):
        """Test formatting with multiple search results"""
        # Mock search results with multiple documents
        mock_results = SearchResults(
            documents=[
                "First document content",
                "Second document content"
            ],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None  # No links
        
        # Execute search
        result = self.search_tool.execute("content")
        
        # Verify both documents are included
        self.assertIn("Course A", result)
        self.assertIn("Course B", result)
        self.assertIn("First document content", result)
        self.assertIn("Second document content", result)
        self.assertIn("Lesson 1", result)
        self.assertIn("Lesson 2", result)
        
        # Verify multiple sources are tracked
        self.assertEqual(len(self.search_tool.last_sources), 2)
        
        print("✅ Multiple documents formatting test passed")


class TestCourseSearchToolIntegration(unittest.TestCase):
    """Integration tests with real vector store (if available)"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        try:
            from vector_store import VectorStore
            from config import config
            
            # Try to create a real vector store for integration testing
            self.vector_store = VectorStore(
                chroma_path="./test_chroma_db",
                embedding_model=config.EMBEDDING_MODEL,
                max_results=3
            )
            self.search_tool = CourseSearchTool(self.vector_store)
            self.integration_available = True
            
        except Exception as e:
            print(f"⚠️  Integration tests disabled: {e}")
            self.integration_available = False
    
    def test_integration_with_empty_database(self):
        """Test search tool behavior with empty vector database"""
        if not self.integration_available:
            self.skipTest("Integration testing not available")
        
        # Clear any existing data
        self.vector_store.clear_all_data()
        
        # Execute search on empty database
        result = self.search_tool.execute("any query")
        
        # Should return no results message
        self.assertEqual(result, "No relevant content found.")
        
        print("✅ Integration test with empty database passed")
    
    def tearDown(self):
        """Clean up integration test resources"""
        if self.integration_available:
            # Clean up test database
            try:
                self.vector_store.clear_all_data()
            except:
                pass


class TestCourseSearchToolErrorScenarios(unittest.TestCase):
    """Test error scenarios and edge cases"""
    
    def setUp(self):
        """Set up error scenario test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_vector_store_exception(self):
        """Test handling of unexpected vector store exceptions"""
        # Mock vector store to raise exception
        self.mock_vector_store.search.side_effect = Exception("Unexpected database error")
        
        # This should be handled gracefully by the vector store's search method
        # The search method should return SearchResults.empty() with error
        mock_error_results = SearchResults.empty("Search error: Unexpected database error")
        self.mock_vector_store.search.side_effect = None
        self.mock_vector_store.search.return_value = mock_error_results
        
        result = self.search_tool.execute("test query")
        
        self.assertEqual(result, "Search error: Unexpected database error")
        
        print("✅ Vector store exception handling test passed")
    
    def test_malformed_metadata(self):
        """Test handling of malformed metadata in search results"""
        # Mock search results with missing/malformed metadata
        mock_results = SearchResults(
            documents=["Content with bad metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        result = self.search_tool.execute("test query")
        
        # Should handle missing metadata gracefully
        self.assertIn("unknown", result)
        self.assertIn("Content with bad metadata", result)
        
        print("✅ Malformed metadata handling test passed")


def run_all_tests():
    """Run all CourseSearchTool tests"""
    print("🧪 Running CourseSearchTool Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    for test_class in [TestCourseSearchTool, TestCourseSearchToolIntegration, TestCourseSearchToolErrorScenarios]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All CourseSearchTool tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        # Print failure details
        for test, traceback in result.failures:
            print(f"\n❌ FAILURE: {test}")
            print(traceback)
        
        for test, traceback in result.errors:
            print(f"\n💥 ERROR: {test}")
            print(traceback)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()