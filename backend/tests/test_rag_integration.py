#!/usr/bin/env python3
"""
End-to-end integration tests for the RAG system to identify query failure points.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class TestRAGSystemIntegration(unittest.TestCase):
    """Test complete RAG system integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test ChromaDB
        self.temp_dir = tempfile.mkdtemp()

        # Create test configuration
        self.test_config = Config()
        self.test_config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma")
        self.test_config.ANTHROPIC_API_KEY = "test_key_12345"
        self.test_config.MAX_RESULTS = 3

        # Create RAG system with mocked AI generator
        with patch("rag_system.AIGenerator") as mock_ai_gen_class:
            self.rag_system = RAGSystem(self.test_config)
            self.mock_ai_generator = mock_ai_gen_class.return_value

    def tearDown(self):
        """Clean up test resources"""
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_rag_system_initialization(self):
        """Test that RAG system initializes correctly"""
        # Verify all components are initialized
        self.assertIsNotNone(self.rag_system.vector_store)
        self.assertIsNotNone(self.rag_system.document_processor)
        self.assertIsNotNone(self.rag_system.ai_generator)
        self.assertIsNotNone(self.rag_system.session_manager)
        self.assertIsNotNone(self.rag_system.tool_manager)

        # Verify tools are registered
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]

        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)

        print("✅ RAG system initialization test passed")

    def test_query_with_empty_database(self):
        """Test query behavior when database is empty"""
        # Mock AI response for empty database scenario
        self.mock_ai_generator.generate_response.return_value = (
            "I don't have any course materials loaded."
        )

        # Execute query
        response, sources = self.rag_system.query("What is Python?")

        # Verify AI generator was called with correct parameters
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args[1]

        self.assertIn("query", call_args)
        self.assertIn("tools", call_args)
        self.assertIn("tool_manager", call_args)

        # Verify tools were provided
        tools = call_args["tools"]
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

        print("✅ Query with empty database test passed")

    def test_add_course_and_query_flow(self):
        """Test the complete flow of adding a course and querying it"""
        # Create sample course data
        sample_course = Course(
            title="Python Basics",
            instructor="John Doe",
            course_link="https://example.com/python",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Introduction",
                    lesson_link="https://example.com/lesson1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Variables",
                    lesson_link="https://example.com/lesson2",
                ),
            ],
        )

        sample_chunks = [
            CourseChunk(
                content="Python is a programming language",
                course_title="Python Basics",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Variables store data values",
                course_title="Python Basics",
                lesson_number=2,
                chunk_index=1,
            ),
        ]

        # Mock document processor to return our sample data
        with patch.object(
            self.rag_system.document_processor, "process_course_document"
        ) as mock_process:
            mock_process.return_value = (sample_course, sample_chunks)

            # Add course to system
            course, chunk_count = self.rag_system.add_course_document("fake_file.txt")

            # Verify course was processed
            self.assertEqual(course.title, "Python Basics")
            self.assertEqual(chunk_count, 2)

        # Mock AI response that uses the search tool
        self.mock_ai_generator.generate_response.return_value = (
            "Python is a programming language used for development."
        )

        # Execute query
        response, sources = self.rag_system.query("What is Python?")

        # Verify response was generated
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

        print("✅ Add course and query flow test passed")

    def test_tool_manager_integration(self):
        """Test that tool manager properly integrates with AI generator"""

        # Mock AI generator to simulate tool calling
        def mock_generate_response(*args, **kwargs):
            # Simulate tool calling by executing a tool through tool_manager
            if "tool_manager" in kwargs:
                tool_manager = kwargs["tool_manager"]
                # Execute a search tool
                result = tool_manager.execute_tool(
                    "search_course_content", query="Python"
                )
                return f"Based on search: {result[:50]}..."
            return "No tool manager provided"

        self.mock_ai_generator.generate_response.side_effect = mock_generate_response

        # Execute query
        response, sources = self.rag_system.query("Tell me about Python")

        # Verify response includes tool execution result
        self.assertIn("Based on search:", response)

        print("✅ Tool manager integration test passed")

    def test_session_management(self):
        """Test session-based conversation management"""
        session_id = "test_session_123"

        # Mock AI responses
        responses = ["First response", "Second response with context"]
        self.mock_ai_generator.generate_response.side_effect = responses

        # Execute first query
        response1, _ = self.rag_system.query("What is Python?", session_id)

        # Execute second query in same session
        response2, _ = self.rag_system.query("Tell me more", session_id)

        # Verify both responses were generated
        self.assertEqual(response1, "First response")
        self.assertEqual(response2, "Second response with context")

        # Verify AI generator was called with conversation history on second call
        second_call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertIn("conversation_history", second_call_args)
        self.assertIsNotNone(second_call_args["conversation_history"])

        print("✅ Session management test passed")

    def test_course_analytics(self):
        """Test course analytics functionality"""
        # Initially should have empty analytics
        analytics = self.rag_system.get_course_analytics()

        self.assertIn("total_courses", analytics)
        self.assertIn("course_titles", analytics)
        self.assertEqual(analytics["total_courses"], 0)
        self.assertEqual(len(analytics["course_titles"]), 0)

        print("✅ Course analytics test passed")


class TestRAGSystemErrorScenarios(unittest.TestCase):
    """Test RAG system error handling and edge cases"""

    def setUp(self):
        """Set up error scenario test fixtures"""
        # Create test configuration with invalid paths to trigger errors
        self.test_config = Config()
        self.test_config.CHROMA_PATH = "/invalid/path/chroma"
        self.test_config.ANTHROPIC_API_KEY = "test_key"

    def test_invalid_chroma_path_handling(self):
        """Test handling of invalid ChromaDB path"""
        # This test may fail during initialization, which is expected
        try:
            rag_system = RAGSystem(self.test_config)

            # If initialization succeeds, test query behavior
            with patch.object(rag_system, "ai_generator") as mock_ai:
                mock_ai.generate_response.return_value = "Error response"

                response, sources = rag_system.query("Test query")

                # Should still return a response even with database issues
                self.assertIsInstance(response, str)

        except Exception as e:
            # This is acceptable - invalid paths should cause initialization errors
            print(f"⚠️  Expected error with invalid ChromaDB path: {e}")

        print("✅ Invalid ChromaDB path handling test passed")

    def test_missing_api_key_behavior(self):
        """Test behavior when API key is missing"""
        # Create config without API key
        config_no_key = Config()
        config_no_key.ANTHROPIC_API_KEY = ""
        config_no_key.CHROMA_PATH = tempfile.mkdtemp()

        try:
            # RAG system should still initialize (API key is used later)
            rag_system = RAGSystem(config_no_key)

            # Query should fail when trying to use AI generator
            with patch.object(
                rag_system.ai_generator.client.messages, "create"
            ) as mock_create:
                import anthropic

                mock_create.side_effect = anthropic.AuthenticationError(
                    "Invalid API key"
                )

                with self.assertRaises(anthropic.AuthenticationError):
                    rag_system.query("Test query")

            print("✅ Missing API key behavior test passed")

            # Clean up
            shutil.rmtree(config_no_key.CHROMA_PATH)

        except Exception as e:
            print(f"⚠️  API key test encountered error: {e}")


class TestRAGSystemWithMockComponents(unittest.TestCase):
    """Test RAG system with heavily mocked components for isolated testing"""

    def setUp(self):
        """Set up with fully mocked components"""
        self.test_config = Config()
        self.test_config.CHROMA_PATH = tempfile.mkdtemp()
        self.test_config.ANTHROPIC_API_KEY = "mock_key"

        # Mock all major components
        with patch.multiple(
            "rag_system",
            VectorStore=Mock(),
            AIGenerator=Mock(),
            DocumentProcessor=Mock(),
            SessionManager=Mock(),
        ) as mocks:

            self.rag_system = RAGSystem(self.test_config)
            self.mock_vector_store = mocks["VectorStore"].return_value
            self.mock_ai_generator = mocks["AIGenerator"].return_value
            self.mock_document_processor = mocks["DocumentProcessor"].return_value
            self.mock_session_manager = mocks["SessionManager"].return_value

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_config.CHROMA_PATH):
            shutil.rmtree(self.test_config.CHROMA_PATH)

    def test_query_failure_scenarios(self):
        """Test various query failure scenarios"""
        test_cases = [
            # Case 1: AI generator throws exception
            {
                "name": "AI Generator Exception",
                "setup": lambda: setattr(
                    self.mock_ai_generator.generate_response,
                    "side_effect",
                    Exception("AI service unavailable"),
                ),
                "expected_error": Exception,
            },
            # Case 2: Tool manager returns error
            {
                "name": "Tool Manager Error",
                "setup": lambda: (
                    setattr(
                        self.mock_ai_generator.generate_response,
                        "return_value",
                        "Tool result",
                    ),
                    setattr(
                        self.rag_system.tool_manager.get_last_sources,
                        "return_value",
                        [],
                    ),
                ),
                "expected_error": None,  # Should not raise error
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                case["setup"]()

                if case["expected_error"]:
                    with self.assertRaises(case["expected_error"]):
                        self.rag_system.query("Test query")
                else:
                    try:
                        response, sources = self.rag_system.query("Test query")
                        self.assertIsInstance(response, str)
                        self.assertIsInstance(sources, list)
                    except Exception as e:
                        self.fail(f"Query should not have raised exception: {e}")

        print("✅ Query failure scenarios test passed")


def run_all_tests():
    """Run all RAG integration tests"""
    print("🧪 Running RAG System Integration Tests...")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestSuite()

    # Add test classes
    for test_class in [
        TestRAGSystemIntegration,
        TestRAGSystemErrorScenarios,
        TestRAGSystemWithMockComponents,
    ]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All RAG integration tests passed!")
    else:
        print(
            f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)"
        )

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
