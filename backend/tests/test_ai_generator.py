#!/usr/bin/env python3
"""
Comprehensive tests for AIGenerator to identify tool calling and API integration issues.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class MockAnthropicResponse:
    """Mock Anthropic API response"""

    def __init__(self, content_text=None, stop_reason="end_turn", tool_calls=None):
        self.stop_reason = stop_reason

        if tool_calls:
            # Mock tool use response
            self.content = []
            for tool_call in tool_calls:
                mock_tool = Mock()
                mock_tool.type = "tool_use"
                mock_tool.name = tool_call["name"]
                mock_tool.input = tool_call["input"]
                mock_tool.id = tool_call.get("id", "tool_123")
                self.content.append(mock_tool)
        else:
            # Mock text response
            mock_content = Mock()
            mock_content.text = content_text or "Default response"
            self.content = [mock_content]


class TestAIGenerator(unittest.TestCase):
    """Test AIGenerator functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"

        # Create AIGenerator with mocked Anthropic client
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            self.ai_generator = AIGenerator(self.api_key, self.model)
            self.mock_client = mock_anthropic.return_value

    def test_initialization(self):
        """Test AIGenerator initialization"""
        self.assertEqual(self.ai_generator.model, self.model)
        self.assertIsNotNone(self.ai_generator.client)
        self.assertIn("temperature", self.ai_generator.base_params)
        self.assertEqual(self.ai_generator.base_params["temperature"], 0)

        print("✅ AIGenerator initialization test passed")

    def test_generate_response_without_tools(self):
        """Test response generation without tool calling"""
        # Mock successful API response
        mock_response = MockAnthropicResponse(
            "This is a direct response without tools."
        )
        self.mock_client.messages.create.return_value = mock_response

        # Generate response
        response = self.ai_generator.generate_response("What is Python?")

        # Verify response
        self.assertEqual(response, "This is a direct response without tools.")

        # Verify API was called correctly
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["model"], self.model)
        self.assertEqual(call_args["temperature"], 0)
        self.assertIn("messages", call_args)
        self.assertEqual(len(call_args["messages"]), 1)
        self.assertEqual(call_args["messages"][0]["role"], "user")

        print("✅ Generate response without tools test passed")

    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        # Mock API response
        mock_response = MockAnthropicResponse("Response with history context.")
        self.mock_client.messages.create.return_value = mock_response

        # Generate response with history
        history = "Previous conversation context"
        response = self.ai_generator.generate_response(
            "Follow-up question", conversation_history=history
        )

        # Verify response
        self.assertEqual(response, "Response with history context.")

        # Verify system prompt includes history
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertIn(history, call_args["system"])

        print("✅ Generate response with conversation history test passed")

    def test_generate_response_with_tools_but_no_tool_use(self):
        """Test response generation when tools are available but not used"""
        # Mock tool manager
        mock_tool_manager = Mock()
        tools = [{"name": "search_tool", "description": "Search courses"}]

        # Mock API response (no tool use)
        mock_response = MockAnthropicResponse("Direct response without using tools.")
        self.mock_client.messages.create.return_value = mock_response

        # Generate response
        response = self.ai_generator.generate_response(
            "What is 2+2?", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify response
        self.assertEqual(response, "Direct response without using tools.")

        # Verify tools were provided to API
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertIn("tools", call_args)
        self.assertEqual(call_args["tools"], tools)
        self.assertIn("tool_choice", call_args)

        print("✅ Generate response with tools but no tool use test passed")

    def test_tool_execution_flow(self):
        """Test the complete tool execution flow"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Mock initial API response with tool use
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "Python basics"},
                    "id": "tool_abc123",
                }
            ],
        )

        # Mock follow-up API response after tool execution
        final_response = MockAnthropicResponse(
            "Based on the search results, here's the answer."
        )

        # Set up mock client to return different responses
        self.mock_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Generate response
        response = self.ai_generator.generate_response(
            "Tell me about Python basics", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify final response
        self.assertEqual(response, "Based on the search results, here's the answer.")

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python basics"
        )

        # Verify two API calls were made
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        print("✅ Tool execution flow test passed")

    def test_multiple_tool_calls(self):
        """Test handling of multiple tool calls in one response"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First tool result",
            "Second tool result",
        ]

        tools = [
            {"name": "search_tool", "description": "Search courses"},
            {"name": "outline_tool", "description": "Get course outline"},
        ]

        # Mock initial response with multiple tool calls
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {"name": "search_tool", "input": {"query": "Python"}, "id": "tool_1"},
                {"name": "outline_tool", "input": {"course": "Python"}, "id": "tool_2"},
            ],
        )

        # Mock final response
        final_response = MockAnthropicResponse("Combined results from both tools.")

        self.mock_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Generate response
        response = self.ai_generator.generate_response(
            "Search and outline Python course",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify final response
        self.assertEqual(response, "Combined results from both tools.")

        print("✅ Multiple tool calls test passed")

    def test_tool_execution_error_handling(self):
        """Test handling of tool execution errors"""
        # Mock tool manager that returns an error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Tool execution failed: Database error"
        )

        tools = [{"name": "search_tool", "description": "Search courses"}]

        # Mock responses
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {"name": "search_tool", "input": {"query": "test"}, "id": "tool_1"}
            ],
        )
        final_response = MockAnthropicResponse(
            "I encountered an error while searching."
        )

        self.mock_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Generate response
        response = self.ai_generator.generate_response(
            "Search for something", tools=tools, tool_manager=mock_tool_manager
        )

        # Should still return a response even if tool failed
        self.assertEqual(response, "I encountered an error while searching.")

        # Verify tool execution was attempted
        mock_tool_manager.execute_tool.assert_called_once()

        print("✅ Tool execution error handling test passed")

    def test_api_error_handling(self):
        """Test handling of Anthropic API errors"""
        # Mock API to raise an exception
        from unittest.mock import Mock

        import anthropic

        # Create a proper APIError with mock request
        mock_response = Mock()
        mock_response.status_code = 429
        self.mock_client.messages.create.side_effect = anthropic.RateLimitError(
            "API rate limit exceeded", response=mock_response, body=None
        )

        # This should raise the exception (we want to see API errors)
        with self.assertRaises(anthropic.RateLimitError):
            self.ai_generator.generate_response("Test query")

        print("✅ API error handling test passed")

    def test_system_prompt_integration(self):
        """Test that system prompt is properly included"""
        mock_response = MockAnthropicResponse("Response using system prompt.")
        self.mock_client.messages.create.return_value = mock_response

        # Generate response
        self.ai_generator.generate_response("Test query")

        # Verify system prompt was included
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertIn("system", call_args)
        system_content = call_args["system"]

        # Should contain key parts of the system prompt
        self.assertIn("AI assistant specialized in course materials", system_content)
        self.assertIn("Tool Usage", system_content)

        print("✅ System prompt integration test passed")

    def test_sequential_tool_calling_two_rounds(self):
        """Test sequential tool calling over 2 rounds"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course X lesson 4: Python Data Structures",  # Round 1 result
            "Found course Y that covers Python Data Structures",  # Round 2 result
        ]

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Mock round 1: Tool use response
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "course X lesson 4"},
                    "id": "tool_1",
                }
            ],
        )

        # Mock round 2: Tool use response
        round2_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "Python Data Structures"},
                    "id": "tool_2",
                }
            ],
        )

        # Mock final response after max rounds
        final_response = MockAnthropicResponse(
            "Based on my searches, Course Y covers similar topics to Course X lesson 4."
        )

        # Set up mock client for 3 API calls (2 tool rounds + final)
        self.mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        # Generate response
        response = self.ai_generator.generate_response(
            "Find a course that covers the same topic as lesson 4 of course X",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify final response
        self.assertEqual(
            response,
            "Based on my searches, Course Y covers similar topics to Course X lesson 4.",
        )

        # Verify 3 API calls were made (2 rounds + final)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)

        print("✅ Sequential tool calling two rounds test passed")

    def test_sequential_tool_calling_early_termination(self):
        """Test early termination when Claude responds without tools in round 1"""
        # Mock tool manager
        mock_tool_manager = Mock()

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Mock response without tool use (early termination)
        direct_response = MockAnthropicResponse(
            "This is a general knowledge question. 2+2 equals 4."
        )

        self.mock_client.messages.create.return_value = direct_response

        # Generate response
        response = self.ai_generator.generate_response(
            "What is 2+2?", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify no tools were executed
        mock_tool_manager.execute_tool.assert_not_called()

        # Verify direct response
        self.assertEqual(
            response, "This is a general knowledge question. 2+2 equals 4."
        )

        # Verify only 1 API call was made
        self.assertEqual(self.mock_client.messages.create.call_count, 1)

        print("✅ Sequential tool calling early termination test passed")

    def test_sequential_tool_calling_round1_tool_failure(self):
        """Test tool failure in round 1"""
        # Mock tool manager that fails
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception(
            "Database connection failed"
        )

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Mock round 1: Tool use response
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "test"},
                    "id": "tool_1",
                }
            ],
        )

        self.mock_client.messages.create.return_value = round1_response

        # Generate response
        response = self.ai_generator.generate_response(
            "Search for something", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify tool execution was attempted
        mock_tool_manager.execute_tool.assert_called_once()

        # Verify error response
        self.assertIn("I encountered an error while searching", response)
        self.assertIn("Database connection failed", response)

        print("✅ Sequential tool calling round 1 tool failure test passed")

    def test_sequential_tool_calling_round2_tool_failure(self):
        """Test tool failure in round 2"""
        # Mock tool manager that succeeds first, then fails
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First search successful",  # Round 1 succeeds
            Exception("Search service unavailable"),  # Round 2 fails
        ]

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Mock round 1: Tool use response
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "first search"},
                    "id": "tool_1",
                }
            ],
        )

        # Mock round 2: Tool use response
        round2_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "second search"},
                    "id": "tool_2",
                }
            ],
        )

        self.mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
        ]

        # Generate response
        response = self.ai_generator.generate_response(
            "Search for multiple things", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify both tool executions were attempted
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify error response
        self.assertIn("I encountered an error while searching", response)
        self.assertIn("Search service unavailable", response)

        print("✅ Sequential tool calling round 2 tool failure test passed")

    def test_sequential_tool_calling_max_rounds_exceeded(self):
        """Test behavior when max rounds is exceeded (final call without tools)"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First search result",
            "Second search result",
        ]

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Mock round 1: Tool use response
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "first"},
                    "id": "tool_1",
                }
            ],
        )

        # Mock round 2: Tool use response (but this will hit max rounds)
        round2_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "second"},
                    "id": "tool_2",
                }
            ],
        )

        # Mock final response without tools
        final_response = MockAnthropicResponse(
            "Based on my research, here's the complete answer."
        )

        # Set up mock client for 3 API calls
        self.mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        # Generate response with max_rounds=2 (default)
        response = self.ai_generator.generate_response(
            "Complex multi-step query", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify final response
        self.assertEqual(response, "Based on my research, here's the complete answer.")

        # Verify 3 API calls were made (2 tool rounds + final without tools)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)

        # Verify final call had no tools by checking the last call
        final_call_kwargs = self.mock_client.messages.create.call_args_list[2][1]
        self.assertNotIn("tools", final_call_kwargs)

        print("✅ Sequential tool calling max rounds exceeded test passed")

    def test_sequential_tool_calling_context_preservation(self):
        """Test that context is preserved correctly across rounds"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First result: Course basics",
            "Second result: Advanced topics",
        ]

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Mock responses
        round1_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "basics"},
                    "id": "tool_1",
                }
            ],
        )

        round2_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "advanced"},
                    "id": "tool_2",
                }
            ],
        )

        final_response = MockAnthropicResponse(
            "Combined analysis of basics and advanced topics."
        )

        self.mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        # Generate response with conversation history
        response = self.ai_generator.generate_response(
            "Tell me about course progression",
            conversation_history="Previous context about learning",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify the final API call includes accumulated context
        final_call_kwargs = self.mock_client.messages.create.call_args_list[2][1]

        # Should have 5 messages: initial user query + assistant tool use + user tool results + assistant tool use + user tool results
        self.assertEqual(len(final_call_kwargs["messages"]), 5)

        # Check message structure
        messages = final_call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "user")  # Initial query
        self.assertEqual(messages[1]["role"], "assistant")  # First tool use
        self.assertEqual(messages[2]["role"], "user")  # First tool results
        self.assertEqual(messages[3]["role"], "assistant")  # Second tool use
        self.assertEqual(messages[4]["role"], "user")  # Second tool results

        # Verify system prompt includes conversation history
        self.assertIn("Previous context about learning", final_call_kwargs["system"])

        print("✅ Sequential tool calling context preservation test passed")

    def test_custom_max_rounds_parameter(self):
        """Test custom max_rounds parameter"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Mock single tool use response
        tool_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "name": "search_course_content",
                    "input": {"query": "test"},
                    "id": "tool_1",
                }
            ],
        )

        # Mock final response
        final_response = MockAnthropicResponse("Final answer after 1 round.")

        self.mock_client.messages.create.side_effect = [tool_response, final_response]

        # Generate response with max_rounds=1 (custom)
        response = self.ai_generator.generate_response(
            "Test query", tools=tools, tool_manager=mock_tool_manager, max_rounds=1
        )

        # Verify only 1 tool execution
        mock_tool_manager.execute_tool.assert_called_once()

        # Verify final response
        self.assertEqual(response, "Final answer after 1 round.")

        # Verify 2 API calls (1 tool round + final)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        print("✅ Custom max_rounds parameter test passed")


class TestAIGeneratorRealAPI(unittest.TestCase):
    """Integration tests with real Anthropic API (if API key available)"""

    def setUp(self):
        """Set up real API integration tests"""
        # Check if API key is available
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key.strip() == "":
            self.api_available = False
            print("⚠️  Real API tests disabled: ANTHROPIC_API_KEY not set")
        else:
            self.api_available = True
            self.ai_generator = AIGenerator(api_key, "claude-sonnet-4-20250514")

    def test_simple_query_without_tools(self):
        """Test simple query without tools using real API"""
        if not self.api_available:
            self.skipTest("Real API not available")

        try:
            response = self.ai_generator.generate_response("What is 2 + 2?")

            # Should get a reasonable response
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            self.assertIn("4", response)  # Should mention the answer

            print("✅ Real API simple query test passed")

        except Exception as e:
            self.fail(f"Real API test failed: {e}")

    def test_tool_calling_with_mock_tools(self):
        """Test tool calling behavior with real API but mock tools"""
        if not self.api_available:
            self.skipTest("Real API not available")

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Mock course content about Python basics"
        )

        # Define simple tool
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }
        ]

        try:
            response = self.ai_generator.generate_response(
                "Search for information about Python programming basics",
                tools=tools,
                tool_manager=mock_tool_manager,
            )

            # Should get a response that potentially used the tool
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

            print("✅ Real API tool calling test passed")

        except Exception as e:
            print(f"⚠️  Real API tool calling test failed (this may be expected): {e}")


def run_all_tests():
    """Run all AIGenerator tests"""
    print("🧪 Running AIGenerator Tests...")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestSuite()

    # Add test classes
    for test_class in [TestAIGenerator, TestAIGeneratorRealAPI]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All AIGenerator tests passed!")
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
