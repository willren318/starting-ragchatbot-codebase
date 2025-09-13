from typing import Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Strategy:
- **Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use when users ask about course structure, lesson lists, course outlines, or want to see what lessons are available in a course
- **Sequential Tool Use**: You may use tools multiple times to gather comprehensive information
  - Use initial searches to understand available content
  - Use follow-up searches to get specific details based on initial results
  - Maximum 2 tool calls per query - use them strategically
- **Reasoning Between Tools**: Analyze results from each tool call to determine if additional searches would help
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Complex questions**: May require multiple searches for complete answers
- **No meta-commentary**: Provide direct answers only — no reasoning process or search explanations

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with up to max_rounds of sequential tool calling.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)

        Returns:
            Generated response as string

        Termination conditions:
            - Max rounds reached
            - Claude's response contains no tool_use blocks
            - Tool execution fails
        """

        # Build system content with conversation history
        system_content = self._build_system_content(conversation_history)

        # Initialize conversation state
        messages = [{"role": "user", "content": query}]

        # Sequential tool calling loop
        for round_num in range(1, max_rounds + 1):
            # Make API call with tools (except potentially the last round)
            response = self._make_api_call(messages, system_content, tools)

            # Check termination conditions
            if response.stop_reason != "tool_use":
                # No tool use - return direct response
                return response.content[0].text

            # Tool execution required but no tool manager available
            if not tool_manager:
                return (
                    response.content[0].text
                    if response.content
                    else "No response available"
                )

            # Execute tools and add results to conversation
            try:
                messages = self._execute_tools_and_update_messages(
                    response, messages, tool_manager
                )
            except Exception as e:
                # Tool execution failed - return error gracefully
                return f"I encountered an error while searching: {str(e)}"

        # Max rounds reached - make final call without tools
        final_response = self._make_api_call(messages, system_content, tools=None)
        return final_response.content[0].text

    def _build_system_content(self, conversation_history: Optional[str]) -> str:
        """Build system prompt with conversation history."""
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

    def _make_api_call(
        self, messages: List[Dict], system_content: str, tools: Optional[List] = None
    ):
        """
        Make a single API call to Claude with consistent parameters.

        Args:
            messages: Conversation messages
            system_content: System prompt with history
            tools: Tool definitions (None for final calls)

        Returns:
            Claude response object
        """
        api_params = {
            **self.base_params,
            "messages": messages.copy(),  # Ensure no mutation
            "system": system_content,
        }

        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.messages.create(**api_params)

    def _execute_tools_and_update_messages(
        self, response, messages: List[Dict], tool_manager
    ) -> List[Dict]:
        """
        Execute all tool calls in response and update message history.

        Args:
            response: Claude's response containing tool use
            messages: Current message history
            tool_manager: Tool executor

        Returns:
            Updated messages list with assistant response and tool results

        Raises:
            Exception: If any tool execution fails
        """
        # Add Claude's tool use response to messages
        messages = messages.copy()  # Avoid mutation
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tools and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Individual tool failure - propagate up
                    raise Exception(f"Tool '{content_block.name}' failed: {str(e)}")

        # Add tool results as user message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages
