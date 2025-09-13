from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI with links

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI with link
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"

            # Try to get lesson link if we have both course and lesson
            lesson_link = None
            if course_title != "unknown" and lesson_num is not None:
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)

            # Create source object with text and optional link
            source = {"text": source_text, "link": lesson_link}
            sources.append(source)

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outlines with course metadata and lesson structure"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get complete course outline including course title, course link, and full lesson list with titles",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    }
                },
                "required": ["course_title"],
            },
        }

    def execute(self, course_title: str) -> str:
        """
        Execute the course outline tool with given course title.

        Args:
            course_title: Course title to find outline for

        Returns:
            Formatted course outline or error message
        """

        # Use the vector store's course name resolution
        resolved_course_title = self.store._resolve_course_name(course_title)
        if not resolved_course_title:
            return f"No course found matching '{course_title}'"

        # Get all courses metadata
        all_courses_metadata = self.store.get_all_courses_metadata()

        # Find the specific course
        course_metadata = None
        for course_meta in all_courses_metadata:
            if course_meta.get("title") == resolved_course_title:
                course_metadata = course_meta
                break

        if not course_metadata:
            return f"Course metadata not found for '{resolved_course_title}'"

        # Format the course outline
        return self._format_course_outline(course_metadata)

    def _format_course_outline(self, course_metadata: Dict[str, Any]) -> str:
        """Format course metadata into a readable outline"""
        title = course_metadata.get("title", "Unknown Course")
        instructor = course_metadata.get("instructor", "Unknown Instructor")
        course_link = course_metadata.get("course_link")
        lessons = course_metadata.get("lessons", [])

        # Build the outline
        outline_parts = [
            f"**Course Title:** {title}",
            f"**Course Instructor:** {instructor}",
        ]

        # Add course link if available
        if course_link:
            outline_parts.append(f"**Course Link:** {course_link}")

        # Add lessons section
        if lessons:
            outline_parts.append(f"**Total Lessons:** {len(lessons)}")
            outline_parts.append("\n**Course Outline:**")

            for lesson in lessons:
                lesson_num = lesson.get("lesson_number", "N/A")
                lesson_title = lesson.get("lesson_title", "Untitled Lesson")
                outline_parts.append(f"Lesson {lesson_num}: {lesson_title}")
        else:
            outline_parts.append("**No lessons found for this course**")

        # Track source for the UI
        source = {"text": title, "link": course_link}
        self.last_sources = [source]

        return "\n".join(outline_parts)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
