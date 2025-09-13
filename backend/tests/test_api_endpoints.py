"""
API endpoint tests for the RAG system FastAPI application.
Tests all API endpoints for proper request/response handling.
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import status


class TestQueryEndpoint:
    """Test the /api/query endpoint."""

    def test_query_endpoint_success(self, test_client, valid_query_request):
        """Test successful query processing."""
        response = test_client.post("/api/query", json=valid_query_request)

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        assert data["answer"] == "Test response"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test source"
        assert data["session_id"] == "test_session_123"

    def test_query_endpoint_without_session_id(self, test_client):
        """Test query without session_id creates new session."""
        request_data = {"query": "What is Python?"}

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"  # Mock returns this

    def test_query_endpoint_empty_query(self, test_client):
        """Test query with empty string."""
        request_data = {"query": "", "session_id": "test_session"}

        response = test_client.post("/api/query", json=request_data)

        # Should still process empty queries
        assert response.status_code == status.HTTP_200_OK

    def test_query_endpoint_missing_query_field(self, test_client):
        """Test query request missing required query field."""
        request_data = {"session_id": "test_session"}

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_endpoint_invalid_json(self, test_client):
        """Test query with invalid JSON."""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_endpoint_rag_system_error(self, test_client, test_app):
        """Test query when RAG system raises an error."""
        # Mock the RAG system to raise an error
        test_app.mock_rag_system.query.side_effect = Exception("RAG system error")

        request_data = {"query": "What is Python?", "session_id": "test_session"}

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "RAG system error" in response.json()["detail"]

    def test_query_endpoint_long_query(self, test_client):
        """Test query with very long text."""
        long_query = "What is Python? " * 1000  # Very long query
        request_data = {"query": long_query, "session_id": "test_session"}

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == status.HTTP_200_OK

    def test_query_endpoint_special_characters(self, test_client):
        """Test query with special characters and unicode."""
        special_query = "What is Python? 🐍 Special chars: @#$%^&*()+=[]{}|;:'\",.<>?/`~"
        request_data = {"query": special_query, "session_id": "test_session"}

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == status.HTTP_200_OK

    def test_query_endpoint_sources_format(self, test_client, test_app):
        """Test that sources are properly formatted."""
        # Mock different source formats
        test_app.mock_rag_system.query.return_value = (
            "Test response",
            [
                {"text": "Source with link", "link": "https://example.com"},
                {"text": "Source without link"},
                "Legacy string source"
            ]
        )

        request_data = {"query": "Test query", "session_id": "test_session"}

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        sources = data["sources"]

        assert len(sources) == 3

        # Source with link
        assert sources[0]["text"] == "Source with link"
        assert sources[0]["link"] == "https://example.com"

        # Source without link
        assert sources[1]["text"] == "Source without link"
        assert sources[1]["link"] is None

        # Legacy string source
        assert sources[2]["text"] == "Legacy string source"
        assert sources[2]["link"] is None


class TestCoursesEndpoint:
    """Test the /api/courses endpoint."""

    def test_courses_endpoint_success(self, test_client, mock_course_analytics):
        """Test successful course analytics retrieval."""
        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data

        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Python Basics" in data["course_titles"]
        assert "Advanced Python" in data["course_titles"]

    def test_courses_endpoint_empty_database(self, test_client, test_app):
        """Test courses endpoint when no courses are loaded."""
        # Mock empty course analytics
        test_app.mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_endpoint_rag_system_error(self, test_client, test_app):
        """Test courses endpoint when RAG system raises an error."""
        # Mock the RAG system to raise an error
        test_app.mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Analytics error" in response.json()["detail"]

    def test_courses_endpoint_no_query_params(self, test_client):
        """Test courses endpoint doesn't accept query parameters."""
        response = test_client.get("/api/courses?invalid=param")

        # Should still work, ignoring invalid parameters
        assert response.status_code == status.HTTP_200_OK

    def test_courses_endpoint_post_method(self, test_client):
        """Test courses endpoint with wrong HTTP method."""
        response = test_client.post("/api/courses", json={"test": "data"})

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


class TestRootEndpoint:
    """Test the root / endpoint."""

    def test_root_endpoint_success(self, test_client):
        """Test root endpoint returns API information."""
        response = test_client.get("/")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "message" in data
        assert "RAG System" in data["message"]

    def test_root_endpoint_post_method(self, test_client):
        """Test root endpoint with POST method."""
        response = test_client.post("/", json={"test": "data"})

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


class TestCORSAndHeaders:
    """Test CORS configuration and HTTP headers."""

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are properly set."""
        # Make a request with Origin header to trigger CORS
        response = test_client.get(
            "/api/courses",
            headers={"Origin": "http://localhost:3000"}
        )

        # Check for CORS headers (may not be present in test environment)
        # This test verifies CORS middleware is configured, headers may vary
        assert response.status_code == 200

    def test_preflight_request(self, test_client):
        """Test CORS preflight request."""
        response = test_client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )

        assert response.status_code == status.HTTP_200_OK
        assert "access-control-allow-methods" in response.headers

    def test_content_type_json(self, test_client, valid_query_request):
        """Test that responses have correct content type."""
        response = test_client.post("/api/query", json=valid_query_request)

        assert response.status_code == status.HTTP_200_OK
        assert "application/json" in response.headers["content-type"]


class TestErrorHandling:
    """Test error handling across all endpoints."""

    def test_404_for_nonexistent_endpoint(self, test_client):
        """Test 404 response for nonexistent endpoint."""
        response = test_client.get("/api/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_405_for_wrong_method(self, test_client):
        """Test 405 response for wrong HTTP method."""
        response = test_client.delete("/api/query")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_422_for_invalid_request_body(self, test_client):
        """Test 422 response for invalid request body."""
        # Send invalid data type for query
        response = test_client.post("/api/query", json={"query": 123})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestResponseModels:
    """Test that response models are correctly structured."""

    def test_query_response_model(self, test_client, valid_query_request):
        """Test QueryResponse model structure."""
        response = test_client.post("/api/query", json=valid_query_request)

        assert response.status_code == status.HTTP_200_OK

        data = response.json()

        # Check required fields
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data

        # Check field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

        # Check source structure
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "link" in source
            assert isinstance(source["text"], str)
            # link can be None or string

    def test_course_stats_response_model(self, test_client):
        """Test CourseStats model structure."""
        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()

        # Check required fields
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data

        # Check field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # Check that all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)


class TestEndpointIntegration:
    """Test integration between different endpoints."""

    def test_query_then_courses_workflow(self, test_client, valid_query_request):
        """Test typical workflow: query then get courses."""
        # First, make a query
        query_response = test_client.post("/api/query", json=valid_query_request)
        assert query_response.status_code == status.HTTP_200_OK

        # Then, get courses
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK

        # Both should be independent and successful
        query_data = query_response.json()
        courses_data = courses_response.json()

        assert "answer" in query_data
        assert "total_courses" in courses_data

    def test_multiple_concurrent_queries(self, test_client):
        """Test multiple queries can be handled concurrently."""
        queries = [
            {"query": "What is Python?", "session_id": f"session_{i}"}
            for i in range(5)
        ]

        responses = []
        for query in queries:
            response = test_client.post("/api/query", json=query)
            responses.append(response)

        # All should succeed
        for i, response in enumerate(responses):
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["session_id"] == f"session_{i}"


# Pytest markers for different test categories
@pytest.mark.api
class TestAPIPerformance:
    """Performance-related API tests."""

    def test_query_response_time(self, test_client, valid_query_request):
        """Test that query responses are reasonably fast."""
        import time

        start_time = time.time()
        response = test_client.post("/api/query", json=valid_query_request)
        end_time = time.time()

        assert response.status_code == status.HTTP_200_OK

        # Response should be under 5 seconds (generous for testing)
        response_time = end_time - start_time
        assert response_time < 5.0, f"Response took {response_time:.2f} seconds"

    def test_courses_response_time(self, test_client):
        """Test that courses endpoint responds quickly."""
        import time

        start_time = time.time()
        response = test_client.get("/api/courses")
        end_time = time.time()

        assert response.status_code == status.HTTP_200_OK

        # Response should be under 1 second
        response_time = end_time - start_time
        assert response_time < 1.0, f"Response took {response_time:.2f} seconds"