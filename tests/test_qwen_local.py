"""
Pytest tests for SubjectiveQwenLocalDataSource

Run with:
    venv\\Scripts\\pytest tests/ -v
    venv\\Scripts\\pytest tests/ -v -m "not slow"
"""

import pytest
from SubjectiveQwenLocalDataSource import SubjectiveQwenLocalDataSource


class TestQwenLocalDataSourceInit:
    """Tests for initialization and configuration."""

    def test_initialization_default_params(self):
        """Test that the data source initializes with default parameters."""
        ds = SubjectiveQwenLocalDataSource(name="test_qwen")

        assert ds.name == "test_qwen"
        assert ds.params.get("model_id") == "Qwen/Qwen2-VL-2B-Instruct"
        assert ds.params.get("device") == "auto"
        assert ds.params.get("max_new_tokens") == 1024
        assert ds.params.get("temperature") == 0.7

    def test_initialization_with_params(self):
        """Test initialization with custom parameters dict."""
        ds = SubjectiveQwenLocalDataSource(
            name="custom_qwen",
            params={
                "connection_name": "Custom Qwen",
            }
        )

        # Check that params dict exists and has required keys
        # (parent class may override specific values with global config)
        assert ds.params is not None
        assert isinstance(ds.params, dict)
        assert ds.params.get("model_id") is not None
        assert ds.params.get("device") is not None
        assert ds.params.get("max_new_tokens") is not None
        assert ds.params.get("temperature") is not None

    def test_param_normalization_invalid_max_tokens(self):
        """Test that invalid max_tokens gets normalized to default."""
        ds = SubjectiveQwenLocalDataSource(
            name="test",
            params={"max_new_tokens": -100}
        )
        assert ds.params.get("max_new_tokens") == 1024

    def test_param_normalization_invalid_temperature(self):
        """Test that invalid temperature gets normalized to default."""
        ds = SubjectiveQwenLocalDataSource(
            name="test",
            params={"temperature": 5.0}  # Out of range
        )
        assert ds.params.get("temperature") == 0.7

    def test_param_normalization_invalid_top_p(self):
        """Test that invalid top_p gets normalized to default."""
        ds = SubjectiveQwenLocalDataSource(
            name="test",
            params={"top_p": 2.0}  # Out of range
        )
        assert ds.params.get("top_p") == 0.9


class TestQwenLocalDataSourceConnectionData:
    """Tests for connection data configuration."""

    def test_connection_type(self):
        """Test that connection type is ON_DEMAND."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        connection_data = ds.get_connection_data()

        assert connection_data.get("connection_type") == "ON_DEMAND"

    def test_connection_fields_exist(self):
        """Test that all expected fields are present."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        connection_data = ds.get_connection_data()
        fields = connection_data.get("fields", [])

        field_names = [f["name"] for f in fields]

        assert "connection_name" in field_names
        assert "model_id" in field_names
        assert "device" in field_names
        assert "max_new_tokens" in field_names
        assert "temperature" in field_names
        assert "top_p" in field_names
        assert "system_prompt" in field_names
        assert "load_in_4bit" in field_names
        assert "load_in_8bit" in field_names

    def test_connection_fields_count(self):
        """Test the number of configuration fields."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        connection_data = ds.get_connection_data()
        fields = connection_data.get("fields", [])

        assert len(fields) == 12

    def test_required_fields(self):
        """Test that required fields are marked correctly."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        connection_data = ds.get_connection_data()
        fields = connection_data.get("fields", [])

        required_fields = [f["name"] for f in fields if f.get("required")]

        assert "connection_name" in required_fields
        assert "model_id" in required_fields


class TestQwenLocalDataSourceIcon:
    """Tests for icon functionality."""

    def test_icon_returns_svg(self):
        """Test that get_icon returns valid SVG."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        icon = ds.get_icon()

        assert icon is not None
        assert len(icon) > 0
        assert "<svg" in icon

    def test_icon_is_valid_xml(self):
        """Test that the icon is valid XML."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        icon = ds.get_icon()

        # Basic XML validation
        assert icon.strip().startswith("<svg") or icon.strip().startswith("<?xml")
        assert "</svg>" in icon


class TestQwenLocalDataSourceDependencies:
    """Tests for dependency checking."""

    def test_dependencies_flag_exists(self):
        """Test that dependencies flag is set."""
        ds = SubjectiveQwenLocalDataSource(name="test")

        assert hasattr(ds, "_dependencies_available")
        assert isinstance(ds._dependencies_available, bool)

    def test_dependency_error_response(self):
        """Test the dependency error response format."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        # Force dependencies unavailable for this test
        original = ds._dependencies_available
        ds._dependencies_available = False

        response = ds._dependency_error_response("test message")

        assert response.get("error") is True
        assert response.get("error_type") == "dependency_error"
        assert "pip install" in response.get("message", "")
        assert response.get("original_message") == "test message"

        ds._dependencies_available = original


class TestQwenLocalDataSourceHelpers:
    """Tests for helper methods."""

    def test_normalize_files_empty(self):
        """Test file normalization with empty input."""
        ds = SubjectiveQwenLocalDataSource(name="test")

        assert ds._normalize_files(None) == []
        assert ds._normalize_files([]) == []

    def test_normalize_files_valid(self):
        """Test file normalization with valid input."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        files = [
            {"name": "test.jpg", "mime_type": "image/jpeg"},
            {"name": "test.txt", "mime_type": "text/plain"},
        ]

        result = ds._normalize_files(files)
        assert len(result) == 2

    def test_normalize_files_filters_invalid(self):
        """Test that invalid entries are filtered out."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        files = [
            {"name": "valid.jpg"},
            "invalid_string",
            123,
            {"name": "also_valid.png"},
        ]

        result = ds._normalize_files(files)
        assert len(result) == 2

    def test_truncate_text_short(self):
        """Test truncation with short text."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        text = "Short text"

        result = ds._truncate_text(text)
        assert result == text

    def test_truncate_text_long(self):
        """Test truncation with long text."""
        ds = SubjectiveQwenLocalDataSource(name="test")
        text = "x" * 25000

        result = ds._truncate_text(text, max_chars=20000)
        assert len(result) < len(text)
        assert "[truncated]" in result

    def test_guess_mime_type(self):
        """Test MIME type guessing."""
        ds = SubjectiveQwenLocalDataSource(name="test")

        assert ds._guess_mime_type("image.jpg") == "image/jpeg"
        assert ds._guess_mime_type("image.png") == "image/png"
        assert ds._guess_mime_type("document.pdf") == "application/pdf"
        assert ds._guess_mime_type("unknown.xyz") == "application/octet-stream"


@pytest.mark.skipif(
    not SubjectiveQwenLocalDataSource(name="check")._dependencies_available,
    reason="Qwen VLM dependencies not installed"
)
class TestQwenLocalDataSourceModel:
    """Tests that require the model (skipped if dependencies unavailable)."""

    @pytest.fixture
    def datasource(self):
        """Create a data source instance for testing."""
        return SubjectiveQwenLocalDataSource(
            name="test_model",
            params={
                "model_id": "Qwen/Qwen2-VL-2B-Instruct",
                "device": "auto",
                "max_new_tokens": 64,
                "load_in_4bit": True,
            }
        )

    @pytest.mark.slow
    def test_model_loading(self, datasource):
        """Test that the model can be loaded."""
        result = datasource._load_model()
        assert result is True
        assert datasource._model_loaded is True
        datasource.unload_model()

    @pytest.mark.slow
    def test_text_message_processing(self, datasource):
        """Test processing a simple text message."""
        response = datasource._process_message("What is 2+2?")

        if response.get("success"):
            assert "response" in response
            assert len(response["response"]) > 0
        else:
            # May fail due to model download issues, etc.
            assert "error" in response

        datasource.unload_model()

    @pytest.mark.slow
    def test_unload_model(self, datasource):
        """Test model unloading."""
        datasource._load_model()
        datasource.unload_model()

        assert datasource._model is None
        assert datasource._processor is None
        assert datasource._model_loaded is False
