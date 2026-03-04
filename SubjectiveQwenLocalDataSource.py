"""
Subjective Qwen Local VLM Data Source
=====================================
An on-demand data source plugin for running Qwen2-VL (Vision-Language Model) locally.
Supports both text and image inputs using the Qwen2-VL family of models.
"""

import os
import base64
import mimetypes
from typing import Any, Dict, List, Optional, Union
from io import BytesIO

from subjective_abstract_data_source_package import SubjectiveOnDemandDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveQwenLocalDataSource(SubjectiveOnDemandDataSource):
    """
    On-demand data source for running Qwen2-VL models locally.

    Supports:
    - Text-only queries
    - Image + text queries (Vision-Language)
    - Multiple image formats (JPEG, PNG, GIF, WebP)
    - Configurable model selection and generation parameters
    """

    # Flag to track if dependencies are available
    _dependencies_available = False
    _model = None
    _processor = None
    _model_loaded = False

    def __init__(self, name: str = "qwen_local", params: Dict = None):
        """
        Initialize the Qwen Local VLM data source.

        Args:
            name: Identifier for this data source instance
            params: Configuration dictionary with the following options:
                - connection_name: Friendly name for the connection
                - model_id: HuggingFace model ID (default: Qwen/Qwen2-VL-2B-Instruct)
                - device: Device to run on (auto, cuda, cpu)
                - max_new_tokens: Maximum tokens to generate (default: 1024)
                - temperature: Sampling temperature (default: 0.7)
                - top_p: Top-p sampling parameter (default: 0.9)
                - do_sample: Whether to use sampling (default: True)
                - system_prompt: Optional system prompt
                - load_in_4bit: Use 4-bit quantization (default: False)
                - load_in_8bit: Use 8-bit quantization (default: False)
                - torch_dtype: Data type (auto, float16, bfloat16, float32)
                - trust_remote_code: Trust remote code from HF (default: True)
        """
        super().__init__(name, params)
        self._normalize_params()
        self._check_dependencies()

    def _normalize_params(self):
        """Normalize and validate parameters with safe defaults."""
        if not self.params:
            self.params = {}

        # Model selection - default to smaller 2B model for accessibility
        if not self.params.get("model_id"):
            self.params["model_id"] = "Qwen/Qwen2-VL-2B-Instruct"

        # Device selection
        if not self.params.get("device"):
            self.params["device"] = "auto"

        # Generation parameters
        try:
            max_tokens = int(self.params.get("max_new_tokens", 1024))
            if max_tokens <= 0:
                max_tokens = 1024
        except (ValueError, TypeError):
            max_tokens = 1024
        self.params["max_new_tokens"] = max_tokens

        try:
            temperature = float(self.params.get("temperature", 0.7))
            if temperature < 0.0 or temperature > 2.0:
                temperature = 0.7
        except (ValueError, TypeError):
            temperature = 0.7
        self.params["temperature"] = temperature

        try:
            top_p = float(self.params.get("top_p", 0.9))
            if top_p < 0.0 or top_p > 1.0:
                top_p = 0.9
        except (ValueError, TypeError):
            top_p = 0.9
        self.params["top_p"] = top_p

        # Boolean parameters
        self.params["do_sample"] = self.params.get("do_sample", True)
        self.params["load_in_4bit"] = self.params.get("load_in_4bit", False)
        self.params["load_in_8bit"] = self.params.get("load_in_8bit", False)
        self.params["trust_remote_code"] = self.params.get("trust_remote_code", True)

        # Torch dtype
        if not self.params.get("torch_dtype"):
            self.params["torch_dtype"] = "auto"

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            self._dependencies_available = True
            BBLogger.log("Qwen VLM dependencies are available")
        except ImportError as e:
            self._dependencies_available = False
            BBLogger.log(
                f"Qwen VLM dependencies not available: {e}. "
                "Please install: pip install transformers torch qwen-vl-utils accelerate"
            )

    def _dependency_error_response(self, original_message: Any) -> Dict:
        """Return error response when dependencies are not available."""
        return {
            "error": True,
            "error_type": "dependency_error",
            "message": (
                "Required dependencies not installed. Please install:\n"
                "pip install transformers torch qwen-vl-utils accelerate\n"
                "For GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
                "For quantization: pip install bitsandbytes"
            ),
            "original_message": original_message
        }

    def _load_model(self) -> bool:
        """
        Load the Qwen2-VL model and processor.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True

        if not self._dependencies_available:
            return False

        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

            model_id = self.params.get("model_id", "Qwen/Qwen2-VL-2B-Instruct")
            device = self.params.get("device", "auto")

            BBLogger.log(f"Loading Qwen2-VL model: {model_id}")

            # Determine torch dtype
            torch_dtype_str = self.params.get("torch_dtype", "auto")
            if torch_dtype_str == "float16":
                torch_dtype = torch.float16
            elif torch_dtype_str == "bfloat16":
                torch_dtype = torch.bfloat16
            elif torch_dtype_str == "float32":
                torch_dtype = torch.float32
            else:
                torch_dtype = "auto"

            # Build model loading kwargs
            model_kwargs = {
                "trust_remote_code": self.params.get("trust_remote_code", True),
            }

            # Handle quantization
            if self.params.get("load_in_4bit"):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
            elif self.params.get("load_in_8bit"):
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = torch_dtype

            # Handle device mapping
            if device == "auto":
                model_kwargs["device_map"] = "auto"
            elif device == "cuda":
                if torch.cuda.is_available():
                    model_kwargs["device_map"] = "cuda"
                else:
                    BBLogger.log("CUDA requested but not available, falling back to CPU")
                    model_kwargs["device_map"] = "cpu"
            else:
                model_kwargs["device_map"] = "cpu"

            # Load model and processor
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                **model_kwargs
            )

            self._processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=self.params.get("trust_remote_code", True)
            )

            self._model_loaded = True
            BBLogger.log(f"Qwen2-VL model loaded successfully on device: {self._model.device}")
            return True

        except Exception as e:
            BBLogger.log(f"Error loading Qwen2-VL model: {e}")
            return False

    def _process_message(self, message: Any) -> Dict:
        """
        Process a text-only or mixed message.

        Args:
            message: Either a string or a dict with 'text' and optional 'files' keys

        Returns:
            Response dictionary with success/error status and response text
        """
        # Check dependencies
        if not self._dependencies_available:
            return self._dependency_error_response(message)

        # Load model if not loaded
        if not self._load_model():
            return {
                "error": True,
                "error_type": "model_load_error",
                "message": "Failed to load Qwen2-VL model. Check logs for details.",
                "original_message": message
            }

        # Handle dict message with potential files
        if isinstance(message, dict):
            user_text = message.get("content", message.get("text", message.get("message", "")))
            files = message.get("files", [])

            if files:
                return self._process_message_with_files(user_text, files)
            else:
                return self._process_text_only(user_text)
        else:
            # Simple string message
            return self._process_text_only(str(message))

    def _process_text_only(self, text: str) -> Dict:
        """
        Process a text-only message (no vision).

        Args:
            text: The user's text input

        Returns:
            Response dictionary
        """
        try:
            from qwen_vl_utils import process_vision_info

            BBLogger.log(f"Processing text-only message with Qwen2-VL")

            # Build messages in Qwen format
            messages = []

            # Add system prompt if configured
            system_prompt = self.params.get("system_prompt", "")
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Add user message
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": text}]
            })

            # Prepare inputs
            prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self._processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self._model.device)

            # Generate response
            import torch

            generation_kwargs = {
                "max_new_tokens": self.params.get("max_new_tokens", 1024),
            }

            if self.params.get("do_sample", True):
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = self.params.get("temperature", 0.7)
                generation_kwargs["top_p"] = self.params.get("top_p", 0.9)
            else:
                generation_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    **generation_kwargs
                )

            # Decode response
            generated_ids = [
                output_ids[i][len(inputs.input_ids[i]):]
                for i in range(len(output_ids))
            ]

            response_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            BBLogger.log(f"Qwen2-VL response generated (length: {len(response_text)} chars)")

            return {
                "success": True,
                "response": response_text,
                "model": self.params.get("model_id"),
                "original_message": text
            }

        except Exception as e:
            BBLogger.log(f"Error processing text message: {e}")
            return {
                "error": True,
                "error_type": "processing_error",
                "message": str(e),
                "original_message": text
            }

    def _process_message_with_files(self, text: str, files: List) -> Dict:
        """
        Process a message with image files (vision-language).

        Args:
            text: The user's text input
            files: List of file dictionaries with 'name', 'mime_type', and 'data'

        Returns:
            Response dictionary
        """
        try:
            import torch
            from PIL import Image
            from qwen_vl_utils import process_vision_info

            BBLogger.log(f"Processing vision message with {len(files)} file(s)")

            # Normalize files
            files = self._normalize_files(files)

            # Build content list
            content = []
            images = []

            # Process each file
            for file_info in files:
                mime_type = file_info.get("mime_type", "")

                # Get image data - check both "data_base64" (from abstract datasource) and "data" (legacy)
                image_data = file_info.get("data_base64") or file_info.get("data", "")

                # If no mime_type, try to detect from base64 signature
                if not mime_type and isinstance(image_data, str) and image_data:
                    # Common base64 signatures for image formats
                    if image_data.startswith("iVBORw"):  # PNG
                        mime_type = "image/png"
                    elif image_data.startswith("/9j/"):  # JPEG
                        mime_type = "image/jpeg"
                    elif image_data.startswith("R0lGOD"):  # GIF
                        mime_type = "image/gif"
                    elif image_data.startswith("UklGR"):  # WebP
                        mime_type = "image/webp"

                if mime_type.startswith("image/"):
                    # Process image

                    if isinstance(image_data, str):
                        # Base64 encoded
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(BytesIO(image_bytes))
                    elif isinstance(image_data, bytes):
                        image = Image.open(BytesIO(image_data))
                    else:
                        continue

                    images.append(image)
                    content.append({
                        "type": "image",
                        "image": image
                    })

                elif mime_type.startswith("text/") or file_info.get("name", "").endswith((".txt", ".md", ".py", ".js", ".json")):
                    # Include text file content in the prompt
                    file_text = file_info.get("data", "")
                    if isinstance(file_text, bytes):
                        file_text = file_text.decode("utf-8", errors="ignore")
                    text += f"\n\n[File: {file_info.get('name', 'unknown')}]\n{self._truncate_text(file_text)}"

            # Add the text prompt
            content.append({
                "type": "text",
                "text": text
            })

            # Build messages
            messages = []

            # Add system prompt if configured
            system_prompt = self.params.get("system_prompt", "")
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            messages.append({
                "role": "user",
                "content": content
            })

            # Prepare inputs using Qwen VL utils
            prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self._processor(
                text=[prompt],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self._model.device)

            # Generate response
            generation_kwargs = {
                "max_new_tokens": self.params.get("max_new_tokens", 1024),
            }

            if self.params.get("do_sample", True):
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = self.params.get("temperature", 0.7)
                generation_kwargs["top_p"] = self.params.get("top_p", 0.9)
            else:
                generation_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    **generation_kwargs
                )

            # Decode response
            generated_ids = [
                output_ids[i][len(inputs.input_ids[i]):]
                for i in range(len(output_ids))
            ]

            response_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            BBLogger.log(f"Qwen2-VL vision response generated (length: {len(response_text)} chars)")

            return {
                "success": True,
                "response": response_text,
                "model": self.params.get("model_id"),
                "images_processed": len(images),
                "original_message": text
            }

        except Exception as e:
            BBLogger.log(f"Error processing vision message: {e}")
            import traceback
            BBLogger.log(traceback.format_exc())
            return {
                "error": True,
                "error_type": "vision_processing_error",
                "message": str(e),
                "original_message": text
            }

    def _normalize_files(self, files: Any) -> List[Dict]:
        """Normalize files input to list of dicts."""
        if not files:
            return []
        if not isinstance(files, list):
            return []
        return [f for f in files if isinstance(f, dict)]

    def _truncate_text(self, text: str, max_chars: int = 20000) -> str:
        """Truncate text to maximum characters."""
        if len(text) > max_chars:
            return text[:max_chars] + "\n[truncated]"
        return text

    def _guess_mime_type(self, filename: str) -> str:
        """Guess MIME type from filename."""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"

    def unload_model(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._model_loaded = False

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            BBLogger.log("Qwen2-VL model unloaded and memory cleared")
        except Exception as e:
            BBLogger.log(f"Error clearing CUDA cache: {e}")

    def get_icon(self) -> str:
        """Return the SVG icon for this data source."""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        try:
            with open(icon_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            BBLogger.log(f"Error reading icon file: {e}")
            # Fallback Qwen-style icon
            return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
                <rect width="100" height="100" rx="20" fill="#6366f1"/>
                <text x="50" y="65" font-family="Arial, sans-serif" font-size="40"
                      font-weight="bold" fill="white" text-anchor="middle">Q</text>
            </svg>'''

    def get_connection_data(self) -> dict:
        """
        Return connection configuration metadata for UI/configuration.

        Returns:
            Dictionary with connection type and configurable fields
        """
        return {
            "connection_type": "ON_DEMAND",
            "fields": [
                {
                    "name": "connection_name",
                    "label": "Connection Name",
                    "type": "text",
                    "required": True,
                    "default": "Qwen Local VLM",
                    "description": "Friendly name for this connection"
                },
                {
                    "name": "model_id",
                    "label": "Model",
                    "type": "select",
                    "required": True,
                    "default": "Qwen/Qwen2-VL-2B-Instruct",
                    "options": [
                        {"value": "Qwen/Qwen2-VL-2B-Instruct", "label": "Qwen2-VL 2B Instruct (Recommended for most users)"},
                        {"value": "Qwen/Qwen2-VL-7B-Instruct", "label": "Qwen2-VL 7B Instruct (Better quality, needs more VRAM)"},
                        {"value": "Qwen/Qwen2-VL-72B-Instruct", "label": "Qwen2-VL 72B Instruct (Best quality, needs A100/H100)"},
                        {"value": "Qwen/Qwen2.5-VL-3B-Instruct", "label": "Qwen2.5-VL 3B Instruct"},
                        {"value": "Qwen/Qwen2.5-VL-7B-Instruct", "label": "Qwen2.5-VL 7B Instruct"},
                        {"value": "Qwen/Qwen2.5-VL-32B-Instruct", "label": "Qwen2.5-VL 32B Instruct"},
                        {"value": "Qwen/Qwen2.5-VL-72B-Instruct", "label": "Qwen2.5-VL 72B Instruct"}
                    ],
                    "description": "Qwen2-VL model to use. Larger models need more GPU memory."
                },
                {
                    "name": "device",
                    "label": "Device",
                    "type": "select",
                    "required": False,
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto (Recommended)"},
                        {"value": "cuda", "label": "CUDA (GPU)"},
                        {"value": "cpu", "label": "CPU (Slow)"}
                    ],
                    "description": "Device to run the model on. Auto will use GPU if available."
                },
                {
                    "name": "max_new_tokens",
                    "label": "Max New Tokens",
                    "type": "number",
                    "required": False,
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "description": "Maximum number of tokens to generate in the response"
                },
                {
                    "name": "temperature",
                    "label": "Temperature",
                    "type": "number",
                    "required": False,
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "Controls randomness. Lower = more deterministic, higher = more creative."
                },
                {
                    "name": "top_p",
                    "label": "Top P",
                    "type": "number",
                    "required": False,
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Nucleus sampling parameter. Lower values = more focused responses."
                },
                {
                    "name": "do_sample",
                    "label": "Use Sampling",
                    "type": "checkbox",
                    "required": False,
                    "default": True,
                    "description": "Enable temperature/top_p sampling. Disable for greedy decoding."
                },
                {
                    "name": "system_prompt",
                    "label": "System Prompt",
                    "type": "textarea",
                    "required": False,
                    "default": "",
                    "description": "Optional system instructions for the model"
                },
                {
                    "name": "load_in_4bit",
                    "label": "4-bit Quantization",
                    "type": "checkbox",
                    "required": False,
                    "default": False,
                    "description": "Load model in 4-bit precision (saves VRAM, requires bitsandbytes)"
                },
                {
                    "name": "load_in_8bit",
                    "label": "8-bit Quantization",
                    "type": "checkbox",
                    "required": False,
                    "default": False,
                    "description": "Load model in 8-bit precision (saves VRAM, requires bitsandbytes)"
                },
                {
                    "name": "torch_dtype",
                    "label": "Torch Data Type",
                    "type": "select",
                    "required": False,
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto"},
                        {"value": "float16", "label": "Float16 (Recommended for GPU)"},
                        {"value": "bfloat16", "label": "BFloat16 (For newer GPUs)"},
                        {"value": "float32", "label": "Float32 (CPU or max precision)"}
                    ],
                    "description": "Data type for model weights. Float16 is recommended for GPUs."
                },
                {
                    "name": "trust_remote_code",
                    "label": "Trust Remote Code",
                    "type": "checkbox",
                    "required": False,
                    "default": True,
                    "description": "Allow execution of model-specific code from HuggingFace"
                }
            ]
        }
