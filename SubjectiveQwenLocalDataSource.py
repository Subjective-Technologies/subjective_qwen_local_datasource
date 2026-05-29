"""
Subjective Qwen Local VLM Data Source
=====================================
An on-demand data source plugin for running Qwen2-VL (Vision-Language Model) locally.
Supports both text and image inputs using the Qwen2-VL family of models.
"""

import os
import base64
import hashlib
import json
import mimetypes
import re
from typing import Any, Dict, List, Optional
from io import BytesIO

from subjective_abstract_data_source_package import SubjectiveDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveQwenLocalDataSource(SubjectiveDataSource):
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

    def __init__(self, **kwargs):
        """
        Initialize the Qwen Local VLM data source.

        Supports both v1-style `params=` initialization and v2-style
        `connection=` initialization via the shared base class.
        """
        super().__init__(**kwargs)
        self._normalize_params()
        self._check_dependencies()

    @classmethod
    def connection_schema(cls):
        return {
            "model_id": {
                "type": "select",
                "label": "Model",
                "required": True,
                "default": "Qwen/Qwen2-VL-2B-Instruct",
                "options": [
                    {"value": "Qwen/Qwen2-VL-2B-Instruct", "label": "Qwen2-VL 2B Instruct (Recommended for most users)"},
                    {"value": "Qwen/Qwen2-VL-7B-Instruct", "label": "Qwen2-VL 7B Instruct (Better quality, needs more VRAM)"},
                    {"value": "Qwen/Qwen2-VL-72B-Instruct", "label": "Qwen2-VL 72B Instruct (Best quality, needs A100/H100)"},
                    {"value": "Qwen/Qwen2.5-VL-3B-Instruct", "label": "Qwen2.5-VL 3B Instruct"},
                    {"value": "Qwen/Qwen2.5-VL-7B-Instruct", "label": "Qwen2.5-VL 7B Instruct"},
                    {"value": "Qwen/Qwen2.5-VL-32B-Instruct", "label": "Qwen2.5-VL 32B Instruct"},
                    {"value": "Qwen/Qwen2.5-VL-72B-Instruct", "label": "Qwen2.5-VL 72B Instruct"},
                ],
                "description": "Qwen vision-language model to load locally.",
            },
            "device": {
                "type": "select",
                "label": "Device",
                "required": False,
                "default": "auto",
                "options": [
                    {"value": "auto", "label": "Auto (Recommended)"},
                    {"value": "cuda", "label": "CUDA (GPU)"},
                    {"value": "cpu", "label": "CPU (Slow)"},
                ],
                "description": "Device to run the model on.",
            },
            "max_new_tokens": {
                "type": "number",
                "label": "Max New Tokens",
                "required": False,
                "default": 1024,
                "description": "Maximum number of tokens to generate.",
            },
            "temperature": {
                "type": "number",
                "label": "Temperature",
                "required": False,
                "default": 0.7,
                "description": "Sampling temperature.",
            },
            "top_p": {
                "type": "number",
                "label": "Top P",
                "required": False,
                "default": 0.9,
                "description": "Top-p sampling parameter.",
            },
            "do_sample": {
                "type": "checkbox",
                "label": "Use Sampling",
                "required": False,
                "default": True,
                "description": "Enable temperature/top_p sampling.",
            },
            "system_prompt": {
                "type": "textarea",
                "label": "System Prompt",
                "required": False,
                "default": "",
                "description": "Optional system instructions for the model.",
            },
            "load_in_4bit": {
                "type": "checkbox",
                "label": "4-bit Quantization",
                "required": False,
                "default": False,
                "description": "Load the model in 4-bit precision.",
            },
            "load_in_8bit": {
                "type": "checkbox",
                "label": "8-bit Quantization",
                "required": False,
                "default": False,
                "description": "Load the model in 8-bit precision.",
            },
            "torch_dtype": {
                "type": "select",
                "label": "Torch Data Type",
                "required": False,
                "default": "auto",
                "options": [
                    {"value": "auto", "label": "Auto"},
                    {"value": "float16", "label": "Float16 (Recommended for GPU)"},
                    {"value": "bfloat16", "label": "BFloat16 (For newer GPUs)"},
                    {"value": "float32", "label": "Float32 (CPU or max precision)"},
                ],
                "description": "Data type for model weights.",
            },
            "trust_remote_code": {
                "type": "checkbox",
                "label": "Trust Remote Code",
                "required": False,
                "default": True,
                "description": "Allow execution of model-specific code from HuggingFace.",
            },
        }

    @classmethod
    def request_schema(cls):
        return {
            "message": {
                "type": "text",
                "label": "Message",
                "required": False,
                "description": "Prompt to send to the local Qwen model.",
            },
        }

    @classmethod
    def output_schema(cls):
        return {
            "success": {"type": "bool", "label": "Success"},
            "response": {"type": "textarea", "label": "Response"},
            "model": {"type": "text", "label": "Model"},
            "images_processed": {"type": "int", "label": "Images Processed"},
            "image_paths_processed": {"type": "json", "label": "Image Paths Processed"},
            "image_sha256s": {"type": "json", "label": "Image SHA256 Hashes"},
            "original_message": {"type": "textarea", "label": "Original Message"},
            "error": {"type": "bool", "label": "Error"},
            "error_type": {"type": "text", "label": "Error Type"},
            "message": {"type": "textarea", "label": "Message"},
        }

    @classmethod
    def icon(cls) -> str:
        """Return the SVG icon for this data source."""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        try:
            with open(icon_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            BBLogger.log(f"Error reading icon file: {e}")
            return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
                <rect width="100" height="100" rx="20" fill="#6366f1"/>
                <text x="50" y="65" font-family="Arial, sans-serif" font-size="40"
                      font-weight="bold" fill="white" text-anchor="middle">Q</text>
            </svg>'''

    def supports_chat(self) -> bool:
        return True

    def handle_message(self, message, files=None):
        if files:
            if isinstance(message, dict):
                payload = dict(message)
                payload["files"] = files
            else:
                payload = {
                    "content": "" if message is None else str(message),
                    "files": files,
                }
            return self._handle_request(payload)
        return self._handle_request(message)

    def run(self, request):
        if isinstance(request, dict):
            # Pipelines commonly wire a screenshot/image producer as `image_path`
            # (single str) or `image_paths` (list[str]). The vision code path
            # only consumes `files`, so bridge here so existing pipes that use
            # `image_path` actually get vision inference instead of silently
            # falling through to text-only.
            request = self._bridge_image_paths(request)
            # When the producer skipped this tick (dedupe), `image_path` is
            # the empty string. Short-circuit so we don't waste a Qwen call
            # producing deterministic template text on no image.
            if self._request_signals_skipped_image(request):
                return self._skipped_image_response(request)
            return self.handle_message(request, files=request.get("files"))
        return self.handle_message(request)

    def _bridge_image_paths(self, request: Dict) -> Dict:
        if request.get("files"):
            return request
        raw = request.get("image_paths")
        if raw is None:
            raw = request.get("image_path")
        if isinstance(raw, str):
            paths = [raw] if raw else []
        elif isinstance(raw, list):
            paths = [p for p in raw if isinstance(p, str) and p]
        else:
            paths = []
        files = []
        for p in paths:
            try:
                if not os.path.isfile(p):
                    continue
                with open(p, "rb") as fh:
                    raw = fh.read()
                files.append({
                    "name": os.path.basename(p),
                    "mime_type": self._guess_mime_type(p),
                    "data_base64": base64.b64encode(raw).decode("ascii"),
                    "source_path": os.path.abspath(p),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                })
            except Exception as exc:
                BBLogger.log(f"image_path bridge failed for {p}: {exc}")
        if files:
            request = dict(request)
            request["files"] = files
        return request

    @staticmethod
    def _request_signals_skipped_image(request: Dict) -> bool:
        if request.get("files"):
            return False
        if "image_path" in request and not request.get("image_path"):
            return True
        if "image_paths" in request:
            raw = request.get("image_paths")
            if not raw:
                return True
            if isinstance(raw, list) and not any(isinstance(p, str) and p for p in raw):
                return True
        return False

    @staticmethod
    def _skipped_image_response(request: Dict) -> Dict:
        # Emit a syntactically-valid WorkAtom that downstream
        # `isPlaceholderEpisode` recognises as low-signal (empty primary_task
        # and dense_summary) so it doesn't poison the episode accumulator.
        payload = (
            '{"work_atom_id":"atom_skipped","timestamp":"","source_type":"screenshot",'
            '"work_relevance":"low_signal","rankable":false,"primary_task":"",'
            '"subtask":"","workflow_stage":"","user_intent":"","applications":[],'
            '"business_objects":[],"domain_knowledge":[],"procedural_skills":[],'
            '"dense_summary":"","uncertainty":{"low_signal":true,"mixed_content":false,'
            '"non_work_signals":[]},"skipped":true}'
        )
        return {
            "success": True,
            "response": payload,
            "model": "skipped",
            "original_message": request.get("message") or request.get("content") or "",
            "skipped": True,
        }

    def _normalize_params(self):
        """Normalize and validate parameters with safe defaults."""
        conn = getattr(self, "_connection", {}) or {}
        if not isinstance(self.params, dict):
            self.params = {}

        # Model selection - default to smaller 2B model for accessibility
        if not conn.get("model_id") and not self.params.get("model_id"):
            self.params["model_id"] = "Qwen/Qwen2-VL-2B-Instruct"
        elif conn.get("model_id"):
            self.params["model_id"] = conn["model_id"]

        # Device selection
        if not conn.get("device") and not self.params.get("device"):
            self.params["device"] = "auto"
        elif conn.get("device"):
            self.params["device"] = conn["device"]

        # Generation parameters - conn value takes priority
        try:
            max_tokens = int(conn.get("max_new_tokens") or self.params.get("max_new_tokens", 1024))
            if max_tokens <= 0:
                max_tokens = 1024
        except (ValueError, TypeError):
            max_tokens = 1024
        self.params["max_new_tokens"] = max_tokens

        try:
            temperature = float(conn.get("temperature") or self.params.get("temperature", 0.7))
            if temperature < 0.0 or temperature > 2.0:
                temperature = 0.7
        except (ValueError, TypeError):
            temperature = 0.7
        self.params["temperature"] = temperature

        try:
            top_p = float(conn.get("top_p") or self.params.get("top_p", 0.9))
            if top_p < 0.0 or top_p > 1.0:
                top_p = 0.9
        except (ValueError, TypeError):
            top_p = 0.9
        self.params["top_p"] = top_p

        # Boolean parameters - conn takes priority
        self.params["do_sample"] = conn.get("do_sample") if "do_sample" in conn else self.params.get("do_sample", True)
        self.params["load_in_4bit"] = conn.get("load_in_4bit") if "load_in_4bit" in conn else self.params.get("load_in_4bit", False)
        self.params["load_in_8bit"] = conn.get("load_in_8bit") if "load_in_8bit" in conn else self.params.get("load_in_8bit", False)
        self.params["trust_remote_code"] = conn.get("trust_remote_code") if "trust_remote_code" in conn else self.params.get("trust_remote_code", True)

        # Torch dtype
        if not conn.get("torch_dtype") and not self.params.get("torch_dtype"):
            self.params["torch_dtype"] = "auto"
        elif conn.get("torch_dtype"):
            self.params["torch_dtype"] = conn["torch_dtype"]

        # System prompt
        if "system_prompt" in conn:
            self.params["system_prompt"] = conn["system_prompt"]

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

    def _handle_request(self, message: Any) -> Dict:
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
            response_text = self._coerce_workatom_response(text, response_text)

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
            response_text = self._coerce_workatom_response(text, response_text)

            BBLogger.log(f"Qwen2-VL vision response generated (length: {len(response_text)} chars)")

            return {
                "success": True,
                "response": response_text,
                "model": self.params.get("model_id"),
                "images_processed": len(images),
                "image_paths_processed": [
                    f.get("source_path") for f in files
                    if isinstance(f, dict) and f.get("source_path")
                ],
                "image_sha256s": [
                    f.get("sha256") for f in files
                    if isinstance(f, dict) and f.get("sha256")
                ],
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

    @classmethod
    def _coerce_workatom_response(cls, prompt: str, response: str) -> str:
        if '"work_atom_id"' not in str(prompt or ""):
            return response
        extracted = cls._extract_json_object(response)
        if extracted:
            return cls._normalize_workatom_payload(extracted)

        summary = str(response or "").strip()
        non_work = cls._looks_non_work(summary)
        payload = {
            "work_atom_id": "atom_001",
            "timestamp": "",
            "source_type": "screenshot",
            "work_relevance": "non_work" if non_work else "possibly_work_related",
            "rankable": False if non_work else True,
            "primary_task": summary,
            "subtask": "",
            "workflow_stage": "",
            "user_intent": "",
            "applications": [],
            "business_objects": [],
            "domain_knowledge": [],
            "procedural_skills": [],
            "dense_summary": summary,
            "uncertainty": {
                "low_signal": not bool(summary),
                "mixed_content": False,
                "non_work_signals": ["entertainment_media"] if non_work else [],
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def _normalize_workatom_payload(cls, raw_json: str) -> str:
        try:
            payload = json.loads(raw_json)
        except Exception:
            return raw_json
        if not isinstance(payload, dict):
            return raw_json
        signal_text = " ".join(
            str(payload.get(key) or "")
            for key in ("primary_task", "dense_summary", "user_intent", "subtask")
        )
        apps = payload.get("applications")
        if isinstance(apps, list):
            signal_text += " " + " ".join(str(item) for item in apps)
        if cls._looks_non_work(signal_text):
            payload["work_relevance"] = "non_work"
            payload["rankable"] = False
            uncertainty = payload.get("uncertainty")
            if not isinstance(uncertainty, dict):
                uncertainty = {}
            signals = uncertainty.get("non_work_signals")
            if not isinstance(signals, list):
                signals = []
            if "entertainment_media" not in signals:
                signals.append("entertainment_media")
            uncertainty["non_work_signals"] = signals
            uncertainty.setdefault("low_signal", False)
            uncertainty.setdefault("mixed_content", False)
            payload["uncertainty"] = uncertainty
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _looks_non_work(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(
            needle in lowered
            for needle in (
                "youtube",
                "video",
                "music",
                "movie",
                "game",
                "social media",
                "entertainment",
            )
        )

    @staticmethod
    def _extract_json_object(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            raw = fenced.group(1).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        start = raw.find("{")
        if start < 0:
            return ""
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return json.dumps(parsed, ensure_ascii=False)
                    except Exception:
                        return ""
        return ""

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
