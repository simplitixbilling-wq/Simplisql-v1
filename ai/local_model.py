"""
Local LLM Module for SimpliSQL
================================
Loads and runs GGUF models directly in Python via llama-cpp-python.
No external services (Ollama, APIs) required.

Models are loaded only from local files in the models/ directory.

Usage:
    from ai.local_model import LocalModelClient

    client = LocalModelClient()
    client.load_model()  # loads from local models/ directory
    response = client.generate("Write a SQL query to ...")
"""

import os
import sys
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LENGTH = 8192
CONTEXT_FALLBACK_SEQUENCE = (65536, 32768, 18432, 8192)


def _build_context_retry_sequence(primary_ctx: int, configured_ctx_length: int) -> list[int]:
    """Build a descending list of safe retry context sizes."""
    sequence = []
    seen = set()

    for value in [primary_ctx, *CONTEXT_FALLBACK_SEQUENCE, configured_ctx_length, DEFAULT_CONTEXT_LENGTH]:
        try:
            ctx = int(value)
        except (TypeError, ValueError):
            continue
        if ctx < 512 or ctx in seen:
            continue
        seen.add(ctx)
        sequence.append(ctx)

    return sorted(sequence, reverse=True)


class GenerationCancelled(Exception):
    """Raised when a local generation is cancelled by the caller."""

# ── PyInstaller frozen-bundle fix for llama_cpp native DLLs ──────────
# When running as a packaged .exe, sys._MEIPASS points to the _internal
# folder.  llama_cpp expects its DLLs in a 'lib' sub-directory relative
# to the package, so we add that path to os.add_dll_directory() (Win10+)
# and to PATH so LoadLibrary can resolve them.
def _fix_llama_dll_path():
    if not getattr(sys, 'frozen', False):
        return
    base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    lib_dir = os.path.join(base, 'llama_cpp', 'lib')
    if os.path.isdir(lib_dir):
        os.environ['PATH'] = lib_dir + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(lib_dir)   # Python 3.8+ Windows
        except (AttributeError, OSError):
            pass
        logger.info(f"llama_cpp DLL path registered: {lib_dir}")
    else:
        logger.warning(f"llama_cpp lib dir not found in bundle: {lib_dir}")

_fix_llama_dll_path()


def get_models_dir() -> Path:
    """Get the local models directory used by the app.

    Preference order:
    1. V1 local models folder (<V1>/models)
    2. Workspace-level models folder (<workspace>/models) as fallback
       when the V1 local folder has no GGUF files.
    """
    v1_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    v1_models_dir = v1_root / "models"
    v1_models_dir.mkdir(exist_ok=True)

    try:
        if any(v1_models_dir.glob("*.gguf")):
            return v1_models_dir
    except OSError:
        return v1_models_dir

    workspace_models_dir = v1_root.parent.parent / "models"
    try:
        if workspace_models_dir.exists() and any(workspace_models_dir.glob("*.gguf")):
            return workspace_models_dir
    except OSError:
        pass

    return v1_models_dir


def _describe_model_file(model_path: Path) -> str:
    stem = model_path.stem.replace("-", " ").replace("_", " ")
    try:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb / 1024:.1f} GB" if size_mb >= 1024 else f"{size_mb:.0f} MB"
        return f"{stem} (~{size_str})"
    except OSError:
        return stem


def _discover_model_entries() -> list[dict]:
    models_dir = get_models_dir()
    result = []
    for model_path in sorted(models_dir.glob("*.gguf")):
        result.append({
            "key": model_path.name,
            "description": _describe_model_file(model_path),
            "path": str(model_path),
        })
    return result


class LocalModelClient:
    """
    Client that loads GGUF models via llama-cpp-python.
    Loads models only from the local models/ directory.
    """

    def __init__(self):
        self.llm = None
        self.model_name: Optional[str] = None
        self.model_path: Optional[str] = None
        self.context_length: int = 4096
        self._loading = False
        logger.info("LocalModelClient initialized")

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    @staticmethod
    def list_available_models() -> list[dict]:
        """Return metadata for local GGUF models present on disk only."""
        return _discover_model_entries()

    @staticmethod
    def get_default_model_key() -> Optional[str]:
        models = _discover_model_entries()
        return models[0]["key"] if models else None

    def resolve_model_path(self, model_key: Optional[str] = None, progress_callback=None) -> str:
        """
        Resolve the local path for a discovered GGUF model.

        Args:
            model_key: GGUF filename key from list_available_models()
            progress_callback: Optional callable(status_text)

        Returns:
            Local file path to the model.
        """
        resolved_key = model_key or self.get_default_model_key()
        models = {m["key"]: m for m in self.list_available_models()}
        info = models.get(resolved_key or "")
        if info:
            local_path = Path(info["path"])
            if local_path.exists():
                logger.info(f"Model available locally: {local_path}")
                if progress_callback:
                    progress_callback(f"Model already available: {info['description']}")
                return str(local_path)

        message = (
            f"Model file not found for key: {resolved_key}. "
            f"Add a .gguf model to '{get_models_dir()}' to use this model in offline mode."
        )
        logger.warning(message)
        if progress_callback:
            progress_callback("Model file not found locally. Add the GGUF model to the models folder.")
        raise FileNotFoundError(message)

    def load_model(self, model_key: Optional[str] = None, progress_callback=None) -> bool:
        """
        Load a model from local disk into memory.

        Returns True on success, False on failure.
        """
        if self._loading:
            return False

        self._loading = True
        try:
            resolved_key = model_key or self.get_default_model_key()
            if not resolved_key:
                logger.error("No GGUF models found in models folder")
                return False

            model_info = {m["key"]: m for m in self.list_available_models()}.get(resolved_key)
            if model_info is None:
                logger.error(f"Unknown model key: {resolved_key}")
                return False

            configured_ctx_length = DEFAULT_CONTEXT_LENGTH
            model_path = self.resolve_model_path(resolved_key, progress_callback)
            ctx_length = configured_ctx_length

            if progress_callback:
                progress_callback("Loading model into memory...")

            from llama_cpp import Llama

            use_flash_attn = os.environ.get("SIMPLISQL_FLASH_ATTN", "").strip().lower() in {"1", "true", "yes", "on"}

            ctx_length = int(ctx_length) if 'ctx_length' in locals() else DEFAULT_CONTEXT_LENGTH

            requested_ctx = os.environ.get("SIMPLISQL_CONTEXT_LENGTH", "").strip()
            if requested_ctx:
                try:
                    ctx_length = max(512, int(requested_ctx))
                except ValueError:
                    logger.warning(f"Ignoring invalid SIMPLISQL_CONTEXT_LENGTH={requested_ctx!r}")

            if not requested_ctx:
                # Load with n_ctx=1 (minimal KV cache) purely to read model metadata.
                # This is ~identical RAM cost to mmap loading the weights; no full
                # context buffer is allocated, so it's fast and cheap.
                _meta_llm = Llama(
                    model_path=model_path,
                    n_ctx=1,
                    n_threads=os.cpu_count() or 4,
                    flash_attn=use_flash_attn,
                    verbose=False,
                )
                try:
                    model_meta = getattr(getattr(_meta_llm, '_ctx', None), 'model', None)
                    if model_meta is not None and hasattr(model_meta, 'n_ctx_train'):
                        try:
                            train_ctx = int(model_meta.n_ctx_train())
                            if train_ctx > ctx_length:
                                ctx_length = train_ctx
                                logger.info(f"Using model's native context length: {ctx_length}")
                        except Exception:
                            pass
                except Exception:
                    pass

                # If the target context is small enough that n_ctx=1 just needs a
                # reload, close the probe. Otherwise reuse it if ctx matches exactly.
                # In practice we always need to reload with the real n_ctx, so close.
                try:
                    _meta_llm.close()
                except Exception:
                    pass

            last_error = None
            retry_contexts = _build_context_retry_sequence(ctx_length, configured_ctx_length)
            for attempt_index, attempt_ctx in enumerate(retry_contexts):
                try:
                    if attempt_index > 0:
                        logger.warning(
                            f"Retrying {resolved_key} with reduced context={attempt_ctx} after load failure: {last_error}"
                        )
                        if progress_callback:
                            progress_callback(
                                f"Large context load failed; retrying with {attempt_ctx} token context..."
                            )

                    self.llm = Llama(
                        model_path=model_path,
                        n_ctx=attempt_ctx,
                        n_threads=os.cpu_count() or 4,
                        flash_attn=use_flash_attn,
                        verbose=False,
                    )
                    ctx_length = attempt_ctx
                    logger.info(f"Loaded {resolved_key} with context={ctx_length}")
                    break
                except Exception as load_error:
                    last_error = load_error
                    self.llm = None
            else:
                raise last_error
            self.model_name = resolved_key
            self.model_path = model_path
            self.context_length = int(
                getattr(getattr(self.llm, 'context_params', None), 'n_ctx', ctx_length)
            )
            logger.info(f"Model loaded: {resolved_key} (context={self.context_length})")
            if progress_callback:
                progress_callback(f"Model ready: {model_info['description']}")
            return True

        except GenerationCancelled:
            logger.info("Streaming chat cancelled by user.")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            if progress_callback:
                progress_callback(f"Error: {e}")
            return False
        finally:
            self._loading = False

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.llm is not None

    def unload_model(self):
        """Free model from memory."""
        self.llm = None
        self.model_name = None
        self.model_path = None
        logger.info("Model unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                 top_p: float = 0.9, stop: Optional[list[str]] = None) -> str:
        """
        Generate text from the loaded model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Generated text string
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_model() first.")

        try:
            if max_tokens is None or max_tokens <= 0:
                max_tokens = min(512, max(128, self.context_length // 4))
            max_tokens = min(max_tokens, max(128, self.context_length - 256))

            result = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                echo=False,
            )
            text = result["choices"][0]["text"].strip()
            logger.info(f"Generated {len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def chat_streaming(self, messages: list[dict], max_tokens: int = 512,
                       temperature: float = 0.7, token_callback=None,
                       stop_callback=None) -> str:
        """
        Streaming chat generation – calls token_callback(chunk: str) for each token.
        Returns the full concatenated text when done.
        Falls back to non-streaming chat() if streaming is unsupported.
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_model() first.")

        if max_tokens is None or max_tokens <= 0:
            max_tokens = min(512, max(128, self.context_length // 4))
        max_tokens = min(max_tokens, max(128, self.context_length - 256))

        cleaned = []
        system_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                cleaned.append(msg)

        if system_parts and cleaned:
            for i, msg in enumerate(cleaned):
                if msg["role"] == "user":
                    prefix = "\n".join(system_parts)
                    cleaned[i] = {
                        "role": "user",
                        "content": f"[Instructions: {prefix}]\n\n{msg['content']}",
                    }
                    break

        try:
            full_text = []
            if callable(stop_callback) and stop_callback():
                raise GenerationCancelled()
            stream = self.llm.create_chat_completion(
                messages=cleaned,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                if callable(stop_callback) and stop_callback():
                    raise GenerationCancelled()
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_text.append(token)
                    if token_callback:
                        callback_result = token_callback(token)
                        if callback_result is False:
                            raise GenerationCancelled()
            result = "".join(full_text).strip()
            logger.info(f"Streaming chat response: {len(result)} chars")
            return result
        except Exception as e:
            logger.warning(f"Streaming failed ({e}), falling back to non-streaming chat()")
            if callable(stop_callback) and stop_callback():
                raise GenerationCancelled()
            # Fall back – collect full response then emit it as one chunk
            result = self.chat(messages, max_tokens=max_tokens, temperature=temperature)
            if token_callback:
                callback_result = token_callback(result)
                if callback_result is False:
                    raise GenerationCancelled()
            return result

    def chat(self, messages: list[dict], max_tokens: int = 512,
             temperature: float = 0.7) -> str:
        """
        Chat-style generation using message list.

        Args:
            messages: List of {"role": "user"/"assistant"/"system", "content": "..."}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Assistant response text
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_model() first.")

        try:
            if max_tokens is None or max_tokens <= 0:
                max_tokens = min(512, max(128, self.context_length // 4))
            max_tokens = min(max_tokens, max(128, self.context_length - 256))

            # Some models (e.g. Gemma) don't support the 'system' role.
            # Merge any system messages into the first user message.
            cleaned = []
            system_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    system_parts.append(msg["content"])
                else:
                    cleaned.append(msg)

            if system_parts and cleaned:
                # Prepend system context to the first user message
                for i, msg in enumerate(cleaned):
                    if msg["role"] == "user":
                        prefix = "\n".join(system_parts)
                        cleaned[i] = {
                            "role": "user",
                            "content": f"[Instructions: {prefix}]\n\n{msg['content']}",
                        }
                        break

            result = self.llm.create_chat_completion(
                messages=cleaned,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = result["choices"][0]["message"]["content"].strip()
            logger.info(f"Chat response: {len(text)} chars")
            return text
        except Exception as e:
            err_str = str(e).lower()
            if "not supported" in err_str or "system" in err_str or "role" in err_str:
                # Fallback for models (e.g. Gemma) whose jinja2 template rejects
                # create_chat_completion even after system-role merging.
                # Re-format all messages into a single raw prompt and use generate().
                logger.warning(f"create_chat_completion failed ({e}), falling back to raw generate()")
                try:
                    prompt_parts = []
                    if system_parts:
                        prompt_parts.append("[Context]\n" + "\n\n".join(system_parts))
                    for m in cleaned:
                        role_label = "User" if m["role"] == "user" else "Assistant"
                        prompt_parts.append(f"{role_label}: {m['content']}")
                    prompt_parts.append("Assistant:")
                    raw_prompt = "\n\n".join(prompt_parts)
                    fallback = self.llm(
                        prompt=raw_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        echo=False,
                    )
                    text = fallback["choices"][0]["text"].strip()
                    logger.info(f"Fallback generate response: {len(text)} chars")
                    return text
                except Exception as fe:
                    logger.error(f"Fallback generate also failed: {fe}")
                    raise fe
            logger.error(f"Chat generation failed: {e}")
            raise
