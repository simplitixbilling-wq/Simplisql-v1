"""
AI Assistant Dialog for SimpliSQL â€“ Local GGUF Edition
=======================================================
Runs models entirely in-process via llama-cpp-python.
No Ollama, no cloud APIs, no external services.

Models are loaded only from local GGUF files in the models/ folder.
"""

import os
import json
import logging
import re
import html
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QComboBox, QCheckBox, QFrame, QTabWidget, QWidget,
    QListWidget, QListWidgetItem, QMessageBox, QApplication, QProgressBar,
    QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor

from ai.local_model import LocalModelClient, GenerationCancelled

logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Auto_Workflow", "ai_config.json")
SAFE_AI_CONFIG_KEYS = {"selected_provider", "default_model"}


def _load_ai_config() -> dict:
    try:
        with open(CONFIG_FILE, "r") as f:
            raw = json.load(f)
            if isinstance(raw, dict):
                return {k: raw[k] for k in SAFE_AI_CONFIG_KEYS if k in raw}
    except Exception:
        return {}
    return {}


def _save_ai_config(cfg: dict):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    safe_cfg = {}
    if isinstance(cfg, dict):
        safe_cfg = {k: cfg[k] for k in SAFE_AI_CONFIG_KEYS if k in cfg}
    with open(CONFIG_FILE, "w") as f:
        json.dump(safe_cfg, f, indent=2)


def _strip_ai_diagnostics(response: str) -> str:
    if not response:
        return ""

    markers = [
        "\n\n---\nðŸ” **Self-check correction:**",
        "\n\n---\nðŸ”§ Issues detected:",
        "\n\n---\nðŸ”Ž Schema checks:",
    ]
    end = len(response)
    for marker in markers:
        idx = response.find(marker)
        if idx != -1:
            end = min(end, idx)
    return response[:end]


def _extract_sql_candidates_from_text(response: str) -> list:
    response = _strip_ai_diagnostics(response)
    candidates = []

    sql_blocks = re.findall(r"```sql\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    candidates.extend([b.strip() for b in sql_blocks if b.strip()])

    if not candidates:
        generic_blocks = re.findall(r"```\s*(.*?)```", response, re.DOTALL)
        for block in generic_blocks:
            if re.search(r"\b(SELECT|WITH)\b", block, re.IGNORECASE):
                candidates.append(block.strip())

    if not candidates:
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        statement = []
        capture = False
        for ln in lines:
            up = ln.upper()
            if not capture and (up.startswith("SELECT") or up.startswith("WITH")):
                capture = True
            if capture:
                statement.append(ln)
                if ln.endswith(";"):
                    break
        if statement:
            candidates.append("\n".join(statement))

    return candidates


def _normalize_sql_text(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip().rstrip(";")).upper()


def _estimate_text_tokens(text: str) -> int:
    return max(1, len((text or "")) // 4)


def _merge_with_overlap(base: str, extra: str, max_overlap: int = 800) -> str:
    """Merge two response chunks while removing repeated overlap at the boundary."""
    if not extra:
        return base or ""
    if not base:
        return extra

    left = base
    right = extra
    max_len = min(len(left), len(right), max_overlap)
    for size in range(max_len, 31, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return left + right


def _has_unclosed_code_fence(text: str) -> bool:
    return (text or "").count("```") % 2 == 1


def _sql_candidate_complete(sql_text: str) -> bool:
    s = (sql_text or "").strip()
    if not s:
        return False
    if s.count("(") > s.count(")"):
        return False
    if re.search(r"(?i)(,|\b(AND|OR|SELECT|FROM|WHERE|JOIN|ON|GROUP\s+BY|ORDER\s+BY|HAVING|QUALIFY|UNION|WITH))\s*$", s):
        return False
    return True


def _response_needs_continuation(response: str, max_tokens: int, mode_hint: str = "sql") -> bool:
    """Enhanced diagnostic heuristic to catch local LLM token truncation."""
    cleaned = _strip_ai_diagnostics(response or "").strip()
    if not cleaned:
        return False

    # Guardrail 1: Code fence is explicitly cut off mid-air
    if _has_unclosed_code_fence(cleaned):
        return True

    # Guardrail 2: Token Proximity Safety Check
    # If the model filled up more than 85% of its allowed chunk budget, it likely choked
    estimated_output_tokens = _estimate_text_tokens(cleaned)
    if estimated_output_tokens >= int(max_tokens * 0.85):
        # If it finished with a formal SQL semicolon, it's genuinely done
        if cleaned.endswith(";"):
            return False
        return True

    # Guardrail 3: Legacy structural heuristics
    mode = (mode_hint or "sql").lower()
    sql_candidates = _extract_sql_candidates_from_text(cleaned)

    if mode == "sql":
        if sql_candidates:
            return not _sql_candidate_complete(sql_candidates[-1])
        if re.search(r"(?i)\b(select|with)\b", cleaned):
            tail = cleaned[-200:]
            if re.search(r"(?i)(,|\b(AND|OR|SELECT|FROM|WHERE|JOIN|ON|GROUP\s+BY|ORDER\s+BY|HAVING|QUALIFY|UNION|WITH))\s*$", tail):
                return True
        return False

    if sql_candidates:
        return not _sql_candidate_complete(sql_candidates[-1])

    return False


def _extract_requested_min_lines(user_text: str) -> int:
    """Parse explicit user requests like 'more than 150 lines' from text."""
    text = (user_text or "").lower()
    if not text:
        return 0

    matches = []
    patterns = [
        r"(?:at\s*least|minimum|min\.?|over|more\s+than)\s*(\d{2,4})\s*line(?:s)?",
        r"\b(\d{2,4})\s*[- ]?line(?:s)?\b",
    ]
    for pattern in patterns:
        for m in re.findall(pattern, text, flags=re.IGNORECASE):
            try:
                value = int(m)
                if 20 <= value <= 1500:
                    matches.append(value)
            except Exception:
                continue

    if not matches:
        return 0
    return max(matches)


def _largest_sql_candidate_line_count(response: str) -> int:
    candidates = _extract_sql_candidates_from_text(_strip_ai_diagnostics(response or ""))
    if not candidates:
        return 0
    best = max(candidates, key=lambda c: len(c or ""))
    return len([ln for ln in best.splitlines() if ln.strip()])


def _needs_more_sql_lines(response: str, requested_min_lines: int) -> bool:
    if requested_min_lines <= 0:
        return False
    return _largest_sql_candidate_line_count(response) < requested_min_lines


# â”€â”€ Background threads â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ModelLoaderThread(QThread):
    """Load a model without freezing the UI."""
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal()
    finished_err = pyqtSignal(str)

    def __init__(self, client: LocalModelClient, model_key: str):
        super().__init__()
        self.client = client
        self.model_key = model_key

    def run(self):
        try:
            ok = self.client.load_model(self.model_key, progress_callback=self.progress.emit)
            if ok:
                self.finished_ok.emit()
            else:
                self.finished_err.emit("Failed to load model.")
        except Exception as e:
            self.finished_err.emit(str(e))


class AIChatThread(QThread):
    """Generate a response in the background, then self-validate."""
    response_ready = pyqtSignal(str)
    token_ready = pyqtSignal(str)   # emitted per token during streaming
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        client: LocalModelClient,
        messages: list,
        user_request: str = "",
        answer_mode: str = "auto",
    ):
        super().__init__()
        self.client = client
        self.messages = messages
        self.user_request = user_request
        self.answer_mode = "sql"
        self.requested_min_lines = 0
        self.requested_min_lines = _extract_requested_min_lines(self.user_request)

    def _build_continuation_messages(self, accumulated_text: str, ctx: int) -> list:
        tail_chars = max(2000, min(24000, ctx * 4))
        assistant_tail = (accumulated_text or "")[-tail_chars:]
        continuation_prompt = (
            "You ran out of output space. Continue generating the exact DuckDB SQL query "
            "above from the very last character without repeating any existing text.\n"
            "Do not start a new query block. Do not provide explanations. Output ONLY the "
            "remaining valid SQL code required to finish the complete query."
        )

        continuation_messages = []
        user_request = (self.user_request or "").strip()
        if user_request:
            continuation_messages.append({"role": "user", "content": user_request})
        continuation_messages.append({"role": "assistant", "content": assistant_tail})
        continuation_messages.append({"role": "user", "content": continuation_prompt})
        return continuation_messages

    def run(self):
        try:
            # Calculate max_tokens from both context window and estimated prompt size.
            # This avoids output truncation/failures for large SQL generations.
            ctx = getattr(self.client, 'context_length', 4096)
            prompt_tokens = 0
            for msg in self.messages:
                prompt_tokens += max(1, len((msg or {}).get("content", "")) // 4) + 20

            reserve_tokens = 128
            available_for_output = max(0, ctx - prompt_tokens - reserve_tokens)
            desired_output_tokens = max(512, min(4096, int(ctx * 0.55)))

            if available_for_output < 64:
                self.error_occurred.emit(
                    "Prompt is too large for the selected model context. "
                    "Reduce selected tables/context, then try again."
                )
                return

            main_max_tokens = max(64, min(desired_output_tokens, available_for_output))

            # Pass 1: Stream the response token-by-token
            def _on_token(token: str):
                if self.isInterruptionRequested():
                    return False
                self.token_ready.emit(token)
                return True

            text = self.client.chat_streaming(
                self.messages,
                max_tokens=main_max_tokens,
                temperature=0.3,
                token_callback=_on_token,
                stop_callback=self.isInterruptionRequested,
            )
            if not text:
                self.error_occurred.emit("Model returned an empty response.")
                return

            # Auto-continuation for long responses: keep requesting follow-up chunks
            # while the output appears truncated by token limits.
            combined_text = text
            max_continuations = 40
            if self.requested_min_lines >= 150:
                max_continuations = 50
            continuation_count = 0

            while continuation_count < max_continuations:
                if self.isInterruptionRequested():
                    return

                needs_completion = _response_needs_continuation(
                    combined_text,
                    main_max_tokens,
                    self.answer_mode,
                )
                needs_line_target = _needs_more_sql_lines(combined_text, self.requested_min_lines)

                if not (needs_completion or needs_line_target):
                    break

                continuation_count += 1
                continuation_messages = self._build_continuation_messages(combined_text, ctx)

                cont_prompt_tokens = 0
                for msg in continuation_messages:
                    cont_prompt_tokens += _estimate_text_tokens((msg or {}).get("content", "")) + 20

                cont_reserve_tokens = 96
                cont_available = max(0, ctx - cont_prompt_tokens - cont_reserve_tokens)
                if cont_available < 64:
                    break

                cont_desired = max(256, min(4096, int(ctx * 0.45)))
                continuation_max_tokens = max(64, min(cont_desired, cont_available))

                continuation_text = self.client.chat_streaming(
                    continuation_messages,
                    max_tokens=continuation_max_tokens,
                    temperature=0.2,
                    # We'll receive the continuation_text as a whole and
                    # stream only the truly new tail to the UI so the
                    # user sees progress without duplicating overlap.
                    token_callback=None,
                    stop_callback=self.isInterruptionRequested,
                )
                if not continuation_text:
                    break

                if continuation_text.strip() == "<DONE>":
                    break

                # Merge while computing the newly added tail so we can
                # stream only the delta to the UI.
                merged_text = _merge_with_overlap(combined_text, continuation_text)
                if merged_text == combined_text:
                    break
                # The new content appended by this continuation
                new_tail = merged_text[len(combined_text):]
                combined_text = merged_text

                # Emit the newly added tail in small chunks so UI updates
                # appear incrementally. Use whitespace-aware chunking.
                import re
                parts = re.findall(r"\s+|\S+", new_tail)
                for part in parts:
                    if self.isInterruptionRequested():
                        return
                    try:
                        self.token_ready.emit(part)
                    except Exception:
                        # If the receiver disappeared or UI is closing, abort.
                        return
                main_max_tokens = continuation_max_tokens

            text = combined_text

            # Check for DuckDB-specific syntax errors and auto-fix
            sql_candidates = _extract_sql_candidates_from_text(text)
            sql_text = "\n".join(sql_candidates)
            if sql_text:
                error_fixes = []
                text_upper = sql_text.upper()
                
                if "WHERE" in text_upper and "ROW_NUMBER()" in text_upper:
                    if "QUALIFY" not in text_upper:
                        error_fixes.append(
                            "âš ï¸ WHERE clause with window function detected. "
                            "DuckDB requires QUALIFY instead of WHERE."
                        )
                
                # Detect and auto-fix QUALIFY used without any window function
                has_qualify = "QUALIFY" in text_upper
                has_over = " OVER " in text_upper or " OVER(" in text_upper
                if has_qualify and not has_over:
                    # Extract SQL blocks and fix them
                    def fix_qualify(match):
                        sql = match.group(1)
                        # Remove QUALIFY clause (it's invalid without window function)
                        fixed = re.sub(
                            r'\s*QUALIFY\b[^;]*?(?=\s*(?:GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING|;|$))',
                            '',
                            sql,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        return f"```sql\n{fixed.strip()}\n```"
                    
                    # Try to fix SQL in code blocks
                    fixed_text = re.sub(
                        r'```sql\s*\n([\s\S]*?)\n```',
                        fix_qualify,
                        text,
                        flags=re.IGNORECASE,
                    )
                    if fixed_text != text:
                        text = fixed_text
                        error_fixes.append(
                            "ðŸ”§ Auto-fixed: Removed invalid QUALIFY clause (QUALIFY requires a window function like ROW_NUMBER() OVER (...))."
                        )
                    else:
                        error_fixes.append(
                            "âš ï¸ QUALIFY used without a window function. "
                            "QUALIFY only works with window functions (e.g. ROW_NUMBER() OVER (...)). "
                            "Use WHERE or HAVING for regular filters."
                        )
                
                if error_fixes:
                    text = text + "\n\n---\nðŸ”§ Issues detected:\n" + "\n".join(error_fixes)

            self.response_ready.emit(text)
        except GenerationCancelled:
            return
        except Exception as e:
            self.error_occurred.emit(str(e))


# â”€â”€ Dialog â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AIAssistantDialog(QDialog):
    """AI Assistant dialog â€“ fully local, no external services."""

    def __init__(self, parent_editor, shared_client=None):
        super().__init__(parent_editor)
        self.parent_editor = parent_editor
        self.current_conversation = []
        self.setWindowTitle("AI Assistant (Local Model)")

        # Core client
        self.client = shared_client if shared_client is not None else LocalModelClient()
        self._force_close = False
        self._chat_thread = None
        self._generation_cancelled = False
        self._last_answer_mode = "sql"

        # Load saved default model preference
        cfg = _load_ai_config()
        self.selected_model_key = cfg.get("default_model") or self.client.get_default_model_key()

        # Window setup
        self.setMinimumSize(760, 520)
        screen = QApplication.primaryScreen().availableGeometry()
        width = min(1100, screen.width() - 80)
        height = min(760, screen.height() - 80)
        self.resize(width, height)
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Center on parent or screen
        parent = self.parent()
        if parent is not None:
            parent_geo = parent.frameGeometry()
            self_geo = self.frameGeometry()
            self_geo.moveCenter(parent_geo.center())
            self.move(self_geo.topLeft())
        else:
            screen_geo = QApplication.primaryScreen().availableGeometry()
            self.move(
                screen_geo.center().x() - self.width() // 2,
                screen_geo.center().y() - self.height() // 2
            )

        self._build_ui()
        self._refresh_table_context_list()
        self._refresh_model_status()

        # Sync the chat tab combo to the saved default
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == self.selected_model_key:
                self.model_combo.setCurrentIndex(i)
                break

        # Auto-load default model in background on startup
        if not self.client.is_loaded() and not getattr(self.client, "_loading", False):
            self._auto_load_default()

    # â”€â”€ UI construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #ccc; }
            QTabBar::tab { padding: 12px 24px; font-size: 14px; font-weight: bold; }
            QTabBar::tab:selected { background-color: #4caf50; color: white; }
        """)

        self.tab_widget.addTab(self._chat_tab(), "Chat")
        self.tab_widget.addTab(self._settings_tab(), "Settings")
        layout.addWidget(self.tab_widget)

        # Visible status strip so users can tell whether generation is running or done.
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self._set_status_badge("Ready", "ready")
        layout.addWidget(self.status_label)

        self._generation_started_at = None
        self._generation_status_timer = QTimer(self)
        self._generation_status_timer.setInterval(500)
        self._generation_status_timer.timeout.connect(self._update_generation_status_tick)

        self.apply_theme()

    # â”€â”€ Chat tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _chat_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(15, 15, 15, 15)
        lay.setSpacing(10)

        # Model row
        model_frame = QFrame()
        model_frame.setFrameShape(QFrame.Shape.StyledPanel)
        ml = QHBoxLayout(model_frame)
        ml.setSpacing(10)
        ml.addWidget(QLabel("<b>Model:</b>"))
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        ml.addWidget(self.model_combo)

        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self._load_selected_model)
        self.load_btn.setStyleSheet(
            "QPushButton{background:#4caf50;color:white;font-weight:bold;padding:7px 16px 9px 16px;border-radius:4px;border:1px solid #2e7d32;border-bottom:3px solid #2e7d32;}"
            "QPushButton:hover{background:#43a047;}"
            "QPushButton:pressed{background:#388e3c;padding:8px 16px 8px 16px;border-bottom:1px solid #2e7d32;}"
            "QPushButton:disabled{background:#a5d6a7;color:#e8f5e9;}"
        )
        ml.addWidget(self.load_btn)
        ml.addWidget(QLabel("|"))

        ml.addWidget(QLabel("Answer mode:"))
        self.answer_mode_combo = QComboBox()
        self.answer_mode_combo.addItem("SQL Only", "sql")
        self.answer_mode_combo.setCurrentIndex(0)
        self.answer_mode_combo.setEnabled(False)
        self.answer_mode_combo.setToolTip(
            "This assistant is locked to SQL generation."
        )
        self.answer_mode_combo.setStyleSheet("padding:4px 8px; font-size:12px;")
        ml.addWidget(self.answer_mode_combo)

        self.auto_paste_check = QCheckBox("Auto-paste")
        self.auto_paste_check.setChecked(True)
        self.auto_paste_check.setToolTip(
            "When checked, generated SQL is automatically pasted to the SQL notepad."
        )
        ml.addWidget(self.auto_paste_check)
        ml.addStretch()
        lay.addWidget(model_frame)

        # Chat display
        lay.addWidget(QLabel("<b style='font-size:16px'>Chat</b>"))
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(400)
        self.chat_display.setStyleSheet(
            "QTextEdit { background:#fff; color:#000; border:2px solid #ddd; "
            "border-radius:6px; padding:12px; font-size:14px; font-family:'Segoe UI',Arial; }"
        )
        lay.addWidget(self.chat_display, stretch=1)

        # Input row
        inp = QHBoxLayout()
        inp.setSpacing(10)
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Ask about SQL, data, or workflows...")
        self.user_input.setStyleSheet("font-size:13px; padding:8px; border:2px solid #ddd; border-radius:4px;")
        self.user_input.returnPressed.connect(self.send_message)
        inp.addWidget(self.user_input)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_message)
        send_btn.setStyleSheet(
            "QPushButton{background:#4caf50;color:white;font-weight:bold;padding:7px 16px 9px 16px;border-radius:4px;border:1px solid #2e7d32;border-bottom:3px solid #2e7d32;}"
            "QPushButton:hover{background:#43a047;}"
            "QPushButton:pressed{background:#388e3c;padding:8px 16px 8px 16px;border-bottom:1px solid #2e7d32;}"
            "QPushButton:disabled{background:#a5d6a7;color:#e8f5e9;}"
        )
        send_btn.setObjectName("send_btn")
        inp.addWidget(send_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#f44336;color:white;font-weight:bold;padding:7px 16px 9px 16px;border-radius:4px;border:1px solid #b71c1c;border-bottom:3px solid #b71c1c;}"
            "QPushButton:hover{background:#e53935;}"
            "QPushButton:pressed{background:#c62828;padding:8px 16px 8px 16px;border-bottom:1px solid #b71c1c;}"
            "QPushButton:disabled{background:#ef9a9a;color:#fff;}"
        )
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        inp.addWidget(self.stop_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_chat)
        clear_btn.setStyleSheet(
            "QPushButton{padding:7px 12px 9px 12px;border-radius:4px;background:#e0e0e0;color:#333;border:1px solid #9e9e9e;border-bottom:3px solid #9e9e9e;}"
            "QPushButton:hover{background:#bdbdbd;}"
            "QPushButton:pressed{background:#9e9e9e;padding:8px 12px 8px 12px;border-bottom:1px solid #9e9e9e;}"
            "QPushButton:disabled{background:#f5f5f5;color:#aaa;}"
        )
        clear_btn.setObjectName("clear_btn")
        inp.addWidget(clear_btn)

        copy_btn = QPushButton("Copy Last SQL")
        copy_btn.clicked.connect(self._copy_last_sql_to_editor)
        copy_btn.setStyleSheet(
            "QPushButton{padding:7px 12px 9px 12px;border-radius:4px;background:#e0e0e0;color:#333;border:1px solid #9e9e9e;border-bottom:3px solid #9e9e9e;}"
            "QPushButton:hover{background:#bdbdbd;}"
            "QPushButton:pressed{background:#9e9e9e;padding:8px 12px 8px 12px;border-bottom:1px solid #9e9e9e;}"
            "QPushButton:disabled{background:#f5f5f5;color:#aaa;}"
        )
        copy_btn.setObjectName("copy_btn")
        inp.addWidget(copy_btn)

        lay.addLayout(inp)

        return w

    # â”€â”€ Settings tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _settings_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        lay.addWidget(QLabel("<b style='font-size:18px'>Settings</b>"))

        # Internal model status label used by refresh/load logic.
        # Intentionally not added to Settings UI.
        self.model_info_label = QLabel("Status: checking...")

        # Context options
        cf = QFrame()
        cf.setFrameShape(QFrame.Shape.StyledPanel)
        cl = QVBoxLayout(cf)
        cl.addWidget(QLabel("<b>Context Options</b>"))
        self.context_query_check = QCheckBox("Include current SQL query as context")
        cl.addWidget(self.context_query_check)

        cl.addWidget(QLabel("<b>Tables To Include In AI Analysis</b>"))
        self.table_context_list = QListWidget()
        self.table_context_list.setMaximumHeight(140)
        
        # Apply stylesheet for checkbox visibility in both light and dark themes
        self.table_context_list.setStyleSheet(self._table_context_list_style())

        cl.addWidget(self.table_context_list)

        table_btn_row = QHBoxLayout()
        refresh_tables_btn = QPushButton("Refresh")
        refresh_tables_btn.clicked.connect(self._refresh_table_context_list)
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self._set_all_table_checks(Qt.CheckState.Checked))
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(lambda: self._set_all_table_checks(Qt.CheckState.Unchecked))
        apply_tables_btn = QPushButton("Apply Selection")
        apply_tables_btn.clicked.connect(self._apply_selected_tables)
        table_btn_row.addWidget(refresh_tables_btn)
        table_btn_row.addWidget(select_all_btn)
        table_btn_row.addWidget(clear_all_btn)
        table_btn_row.addWidget(apply_tables_btn)
        table_btn_row.addStretch()
        cl.addLayout(table_btn_row)
        lay.addWidget(cf)

        # Default model selector
        df = QFrame()
        df.setFrameShape(QFrame.Shape.StyledPanel)
        dl = QVBoxLayout(df)
        dl.addWidget(QLabel("<b>Default Model (auto-loads on app start)</b>"))
        self.default_model_combo = QComboBox()
        self._refresh_default_model_combo()
        self.save_default_btn = QPushButton("Save as Default")
        self.save_default_btn.setStyleSheet(
            "QPushButton{background:#2196F3;color:white;font-weight:bold;padding:7px 16px 9px 16px;border-radius:4px;border:1px solid #0d47a1;border-bottom:3px solid #0d47a1;}"
            "QPushButton:hover{background:#1e88e5;}"
            "QPushButton:pressed{background:#1565c0;padding:8px 16px 8px 16px;border-bottom:1px solid #0d47a1;}"
        )
        self.save_default_btn.clicked.connect(self._save_default_model)
        drow = QHBoxLayout()
        drow.addWidget(self.default_model_combo)
        drow.addWidget(self.save_default_btn)
        dl.addLayout(drow)
        lay.addWidget(df)

        note = QLabel(
            "<i>Models are loaded from local files in the models folder.<br>"
            "No internet or external service is required.</i>"
        )
        note.setStyleSheet("color:#888;")
        note.setWordWrap(True)
        lay.addWidget(note)

        lay.addStretch()
        scroll.setWidget(w)
        return scroll

    # â”€â”€ Library status (Milestone 3) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _is_dark_theme(self) -> bool:
        return getattr(self.parent_editor, "current_theme", "dark") == "dark"

    def _table_context_list_style(self) -> str:
        if self._is_dark_theme():
            bg = "#3c3c3c"
            text = "#ffffff"
            border = "#555555"
            unchecked_bg = "#2b2b2b"
            unchecked_border = "#b8b8b8"
        else:
            bg = "#ffffff"
            text = "#1f1f1f"
            border = "#cccccc"
            unchecked_bg = "#ffffff"
            unchecked_border = "#666666"

        return f"""
            QListWidget {{
                background-color: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item {{
                color: {text};
                padding: 4px;
                margin: 2px 0px;
            }}
            QListWidget::item:hover {{
                background-color: rgba(0, 120, 215, 0.2);
            }}
            QListWidget::item:selected {{
                background-color: rgba(0, 120, 215, 0.3);
                border: 1px solid #0078d7;
            }}
            QListWidget::indicator:unchecked {{
                width: 18px;
                height: 18px;
                border: 2px solid {unchecked_border};
                border-radius: 3px;
                background-color: {unchecked_bg};
            }}
            QListWidget::indicator:checked {{
                width: 18px;
                height: 18px;
                border: 2px solid #0078d7;
                border-radius: 3px;
                background-color: #0078d7;
                background-image: url(data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><path fill='white' d='M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z'/></svg>);
                background-repeat: no-repeat;
                background-position: center;
            }}
        """

    def _refresh_library_status(self):
        """No-op in SQL-only mode."""
        return

    # â”€â”€ Theme â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def apply_theme(self):
        if hasattr(self.parent_editor, 'current_theme') and self.parent_editor.current_theme == 'dark':
            self.setStyleSheet("""
                QDialog { background-color: #2b2b2b; color: #ffffff; }
                QLabel { color: #ffffff; }
                QLineEdit { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555; padding: 5px; }
                QTextEdit { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555; }
                QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555; padding: 5px; }
                QPushButton { background-color: #4caf50; color: white; border: none; padding: 8px 15px; border-radius: 4px; }
                QPushButton:hover { background-color: #45a049; }
                QFrame { border: 1px solid #555; background-color: #333; }
                QListWidget { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555; }
            """)

    # â”€â”€ Model management â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _describe_model_key(self, model_key: str) -> str:
        """Return a human-friendly model description for UI/status messages."""
        if not model_key:
            return "Unknown model"

        for m in self.client.list_available_models():
            if m.get("key") == model_key:
                return m.get("description", model_key)

        return (model_key or "Unknown model").replace("-", " ").replace("_", " ").replace(".gguf", "")

    def _refresh_model_status(self):
        models = self.client.list_available_models()
        model_keys = {m["key"] for m in models}

        # Refresh the combo box with local GGUF models only.
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for m in models:
            self.model_combo.addItem(m["description"], m["key"])
        self.model_combo.setEnabled(bool(models))

        # Sync combo to loaded/default key, falling back to first local model.
        target_key = None
        if self.client.is_loaded() and self.client.model_name in model_keys:
            target_key = self.client.model_name
        elif self.selected_model_key in model_keys:
            target_key = self.selected_model_key
        elif models:
            target_key = models[0]["key"]

        if target_key:
            self.selected_model_key = target_key
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == target_key:
                    self.model_combo.setCurrentIndex(i)
                    break
        else:
            self.selected_model_key = None

        self.model_combo.blockSignals(False)

        # Keep the default-model combo in Settings tab in sync too
        self._refresh_default_model_combo()

        self.load_btn.setEnabled(bool(models))

        if not models:
            self.status_label.setText("No local model found in models folder")
            if hasattr(self, 'model_info_label'):
                self.model_info_label.setText("No local model found")
            self.chat_display.setHtml(
                "<div style='padding:12px;background:#fff3e0;border-radius:4px;color:#e65100;'>"
                "<b>No local model found</b><br>"
                "Place at least one <b>.gguf</b> file in the local <b>models</b> folder, then reopen AI settings.<br><br>"
                "<b>Offline only:</b> models are read only from local files."
                "</div>"
            )
            return

        if self.client.is_loaded():
            model_name = self.client.model_name
            desc = self._describe_model_key(model_name)
            self.status_label.setText(f"Model loaded: {desc}")
            if hasattr(self, 'model_info_label'):
                self.model_info_label.setText(f"Loaded: {desc}")
            self.load_btn.setText("Reload Model")
            self._add_welcome_message()
        else:
            self.status_label.setText("No model loaded - click 'Load Model'")
            if hasattr(self, 'model_info_label'):
                self.model_info_label.setText("No model loaded")
            self.chat_display.setHtml(
                "<div style='padding:12px;background:#fff3e0;border-radius:4px;color:#e65100;'>"
                "<b>No model loaded</b><br>"
                "Place a GGUF model file in the local <b>models</b> folder, then click <b>Load Model</b>.<br><br>"
                "<b>No internet or external application needed</b> - everything runs locally inside SimpliSQL."
                "</div>"
            )

    def _on_model_combo_changed(self):
        self.selected_model_key = self.model_combo.currentData()

    def _set_model_loading_ui(self, is_loading: bool, status_text: str | None = None):
        """Lock selectors and chat controls while model loading is in progress."""
        if status_text is not None:
            self.status_label.setText(status_text)

        self.load_btn.setEnabled(not is_loading)
        self.model_combo.setEnabled((not is_loading) and self.model_combo.count() > 0)

        if hasattr(self, "default_model_combo"):
            has_default_models = (
                self.default_model_combo.count() > 0
                and self.default_model_combo.itemData(0) is not None
            )
            self.default_model_combo.setEnabled((not is_loading) and has_default_models)

        if hasattr(self, "save_default_btn"):
            self.save_default_btn.setEnabled(not is_loading)

        self.user_input.setEnabled(not is_loading)
        self.user_input.setPlaceholderText(
            "Loading model... Please wait." if is_loading else "Ask about SQL, data, or workflows..."
        )

        send_btn = self.findChild(QPushButton, "send_btn")
        if send_btn:
            send_btn.setEnabled(not is_loading)
        clear_btn = self.findChild(QPushButton, "clear_btn")
        if clear_btn:
            clear_btn.setEnabled(not is_loading)
        copy_btn = self.findChild(QPushButton, "copy_btn")
        if copy_btn:
            copy_btn.setEnabled(not is_loading)

    def _auto_load_default(self):
        """Auto-load the default model in background on startup."""
        models = self.client.list_available_models()
        if not models:
            self._refresh_model_status()
            return

        model_keys = {m["key"] for m in models}
        key = self.selected_model_key
        if key not in model_keys:
            key = models[0]["key"]
            self.selected_model_key = key

        desc = self._describe_model_key(key)
        self._model_load_start_time = datetime.now()
        self._set_model_loading_ui(True, "Loading model into memory...")
        QApplication.processEvents()

        # Show loading message in chat
        self.chat_display.append(
            f"<div style='margin:8px 0;padding:10px;background:#fff3e0;"
            f"border-left:3px solid #ff9800;border-radius:4px;'>"
            f"<b style='color:#e65100;'>Loading Model:</b><br>"
            f"<div style='color:#000;margin-top:4px;'>Auto-loading {desc}... This may take a few minutes.</div></div>"
        )
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

        self._loader = ModelLoaderThread(self.client, key)
        self._loader.progress.connect(lambda msg: self.status_label.setText(msg))
        self._loader.progress.connect(self._update_loading_progress)
        self._loader.finished_ok.connect(self._on_model_loaded)
        self._loader.finished_err.connect(self._on_model_load_error)
        self._loader.start()

    def _load_selected_model(self):
        """Load or reload the selected local model."""
        key = self.selected_model_key
        if not key:
            QMessageBox.information(
                self,
                "No Local Model",
                "No local GGUF model is available. Add a model file to the models folder first."
            )
            return
        # If same model is already loaded, do nothing
        if self.client.is_loaded() and self.client.model_name == key:
            self.status_label.setText("Model already loaded")
            return

        # Unload current model before loading new one
        if self.client.is_loaded():
            self.client.unload_model()

        desc = self._describe_model_key(key)
        self._model_load_start_time = datetime.now()
        self._set_model_loading_ui(True, f"Preparing {desc}...")
        QApplication.processEvents()

        # Show loading message in chat
        self.chat_display.append(
            f"<div style='margin:8px 0;padding:10px;background:#fff3e0;"
            f"border-left:3px solid #ff9800;border-radius:4px;'>"
            f"<b style='color:#e65100;'>Loading Model:</b><br>"
            f"<div style='color:#000;margin-top:4px;'>Preparing {desc}... This may take a few minutes.</div></div>"
        )
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

        self._loader = ModelLoaderThread(self.client, key)
        self._loader.progress.connect(lambda msg: self.status_label.setText(msg))
        self._loader.progress.connect(self._update_loading_progress)
        self._loader.finished_ok.connect(self._on_model_loaded)
        self._loader.finished_err.connect(self._on_model_load_error)
        self._loader.start()

    def _update_loading_progress(self, msg):
        """Update the loading message in chat with current progress."""
        # Replace the last loading message
        html = self.chat_display.toHtml()
        # Find and replace the loading div
        if "Loading Model:" in html:
            new_msg = (
                f"<div style='margin:8px 0;padding:10px;background:#fff3e0;"
                f"border-left:3px solid #ff9800;border-radius:4px;'>"
                f"<b style='color:#e65100;'>Loading Model:</b><br>"
                f"<div style='color:#000;margin-top:4px;'>{msg}</div></div>"
            )
            # Simple replacement - replace the entire loading div
            start = html.find("<div style='margin:8px 0;padding:10px;background:#fff3e0;")
            if start != -1:
                end = html.find("</div>", start) + 6
                html = html[:start] + new_msg + html[end:]
                self.chat_display.setHtml(html)
                sb = self.chat_display.verticalScrollBar()
                sb.setValue(sb.maximum())

    def _on_model_loaded(self):
        self._set_model_loading_ui(False)
        self._refresh_model_status()

        load_elapsed = (datetime.now() - getattr(self, '_model_load_start_time', datetime.now())).total_seconds()
        load_time_text = f"{load_elapsed:.1f}s"

        # Keep timing visible in status areas after refresh.
        current_status = self.status_label.text() or ""
        if current_status.startswith("Model loaded:"):
            self.status_label.setText(f"{current_status} ({load_time_text})")
        if hasattr(self, 'model_info_label'):
            current_info = self.model_info_label.text() or ""
            if current_info.startswith("Loaded:"):
                self.model_info_label.setText(f"{current_info} ({load_time_text})")

        # Show success message in chat
        desc = self._describe_model_key(self.client.model_name or self.selected_model_key)
        self.chat_display.append(
            f"<div style='margin:8px 0;padding:10px;background:#e8f5e8;"
            f"border-left:3px solid #4caf50;border-radius:4px;'>"
            f"<b style='color:#2e7d32;'>Model Loaded:</b><br>"
            f"<div style='color:#000;margin-top:4px;'>{desc} is ready in <b>{load_time_text}</b>! You can now ask questions.</div></div>"
        )
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_model_load_error(self, error):
        self._set_model_loading_ui(False, f"Error: {error}")
        QMessageBox.warning(self, "Error", f"Failed to load model:\n{error}")

        # Show error message in chat
        self.chat_display.append(
            f"<div style='margin:8px 0;padding:10px;background:#ffebee;"
            f"border-left:3px solid #f44336;border-radius:4px;'>"
            f"<b style='color:#c62828;'>Model Load Failed:</b><br>"
            f"<div style='color:#000;margin-top:4px;'>{error}</div></div>"
        )
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _refresh_default_model_combo(self):
        """Populate the default-model combo with local models only."""
        self.default_model_combo.blockSignals(True)
        self.default_model_combo.clear()
        models = self.client.list_available_models()
        for m in models:
            self.default_model_combo.addItem(m["description"], m["key"])

        if not models:
            self.default_model_combo.addItem("No local models found", None)
            self.default_model_combo.setEnabled(False)
            self.default_model_combo.blockSignals(False)
            return

        self.default_model_combo.setEnabled(True)

        # Restore saved selection if still available, else fallback.
        model_keys = {m["key"] for m in models}
        cfg = _load_ai_config()
        saved = cfg.get("default_model")
        if saved in model_keys:
            target = saved
        elif self.selected_model_key in model_keys:
            target = self.selected_model_key
        else:
            target = models[0]["key"]

        for i in range(self.default_model_combo.count()):
            if self.default_model_combo.itemData(i) == target:
                self.default_model_combo.setCurrentIndex(i)
                break
        self.default_model_combo.blockSignals(False)

    def _save_default_model(self):
        """Save the selected default model to config."""
        key = self.default_model_combo.currentData()
        if key:
            cfg = _load_ai_config()
            cfg["default_model"] = key
            _save_ai_config(cfg)
            desc = self.default_model_combo.currentText()
            self.status_label.setText(f"Default model saved: {desc}")
            self.selected_model_key = key
            # Update main combo to match
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == key:
                    self.model_combo.setCurrentIndex(i)
                    break

    # AFTER
    def _refresh_table_context_list(self):
        if not hasattr(self, "table_context_list"):
            return

        # Disconnect during population so setCheckState doesn't trigger sync per row
        try:
            self.table_context_list.itemChanged.disconnect(
                self._sync_selected_tables_to_parent
            )
        except Exception:
            pass

        self.table_context_list.clear()
        editor_tables = list(getattr(self.parent_editor, "uploaded_display_names", []) or [])
        selected = set(getattr(self.parent_editor, "selected_tables_for_ai", []) or [])

        if not editor_tables:
            self.table_context_list.addItem("No uploaded tables available")
            item = self.table_context_list.item(0)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            item.setForeground(QColor("#b8b8b8" if self._is_dark_theme() else "#666666"))
            return

        for table_name in editor_tables:
            item = QListWidgetItem(table_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if table_name in selected else Qt.CheckState.Unchecked)
            item.setForeground(QColor("#ffffff" if self._is_dark_theme() else "#1f1f1f"))
            self.table_context_list.addItem(item)

        # Reconnect â€” any checkbox change from here on syncs immediately to parent
        self.table_context_list.itemChanged.connect(self._sync_selected_tables_to_parent)

    def _set_all_table_checks(self, state):
        if not hasattr(self, "table_context_list"):
            return
        for i in range(self.table_context_list.count()):
            item = self.table_context_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)
        self._sync_selected_tables_to_parent()

    def _get_selected_tables_for_ai(self):
        editor_tables = list(getattr(self.parent_editor, "uploaded_display_names", []) or [])
        if not hasattr(self, "table_context_list") or self.table_context_list.count() == 0:
            return []

        selected = []
        for i in range(self.table_context_list.count()):
            item = self.table_context_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())

        # Return only explicitly selected tables (no fallback to all)
        return selected

    def _sync_selected_tables_to_parent(self):
        if self.parent_editor is None:
            return
        self.parent_editor.selected_tables_for_ai = self._get_selected_tables_for_ai()

    def _apply_selected_tables(self):
        self._sync_selected_tables_to_parent()
        selected = list(getattr(self.parent_editor, "selected_tables_for_ai", []) or [])

        if selected:
            selected_html = "<br>".join(html.escape(name) for name in selected)
            message = (
                "<b>Selected tables for AI context:</b><br><br>"
                f"{selected_html}"
            )
        else:
            message = "No tables are currently selected for AI context."

        QMessageBox.information(self, "AI Table Selection Applied", message)

    def _estimate_tokens(self, text: str) -> int:
        # Fast heuristic: most LLM tokenizers are roughly 3-4 chars/token on mixed text.
        return max(1, len(text) // 4)

    def _get_model_context_limit(self) -> int:
        loaded_context = int(getattr(self.client, "context_length", 0) or 0)
        if loaded_context > 0:
            return loaded_context
        return 8192

    def _estimate_messages_tokens(self, messages: list) -> int:
        total = 0
        for msg in messages:
            total += self._estimate_tokens(msg.get("content", "")) + 20
        return total

    def _desired_output_tokens(self, context_limit: int) -> int:
        """Target output token budget for large SQL responses."""
        return max(512, min(4096, int(context_limit * 0.55)))

    def _get_prompt_budget_tiers(self, context_limit: int, mode_key: str) -> list[tuple[int, int]]:
        """Return ordered (prompt_budget, reserved_output_tokens) tiers for overflow handling.

        Tiers progressively reduce reserved output to preserve more input context,
        up to this model's maximum context limit.
        """
        mode = (mode_key or "auto").lower()
        default_reserved = self._desired_output_tokens(context_limit)

        reserve_candidates = [default_reserved]
        if mode == "sql":
            reserve_candidates.extend([
                int(context_limit * 0.45),
                int(context_limit * 0.35),
                int(context_limit * 0.28),
                int(context_limit * 0.22),
                int(context_limit * 0.16),
                256,
                192,
                128,
            ])
        else:
            reserve_candidates.extend([
                int(context_limit * 0.45),
                int(context_limit * 0.35),
                int(context_limit * 0.25),
                int(context_limit * 0.18),
                256,
                192,
                128,
            ])

        tiers = []
        seen = set()
        max_safe_reserve = max(128, context_limit - 256)

        for reserve in reserve_candidates:
            reserve = max(128, min(int(reserve), max_safe_reserve))
            prompt_budget = max(768, context_limit - reserve - 200)
            key = (prompt_budget, reserve)
            if key in seen:
                continue
            seen.add(key)
            tiers.append(key)

        tiers.sort(key=lambda x: x[0])
        return tiers

    def _fit_messages_to_budget(self, source_messages: list, target_prompt_budget: int) -> tuple[list, list, bool]:
        """Fit messages to a target prompt budget by trimming history/system/user in that order."""
        messages = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in source_messages]
        notes = []

        history_trimmed = False
        while len(messages) > 2 and self._estimate_messages_tokens(messages) > target_prompt_budget:
            # Remove oldest user/assistant turn pair but keep system + latest user.
            if len(messages) >= 4:
                del messages[1:3]
                history_trimmed = True
            else:
                break
        if history_trimmed:
            notes.append("Older chat turns were trimmed to prioritize your latest request.")

        # Compact system prompt next.
        if self._estimate_messages_tokens(messages) > target_prompt_budget and messages:
            non_system_tokens = self._estimate_messages_tokens(messages[1:])
            system_budget = max(180, target_prompt_budget - non_system_tokens - 20)
            compact_system = self._compact_system_prompt(messages[0].get("content", ""), system_budget)
            if compact_system != messages[0].get("content", ""):
                messages[0]["content"] = compact_system
                notes.append("System context was compacted to fit model limits.")

        # Truncate latest user prompt as final fallback.
        if self._estimate_messages_tokens(messages) > target_prompt_budget and messages:
            without_latest_user = self._estimate_messages_tokens(messages[:-1])
            user_budget = max(120, target_prompt_budget - without_latest_user - 20)
            trimmed_user, was_trimmed = self._truncate_user_prompt_for_budget(messages[-1].get("content", ""), user_budget)
            if was_trimmed:
                messages[-1]["content"] = trimmed_user
                notes.append("Your prompt was truncated for this run to prevent context overflow.")

        fits = self._estimate_messages_tokens(messages) <= target_prompt_budget
        return messages, notes, fits

    def _compact_system_prompt(self, prompt: str, max_tokens: int) -> str:
        """Trim system prompt to fit within a token budget while keeping leading rules/context."""
        if not prompt:
            return ""
        if self._estimate_tokens(prompt) <= max_tokens:
            return prompt

        lines = prompt.splitlines()
        kept = []
        for line in lines:
            trial = "\n".join(kept + [line])
            if self._estimate_tokens(trial) > max_tokens:
                break
            kept.append(line)

        if not kept:
            max_chars = max(400, max_tokens * 4)
            compact = prompt[:max_chars]
            if len(prompt) > len(compact):
                compact += "\n[System context truncated due token budget.]"
            return compact

        if len(kept) < len(lines):
            kept.append("[System context truncated due token budget.]")
        return "\n".join(kept)

    def _truncate_user_prompt_for_budget(self, text: str, max_tokens: int) -> tuple[str, bool]:
        """Trim very large user prompts while preserving both beginning and end context."""
        text = text or ""
        if self._estimate_tokens(text) <= max_tokens:
            return text, False

        max_chars = max(300, max_tokens * 4)
        if len(text) <= max_chars:
            return text, False

        head_chars = int(max_chars * 0.75)
        tail_chars = max_chars - head_chars
        truncated = (
            text[:head_chars].rstrip()
            + "\n\n...[User prompt truncated due model context limit]...\n\n"
            + text[-tail_chars:].lstrip()
        )
        return truncated, True

    def _extract_sql_candidates(self, response: str) -> list:
        return _extract_sql_candidates_from_text(response)

    def _extract_cte_names(self, sql_text: str) -> set[str]:
        return {
            name.lower()
            for name in re.findall(r"(?i)(?:WITH|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", sql_text or "")
        }

    def _build_explain_stmt_for_validation(self, stmt: str, table_sources: dict, cte_names: set[str]) -> str:
        explain_stmt = stmt
        physical_names = {
            name.lower(): path
            for name, path in (table_sources or {}).items()
            if name.lower() not in (cte_names or set())
        }

        def repl(match):
            keyword = match.group(1)
            table_ref = match.group(2)
            suffix = match.group(3) or ""
            table_key = table_ref.lower()
            if table_key in cte_names:
                return match.group(0)
            source_path = physical_names.get(table_key)
            if not source_path:
                return match.group(0)
            return f"{keyword} read_parquet('{source_path}'){suffix}"

        pattern = re.compile(
            r"(?is)\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
            r"((?:\s+(?:AS\s+)?\"?[A-Za-z_][A-Za-z0-9_]*\"?)?)"
        )
        explain_stmt = pattern.sub(repl, explain_stmt)
        return explain_stmt

    def _guard_explain_stmt(self, explain_stmt: str, cte_names: set[str], table_sources: dict) -> str | None:
        guard_patterns = [
            "TRY_CAST(COALESCE AS DOUBLE)",
            "TRY_CAST(FROM AS DOUBLE)",
            "TRY_CAST(B. AS DOUBLE)",
        ]
        upper_stmt = (explain_stmt or "").upper()
        for pattern in guard_patterns:
            if pattern in upper_stmt:
                return f"Rejected invalid validation SQL fragment: {pattern}"

        for cte_name in (cte_names or set()):
            cte_path_pattern = re.compile(
                rf"(?i)read_parquet\('([^']*[\\/])?{re.escape(cte_name)}\.parquet'\)"
            )
            if cte_path_pattern.search(explain_stmt or ""):
                return f"Rejected validation rewrite for CTE name: {cte_name}"

            cte_source = (table_sources or {}).get(cte_name)
            if cte_source and f"read_parquet('{cte_source}')" in (explain_stmt or ""):
                return f"Rejected validation rewrite for CTE source: {cte_name}"

        return None

    def _schema_validate_response(self, response: str) -> list:
        warnings = []
        editor = self.parent_editor
        if not hasattr(editor, "conn"):
            return warnings

        sql_candidates = self._extract_sql_candidates(response)
        if not sql_candidates:
            return warnings

        selected_tables = set(t.lower() for t in (getattr(editor, "selected_tables_for_ai", []) or []))
        uploaded_tables = set(t.lower() for t in (getattr(editor, "uploaded_display_names", []) or []))
        display_names = list(getattr(editor, "uploaded_display_names", []) or [])
        uploaded_files = list(getattr(editor, "uploaded_files", []) or [])
        doc_dir = getattr(editor, "doc_dir", "")

        table_sources = {}
        for idx, dname in enumerate(display_names):
            if idx < len(uploaded_files) and uploaded_files[idx]:
                table_sources[dname.lower()] = uploaded_files[idx].replace("\\", "/")
            elif doc_dir:
                table_sources[dname.lower()] = os.path.join(doc_dir, f"{dname}.parquet").replace("\\", "/")

        for tname in selected_tables:
            if tname not in table_sources and doc_dir:
                table_sources[tname] = os.path.join(doc_dir, f"{tname}.parquet").replace("\\", "/")

        for sql in sql_candidates:
            stmt = sql.strip().rstrip(";")
            if not stmt:
                continue

            refs = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", stmt, re.IGNORECASE)
            for table_ref in refs:
                table_l = table_ref.lower()
                if selected_tables and table_l in uploaded_tables and table_l not in selected_tables:
                    warnings.append(
                        f"âš ï¸ Table '{table_ref}' is referenced but not selected in AI table settings."
                    )

            # Binder-level validation catches wrong table/column names without running the query.
            # Must resolve bare table names â†’ read_parquet() since tables aren't registered in DuckDB.
            if stmt.upper().startswith("SELECT") or stmt.upper().startswith("WITH"):
                try:
                    cte_names = self._extract_cte_names(stmt)
                    explain_stmt = self._build_explain_stmt_for_validation(stmt, table_sources, cte_names)
                    logger.debug("Schema validation original stmt: %s", stmt)
                    logger.debug("Schema validation detected CTE names: %s", sorted(cte_names))
                    logger.debug("Schema validation table_sources: %s", table_sources)
                    logger.debug("Schema validation explain_stmt: %s", explain_stmt)
                    guard_error = self._guard_explain_stmt(explain_stmt, cte_names, table_sources)
                    if guard_error:
                        warnings.append(f"âš ï¸ Schema validation skipped: {guard_error}")
                        continue
                    editor.conn.execute(f"EXPLAIN {explain_stmt}")
                except Exception as e:
                    err_msg = str(e)
                    # Auto-fix: if QUALIFY is present without any window function, strip it
                    qual_upper = stmt.upper()
                    has_qualify = "QUALIFY" in qual_upper
                    has_over = " OVER " in qual_upper or " OVER(" in qual_upper
                    if has_qualify and not has_over:
                        try:
                            # Strip the QUALIFY clause (no window function = always wrong usage)
                            fixed_stmt = re.sub(
                                r'\s*QUALIFY\b.*?(?=\s*(?:GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b|\s*;?\s*$)',
                                '',
                                stmt,
                                flags=re.IGNORECASE | re.DOTALL,
                            ).strip()
                            cte_names = self._extract_cte_names(fixed_stmt)
                            fixed_explain = self._build_explain_stmt_for_validation(fixed_stmt, table_sources, cte_names)
                            logger.debug("Schema validation fixed stmt: %s", fixed_stmt)
                            logger.debug("Schema validation fixed detected CTE names: %s", sorted(cte_names))
                            logger.debug("Schema validation fixed table_sources: %s", table_sources)
                            logger.debug("Schema validation fixed explain_stmt: %s", fixed_explain)
                            guard_error = self._guard_explain_stmt(fixed_explain, cte_names, table_sources)
                            if guard_error:
                                warnings.append(f"âš ï¸ Schema validation skipped: {guard_error}")
                                continue
                            editor.conn.execute(f"EXPLAIN {fixed_explain}")
                            warnings.append(
                                "âš ï¸ Invalid QUALIFY removed (QUALIFY requires a window function like "
                                "ROW_NUMBER() OVER (...)). Auto-corrected SQL:\n"
                                f"```sql\n{fixed_stmt}\n```"
                            )
                        except Exception:
                            warnings.append(f"âš ï¸ Schema validation: {err_msg}")
                    else:
                        warnings.append(f"âš ï¸ Schema validation: {err_msg}")

        # Deduplicate while preserving order
        dedup = []
        seen = set()
        for w in warnings:
            if w not in seen:
                dedup.append(w)
                seen.add(w)
        return dedup

    # â”€â”€ Chat logic â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _normalize_table_match_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    def _sql_identifier_display(self, name: str) -> str:
        """Render identifiers the way SQL should reference them."""
        text = str(name or "")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
            return text
        return '"' + text.replace('"', '""') + '"'

    def _table_aliases_for_prompt(self, table_name: str, file_path: str = "") -> list:
        """Build compact aliases so user terms map to actual selected table names."""
        raw_values = {table_name}
        if file_path:
            raw_values.add(os.path.splitext(os.path.basename(file_path))[0])

        aliases = set()
        for raw in raw_values:
            raw = (raw or "").strip()
            if not raw:
                continue
            aliases.add(raw)
            aliases.add(raw.replace("_", " "))
            aliases.add(raw.replace("-", " "))
            aliases.add(raw.replace("_", "").replace("-", ""))

            norm = self._normalize_table_match_text(raw)
            if norm:
                aliases.add(norm)
                if "2b" in norm:
                    aliases.update(["2b", "2b file", "2b files", "gstr 2b", "gstr_2b", "gstr_2b_file"])
                if norm.startswith("pr") or "purchase" in norm:
                    aliases.update(["pr", "pr file", "pr files", "pr_file", "pr_files", "purchase register"])

        aliases.discard(table_name)
        return sorted(aliases, key=lambda x: (len(x), x))[:10]

    def _selected_table_alias_map(self) -> dict:
        """Return normalized alias -> selected table name for generated SQL repair."""
        editor = self.parent_editor
        selected_tables = self._get_selected_tables_for_ai()
        display_to_path = {}

        if hasattr(editor, "uploaded_files") and hasattr(editor, "uploaded_display_names"):
            for i, fpath in enumerate(editor.uploaded_files or []):
                if i < len(editor.uploaded_display_names or []):
                    display_to_path[editor.uploaded_display_names[i]] = fpath

        alias_candidates = {}
        for table_name in selected_tables:
            values = [table_name] + self._table_aliases_for_prompt(table_name, display_to_path.get(table_name, ""))
            for value in values:
                norm = self._normalize_table_match_text(value)
                if norm:
                    alias_candidates.setdefault(norm, set()).add(table_name)
        return {
            alias: next(iter(tables))
            for alias, tables in alias_candidates.items()
            if len(tables) == 1
        }

    def _selected_table_columns(self) -> dict:
        """Return selected table -> ordered column names from schema."""
        editor = self.parent_editor
        selected_tables = self._get_selected_tables_for_ai()
        display_to_path = {}
        columns_map = {}

        if hasattr(editor, "uploaded_files") and hasattr(editor, "uploaded_display_names"):
            for i, fpath in enumerate(editor.uploaded_files or []):
                if i < len(editor.uploaded_display_names or []):
                    display_to_path[editor.uploaded_display_names[i]] = fpath

        for table_name in selected_tables:
            try:
                fpath = display_to_path.get(table_name)
                if not fpath:
                    doc_dir = getattr(editor, "doc_dir", "")
                    fpath = os.path.join(doc_dir, f"{table_name}.parquet")
                source = f"read_parquet('{fpath.replace(chr(92), '/')}')"
                cols = editor.conn.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
                columns_map[table_name] = [str(c[0]) for c in cols if c and c[0]]
            except Exception:
                columns_map[table_name] = []

        return columns_map

    def _rewrite_user_prompt_with_selected_tables(self, user_text: str) -> tuple[str, list]:
        """Rewrite alias-like table references in the user prompt to actual selected tables."""
        rewritten = user_text or ""
        alias_map = self._selected_table_alias_map()
        columns_map = self._selected_table_columns()
        replacements = []

        alias_items = []
        for norm_alias, table_name in alias_map.items():
            if norm_alias == self._normalize_table_match_text(table_name):
                continue
            for raw_alias in self._table_aliases_for_prompt(table_name):
                if self._normalize_table_match_text(raw_alias) == norm_alias:
                    # Skip ultra-short generic aliases in user-prompt rewriting so
                    # column names like "2B Document No" are not partially rewritten.
                    if len(raw_alias.strip()) < 4:
                        continue
                    alias_items.append((raw_alias, table_name))

        seen_pairs = set()
        alias_items = [
            (alias, table_name)
            for alias, table_name in alias_items
            if not ((alias.lower(), table_name) in seen_pairs or seen_pairs.add((alias.lower(), table_name)))
        ]
        alias_items.sort(key=lambda item: len(item[0]), reverse=True)

        for alias, table_name in alias_items:
            pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])")
            if pattern.search(rewritten):
                rewritten = pattern.sub(table_name, rewritten)
                replacements.append(f"{alias} -> {table_name}")

        if not replacements:
            return user_text, []

        resolved_tables = []
        seen_tables = set()
        for item in replacements:
            table_name = item.split("->", 1)[1].strip()
            if table_name not in seen_tables:
                resolved_tables.append(table_name)
                seen_tables.add(table_name)

        resolved_lines = []
        for table_name in resolved_tables:
            columns = columns_map.get(table_name, [])
            if columns:
                resolved_lines.append(
                    f"- {table_name} columns: {', '.join(self._sql_identifier_display(col) for col in columns)}"
                )
            else:
                resolved_lines.append(f"- {table_name} columns: (schema unavailable)")

        rewritten = (
            rewritten
            + "\n\nResolved selected tables for this request:\n"
            + "\n".join(resolved_lines)
        )
        return rewritten, replacements

    def _resolve_generated_table_ref(self, table_ref: str) -> str | None:
        """Resolve an invented table reference to a selected table when confidence is high."""
        norm_ref = self._normalize_table_match_text(table_ref)
        if not norm_ref:
            return None

        alias_map = self._selected_table_alias_map()
        if norm_ref in alias_map:
            return alias_map[norm_ref]

        # Keep generated table repair conservative. Do not use substring or
        # generic prefix matching here because names like "pr_prep",
        # "joined_data", or "reconciliation" may be intended CTEs or derived
        # aliases rather than physical uploaded tables.
        return None

    def _repair_generated_table_refs(self, response: str) -> tuple[str, list]:
        """Replace obvious hallucinated FROM/JOIN table names with selected table names."""
        if not response:
            return response, []

        selected = set(self._get_selected_tables_for_ai())
        uploaded = set(getattr(self.parent_editor, "uploaded_display_names", []) or [])
        replacements = []

        def repair_sql(sql_text: str) -> str:
            cte_names = {
                name.lower()
                for name in re.findall(r"(?i)(?:WITH|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", sql_text)
            }

            def repl(match):
                keyword = match.group(1)
                table_ref = match.group(2)
                if table_ref in selected or table_ref in uploaded or table_ref.lower() in cte_names:
                    return match.group(0)

                resolved = self._resolve_generated_table_ref(table_ref)
                if resolved and resolved != table_ref:
                    replacements.append(f"{table_ref} -> {resolved}")
                    return f"{keyword} {resolved}"
                return match.group(0)

            return re.sub(
                r"(?i)\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                repl,
                sql_text,
            )

        def block_repl(match):
            return "```sql\n" + repair_sql(match.group(1)) + "```"

        repaired = re.sub(r"```sql\s*(.*?)```", block_repl, response, flags=re.IGNORECASE | re.DOTALL)
        if repaired == response:
            repaired = repair_sql(response)

        deduped = []
        seen = set()
        for item in replacements:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return repaired, deduped

    def _add_welcome_message(self):
        model_name = self.client.model_name
        desc = ""
        for m in self.client.list_available_models():
            if m["key"] == model_name:
                desc = m["description"]
                break
        if not desc:
            desc = (model_name or "Unknown model").replace("-", " ").replace("_", " ").replace(".gguf", "")
        self.chat_display.setHtml(
            "<div style='padding:12px;background:#e8f5e9;border-radius:4px;color:#1b5e20;'>"
            f"<b>Welcome to SimpliSQL AI Assistant</b><br>"
            f"Powered by local model: <b>{desc}</b><br><br>"
            "<i>You can ask me to:</i><br>"
            "- Generate SQL queries (simple or complex)<br>"
            "- Use subqueries, CTEs, window functions<br>"
            "- Explain SQL syntax and DuckDB features<br>"
            "- Optimize queries and suggest improvements<br>"
            "- Help with data analysis and workflows<br><br>"
            "<b>Note:</b> I can generate ANY valid DuckDB SQL query - don't hesitate to ask for complex operations!<br><br>"
            "All processing happens locally - no data leaves your machine."
            "</div>"
        )

    def _is_simple_chat_message(self, text: str) -> bool:
        cleaned = re.sub(r"[^\w\s]", "", (text or "").strip().lower())
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned in {
            "hi",
            "hello",
            "hey",
            "hii",
            "helo",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
            "ok",
            "okay",
        }

    def _append_user_message(self, user_text: str):
        safe_user = html.escape(user_text or "")
        self.chat_display.append(
            f"<div style='margin:8px 0;padding:10px;background:#e3f2fd;"
            f"border-left:3px solid #2196F3;border-radius:4px;'>"
            f"<b>You:</b><br><div style='color:#000;margin-top:4px;'>"
            f"{safe_user}</div></div>"
        )

    def _append_ai_message(self, response: str):
        safe_response = html.escape(response or "").replace("\n", "<br>")
        self.chat_display.append(
            f"<div style='margin:8px 0;padding:10px;background:#f1f8e9;"
            f"border-left:3px solid #8bc34a;border-radius:4px;'>"
            f"<b style='color:#558b2f;'>AI:</b><br>"
            f"<div style='color:#000;margin-top:4px;white-space:pre-wrap;word-break:break-word;'>"
            f"{safe_response}</div>"
            f"</div>"
        )

    def send_message(self):
        if self._chat_thread is not None and self._chat_thread.isRunning():
            elapsed = ""
            if self._generation_started_at is not None:
                sec = int((datetime.now() - self._generation_started_at).total_seconds())
                elapsed = f" ({sec}s elapsed)"
            self._set_status_badge(
                f"Generation already in progress{elapsed}. Click Stop or wait for completion.",
                "busy",
            )
            return

        user_text = self.user_input.text().strip()
        if not user_text:
            return

        if self._is_simple_chat_message(user_text):
            self._append_user_message(user_text)
            self.user_input.clear()
            response = "Hi! Tell me what SQL query or data analysis you want to build."
            self._append_ai_message(response)
            sb = self.chat_display.verticalScrollBar()
            sb.setValue(sb.maximum())
            return

        if not self.client.is_loaded():
            QMessageBox.warning(self, "No Model",
                                "Please load a local model first (click 'Load Model').")
            return

        self._append_user_message(user_text)
        self.user_input.clear()

        self.chat_display.append(
            "<div style='margin:6px 0;font-style:italic;color:#888;'>Thinking...</div>"
        )
        QApplication.processEvents()

        self._sync_selected_tables_to_parent()

        mode_key = "sql"
        self._last_answer_mode = "sql"
        system_prompt = self._build_system_prompt()
        ai_user_text, _prompt_table_repairs = self._rewrite_user_prompt_with_selected_tables(user_text)

        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.current_conversation[-6:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["ai"]})
        messages.append({"role": "user", "content": ai_user_text})

        # Guardrail: adapt prompt budget across higher tiers up to this model's max context.
        context_limit = self._get_model_context_limit()
        default_reserved_output = self._desired_output_tokens(context_limit)
        budget_tiers = self._get_prompt_budget_tiers(context_limit, mode_key)

        fitted_messages = None
        compaction_notes = []
        selected_tier_index = None
        selected_reserved_output = None
        last_trial_messages = messages
        last_trial_notes = []

        for idx, (target_prompt_budget, reserved_output_tokens) in enumerate(budget_tiers, start=1):
            trial_messages, trial_notes, fits = self._fit_messages_to_budget(messages, target_prompt_budget)
            last_trial_messages = trial_messages
            last_trial_notes = trial_notes
            if fits:
                fitted_messages = trial_messages
                compaction_notes = trial_notes
                selected_tier_index = idx
                selected_reserved_output = reserved_output_tokens
                break

        if fitted_messages is None:
            messages = last_trial_messages
            compaction_notes = last_trial_notes
            html = self.chat_display.toHtml()
            html = html.replace("â³ Thinkingâ€¦", "")
            self.chat_display.setHtml(html)
            self.chat_display.append(
                "<div style='margin:8px 0;padding:10px;background:#ffebee;"
                "border-left:3px solid #f44336;border-radius:4px;'>"
                "<b style='color:#b71c1c;'>Prompt too large for current model context.</b><br>"
                "<div style='color:#000;margin-top:4px;'>"
                "Reached this model's maximum context limit after adaptive expansion. "
                "Reduce selected tables/current-query context, or switch to a higher-context model."
                "</div></div>"
            )
            return

        messages = fitted_messages

        if selected_tier_index is not None and selected_tier_index > 1:
            compaction_notes.append(
                f"Context budget was escalated to a higher tier ({selected_tier_index}/{len(budget_tiers)}) "
                "to fit your request within this model's maximum context."
            )

        if selected_reserved_output is not None and selected_reserved_output < default_reserved_output:
            compaction_notes.append(
                f"Output reservation was reduced to {selected_reserved_output} tokens to preserve more input context."
            )

        if compaction_notes:
            # Deduplicate while preserving order.
            dedup = []
            seen = set()
            for note in compaction_notes:
                if note not in seen:
                    dedup.append(note)
                    seen.add(note)
            compaction_notes = dedup

        if compaction_notes:
            safe_notes = "<br>".join(
                n.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for n in compaction_notes
            )
            self.chat_display.append(
                "<div style='margin:8px 0;padding:10px;background:#fff8e1;"
                "border-left:3px solid #ffb300;border-radius:4px;'>"
                "<b style='color:#8d6e00;'>Context adjusted for large request:</b><br>"
                f"<div style='color:#000;margin-top:4px;'>{safe_notes}</div>"
                "</div>"
            )

        self._current_user_text = user_text
        self._generation_cancelled = False
        self._response_start_time = datetime.now()
        self._start_generation_status()
        self._streaming_buffer = []          # accumulates streamed tokens
        self._streaming_block_inserted = False
        selected_tables = self._get_selected_tables_for_ai()

        self._chat_thread = AIChatThread(
            self.client,
            messages,
            user_request=user_text,
            answer_mode=mode_key,
        )
        
        print("\n" + "=" * 80)
        print("FINAL MESSAGES SENT TO AI")
        print("=" * 80)

        for i, msg in enumerate(messages):
            print(f"\nMESSAGE {i}")
            print("ROLE:", msg["role"])
            print(msg["content"][:10000])

        print("=" * 80)

        self._chat_thread.token_ready.connect(self._on_token_ready)
        self._chat_thread.response_ready.connect(self._on_ai_response)
        self._chat_thread.error_occurred.connect(self._on_ai_error)
        self._chat_thread.finished.connect(self._on_chat_thread_finished)
        self._set_generation_controls(True)
        self._chat_thread.start()

    def _set_generation_controls(self, is_generating: bool):
        send_btn = self.findChild(QPushButton, "send_btn")
        if send_btn:
            send_btn.setEnabled(not is_generating)
        if hasattr(self, "stop_btn") and self.stop_btn:
            self.stop_btn.setEnabled(is_generating)

    def _set_status_badge(self, text: str, level: str = "info"):
        palette = {
            "ready": ("#1b5e20", "#e8f5e9", "#4caf50"),
            "busy": ("#8d6e00", "#fff8e1", "#ffb300"),
            "error": ("#b71c1c", "#ffebee", "#f44336"),
            "info": ("#0d47a1", "#e3f2fd", "#2196f3"),
        }
        fg, bg, border = palette.get(level, palette["info"])
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color: {fg}; background-color: {bg}; border: 1px solid {border}; "
            "border-radius: 4px; padding: 6px 10px; font-size: 12px;"
        )

    def _start_generation_status(self):
        self._generation_started_at = datetime.now()
        self._set_status_badge("Generating... 0s (click Stop to cancel)", "busy")
        if hasattr(self, "_generation_status_timer"):
            self._generation_status_timer.start()

    def _stop_generation_status(self, text: str = "Ready", level: str = "ready"):
        if hasattr(self, "_generation_status_timer"):
            self._generation_status_timer.stop()
        self._generation_started_at = None
        self._set_status_badge(text, level)

    def _update_generation_status_tick(self):
        if self._chat_thread is None or not self._chat_thread.isRunning() or self._generation_started_at is None:
            if hasattr(self, "_generation_status_timer"):
                self._generation_status_timer.stop()
            return
        sec = int((datetime.now() - self._generation_started_at).total_seconds())
        self._set_status_badge(f"Generating... {sec}s (click Stop to cancel)", "busy")

    def _stop_generation(self):
        if self._chat_thread is None or not self._chat_thread.isRunning():
            return

        self._generation_cancelled = True
        # Politely request interruption. Avoid calling terminate() which can
        # destabilize the interpreter or the Qt event loop; rely on the
        # worker to check isInterruptionRequested() and exit cleanly.
        try:
            self._chat_thread.requestInterruption()
        except Exception:
            pass

        html = self.chat_display.toHtml()
        html = html.replace("Thinking...", "")
        self.chat_display.setHtml(html)
        self.chat_display.append(
            "<div style='margin:8px 0;padding:10px;background:#fff8e1;"
            "border-left:3px solid #ffb300;border-radius:4px;'>"
            "<b style='color:#8d6e00;'>Generation stopped.</b></div>"
        )
        # Immediately update controls; thread will clear _chat_thread when it
        # naturally ends or emits response/error.
        self._set_generation_controls(False)
        self._stop_generation_status("Generation stopped by user.", "info")

    def _on_token_ready(self, token: str):
        """Called for each streamed token â€“ appends incrementally without full repaint."""
        if self._generation_cancelled:
            return

        self._streaming_buffer.append(token)

        if not self._streaming_block_inserted:
            # Remove the Thinking... placeholder (one-time setHtml is acceptable here)
            html = self.chat_display.toHtml()
            html = html.replace("Thinking...", "")
            self.chat_display.setHtml(html)

            # Insert the AI header block using cursor (no full repaint)
            cursor = self.chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertHtml(
                "<div style='margin:8px 0;padding:10px;background:#f1f8e9;"
                "border-left:3px solid #8bc34a;border-radius:4px;'>"
                "<b style='color:#558b2f;'>AI:</b><br></div>"
            )
            # Move to end and remember this position as our write anchor
            cursor.movePosition(cursor.MoveOperation.End)
            self._stream_cursor_pos = cursor.position()
            self.chat_display.setTextCursor(cursor)
            self._streaming_block_inserted = True

        # Append only the new token text at the stored position (no setHtml)
        cursor = self.chat_display.textCursor()
        cursor.setPosition(self._stream_cursor_pos)
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self._stream_cursor_pos = cursor.position()
        self.chat_display.setTextCursor(cursor)

        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_ai_response(self, response):
        self._set_generation_controls(False)

        if self._generation_cancelled:
            self._generation_cancelled = False
            self._chat_thread = None
            self._stop_generation_status("Generation stopped by user.", "info")
            return

        elapsed = (datetime.now() - getattr(self, '_response_start_time', datetime.now())).total_seconds()

        # 1. Run our DuckDB syntax and binder validation checks
        schema_warnings = self._schema_validate_response(response)
        if schema_warnings:
            response = response + "\n\n---\nSchema checks:\n" + "\n".join(schema_warnings)
        pure_sql_candidates = self._extract_sql_candidates(response)
        self._stop_generation_status(f"Response complete in {elapsed:.1f}s", "ready")

        auto_paste = getattr(self, 'auto_paste_check', None)
        auto_pasted = False
        if auto_paste and auto_paste.isChecked() and pure_sql_candidates:
            sql = pure_sql_candidates[-1].strip()
            if sql:
                editor = self.parent_editor
                if hasattr(editor, 'switch_notepad_mode'):
                    editor.switch_notepad_mode("sql")
                editor.sql_text.setPlainText(sql)
                auto_pasted = True
        QApplication.processEvents()

        # 3. Clean the display and push the unified canonical response block
        # This completely bypasses the fragile string matching/stitching bugs
        if not self._streaming_block_inserted:
            html = self.chat_display.toHtml()
            html = html.replace("Thinking...", "")
            self.chat_display.setHtml(html)

        # Only append the full response if streaming never fired
        # (If streaming fired, tokens were already displayed incrementally)
        if not self._streaming_block_inserted:
            safe_final = response.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            self.chat_display.append(
                f"<div style='margin:8px 0;padding:10px;background:#f1f8e9;"
                f"border-left:3px solid #8bc34a;border-radius:4px;'>"
                f"<b style='color:#558b2f;'>AI:</b><br>"
                f"<div style='color:#000;margin-top:4px;white-space:pre-wrap;word-break:break-word;'>"
                f"{safe_final}</div>"
                f"</div>"
            )

        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

        # 4. Log performance metrics
        self.chat_display.append(
            f"<div style='margin:0 0 6px 0;color:#2e7d32;font-size:0.85em;text-align:right;'>Done - {elapsed:.1f}s</div>"
        )
        sb.setValue(sb.maximum())

        self.current_conversation.append({
            "user": getattr(self, '_current_user_text', ''),
            "ai": response,
            "timestamp": datetime.now().isoformat(),
        })

        # 5. Auto-paste confirmation note
        if auto_pasted:
            self.chat_display.append(
                "<div style='margin:4px 0 6px 0;padding:6px 10px;background:#e3f2fd;"
                "border-left:3px solid #2196F3;border-radius:4px;color:#0d47a1;font-size:0.9em;'>"
                "SQL query auto-pasted to <b>SQL Notepad</b>.</div>"
            )
            sb.setValue(sb.maximum())

        self._chat_thread = None

    def _on_ai_error(self, error):
        self._set_generation_controls(False)
        self._stop_generation_status("Generation failed. See error in chat.", "error")
        html = self.chat_display.toHtml()
        html = html.replace("Thinking...", "")
        self.chat_display.setHtml(html)

        self.chat_display.append(
            f"<div style='margin:8px 0;padding:10px;background:#ffebee;"
            f"border-left:3px solid #f44336;border-radius:4px;'>"
            f"<b style='color:#c62828;'>Error:</b><br>"
            f"<div style='color:#000;margin-top:4px;'>{error}</div></div>"
        )
        self._chat_thread = None

    def _on_chat_thread_finished(self):
        if self._chat_thread is not None and not self._chat_thread.isRunning():
            self._chat_thread = None

    def _copy_last_sql_to_editor(self):
        """Extract the last SQL query from the chat and copy it to the main SQL editor."""
        text = self.chat_display.toPlainText()
        
        # Use the same extraction logic as schema validation for consistency
        sql_candidates = self._extract_sql_candidates(text)
        
        if sql_candidates:
            # Use the last SQL candidate (most recent)
            sql = sql_candidates[-1].strip()
            if sql:
                self.parent_editor.sql_text.setPlainText(sql)
                QMessageBox.information(self, "Copied", "SQL query copied to the main editor.")
                return
        
        # Fallback: try to find SQL in last AI response only
        lines = text.split('\n')
        ai_start = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith('AI:'):
                ai_start = i
                break
        
        if ai_start == -1:
            QMessageBox.information(self, "No SQL Found", "No AI response found in the chat.")
            return
        
        # Extract SQL from the AI response section only
        ai_response = '\n'.join(lines[ai_start + 1:])
        ai_sql = self._extract_sql_candidates(ai_response)
        
        if ai_sql:
            sql = ai_sql[-1].strip()
            if sql:
                self.parent_editor.sql_text.setPlainText(sql)
                QMessageBox.information(self, "Copied", "SQL query copied to the main editor.")
                return
        
        QMessageBox.information(self, "No SQL Found", "No SQL code block found in the last AI response.")

    def _extract_query_paths(self, query: str) -> list:
        """Extract file paths mentioned in the query."""
        import re
        paths = []
        
        # Pattern for read_parquet('path'), read_csv_auto('path'), etc.
        pattern1 = r"read_\w+\('([^']+)'\)"
        for match in re.finditer(pattern1, query, re.IGNORECASE):
            path = match.group(1)
            if path and path.replace('\\', '/'):
                paths.append(path.replace('\\', '/'))
        
        # Pattern for FROM 'path' or SELECT * FROM 'path'
        pattern2 = r"FROM\s+'([^']+)'"
        for match in re.finditer(pattern2, query, re.IGNORECASE):
            path = match.group(1)
            if path:
                paths.append(path.replace('\\', '/'))
        
        return list(set(paths))  # Remove duplicates

    def _get_sample_from_paths(self, paths: list) -> list:
        """Get sample data from specific file paths."""
        samples = []
        editor = self.parent_editor
        
        if not hasattr(editor, 'conn'):
            return samples
        
        for path in paths:
            try:
                # Try to read and get sample from the file
                if path.endswith('.parquet'):
                    query = f"SELECT * FROM read_parquet('{path}') LIMIT 1"
                elif path.endswith('.csv'):
                    query = f"SELECT * FROM read_csv_auto('{path}') LIMIT 1"
                elif path.endswith('.json'):
                    query = f"SELECT * FROM read_json_auto('{path}') LIMIT 1"
                else:
                    # Try auto-detection
                    query = f"SELECT * FROM '{path}' LIMIT 1"
                
                rows = editor.conn.execute(query).fetchall()
                if rows:
                    # Get column names
                    col_query = f"DESCRIBE SELECT * FROM read_parquet('{path}')" if path.endswith('.parquet') else query.replace('LIMIT 1', 'LIMIT 0')
                    try:
                        cols = editor.conn.execute(col_query).fetchall()
                        col_names = [c[0] for c in cols]
                    except:
                        col_names = [f"col_{i}" for i in range(len(rows[0]))]
                    
                    sample_lines = [", ".join(col_names)]
                    for r in rows:
                        sample_lines.append(", ".join(str(v) for v in r))
                    samples.append(f"Sample data from '{path}':\n" + "\n".join(sample_lines))
            except Exception:
                pass
        
        return samples


    def _build_system_prompt(self) -> str:
        """Build a compact, schema-aware system prompt for the AI model."""

        # â”€â”€ 1. Gather table schema & mappings FIRST (most important data) â”€â”€
        schema_lines = []
        sample_parts = []
        table_mapping_lines = []
        alias_lines = []
        editor = self.parent_editor
        selected_tables = self._get_selected_tables_for_ai()
        display_to_path = {}

        if hasattr(editor, 'conn') and hasattr(editor, 'uploaded_display_names'):
            # Build display-name â†’ file-path lookup
            if hasattr(editor, 'uploaded_files') and editor.uploaded_files:
                for i, fpath in enumerate(editor.uploaded_files):
                    if i < len(editor.uploaded_display_names):
                        display_to_path[editor.uploaded_display_names[i]] = fpath

            # Helper: resolve a display name to a read_parquet() source expression.
            # Tables are NOT registered in DuckDB; they're parquet files on disk.
            def _parquet_source(tname):
                fpath = display_to_path.get(tname)
                if not fpath:
                    doc_dir = getattr(editor, 'doc_dir', '')
                    fpath = os.path.join(doc_dir, f"{tname}.parquet")
                return f"read_parquet('{fpath.replace(chr(92), '/')}')"

            for table_name in selected_tables:
                source = _parquet_source(table_name)
                # Schema
                try:
                    cols = editor.conn.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
                    col_list = ", ".join(f"{self._sql_identifier_display(c[0])} ({c[1]})" for c in cols)
                    schema_lines.append(f"  {table_name}: {col_list}")
                except Exception:
                    schema_lines.append(f"  {table_name}: (schema unavailable)")

                # File mapping
                fpath = display_to_path.get(table_name)
                if fpath:
                    table_mapping_lines.append(f"  {table_name} -> {fpath.replace(chr(92), '/')}")

                aliases = self._table_aliases_for_prompt(table_name, fpath or "")
                if aliases:
                    alias_lines.append(f"  {table_name}: {', '.join(aliases)}")

                # Sample rows (max 1 per table, compact CSV format)
                try:
                    col_names = [d[0] for d in editor.conn.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
                    rows = editor.conn.execute(f"SELECT * FROM {source} LIMIT 1").fetchall()
                    if rows:
                        header = ", ".join(self._sql_identifier_display(col) for col in col_names)
                        data_lines = [", ".join(str(v) for v in r) for r in rows]
                        sample_parts.append(f"{table_name}:\n  {header}\n  " + "\n  ".join(data_lines))
                except Exception:
                    pass

        # â”€â”€ 2. Build compact system prompt â”€â”€
        parts = []

        # Core identity + critical rules (kept tight)
        parts.append(
            "You are a DuckDB SQL assistant for SimpliSQL.\n"
            "RULES:\n"
            "- ONLY use tables/columns listed below. Never invent names.\n"
            "- If a column name contains spaces, starts with a number, or includes special characters, reference it exactly with double quotes, for example \"2B Document No\".\n"
            "- If the user says PR files, gstr_2b_file, 2B file, or similar wording, match it to the closest selected table filename in TABLE ALIASES. Do not create table names from the user's wording.\n"
            "- If a CTE or subquery renames columns with AS, all later references to that CTE/subquery MUST use the renamed output columns, not the original source column names. Example: if a CTE selects \"PR Vendor GSTIN\" AS pr_vendor_gstin, later joins must use cte_alias.pr_vendor_gstin, not cte_alias.\"PR Vendor GSTIN\".\n"
            "- Use DuckDB syntax.\n"
            "- For date/time questions, ALWAYS use DuckDB date/time functions and INTERVAL syntax.\n"
            "- Prefer explicit casting for mixed text/date columns: TRY_CAST(col AS DATE) or TRY_CAST(col AS TIMESTAMP).\n"
            "- 'total'/'sum' requests MUST use SUM() aggregate. 'count' uses COUNT(). 'average' uses AVG().\n"
            "- When grouping, all non-aggregated columns MUST appear in GROUP BY.\n"
            "- QUALIFY is ONLY for filtering window function results (e.g. ROW_NUMBER() OVER (...)). NEVER use QUALIFY without a window function (OVER keyword). For normal filters use WHERE or HAVING.\n"
            "- QUALIFY must come AFTER GROUP BY and HAVING (clause order: FROMâ†’WHEREâ†’GROUP BYâ†’HAVINGâ†’QUALIFYâ†’ORDER BY).\n"
            "- Use simple table names in queries, not file paths or read_parquet().\n"
            "- If a table is not listed, tell the user to upload it.\n"
            "- 'last'/'first' = ordering (ROW_NUMBER, arg_max), NOT MAX/MIN.\n"
            "- 'all X last Y' = PARTITION BY X ORDER BY ... DESC with ROW_NUMBER()=1.\n"
            "- DuckDB can query files directly: SELECT * FROM 'path/file.csv'\n"
            "- If the user requests export to a file path, generate a COPY statement: COPY (<query>) TO '<path>' (HEADER, DELIMITER ',');\n"
            "- Use the exact file paths and filenames the user provides. Do not invent alternate paths or rename the output files.\n"
            "- If the user provides multiple export targets or asks for multiple output files, create a single reusable view or CTE and then emit multiple COPY statements for each target in the same SQL script.\n"
            "- Example for multiple exports: CREATE OR REPLACE VIEW final_recon AS (...); COPY (SELECT * FROM final_recon WHERE condition1) TO '.../mismatch.csv' (HEADER, DELIMITER ','); COPY (SELECT * FROM final_recon WHERE condition2) TO '.../nomatch.csv' (HEADER, DELIMITER ',');\n"
            "- Return ONE complete SQL script in a single ```sql``` block ending with ';'. Never use ellipsis ('...') or placeholders.\n"
            "GROUPING SIGNALS:\n"
            "- 'each X', 'per X', 'for every X', 'by X level' = needs GROUP BY X or PARTITION BY X.\n"
            "- 'last/first per group' = use ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) with QUALIFY or subquery.\n"
            "- LIMIT N limits total rows, NOT rows per group. For N per group, use window functions."
        )

        # Table schemas (essential - AI needs this to write correct queries)
        if schema_lines:
            parts.append("TABLES:\n" + "\n".join(schema_lines))
        else:
            parts.append("No tables loaded. User can query files by path.")

        # Table-to-file mapping (single unified section, not duplicated)
        if table_mapping_lines:
            parts.append("TABLE PATHS:\n" + "\n".join(table_mapping_lines))

        if alias_lines:
            parts.append("TABLE ALIASES (user wording -> actual table):\n" + "\n".join(alias_lines))

        # Sample data (budget-controlled)
        context_limit = self._get_model_context_limit()
        prompt_budget = max(1200, context_limit - 700)

        if sample_parts:
            sample_block = "SAMPLES (1 row each):\n" + "\n".join(sample_parts)
            candidate = "\n\n".join(parts) + "\n\n" + sample_block
            if self._estimate_tokens(candidate) <= prompt_budget:
                parts.append(sample_block)

        # DuckDB-specific cheat sheet (only differences from standard SQL)
        parts.append(
            "DUCKDB QUICK REFERENCE:\n"
            "- QUALIFY: ONLY for window function results. Example: QUALIFY ROW_NUMBER() OVER (PARTITION BY x ORDER BY y DESC) = 1\n"
            "  DO NOT use QUALIFY for plain column filters â€” use WHERE or HAVING instead.\n"
            "- arg_max(val, order), arg_min(val, order): value at max/min of order col\n"
            "- DATE/TIME (DuckDB):\n"
            "  date_trunc('month', ts_col), extract('year' FROM ts_col), strftime(ts_col, '%Y-%m')\n"
            "  current_date, current_timestamp, now()\n"
            "  date_diff('day', start_date, end_date), date_add(date_col, INTERVAL 7 DAY), date_sub('day', start_date, end_date)\n"
            "  date_col >= current_date - INTERVAL 30 DAY\n"
            "- Avoid non-DuckDB dialect functions: DATE_FORMAT(), TIMESTAMPDIFF(), GETDATE(), TO_CHAR(), ILIKE ANY\n"
            "- TRY_CAST(x AS type): safe cast returning NULL on error\n"
            "- SELECT * EXCLUDE (col), SELECT * REPLACE (expr AS col)\n"
            "- PIVOT / UNPIVOT, FILTER clause for conditional aggregation\n"
            "- read_parquet(), read_csv_auto(), read_json_auto() for file queries\n"
            "- FILE_BASENAME(p), FILE_DIRNAME(p), FILE_NAME_NO_EXT(p), FILE_EXTENSION(p)"
        )

        base = "\n\n".join(parts)

        # Schema budget guardrail: trim if still too large
        if self._estimate_tokens(base) > prompt_budget and schema_lines:
            # Keep as many schemas as fit
            trimmed = []
            for line in schema_lines:
                trial = base.replace("\n".join(schema_lines), "\n".join(trimmed + [line]))
                if self._estimate_tokens(trial) > prompt_budget:
                    break
                trimmed.append(line)
            if not trimmed:
                trimmed = [schema_lines[0]]
            omitted = len(schema_lines) - len(trimmed)
            if omitted > 0:
                trimmed.append(f"  ... {omitted} more tables omitted (select fewer in Settings)")
            base = base.replace("\n".join(schema_lines), "\n".join(trimmed))

        # Include current editor query if checkbox is on
        if self.context_query_check.isChecked():
            query = editor.sql_text.toPlainText().strip()
            if query:
                base += f"\n\nCURRENT QUERY:\n{query}"
                query_paths = self._extract_query_paths(query)
                if query_paths:
                    samples = self._get_sample_from_paths(query_paths)
                    if samples:
                        base += "\n" + "\n".join(samples)

        return base

    def clear_chat(self):
        reply = QMessageBox.question(self, "Clear Chat", "Clear chat history?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.chat_display.clear()
            self.current_conversation.clear()
            if self.client.is_loaded():
                self._add_welcome_message()

    def closeEvent(self, event):
        if self._force_close:
            event.accept()
        else:
            event.ignore()
            self.hide()

