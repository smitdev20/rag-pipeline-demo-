"""NiceGUI chat interface with SSE streaming support."""

import json
import os
import uuid
from collections.abc import Callable
from datetime import datetime

import httpx
from nicegui import events, ui

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"
      rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
    * { font-family: 'Inter', sans-serif; }
    
    body { background: #f5f5f5; min-height: 100vh; }
    
    .app-container {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    
    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    .message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
    }
    
    .message-assistant {
        background: #f3f4f6;
        color: #1f2937;
        border-radius: 18px 18px 18px 4px;
    }
    
    .avatar-user { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .avatar-assistant { background: #6b7280; }
    
    .typing-dot {
        width: 8px; height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-6px); }
    }
    
    .input-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        transition: border-color 0.2s;
    }
    .input-box:focus-within { border-color: #667eea; }
    
    .send-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; }
    
    .attach-btn { 
        background: transparent !important;
        color: #9ca3af !important;
        min-width: 32px !important;
        min-height: 32px !important;
        padding: 0 !important;
    }
    .attach-btn:hover { color: #667eea !important; }
    
    .upload-progress {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px 12px;
        animation: fadeIn 0.2s ease;
    }
    .upload-progress-bar {
        height: 4px;
        background: #e5e7eb;
        border-radius: 2px;
        overflow: hidden;
    }
    .upload-progress-fill {
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: progress 1.5s ease-in-out infinite;
    }
    @keyframes progress {
        0% { width: 0%; margin-left: 0%; }
        50% { width: 50%; margin-left: 25%; }
        100% { width: 0%; margin-left: 100%; }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message-system {
        background: #e0e7ff;
        color: #4338ca;
        border-radius: 8px;
        font-size: 0.875rem;
    }
    
    /* Markdown styling for assistant messages */
    .message-assistant .prose { color: inherit; }
    .message-assistant .prose strong { font-weight: 600; }
    .message-assistant .prose em { font-style: italic; }
    .message-assistant .prose pre { 
        background: #1f2937; 
        color: #f3f4f6;
        border-radius: 0.5rem; 
        padding: 0.75rem;
        margin: 0.5rem 0;
        overflow-x: auto;
        font-size: 0.75rem;
    }
    .message-assistant .prose code { 
        font-family: 'Menlo', 'Monaco', monospace;
        font-size: 0.8em;
    }
    .message-assistant .prose :not(pre) > code {
        background: #e5e7eb;
        color: #db2777;
        padding: 0.125rem 0.375rem;
        border-radius: 0.25rem;
    }
    .message-assistant .prose ul { 
        list-style-type: disc; 
        padding-left: 1.25rem;
        margin: 0.5rem 0;
    }
    .message-assistant .prose ol { 
        list-style-type: decimal; 
        padding-left: 1.25rem;
        margin: 0.5rem 0;
    }
    .message-assistant .prose li { margin: 0.25rem 0; }
    .message-assistant .prose li > ul,
    .message-assistant .prose li > ol { margin: 0.25rem 0; }
    .message-assistant .prose a { color: #4f46e5; text-decoration: underline; }
    .message-assistant .prose h3 { 
        font-weight: 600; 
        font-size: 1rem;
        margin: 0.75rem 0 0.5rem 0;
    }
    .message-assistant .prose p { margin: 0.5rem 0; }
    .message-assistant .prose p:first-child { margin-top: 0; }
    .message-assistant .prose p:last-child { margin-bottom: 0; }
</style>
"""


class ChatSession:
    """Manages chat state for a user session."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.session_id: str = str(uuid.uuid4())
        self.is_streaming: bool = False

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%I:%M %p"),
        })


async def stream_chat_response(
    message: str,
    session_id: str,
    on_chunk: Callable[[str], None],
    on_status: Callable[[str], None],
    on_complete: Callable[[], None],
    on_error: Callable[[str], None],
) -> None:
    """Consume SSE stream from /chat/stream endpoint."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/chat/stream",
                json={"message": message, "session_id": session_id},
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if data.get("error"):
                        on_error(data["error"])
                        return
                    if data.get("done"):
                        on_complete()
                        return
                    if status := data.get("status"):
                        on_status(status)
                    if content := data.get("content"):
                        on_chunk(content)
        except httpx.HTTPStatusError as e:
            on_error(f"HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            on_error(f"Connection failed: {e}")


@ui.page("/")
def chat_page() -> None:
    """Main chat page."""
    ui.add_head_html(CUSTOM_CSS)
    session = ChatSession()

    messages_container: ui.column
    response_label: ui.markdown
    input_field: ui.textarea
    send_btn: ui.button
    upload_progress: ui.element

    def render_avatar(is_user: bool) -> None:
        css = "avatar-user" if is_user else "avatar-assistant"
        icon = "person" if is_user else "robot"
        avatar_classes = f"w-9 h-9 rounded-full flex items-center justify-center {css}"
        with ui.element("div").classes(avatar_classes):
            ui.icon(icon).classes("text-white text-lg")

    def render_message(msg: dict) -> None:
        role = msg["role"]
        
        # System messages (like file uploads) - centered with icon
        if role == "system":
            with (
                ui.row().classes("w-full justify-center"),
                ui.element("div").classes("message-system px-4 py-2 flex items-center gap-2"),
            ):
                ui.icon("description").classes("text-indigo-600")
                ui.label(msg["content"]).classes("text-sm")
            return
        
        is_user = role == "user"
        align = "justify-end" if is_user else "justify-start"
        bubble = "message-user" if is_user else "message-assistant"

        with ui.row().classes(f"w-full {align} gap-3 items-end"):
            if not is_user:
                render_avatar(False)
            with ui.column().classes("max-w-[70%] gap-1"):
                with ui.element("div").classes(f"px-4 py-3 {bubble}"):
                    # Render markdown for assistant, plain text for user
                    if is_user:
                        ui.html(
                            msg["content"].replace("\n", "<br>"), sanitize=False
                        ).classes("text-sm leading-relaxed")
                    else:
                        ui.markdown(msg["content"]).classes(
                            "text-sm leading-relaxed prose prose-sm max-w-none"
                        )
                ui.label(msg["time"]).classes(
                    f"text-[10px] text-gray-400 {'self-end' if is_user else 'self-start'}"
                )
            if is_user:
                render_avatar(True)

    def refresh_messages() -> None:
        messages_container.clear()
        with messages_container:
            if not session.messages:
                with ui.column().classes("w-full h-64 items-center justify-center gap-3"):
                    ui.icon("forum").classes("text-5xl text-gray-300")
                    ui.label("Start a conversation").classes("text-lg text-gray-400")
            else:
                for msg in session.messages:
                    render_message(msg)

    def render_status_indicator(status_text: str = "Thinking") -> tuple[ui.row, ui.label]:
        """Render status indicator with animated dots and status text."""
        with ui.row().classes("w-full justify-start gap-3 items-end") as row:
            render_avatar(False)
            with (
                ui.element("div").classes("message-assistant px-4 py-3"),
                ui.row().classes("items-center gap-2"),
            ):
                with ui.row().classes("gap-1"):
                    for _ in range(3):
                        ui.element("div").classes("typing-dot")
                status_label = ui.label(status_text).classes(
                    "text-sm text-gray-500 italic"
                )
        return row, status_label

    def show_upload_progress(filename: str) -> None:
        """Show upload progress indicator."""
        upload_progress.clear()
        upload_progress.set_visibility(True)
        with upload_progress:
            with ui.row().classes("items-center gap-2 w-full"):
                ui.icon("description").classes("text-indigo-500")
                ui.label(filename).classes("text-sm text-gray-700 flex-grow truncate")
                ui.spinner(size="sm").classes("text-indigo-500")
            with ui.element("div").classes("upload-progress-bar mt-2"):
                ui.element("div").classes("upload-progress-fill w-full")

    def hide_upload_progress() -> None:
        """Hide upload progress indicator."""
        upload_progress.clear()
        upload_progress.set_visibility(False)

    async def handle_upload(e: events.UploadEventArguments) -> None:
        """Handle PDF file upload."""
        filename = e.file.name

        # Client-side validation (also enforced by accept prop)
        if not filename.lower().endswith(".pdf"):
            ui.notify("Only PDF files are allowed", type="negative")
            return

        # Show progress indicator
        show_upload_progress(filename)

        try:
            content = await e.file.read()

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{API_BASE_URL}/upload/pdf",
                    files={"file": (filename, content, "application/pdf")},
                )

                hide_upload_progress()

                if response.status_code == 200:
                    data = response.json()
                    pages = data.get("pages", 0)

                    # Add system message showing upload
                    session.add_message(
                        "system",
                        f"Uploaded: {filename} ({pages} pages)",
                    )
                    refresh_messages()

                    ui.notify(
                        f"Uploaded {filename} ({pages} pages)",
                        type="positive",
                    )
                else:
                    error_detail = response.json().get("detail", "Upload failed")
                    ui.notify(f"Error: {error_detail}", type="negative")

        except httpx.RequestError as err:
            hide_upload_progress()
            ui.notify(f"Connection error: {err}", type="negative")
        except Exception as err:
            hide_upload_progress()
            ui.notify(f"Upload failed: {err}", type="negative")

    async def send_message() -> None:
        nonlocal response_label
        text = input_field.value.strip()
        if not text or session.is_streaming:
            return

        input_field.value = ""
        session.is_streaming = True
        send_btn.disable()

        session.add_message("user", text)
        refresh_messages()

        with messages_container:
            status_row, status_label = render_status_indicator()

        accumulated = ""
        msg_time = datetime.now().strftime("%I:%M %p")

        status_messages = {
            "thinking": "Thinking...",
            "searching": "Searching documents...",
            "generating": "Generating response...",
        }

        def on_status(status: str) -> None:
            """Update the status indicator text."""
            if status in status_messages:
                status_label.set_text(status_messages[status])

        def on_chunk(content: str) -> None:
            nonlocal accumulated, response_label
            if not accumulated:
                status_row.delete()
                with (
                    messages_container,
                    ui.row().classes("w-full justify-start gap-3 items-end"),
                ):
                    render_avatar(False)
                    with ui.column().classes("max-w-[70%] gap-1"):
                        with ui.element("div").classes("message-assistant px-4 py-3"):
                            response_label = ui.markdown("").classes(
                                "text-sm leading-relaxed prose prose-sm max-w-none"
                            )
                        ui.label(msg_time).classes("text-[10px] text-gray-400")
            accumulated += content
            response_label.set_content(accumulated)

        def on_complete() -> None:
            session.add_message("assistant", accumulated)
            session.is_streaming = False
            send_btn.enable()
            refresh_messages()

        def on_error(error: str) -> None:
            status_row.delete()
            session.add_message("assistant", f"Error: {error}")
            session.is_streaming = False
            send_btn.enable()
            refresh_messages()
            ui.notify(error, type="negative")

        await stream_chat_response(
            text, session.session_id, on_chunk, on_status, on_complete, on_error
        )

    def new_chat() -> None:
        session.messages.clear()
        session.session_id = str(uuid.uuid4())
        refresh_messages()

    # === UI Layout ===
    with (
        ui.element("div").classes("w-full min-h-screen p-4 md:p-8"),
        ui.column().classes("w-full max-w-3xl mx-auto app-container").style(
            "height: calc(100vh - 4rem)"
        ),
    ):
            # Header
            with ui.row().classes("w-full header px-5 py-4 items-center justify-between"):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("smart_toy").classes("text-white text-3xl")
                    ui.label("RAG Assistant").classes("text-lg font-semibold text-white")
                with ui.row().classes("items-center gap-3"):
                    with ui.element("div").classes(
                        "bg-white/20 rounded-full px-3 py-1 flex items-center gap-2"
                    ):
                        ui.icon("tag").classes("text-white/80 text-sm")
                        ui.label().bind_text_from(
                            session, "session_id", lambda s: s[:8].upper()
                        ).classes("text-xs text-white/80 font-mono")
                    ui.button(icon="add", on_click=new_chat).props("flat round color=white")

            # Messages
            with (
                ui.scroll_area().classes("flex-grow w-full bg-gray-50"),
                ui.column().classes("w-full p-5"),
            ):
                messages_container = ui.column().classes("w-full gap-4")
                refresh_messages()

            # Input area with upload progress
            with ui.column().classes("w-full bg-white border-t"):
                # Upload progress (hidden by default)
                upload_progress = ui.element("div").classes(
                    "upload-progress mx-4 mt-3"
                )
                upload_progress.set_visibility(False)

                # Input row
                with ui.row().classes("w-full p-4 pt-2 gap-2 items-end"):
                    with (
                        ui.element("div").classes("flex-grow input-box px-3 py-2"),
                        ui.row().classes("w-full items-center gap-2"),
                    ):
                        # Attachment button
                        upload = ui.upload(
                            on_upload=handle_upload,
                            on_rejected=lambda: ui.notify(
                                "Only PDFs under 10MB", type="warning"
                            ),
                            max_file_size=10_000_000,
                            auto_upload=True,
                        ).props('accept=.pdf').classes("hidden")

                        ui.button(
                            icon="attach_file",
                            on_click=lambda: upload.run_method('pickFiles')
                        ).props("flat dense round").classes("attach-btn").tooltip("Upload PDF")

                        # Text input
                        input_field = (
                            ui.textarea(placeholder="Type a message...")
                            .props("autogrow borderless dense rows=1")
                            .classes("flex-grow")
                            .on("keydown.enter.prevent", send_message)
                        )
                    send_btn = (
                        ui.button(icon="send", on_click=send_message)
                        .props("round unelevated")
                        .classes("send-btn")
                    )


def main() -> None:
    ui.run(title="RAG Assistant", port=8080, reload=False)


if __name__ == "__main__":
    main()
