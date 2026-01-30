"""NiceGUI chat interface with SSE streaming support."""

import json
import uuid
from collections.abc import Callable
from datetime import datetime

import httpx
from nicegui import ui

API_BASE_URL = "http://localhost:8000"

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
    response_label: ui.html
    input_field: ui.textarea
    send_btn: ui.button

    def render_avatar(is_user: bool) -> None:
        css = "avatar-user" if is_user else "avatar-assistant"
        icon = "person" if is_user else "robot"
        avatar_classes = f"w-9 h-9 rounded-full flex items-center justify-center {css}"
        with ui.element("div").classes(avatar_classes):
            ui.icon(icon).classes("text-white text-lg")

    def render_message(msg: dict) -> None:
        is_user = msg["role"] == "user"
        align = "justify-end" if is_user else "justify-start"
        bubble = "message-user" if is_user else "message-assistant"

        with ui.row().classes(f"w-full {align} gap-3 items-end"):
            if not is_user:
                render_avatar(False)
            with ui.column().classes("max-w-[70%] gap-1"):
                with ui.element("div").classes(f"px-4 py-3 {bubble}"):
                    ui.html(msg["content"].replace("\n", "<br>"), sanitize=False).classes("text-sm")
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

    def render_typing() -> ui.row:
        with ui.row().classes("w-full justify-start gap-3 items-end") as row:
            render_avatar(False)
            with (
                ui.element("div").classes("message-assistant px-4 py-3"),
                ui.row().classes("gap-1"),
            ):
                for _ in range(3):
                    ui.element("div").classes("typing-dot")
        return row

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
            typing = render_typing()

        accumulated = ""
        msg_time = datetime.now().strftime("%I:%M %p")

        def on_chunk(content: str) -> None:
            nonlocal accumulated, response_label
            if not accumulated:
                typing.delete()
                with (
                    messages_container,
                    ui.row().classes("w-full justify-start gap-3 items-end"),
                ):
                    render_avatar(False)
                    with ui.column().classes("max-w-[70%] gap-1"):
                        with ui.element("div").classes("message-assistant px-4 py-3"):
                            response_label = ui.html("", sanitize=False).classes("text-sm")
                        ui.label(msg_time).classes("text-[10px] text-gray-400")
            accumulated += content
            response_label.set_content(accumulated.replace("\n", "<br>"))

        def on_complete() -> None:
            session.add_message("assistant", accumulated)
            session.is_streaming = False
            send_btn.enable()
            refresh_messages()

        def on_error(error: str) -> None:
            typing.delete()
            session.add_message("assistant", f"Error: {error}")
            session.is_streaming = False
            send_btn.enable()
            refresh_messages()
            ui.notify(error, type="negative")

        await stream_chat_response(text, session.session_id, on_chunk, on_complete, on_error)

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

            # Input
            with ui.row().classes("w-full p-4 gap-3 items-end bg-white border-t"):
                with ui.element("div").classes("flex-grow input-box px-3 py-2"):
                    input_field = (
                        ui.textarea(placeholder="Type a message...")
                        .props("autogrow borderless dense rows=1")
                        .classes("w-full")
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
