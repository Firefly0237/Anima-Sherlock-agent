"""Standard-library HTTP wrapper for the minimal Sherlock case-game demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from anima.case_game.core.demo import (
    CaseGameDemo,
    CaseGameDemoError,
    CaseGamePersistenceError,
    CaseGameVersionConflictError,
    HostAnswerer,
)
from anima.case_game.runtime.model import LiveCaseAnswerer, load_expected_model_identity
from anima.case_game.runtime.player_memory import (
    DEFAULT_MEMORY_MIN_SCORE,
    DEFAULT_MEMORY_TOP_K,
    PlayerMemoryError,
    PlayerMemoryService,
)
from anima.case_game.runtime.session_store import CaseSessionStore, PostgresCaseSessionStore
from anima.case_game.runtime.tools import LiveCaseToolProposer
from anima.persona.contracts.pack import load_pack
from anima.serve.core.repositories import PostgresRepository
from anima.serve.inference.embedding import BGEEmbedder
from anima.serve.inference.model_client import GenerationConfig, OpenAIChatClient

WEB_ROOT = Path(__file__).with_name("web")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/app.css": ("app.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/adventures-cover.jpg": (
        "assets/adventures-of-sherlock-holmes-cover.jpg",
        "image/jpeg",
    ),
    "/favicon.ico": (
        "assets/adventures-of-sherlock-holmes-cover.jpg",
        "image/jpeg",
    ),
}


def run_demo_server(
    levels_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    host_answerer: HostAnswerer | None = None,
    memory_service: PlayerMemoryService | None = None,
    session_store: CaseSessionStore | None = None,
    runtime_info: Mapping[str, Any] | None = None,
    tool_proposer: LiveCaseToolProposer | None = None,
) -> None:
    demo = CaseGameDemo(
        levels_root,
        host_answerer=host_answerer,
        memory_service=memory_service,
        session_store=session_store,
        runtime_info=runtime_info,
        tool_proposer=tool_proposer,
    )
    server = ThreadingHTTPServer((host, port), create_demo_handler(demo))
    print(
        f"Sherlock case-game demo ({demo.runtime_info['mode']}) listening on http://{host}:{port}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if memory_service is not None:
            memory_service.close()
        elif session_store is not None:
            close = getattr(session_store, "close", None)
            if callable(close):
                close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "levels_root",
        nargs="?",
        type=Path,
        default=Path("assets/cases/sherlock/levels"),
    )
    parser.add_argument("host", nargs="?", default="127.0.0.1")
    parser.add_argument("port", nargs="?", type=int, default=8765)
    parser.add_argument("--model-base-url")
    parser.add_argument("--model-identity", type=Path)
    parser.add_argument(
        "--persona-pack-dir",
        type=Path,
        default=Path("persona_packs/public/sherlock_holmes"),
    )
    parser.add_argument("--model-timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-history-turns", type=int, default=12)
    parser.add_argument(
        "--memory-dsn",
        default=os.environ.get("ANIMA_MEMORY_DSN"),
        help="PostgreSQL/pgvector DSN; enables cross-session player memory",
    )
    parser.add_argument(
        "--session-dsn",
        default=os.environ.get("ANIMA_GAME_DSN"),
        help="PostgreSQL DSN for persistent game sessions/turns; defaults to --memory-dsn",
    )
    parser.add_argument("--memory-device", default="cpu")
    parser.add_argument("--memory-top-k", type=int, default=DEFAULT_MEMORY_TOP_K)
    parser.add_argument(
        "--memory-min-score",
        type=float,
        default=DEFAULT_MEMORY_MIN_SCORE,
    )
    parsed = parser.parse_args(sys.argv[1:] if argv is None else argv)

    answerer = None
    memory_service = None
    session_store = None
    runtime_info = None
    tool_proposer = None
    repository = None
    session_dsn = parsed.session_dsn or parsed.memory_dsn
    if parsed.memory_dsn and session_dsn != parsed.memory_dsn:
        parser.error("--memory-dsn and --session-dsn must match for atomic game+memory commit")
    if session_dsn:
        repository = PostgresRepository(session_dsn)
        repository.migrate()
        session_store = PostgresCaseSessionStore(repository)
    if parsed.model_base_url or parsed.model_identity:
        if not parsed.model_base_url or parsed.model_identity is None:
            parser.error("live DPO mode requires --model-base-url and --model-identity")
        if parsed.max_history_turns < 0:
            parser.error("--max-history-turns must be >= 0")
        expected_identity = load_expected_model_identity(parsed.model_identity)
        client = OpenAIChatClient(
            base_url=parsed.model_base_url,
            model=str(expected_identity["base_model"]),
            adapter=(
                str(expected_identity["adapter_id"])
                if expected_identity.get("adapter_id") is not None
                else None
            ),
            timeout_s=parsed.model_timeout_s,
            generation=GenerationConfig(
                temperature=parsed.temperature,
                top_p=parsed.top_p,
                max_tokens=parsed.max_tokens,
                upstream_stream=True,
            ),
        )
        if parsed.memory_dsn:
            assert repository is not None  # session_dsn inherits memory_dsn above
            memory_service = PlayerMemoryService(
                repository=repository,
                embedder=BGEEmbedder(device=parsed.memory_device),
                pack=load_pack(parsed.persona_pack_dir),
                top_k=parsed.memory_top_k,
                min_score=parsed.memory_min_score,
            )
        answerer = LiveCaseAnswerer(
            model_client=client,
            persona_pack_dir=parsed.persona_pack_dir,
            expected_identity=expected_identity,
            memory_service=memory_service,
            max_history_turns=parsed.max_history_turns,
        )
        runtime_info = answerer.runtime_info
        tool_proposer = LiveCaseToolProposer(client)
    elif parsed.memory_dsn:
        parser.error("persistent memory is available only in live DPO mode")

    run_demo_server(
        parsed.levels_root,
        host=parsed.host,
        port=parsed.port,
        host_answerer=answerer,
        memory_service=memory_service,
        session_store=session_store,
        runtime_info=runtime_info,
        tool_proposer=tool_proposer,
    )


def create_demo_handler(demo: CaseGameDemo):
    class DemoHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                path = urlparse(self.path).path
                static_file = STATIC_FILES.get(path)
                if static_file is not None:
                    relative_path, content_type = static_file
                    _send_file(self, WEB_ROOT / relative_path, content_type=content_type)
                    return
                if path == "/case-game/cases":
                    _send_json(self, demo.list_cases())
                    return
                parts = _parts(path)
                if len(parts) == 3 and parts[:2] == ["case-game", "sessions"]:
                    _send_json(self, demo.get_session(parts[2]))
                    return
                if (
                    len(parts) == 4
                    and parts[:2] == ["case-game", "sessions"]
                    and parts[3] == "transcript"
                ):
                    _send_json(self, demo.export_transcript(parts[2]))
                    return
                _send_json(self, {"error": "not_found"}, status=404)
            except CaseGameVersionConflictError as exc:
                _send_json(self, {"error": str(exc)}, status=409)
            except CaseGamePersistenceError as exc:
                _send_json(self, {"error": str(exc)}, status=503)
            except CaseGameDemoError as exc:
                _send_json(self, {"error": str(exc)}, status=404)
            except PlayerMemoryError as exc:
                _send_json(self, {"error": str(exc)}, status=503)

        def do_POST(self) -> None:
            try:
                path = urlparse(self.path).path
                body = _read_json_body(self)
                if path == "/case-game/sessions":
                    _send_json(
                        self,
                        demo.start_session(
                            str(body.get("case_id", "")),
                            player_id=(
                                str(body["player_id"])
                                if body.get("player_id") is not None
                                else None
                            ),
                        ),
                    )
                    return
                parts = _parts(path)
                if (
                    len(parts) == 4
                    and parts[:2] == ["case-game", "sessions"]
                    and parts[3] == "turns"
                ):
                    _send_json(
                        self,
                        demo.submit_turn(
                            parts[2],
                            action=str(body.get("action", "")),
                            player_text=str(body.get("player_text", "")),
                            target_id=(str(body["target_id"]) if body.get("target_id") else None),
                            request_id=(
                                str(body["request_id"])
                                if body.get("request_id") is not None
                                else None
                            ),
                            expected_state_version=(
                                int(body["state_version"])
                                if body.get("state_version") is not None
                                else None
                            ),
                            input_mode=(
                                str(body["input_mode"])
                                if body.get("input_mode") is not None
                                else None
                            ),
                        ),
                    )
                    return
                _send_json(self, {"error": "not_found"}, status=404)
            except CaseGameVersionConflictError as exc:
                _send_json(self, {"error": str(exc)}, status=409)
            except CaseGamePersistenceError as exc:
                _send_json(self, {"error": str(exc)}, status=503)
            except CaseGameDemoError as exc:
                _send_json(self, {"error": str(exc)}, status=404)
            except PlayerMemoryError as exc:
                _send_json(self, {"error": str(exc)}, status=503)
            except ValueError as exc:
                _send_json(self, {"error": str(exc)}, status=422)

        def log_message(self, _format: str, *args: Any) -> None:
            return

    return DemoHandler


def _parts(path: str) -> list[str]:
    return [part for part in path.strip("/").split("/") if part]


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length).decode("utf-8") if length else "{}"
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON body must be an object")
    return value


def _send_json(
    handler: BaseHTTPRequestHandler, payload: dict[str, Any], *, status: int = 200
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_file(
    handler: BaseHTTPRequestHandler,
    path: Path,
    *,
    content_type: str,
) -> None:
    if not path.is_file():
        _send_json(handler, {"error": "static_file_missing"}, status=404)
        return
    body = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


if __name__ == "__main__":
    main()
