from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from anima.case_game import CaseGameDemo, CaseGameDemoError
from anima.case_game.runtime.demo_server import create_demo_handler
from tests.support.environment.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
LEVELS = ROOT / "assets" / "cases" / "sherlock" / "levels"


def _demo() -> CaseGameDemo:
    counter = iter(("session-1", "session-2", "session-3"))
    return CaseGameDemo(LEVELS, session_id_factory=lambda: next(counter))


def test_demo_lists_cases_and_starts_session() -> None:
    demo = _demo()

    cases = demo.list_cases()["cases"]
    started = demo.start_session("red_headed_league")

    assert {case["case_id"] for case in cases} == {"red_headed_league", "speckled_band"}
    assert started["session_id"] == "session-1"
    assert started["case_id"] == "red_headed_league"
    assert started["state"]["stage"] == "premise"
    assert started["progress"]["unlocked_evidence"] == 4
    assert started["evidence_board"]
    assert started["investigation_leads"]
    assert "提交最终推理" in started["objective"]


def test_default_session_identifier_is_not_sequential() -> None:
    demo = CaseGameDemo(LEVELS)

    first = demo.start_session("red_headed_league")["session_id"]
    second = demo.start_session("speckled_band")["session_id"]

    assert first.startswith("game-") and len(first) == 37
    assert second.startswith("game-") and len(second) == 37
    assert first != second
    assert first not in {"game-1", "game-2"}


def test_demo_turn_updates_evidence_board_and_transcript() -> None:
    demo = _demo()
    session_id = demo.start_session("red_headed_league")["session_id"]

    turn = demo.submit_turn(session_id, action="ask", player_text="助手有什么问题？")
    transcript = demo.export_transcript(session_id)

    assert turn["turn"]["accepted"] is True
    assert turn["state"]["stage"] == "investigation"
    assert turn["turn"]["unlocked_evidence_ids"] == ["rhl.ev.004", "rhl.ev.005"]
    assert any(row["evidence_id"] == "rhl.ev.004" for row in turn["evidence_board"])
    assert transcript["turns"][0]["action"] == "ask"
    assert transcript["turns"][0]["trace_metadata"]["case_id"] == "red_headed_league"


def test_demo_structured_lead_round_trips_target_id() -> None:
    demo = _demo()
    started = demo.start_session("red_headed_league")
    lead = next(row for row in started["investigation_leads"] if row["action"] == "ask")

    turn = demo.submit_turn(
        started["session_id"],
        action=lead["action"],
        player_text=lead["player_text"],
        target_id=lead["target_id"],
    )

    assert turn["turn"]["accepted"] is True
    assert turn["turn"]["selected_target_id"] == lead["target_id"]
    assert turn["turn"]["trace_metadata"]["game_selected_target_id"] == lead["target_id"]
    assert turn["turn"]["unlocked_evidence_ids"]


def test_demo_can_solve_and_export_post_case_transcript() -> None:
    demo = _demo()
    session_id = demo.start_session("speckled_band")["session_id"]

    for action, text in (
        ("ask", "遗产有什么关系？"),
        ("ask", "为什么说是密室？"),
        ("ask", "为什么通风口连到隔壁？"),
        ("ask", "床为什么固定？"),
        ("ask", "保险柜和牛奶说明什么？"),
        ("ask", "短鞭为什么打结？"),
    ):
        demo.submit_turn(session_id, action=action, player_text=text)
    solved = demo.submit_turn(
        session_id,
        action="solve",
        player_text=(
            "Roylott 是凶手，他为了钱阻止继女结婚，让 Helen 面临和 Julia 同样危险。"
            "维修是借口，门窗锁住说明普通入口不成立；通风口、假铃绳、固定床、保险柜、牛奶碟和短鞭是一组物证。"
            "哨声和金属声对应夜间控制，吉普赛人是误导；斑点带子是 Roylott 控制的毒蛇。"
        ),
    )
    transcript = demo.export_transcript(session_id)

    assert solved["state"]["stage"] == "post_case"
    assert solved["state"]["solved"] is True
    assert solved["turn"]["solve_status"] == "pass"
    assert transcript["final_state"]["solved"] is True
    assert len(transcript["turns"]) == 7
    assert transcript["case_pack_sha256"] == demo.engines["speckled_band"].pack.manifest_sha256


def test_scripted_post_case_recap_does_not_invite_another_investigation_step() -> None:
    demo = _demo()
    session_id = demo.start_session("red_headed_league")["session_id"]
    demo.submit_turn(
        session_id,
        action="solve",
        player_text=(
            "红发会是骗局，用高薪把 Wilson 固定支开。Spaulding 就是 John Clay，"
            "他从地下室挖向相邻银行；解散说明准备完成，应立即设伏。"
        ),
    )

    recap = demo.submit_turn(session_id, action="recap", player_text="请复盘。")

    assert recap["state"]["solved"] is True
    assert recap["turn"]["host_answer"].startswith("案件已经结案")
    assert "再选择下一步行动" not in recap["turn"]["host_answer"]


def test_demo_rejects_unknown_case_and_session() -> None:
    demo = _demo()

    with pytest.raises(CaseGameDemoError, match="unknown case_id"):
        demo.start_session("missing")
    with pytest.raises(CaseGameDemoError, match="unknown session_id"):
        demo.get_session("missing")


def test_demo_http_server_serves_browser_app_and_api() -> None:
    demo = _demo()
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_demo_handler(demo))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base_url}/", timeout=3) as response:
            html = response.read().decode("utf-8")
            assert response.headers.get_content_type() == "text/html"
        with urllib.request.urlopen(f"{base_url}/static/app.css", timeout=3) as response:
            css = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{base_url}/static/app.js", timeout=3) as response:
            javascript = response.read().decode("utf-8")
        with urllib.request.urlopen(
            f"{base_url}/static/adventures-cover.jpg", timeout=3
        ) as response:
            cover = response.read()
            assert response.headers.get_content_type() == "image/jpeg"
        with urllib.request.urlopen(f"{base_url}/case-game/cases", timeout=3) as response:
            cases = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert "Sherlock Case Desk" in html
    assert ".workspace" in css
    assert "可执行线索" in html
    assert "reset-button" in html
    assert "sessionsByCase" in javascript
    assert "selectedActionsByCase" in javascript
    assert "localStorage" in javascript
    assert "player_id: getPlayerId()" in javascript
    assert "request_id: state.pendingTurn.requestId" in javascript
    assert "state_version: state.session.state_version" in javascript
    assert "state.pendingTurn && state.pendingTurn.key === pendingKey" in javascript
    assert "error.status === 409" in javascript
    assert "回复已按当前可见证据收束" in javascript
    assert "模型回复越过当前剧透边界" not in javascript
    assert cover.startswith(b"\xff\xd8")
    assert cases["runtime"]["mode"] == "scripted_smoke"
    assert len(cases["cases"]) == 2
