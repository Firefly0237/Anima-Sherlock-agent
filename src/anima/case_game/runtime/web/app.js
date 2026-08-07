"use strict";

const PLAYER_ID_STORAGE_KEY = "anima.caseGame.playerId.v1";
const PLAYER_ID_PATTERN = /^[A-Za-z0-9_-]{16,128}$/;
let ephemeralPlayerId = null;

function mintPlayerId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  const random = Math.random().toString(36).slice(2);
  return `player_${Date.now().toString(36)}_${random.padEnd(12, "0")}`;
}

function mintRequestId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return `turn-${globalThis.crypto.randomUUID()}`;
  }
  const random = Math.random().toString(36).slice(2);
  return `turn-${Date.now().toString(36)}-${random.padEnd(12, "0")}`;
}

function getPlayerId() {
  if (ephemeralPlayerId) return ephemeralPlayerId;
  try {
    const stored = globalThis.localStorage.getItem(PLAYER_ID_STORAGE_KEY);
    if (stored && PLAYER_ID_PATTERN.test(stored)) {
      ephemeralPlayerId = stored;
      return stored;
    }
    ephemeralPlayerId = mintPlayerId();
    globalThis.localStorage.setItem(PLAYER_ID_STORAGE_KEY, ephemeralPlayerId);
  } catch (_error) {
    ephemeralPlayerId = mintPlayerId();
  }
  return ephemeralPlayerId;
}

const state = {
  cases: [],
  runtime: null,
  session: null,
  sessionsByCase: new Map(),
  selectedActionsByCase: new Map(),
  completedCaseIds: new Set(),
  selectedAction: "auto",
  selectedTarget: null,
  pendingTurn: null,
  activeView: "evidence",
  busy: false,
};

const actionCopy = {
  auto: { label: "模型工具", placeholder: "用自然语言说明你想做什么", defaultText: "" },
  ask: { label: "提问", placeholder: "向福尔摩斯提出一个可检验的问题", defaultText: "" },
  inspect: { label: "检查", placeholder: "说明你要检查的人、物件或地点", defaultText: "" },
  hypothesize: { label: "假设", placeholder: "提出你的案情假设，并说明依据", defaultText: "" },
  hint: { label: "提示", placeholder: "请求一个不会泄露结论的提示", defaultText: "给我一个不会剧透的提示。" },
  solve: { label: "结案", placeholder: "提交完整推理：人物、动机、方法与证据链", defaultText: "" },
  recap: { label: "复盘", placeholder: "整理当前证据与尚未解释的问题", defaultText: "请复盘当前已经确认的证据与疑点。" },
};

const stageLabels = {
  premise: "案情陈述",
  investigation: "调查",
  hypothesis: "假设检验",
  solution: "结案推理",
  post_case: "案件已结",
};

const difficultyLabels = {
  1: "入门",
  2: "进阶",
};

const elements = {
  runtimeBadge: document.querySelector("#runtime-badge"),
  resetButton: document.querySelector("#reset-button"),
  exportButton: document.querySelector("#export-button"),
  caseList: document.querySelector("#case-list"),
  caseEyebrow: document.querySelector("#case-eyebrow"),
  caseTitle: document.querySelector("#case-title"),
  caseObjective: document.querySelector("#case-objective"),
  stageLabel: document.querySelector("#stage-label"),
  progressFill: document.querySelector("#progress-fill"),
  progressLabel: document.querySelector("#progress-label"),
  leadCount: document.querySelector("#lead-count"),
  leadList: document.querySelector("#lead-list"),
  dialogueLog: document.querySelector("#dialogue-log"),
  turnForm: document.querySelector("#turn-form"),
  actionControl: document.querySelector("#action-control"),
  playerInput: document.querySelector("#player-input"),
  submitButton: document.querySelector("#submit-button"),
  turnStatus: document.querySelector("#turn-status"),
  evidenceCount: document.querySelector("#evidence-count"),
  hintTier: document.querySelector("#hint-tier"),
  evidenceContent: document.querySelector("#evidence-content"),
  evidenceTabs: document.querySelector(".evidence-tabs"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = { error: `HTTP ${response.status}` };
  }
  if (!response.ok) {
    const error = new Error(payload.error || payload.detail || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function setBusy(value, status = "") {
  state.busy = value;
  elements.turnStatus.textContent = status;
  elements.playerInput.disabled = value || !state.session;
  elements.submitButton.disabled = value || !state.session;
  elements.exportButton.disabled = value || !state.session;
  elements.resetButton.disabled = value || !state.session;
  document.querySelectorAll(".case-button, .action-control button, .lead-button").forEach((button) => {
    button.disabled = value;
  });
}

function renderRuntime(runtime) {
  state.runtime = runtime;
  elements.runtimeBadge.className = "runtime-badge";
  if (runtime && String(runtime.mode || "").startsWith("live_") && runtime.identity_verified) {
    elements.runtimeBadge.classList.add("is-live");
    elements.runtimeBadge.textContent = `模型已核验 · ${runtime.served_model}`;
    elements.runtimeBadge.title = [
      runtime.served_model,
      runtime.adapter_sha256,
      runtime.base_model_revision,
    ].filter(Boolean).join(" · ");
    return;
  }
  elements.runtimeBadge.classList.add("is-scripted");
  elements.runtimeBadge.textContent = "脚本冒烟模式";
  elements.runtimeBadge.title = runtime && runtime.label ? runtime.label : "Not live model inference";
}

function renderCases() {
  elements.caseList.replaceChildren();
  state.cases.forEach((caseRow, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-button";
    button.dataset.caseId = caseRow.case_id;
    button.setAttribute("aria-label", `开启案件：${caseRow.title_zh}`);

    const title = document.createElement("strong");
    title.textContent = caseRow.title_zh;
    const english = document.createElement("span");
    english.textContent = caseRow.title_en;
    const meta = document.createElement("small");
    const baseMeta = `CASE 0${index + 1} · ${difficultyLabels[caseRow.difficulty] || caseRow.difficulty}`;
    button.dataset.baseMeta = baseMeta;
    meta.textContent = baseMeta;
    button.append(title, english, meta);
    button.addEventListener("click", () => startCase(caseRow.case_id));
    elements.caseList.append(button);
  });
  updateCaseButtons();
}

function updateCaseButtons() {
  document.querySelectorAll(".case-button").forEach((button) => {
    const caseId = button.dataset.caseId;
    const completed = state.completedCaseIds.has(caseId);
    button.classList.toggle("is-active", state.session && state.session.case_id === caseId);
    button.classList.toggle("is-complete", completed);
    const meta = button.querySelector("small");
    if (meta) meta.textContent = `${button.dataset.baseMeta}${completed ? " · 已结案" : ""}`;
  });
}

function renderDialogue(session, turns = []) {
  elements.dialogueLog.replaceChildren();
  appendMessage({
    kind: "system",
    speaker: "案卷记录",
    meta: stageLabels[session.state.stage] || session.state.stage,
    text: `${session.title_zh}已开启。${session.objective}`,
  });
  turns.forEach((turn) => {
    appendMessage({
      kind: "user",
      speaker: "华生",
      meta: actionCopy[turn.action] ? actionCopy[turn.action].label : turn.action,
      text: turn.player_text,
    });
    appendMessage({
      kind: "sherlock",
      speaker: "Sherlock Holmes",
      meta: turn.degraded ? "运行时降级" : (stageLabels[turn.stage] || turn.stage),
      text: turn.host_answer,
      degraded: turn.degraded,
    });
  });
}

async function startCase(caseId, { forceNew = false } = {}) {
  if (!forceNew && state.session && state.session.case_id === caseId) return;
  if (forceNew) state.selectedActionsByCase.delete(caseId);
  setBusy(true, "正在开启案卷...");
  try {
    let session = null;
    let transcript = null;
    const existingSessionId = forceNew ? null : state.sessionsByCase.get(caseId);
    if (existingSessionId) {
      try {
        [session, transcript] = await Promise.all([
          api(`/case-game/sessions/${existingSessionId}`),
          api(`/case-game/sessions/${existingSessionId}/transcript`),
        ]);
      } catch (_error) {
        state.sessionsByCase.delete(caseId);
        state.selectedActionsByCase.delete(caseId);
      }
    }
    if (!session) {
      session = await api("/case-game/sessions", {
        method: "POST",
        body: JSON.stringify({ case_id: caseId, player_id: getPlayerId() }),
      });
    }
    state.session = session;
    state.sessionsByCase.set(caseId, session.session_id);
    if (session.state.solved) state.completedCaseIds.add(caseId);
    state.selectedTarget = null;
    renderRuntime(session.runtime);
    updateCaseButtons();
    renderDialogue(session, transcript ? transcript.turns : []);
    renderSession(session);
    elements.playerInput.value = "";
    selectAction(state.selectedActionsByCase.get(caseId) || "auto");
    setBusy(false, session.state.solved ? "案件已结，可导出完整记录。" : "");
  } catch (error) {
    setBusy(false, error.message);
  }
}

function renderSession(session) {
  state.session = session;
  if (session.state.solved) state.completedCaseIds.add(session.case_id);
  updateCaseButtons();
  elements.caseEyebrow.textContent = `${session.case_id} · ${session.title_en}`;
  elements.caseTitle.textContent = session.title_zh;
  elements.caseObjective.textContent = session.objective;
  elements.stageLabel.textContent = stageLabels[session.state.stage] || session.state.stage;

  const unlocked = session.progress.unlocked_evidence || 0;
  const total = session.progress.total_evidence || 0;
  const percentage = total ? Math.min(100, Math.round((unlocked / total) * 100)) : 0;
  elements.progressFill.style.width = `${percentage}%`;
  elements.progressLabel.textContent = `${unlocked} / ${total} 条线索`;
  elements.evidenceCount.textContent = `${session.evidence_board.length} 条可见线索`;
  elements.hintTier.textContent = `提示 ${session.state.hint_tier || 0} 级`;
  renderLeads(session.investigation_leads || []);
  renderEvidencePanel();
}

function renderLeads(leads) {
  elements.leadList.replaceChildren();
  elements.leadCount.textContent = `${leads.length} 项`;
  if (!leads.length) {
    const empty = document.createElement("p");
    empty.className = "lead-empty";
    empty.textContent = state.session && state.session.state.solved ? "案件已结。" : "当前没有新的可执行线索。";
    elements.leadList.append(empty);
    return;
  }

  leads.forEach((lead) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "lead-button";
    button.dataset.targetId = lead.target_id;
    button.dataset.action = lead.action;
    button.textContent = lead.label;
    button.title = lead.player_text;
    button.setAttribute("aria-pressed", String(state.selectedTarget && state.selectedTarget.target_id === lead.target_id));
    if (state.selectedTarget && state.selectedTarget.target_id === lead.target_id) {
      button.classList.add("is-selected");
    }
    button.addEventListener("click", () => selectLead(lead));
    elements.leadList.append(button);
  });
}

function selectLead(lead) {
  state.selectedTarget = lead;
  selectAction(lead.action, { preserveTarget: true });
  elements.playerInput.value = lead.player_text;
  elements.leadList.querySelectorAll(".lead-button").forEach((button) => {
    const selected = button.dataset.targetId === lead.target_id;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  elements.playerInput.focus();
}

function clearSelectedTarget() {
  state.selectedTarget = null;
  elements.leadList.querySelectorAll(".lead-button").forEach((button) => {
    button.classList.remove("is-selected");
    button.setAttribute("aria-pressed", "false");
  });
}

function appendMessage({ kind, speaker, meta, text, degraded = false }) {
  const article = document.createElement("article");
  article.className = `message is-${kind}`;
  if (degraded) article.classList.add("is-degraded");

  const header = document.createElement("div");
  header.className = "message-meta";
  const name = document.createElement("strong");
  name.textContent = speaker;
  const detail = document.createElement("span");
  detail.textContent = meta;
  header.append(name, detail);

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.append(header, paragraph);
  elements.dialogueLog.append(article);
  elements.dialogueLog.scrollTop = elements.dialogueLog.scrollHeight;
}

async function submitTurn(event) {
  event.preventDefault();
  if (!state.session || state.busy) return;
  const playerText = elements.playerInput.value.trim();
  if (!playerText) {
    elements.turnStatus.textContent = "请写下本轮行动内容。";
    elements.playerInput.focus();
    return;
  }

  const targetId = state.selectedTarget ? state.selectedTarget.target_id : null;
  const pendingKey = JSON.stringify([
    state.session.session_id,
    state.session.state_version,
    state.selectedAction,
    targetId,
    playerText,
  ]);
  const isRetry = state.pendingTurn && state.pendingTurn.key === pendingKey;
  if (!isRetry) {
    state.pendingTurn = { key: pendingKey, requestId: mintRequestId() };
    appendMessage({
      kind: "user",
      speaker: "华生",
      meta: actionCopy[state.selectedAction].label,
      text: playerText,
    });
  }
  setBusy(true, "福尔摩斯正在检视证据...");
  try {
    const session = await api(`/case-game/sessions/${state.session.session_id}/turns`, {
      method: "POST",
      body: JSON.stringify({
        action: state.selectedAction,
        player_text: playerText,
        target_id: targetId,
        request_id: state.pendingTurn.requestId,
        state_version: state.session.state_version,
        input_mode: state.selectedAction === "auto" ? "model" : "button",
      }),
    });
    state.pendingTurn = null;
    const turn = session.turn;
    appendMessage({
      kind: "sherlock",
      speaker: "Sherlock Holmes",
      meta: turn.degraded ? "运行时降级" : (stageLabels[turn.stage] || turn.stage),
      text: turn.host_answer,
      degraded: turn.degraded,
    });
    renderSession(session);
    clearSelectedTarget();
    elements.playerInput.value = actionCopy[state.selectedAction].defaultText;
    setBusy(false, turn.guard_blocked ? "回复已按当前可见证据收束。" : "");
    if (session.state.solved) {
      state.completedCaseIds.add(session.case_id);
      updateCaseButtons();
      elements.turnStatus.textContent = "案件已结，可导出完整记录。";
    }
  } catch (error) {
    if (error.status === 409) {
      try {
        const [session, transcript] = await Promise.all([
          api(`/case-game/sessions/${state.session.session_id}`),
          api(`/case-game/sessions/${state.session.session_id}/transcript`),
        ]);
        state.pendingTurn = null;
        renderDialogue(session, transcript.turns);
        renderSession(session);
        setBusy(false, "状态已从服务器刷新，请重新提交本轮行动。");
      } catch (refreshError) {
        setBusy(false, `状态冲突，且刷新失败：${refreshError.message}`);
      }
      return;
    }
    appendMessage({
      kind: "degraded",
      speaker: "运行时",
      meta: "请求失败",
      text: error.message,
      degraded: true,
    });
    setBusy(false, error.message);
  }
}

function selectAction(action, { preserveTarget = false } = {}) {
  if (!preserveTarget) clearSelectedTarget();
  state.selectedAction = action;
  if (state.session) state.selectedActionsByCase.set(state.session.case_id, action);
  elements.actionControl.querySelectorAll("button").forEach((button) => {
    const selected = button.dataset.action === action;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  const copy = actionCopy[action];
  elements.playerInput.placeholder = copy.placeholder;
  if (!elements.playerInput.value.trim()) {
    elements.playerInput.value = copy.defaultText;
  }
  elements.playerInput.focus();
}

function renderEvidencePanel() {
  elements.evidenceContent.replaceChildren();
  if (!state.session) {
    const empty = document.createElement("p");
    empty.className = "panel-empty";
    empty.textContent = "暂无案件资料。";
    elements.evidenceContent.append(empty);
    return;
  }

  const rows = state.activeView === "evidence"
    ? state.session.evidence_board
    : state.session.timeline;
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "panel-empty";
    empty.textContent = state.activeView === "evidence" ? "暂无可见证据。" : "暂无可见时间线。";
    elements.evidenceContent.append(empty);
    return;
  }

  rows.forEach((row) => {
    const article = document.createElement("article");
    article.className = state.activeView === "evidence" ? "evidence-card" : "timeline-row";
    const header = document.createElement("header");
    const title = document.createElement("h3");
    title.textContent = row.title || row.label || "未命名记录";
    const id = document.createElement("span");
    id.className = state.activeView === "evidence" ? "evidence-id" : "timeline-order";
    id.textContent = state.activeView === "evidence" ? row.evidence_id : `#${row.order}`;
    const text = document.createElement("p");
    text.textContent = row.text || "";
    header.append(title, id);
    article.append(header, text);
    elements.evidenceContent.append(article);
  });
}

function selectEvidenceView(view) {
  state.activeView = view;
  elements.evidenceTabs.querySelectorAll("button").forEach((button) => {
    const selected = button.dataset.view === view;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  renderEvidencePanel();
}

async function exportTranscript() {
  if (!state.session) return;
  setBusy(true, "正在导出记录...");
  try {
    const transcript = await api(`/case-game/sessions/${state.session.session_id}/transcript`);
    const blob = new Blob([`${JSON.stringify(transcript, null, 2)}\n`], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${transcript.case_id}-${transcript.session_id}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    setBusy(false, "记录已导出。" );
  } catch (error) {
    setBusy(false, error.message);
  }
}

async function resetCurrentCase() {
  if (!state.session || state.busy) return;
  const caseId = state.session.case_id;
  const confirmed = window.confirm("重新开始将清空当前案件在本页中的进度与对话。是否继续？");
  if (!confirmed) return;
  state.sessionsByCase.delete(caseId);
  state.selectedActionsByCase.delete(caseId);
  state.completedCaseIds.delete(caseId);
  await startCase(caseId, { forceNew: true });
}

async function initialize() {
  try {
    const payload = await api("/case-game/cases");
    state.cases = payload.cases || [];
    renderRuntime(payload.runtime || null);
    renderCases();
  } catch (error) {
    elements.caseList.textContent = `无法读取案件：${error.message}`;
    elements.runtimeBadge.textContent = "服务不可用";
    elements.runtimeBadge.className = "runtime-badge is-scripted";
  }
}

elements.turnForm.addEventListener("submit", submitTurn);
elements.actionControl.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (button) selectAction(button.dataset.action);
});
elements.evidenceTabs.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-view]");
  if (button) selectEvidenceView(button.dataset.view);
});
elements.exportButton.addEventListener("click", exportTranscript);
elements.resetButton.addEventListener("click", resetCurrentCase);
elements.playerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    elements.turnForm.requestSubmit();
  }
});
elements.playerInput.addEventListener("input", () => {
  if (state.selectedTarget && elements.playerInput.value !== state.selectedTarget.player_text) {
    clearSelectedTarget();
  }
});

initialize();
