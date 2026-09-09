"use strict";

const elements = {
  email: document.querySelector("#email-input"),
  generate: document.querySelector("#generate-button"),
  example: document.querySelector("#example-button"),
  copy: document.querySelector("#copy-button"),
  evaluate: document.querySelector("#evaluate-button"),
  model: document.querySelector("#model-chip"),
  corpus: document.querySelector("#corpus-size"),
  status: document.querySelector("#request-status"),
  error: document.querySelector("#error-row"),
  empty: document.querySelector("#empty-state"),
  grid: document.querySelector("#result-grid"),
  suggested: document.querySelector("#suggested-reply"),
  baseline: document.querySelector("#baseline-reply"),
  elapsed: document.querySelector("#elapsed"),
  grounding: document.querySelector("#grounding"),
  groundingCount: document.querySelector("#grounding-count"),
  groundingList: document.querySelector("#grounding-list"),
  evaluation: document.querySelector("#evaluation"),
  evaluationNote: document.querySelector("#evaluation-note"),
  judgeError: document.querySelector("#judge-error"),
  verdict: document.querySelector("#verdict"),
  warning: document.querySelector("#critical-warning"),
  judgeScore: document.querySelector("#judge-score"),
  bars: document.querySelector("#rubric-bars"),
  reason: document.querySelector("#judge-reason"),
  metrics: document.querySelector("#metrics-line"),
};

let examples = [];
let exampleIndex = 0;
let referenceReply = "";
let referenceEmail = "";
let currentReply = "";

function showError(target, message) {
  target.textContent = message;
  target.hidden = !message;
}

function setGenerating(active) {
  elements.generate.disabled = active;
  elements.example.disabled = active;
  elements.generate.textContent = active ? "Generating…" : "Generate reply";
  elements.status.textContent = active ? "Generating a grounded reply. This can take 10–30 seconds." : "";
}

function resetResult() {
  elements.grid.hidden = true;
  elements.empty.hidden = false;
  elements.grounding.hidden = true;
  elements.evaluation.hidden = true;
  elements.verdict.hidden = true;
  elements.elapsed.textContent = "";
  currentReply = "";
}

function makeGroundingItem(item, index) {
  const article = document.createElement("article");
  article.className = "grounding-item";

  const heading = document.createElement("div");
  heading.className = "grounding-heading";
  const title = document.createElement("h3");
  title.textContent = `Historical match ${index + 1}`;
  const similarity = document.createElement("span");
  similarity.className = "similarity";
  similarity.textContent = Number(item.similarity).toFixed(3);
  heading.append(title, similarity);

  const incomingLabel = document.createElement("p");
  incomingLabel.className = "grounding-label";
  incomingLabel.textContent = "Incoming email";
  const incoming = document.createElement("p");
  incoming.className = "clamped-email";
  incoming.textContent = item.incoming_email;

  const replyLabel = document.createElement("p");
  replyLabel.className = "grounding-label";
  replyLabel.textContent = "Reply actually sent";
  const reply = document.createElement("p");
  reply.className = "grounding-reply";
  reply.textContent = item.human_reply;

  article.append(heading, incomingLabel, incoming, replyLabel, reply);
  return article;
}

function renderResult(data) {
  currentReply = data.suggested_reply;
  elements.suggested.textContent = data.suggested_reply;
  elements.baseline.textContent = data.baseline_reply || "No similar historical reply was found.";
  elements.elapsed.textContent = `${Number(data.elapsed_seconds).toFixed(1)}s · ${data.model}`;
  elements.empty.hidden = true;
  elements.grid.hidden = false;

  elements.groundingList.replaceChildren();
  data.retrieved.forEach((item, index) => {
    elements.groundingList.append(makeGroundingItem(item, index));
  });
  elements.groundingCount.textContent = String(data.retrieved.length);
  elements.grounding.hidden = data.retrieved.length === 0;

  const hasReference = referenceReply && elements.email.value === referenceEmail;
  elements.evaluation.hidden = false;
  elements.evaluate.disabled = !hasReference;
  elements.evaluationNote.textContent = hasReference
    ? "Compare against the known reply for this held-out example."
    : "Load a held-out example to evaluate against a known human reply.";
}

async function generateReply() {
  const email = elements.email.value.trim();
  if (!email) {
    showError(elements.error, "Enter an incoming email before generating a reply.");
    elements.email.focus();
    return;
  }
  showError(elements.error, "");
  showError(elements.judgeError, "");
  elements.verdict.hidden = true;
  setGenerating(true);
  try {
    const response = await fetch("/api/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, top_k: 3 }),
    });
    if (!response.ok) throw new Error(`Request failed with status ${response.status}.`);
    const data = await response.json();
    if (data.error) {
      resetResult();
      showError(elements.error, data.error);
      return;
    }
    renderResult(data);
  } catch (error) {
    resetResult();
    showError(elements.error, error instanceof Error ? error.message : "Unable to generate a reply.");
  } finally {
    setGenerating(false);
  }
}

async function loadExample() {
  showError(elements.error, "");
  try {
    if (!examples.length) {
      elements.example.disabled = true;
      elements.example.textContent = "Loading…";
      const response = await fetch("/api/examples?n=8");
      if (!response.ok) throw new Error(`Request failed with status ${response.status}.`);
      examples = await response.json();
    }
    if (!examples.length) throw new Error("No held-out examples are available.");
    const selected = examples[exampleIndex % examples.length];
    exampleIndex += 1;
    elements.email.value = selected.incoming_email;
    referenceEmail = selected.incoming_email;
    referenceReply = selected.human_reply;
    resetResult();
    elements.email.focus();
  } catch (error) {
    showError(elements.error, error instanceof Error ? error.message : "Unable to load an example.");
  } finally {
    elements.example.disabled = false;
    elements.example.textContent = "Load an example";
  }
}

function addRubricBar(label, value) {
  const row = document.createElement("div");
  row.className = "rubric-row";
  const text = document.createElement("span");
  text.textContent = label;
  const progress = document.createElement("progress");
  progress.max = 5;
  progress.value = value;
  progress.setAttribute("aria-label", `${label}: ${value} out of 5`);
  const score = document.createElement("span");
  score.className = "dimension-score";
  score.textContent = `${value}/5`;
  row.append(text, progress, score);
  elements.bars.append(row);
}

async function evaluateReply() {
  if (!referenceReply || !currentReply || elements.email.value !== referenceEmail) return;
  elements.evaluate.disabled = true;
  elements.evaluate.textContent = "Evaluating…";
  elements.verdict.hidden = true;
  showError(elements.judgeError, "");
  try {
    const response = await fetch("/api/judge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: referenceEmail,
        human_reply: referenceReply,
        candidate_reply: currentReply,
      }),
    });
    if (!response.ok) throw new Error(`Request failed with status ${response.status}.`);
    const data = await response.json();
    if (data.error) throw new Error(data.error);
    elements.judgeScore.textContent = `${Number(data.judge_score).toFixed(1)} / 100`;
    elements.bars.replaceChildren();
    addRubricBar("Task fulfillment", data.task_fulfillment);
    addRubricBar("Action alignment", data.action_alignment);
    addRubricBar("Completeness", data.completeness);
    addRubricBar("Tone", data.tone);
    elements.reason.textContent = data.reason;
    elements.warning.hidden = !data.critical_error;
    elements.verdict.hidden = false;
  } catch (error) {
    showError(elements.judgeError, error instanceof Error ? error.message : "Unable to evaluate this reply.");
  } finally {
    elements.evaluate.disabled = false;
    elements.evaluate.textContent = "Evaluate this reply";
  }
}

async function copyReply() {
  try {
    await navigator.clipboard.writeText(currentReply);
    elements.copy.textContent = "Copied";
    window.setTimeout(() => { elements.copy.textContent = "Copy"; }, 1400);
  } catch {
    showError(elements.error, "Clipboard access was unavailable. Select and copy the reply manually.");
  }
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    const data = await response.json();
    elements.model.textContent = data.gen_model;
    elements.corpus.textContent = Number(data.corpus_size).toLocaleString();
  } catch {
    elements.model.textContent = "Unavailable";
  }
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/metrics");
    if (!response.ok) return;
    const data = await response.json();
    const values = [data.n_examples, data.main_mean_overall_score, data.baseline_mean_overall_score];
    if (data.available === false || !values.every(Number.isFinite)) return;
    elements.metrics.textContent = `Measured on ${data.n_examples} held-out examples: overall ${data.main_mean_overall_score.toFixed(1)} vs baseline ${data.baseline_mean_overall_score.toFixed(1)}.`;
    elements.metrics.hidden = false;
  } catch {
    elements.metrics.hidden = true;
  }
}

elements.generate.addEventListener("click", generateReply);
elements.example.addEventListener("click", loadExample);
elements.evaluate.addEventListener("click", evaluateReply);
elements.copy.addEventListener("click", copyReply);
elements.email.addEventListener("input", () => {
  if (elements.email.value !== referenceEmail) referenceReply = "";
});
elements.email.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    generateReply();
  }
});

loadHealth();
loadMetrics();
