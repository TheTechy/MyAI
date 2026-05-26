// ── Constants & state ────────────────────────────────────────────────────────
const STORAGE_KEY = 'myai_selected_name';
const userName    = localStorage.getItem(STORAGE_KEY) || 'default';

const chatEl        = document.getElementById('chat');
const inputEl       = document.getElementById('prompt-input');
const sendBtn       = document.getElementById('send-btn');
const ctxFill       = document.getElementById('ctx-fill');
const ctxPct        = document.getElementById('ctx-pct');
const ctxNums       = document.getElementById('ctx-nums');
const convListEl    = document.getElementById('conv-list');
const sidebar       = document.getElementById('sidebar');
const overlay       = document.getElementById('sidebar-overlay');
const newChatBtn    = document.getElementById('new-chat-btn');
const sidebarToggle = document.getElementById('sidebar-toggle');
const attachBtn     = document.getElementById('attach-btn');

document.getElementById('user-label').textContent = userName;

let isStreaming   = false;
let currentConvId = null;

// ── Sidebar open / close (mobile) ────────────────────────────────────────────
function openSidebar()  { sidebar.classList.add('open');    overlay.classList.add('visible'); }
function closeSidebar() { sidebar.classList.remove('open'); overlay.classList.remove('visible'); }

sidebarToggle.addEventListener('click', () =>
  sidebar.classList.contains('open') ? closeSidebar() : openSidebar()
);
overlay.addEventListener('click', closeSidebar);

// ── Context bar ──────────────────────────────────────────────────────────────
function updateCtxBar({ used, max, remaining, pct }) {
  ctxFill.style.width = pct + '%';
  ctxFill.classList.remove('warn', 'crit');
  if (pct >= 90)      ctxFill.classList.add('crit');
  else if (pct >= 70) ctxFill.classList.add('warn');
  ctxPct.textContent  = pct + '%';
  ctxNums.textContent = `${used.toLocaleString()} / ${max.toLocaleString()} · ${remaining.toLocaleString()} free`;
}

// ── Auto-grow textarea ───────────────────────────────────────────────────────
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!isStreaming) handleSend();
  }
});

sendBtn.addEventListener('click', () => { if (!isStreaming) handleSend(); });

// ── HTML / Markdown helpers ──────────────────────────────────────────────────
function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * Renders inline Markdown within a single line:
 * inline `code`, **bold**, *italic*, ~~strikethrough~~.
 * Splits on backtick spans first to avoid double-escaping their contents.
 */
function renderInline(text) {
  return text.split(/(`[^`\n]+`)/g).map((part, i) => {
    if (i % 2 === 1) {
      return `<code>${escHtml(part.slice(1, -1))}</code>`;
    }
    return escHtml(part)
      .replace(/\*\*(.+?)\*\*/g,  '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g,      '<em>$1</em>')
      .replace(/~~(.+?)~~/g,      '<del>$1</del>');
  }).join('');
}

/**
 * Renders a block of prose lines (no fenced code).
 * Handles: headings, blockquotes, unordered & ordered lists, blank-line paragraphs.
 */
function renderBlock(text) {
  const lines  = text.split('\n');
  let html     = '';
  let listType = null;   // 'ul' | 'ol' | null

  function closeList() {
    if (listType) { html += `</${listType}>`; listType = null; }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Heading
    const hm = line.match(/^(#{1,6})\s+(.*)/);
    if (hm) {
      closeList();
      const lvl = hm[1].length;
      html += `<h${lvl}>${renderInline(hm[2])}</h${lvl}>`;
      continue;
    }

    // Blockquote
    const bq = line.match(/^>\s?(.*)/);
    if (bq) {
      closeList();
      html += `<blockquote>${renderInline(bq[1])}</blockquote>`;
      continue;
    }

    // Unordered list item
    const ul = line.match(/^[\*\-]\s+(.*)/);
    if (ul) {
      if (listType !== 'ul') { closeList(); html += '<ul>'; listType = 'ul'; }
      html += `<li>${renderInline(ul[1])}</li>`;
      continue;
    }

    // Ordered list item
    const ol = line.match(/^\d+\.\s+(.*)/);
    if (ol) {
      if (listType !== 'ol') { closeList(); html += '<ol>'; listType = 'ol'; }
      html += `<li>${renderInline(ol[1])}</li>`;
      continue;
    }

    // Blank line → paragraph break
    if (line.trim() === '') {
      closeList();
      html += '<br>';
      continue;
    }

    // Normal prose line
    closeList();
    html += renderInline(line) + '<br>';
  }

  closeList();
  // Trim leading/trailing stray <br>
  return html.replace(/^(<br>)+/, '').replace(/(<br>)+$/, '');
}

/**
 * Full Markdown renderer.
 * Extracts fenced code blocks first, renders prose blocks between them,
 * then highlights the code with highlight.js.
 */
function renderMarkdown(text) {
  const fenceRe = /```(\w*)\n?([\s\S]*?)```/g;
  let html = '', last = 0, m;

  while ((m = fenceRe.exec(text)) !== null) {
    // Prose before this fence
    if (m.index > last) html += renderBlock(text.slice(last, m.index));

    const lang     = m[1].trim();
    const rawCode  = m[2].replace(/\n$/, '');
    const escaped  = escHtml(rawCode);

    // Copy button
    const copyBtn = `
      <button class="copy-btn" onclick="copyCode(this)" title="Copy code">
        <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
          <rect x="5" y="5" width="9" height="11" rx="1.5"/>
          <path d="M11 5V3.5A1.5 1.5 0 0 0 9.5 2h-6A1.5 1.5 0 0 0 2 3.5v8A1.5 1.5 0 0 0 3.5 13H5"/>
        </svg>
        copy
      </button>`;

    html += `<div class="code-block">
      <div class="code-header">
        <span class="code-lang">${escHtml(lang)}</span>
        ${copyBtn}
      </div>
      <pre><code class="${lang ? 'language-' + escHtml(lang) : ''}">${escaped}</code></pre>
    </div>`;

    last = m.index + m[0].length;
  }

  if (last < text.length) html += renderBlock(text.slice(last));
  return html;
}

/** Called by copy buttons embedded in code blocks. */
function copyCode(btn) {
  const code = btn.closest('.code-block').querySelector('code').textContent;
  navigator.clipboard.writeText(code).then(() => {
    btn.classList.add('copied');
    btn.querySelector('svg + *') || (btn.lastChild.textContent = ' copied!');
    btn.innerHTML = btn.innerHTML.replace(/>copy</, '>copied!<');
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = btn.innerHTML.replace(/>copied!</, '>copy<');
    }, 2000);
  });
}

/** Run highlight.js on all un-highlighted code blocks inside an element. */
function highlightAll(el) {
  el.querySelectorAll('pre code').forEach(block => {
    if (!block.dataset.highlighted) hljs.highlightElement(block);
  });
}

function scrollToBottom() { chatEl.scrollTop = chatEl.scrollHeight; }

function setLocked(locked) {
  isStreaming        = locked;
  sendBtn.disabled   = locked;
  inputEl.disabled   = locked;
  if (attachBtn) attachBtn.disabled = locked;
}

// ── Empty state ──────────────────────────────────────────────────────────────
function showEmptyState() {
  chatEl.innerHTML = `
    <div class="empty-state" id="empty">
      <div class="empty-icon">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <p>How can I help today?</p>
    </div>`;
}

function hideEmpty() {
  const el = document.getElementById('empty');
  if (el) el.style.display = 'none';
}

// ── Message bubble builders ──────────────────────────────────────────────────
function appendUserBubble(text, attachmentFilename = null) {
  hideEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row user';
  const attachmentHtml = attachmentFilename
    ? `<div class="bubble-attachment">📎 ${escHtml(attachmentFilename)}</div>`
    : '';
  row.innerHTML = `
    <div class="bubble">${escHtml(text)}${attachmentHtml}</div>
    <div class="avatar usr">You</div>`;
  chatEl.appendChild(row);
  scrollToBottom();
}

/**
 * Creates a streaming AI bubble.
 * Raw text is accumulated in bubble.dataset.raw.
 * textContent is updated each token so the cursor stays visible.
 * On 'done', renderMarkdown() is called and innerHTML is replaced.
 */
function createAiBubble() {
  hideEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row ai';
  row.innerHTML = `<div class="avatar ai">MyAI</div><div class="bubble streaming"></div>`;
  chatEl.appendChild(row);
  scrollToBottom();
  return row.querySelector('.bubble');
}

/**
 * Static bubble used when loading a past conversation.
 * Renders full Markdown immediately.
 */
function appendAiBubbleStatic(text) {
  hideEmpty();
  const row = document.createElement('div');
  row.className = 'msg-row ai';
  row.innerHTML = `<div class="avatar ai"> MyAI </div><div class="bubble">${renderMarkdown(text)}</div>`;
  chatEl.appendChild(row);
  highlightAll(row);
}

// ── Status pill ──────────────────────────────────────────────────────────────
function showStatus(msg) {
  removeStatus();
  const pill = document.createElement('div');
  pill.className = 'status-pill';
  pill.id        = 'status-pill';
  pill.innerHTML = `<div class="spinner"></div><span>${escHtml(msg)}</span>`;
  chatEl.appendChild(pill);
  scrollToBottom();
}

function removeStatus() {
  const old = document.getElementById('status-pill');
  if (old) old.remove();
}

function showError(msg) {
  removeStatus();
  const el = document.createElement('div');
  el.className   = 'error-bubble';
  el.textContent = '⚠ ' + msg;
  chatEl.appendChild(el);
  scrollToBottom();
}

// ── Sidebar: conversation list ────────────────────────────────────────────────
async function loadConversations() {
  try {
    const res   = await fetch(`/conversations/${encodeURIComponent(userName)}`);
    const convs = await res.json();
    renderConvList(convs);

    if (convs.length > 0 && !currentConvId) {
      await switchConversation(convs[0].conv_id);
    } else if (convs.length === 0) {
      showEmptyState();
    }
  } catch (e) {
    console.error('Failed to load conversations:', e);
    showEmptyState();
  }
}

function renderConvList(convs) {
  convListEl.innerHTML = '';
  if (convs.length === 0) return;

  const now      = new Date();
  const todayStr = now.toDateString();
  const yestStr  = new Date(now - 86400000).toDateString();

  const groups = { Today: [], Yesterday: [], Older: [] };
  for (const c of convs) {
    const d = new Date(c.updated_at).toDateString();
    if (d === todayStr)     groups.Today.push(c);
    else if (d === yestStr) groups.Yesterday.push(c);
    else                    groups.Older.push(c);
  }

  for (const [label, items] of Object.entries(groups)) {
    if (items.length === 0) continue;
    const heading = document.createElement('div');
    heading.className   = 'conv-section-label';
    heading.textContent = label;
    convListEl.appendChild(heading);
    items.forEach(c => convListEl.appendChild(makeConvItem(c)));
  }
}

function makeConvItem({ conv_id, title }) {
  const item = document.createElement('div');
  item.className      = 'conv-item' + (conv_id === currentConvId ? ' active' : '');
  item.dataset.convId = conv_id;
  item.innerHTML = `
    <span class="conv-title">${escHtml(title)}</span>
    <button class="conv-delete" title="Delete">
      <svg viewBox="0 0 12 12" xmlns="http://www.w3.org/2000/svg">
        <path d="M1 1l10 10M11 1L1 11"/>
      </svg>
    </button>`;

  item.addEventListener('click', e => {
    if (!e.target.closest('.conv-delete') && !isStreaming) {
      switchConversation(conv_id);
      closeSidebar();
    }
  });

  item.querySelector('.conv-delete').addEventListener('click', async e => {
    e.stopPropagation();
    await deleteConversation(conv_id, item);
  });

  return item;
}

function setActiveConv(conv_id) {
  currentConvId = conv_id;
  document.querySelectorAll('.conv-item').forEach(el => {
    el.classList.toggle('active', el.dataset.convId === conv_id);
  });
}

function updateConvTitle(conv_id, title) {
  const item = convListEl.querySelector(`[data-conv-id="${conv_id}"]`);
  if (item) item.querySelector('.conv-title').textContent = title;
}

// ── Switch to a past conversation ─────────────────────────────────────────────
async function switchConversation(conv_id) {
  if (isStreaming) return;
  setActiveConv(conv_id);
  clearAllAttachments();
  chatEl.innerHTML = '';

  try {
    const res      = await fetch(`/conversations/${conv_id}/messages`);
    const messages = await res.json();

    if (messages.length === 0) {
      showEmptyState();
    } else {
      for (const msg of messages) {
        if (msg.role === 'user') appendUserBubble(msg.content);
        else                     appendAiBubbleStatic(msg.content);
      }
      scrollToBottom();
    }
  } catch (e) {
    showError('Failed to load conversation.');
  }
}

// ── New chat ──────────────────────────────────────────────────────────────────
async function startNewConversation() {
  if (isStreaming) return;

  try {
    const res  = await fetch('/conversations', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ user_id: userName }),
    });
    const data = await res.json();
    currentConvId = data.conv_id;

    const firstHeading = convListEl.querySelector('.conv-section-label');
    const item = makeConvItem({ conv_id: currentConvId, title: 'New conversation' });

    if (firstHeading && firstHeading.textContent === 'Today') {
      firstHeading.after(item);
    } else {
      const heading = document.createElement('div');
      heading.className   = 'conv-section-label';
      heading.textContent = 'Today';
      convListEl.prepend(item);
      convListEl.prepend(heading);
    }

    setActiveConv(currentConvId);
    showEmptyState();
    clearAllAttachments();
    closeSidebar();
    inputEl.focus();
  } catch (e) {
    showError('Failed to create conversation.');
  }
}

newChatBtn.addEventListener('click', startNewConversation);

// ── Delete a conversation ─────────────────────────────────────────────────────
async function deleteConversation(conv_id, itemEl) {
  try {
    await fetch(`/conversations/${conv_id}`, { method: 'DELETE' });
    itemEl.remove();

    convListEl.querySelectorAll('.conv-section-label').forEach(label => {
      const next = label.nextElementSibling;
      if (!next || next.classList.contains('conv-section-label')) label.remove();
    });

    if (conv_id === currentConvId) {
      const firstItem = convListEl.querySelector('.conv-item');
      if (firstItem) {
        await switchConversation(firstItem.dataset.convId);
      } else {
        currentConvId = null;
        showEmptyState();
      }
    }
  } catch (e) {
    showError('Failed to delete conversation.');
  }
}

// ── File download card ────────────────────────────────────────────────────────
function renderFileCard({ filename, mime_type, data, size }, bubble) {
  const bytes  = Uint8Array.from(atob(data), c => c.charCodeAt(0));
  const blob   = new Blob([bytes], { type: mime_type });
  const url    = URL.createObjectURL(blob);

  const kb     = size < 1024
    ? size + ' B'
    : size < 1048576
      ? (size / 1024).toFixed(1) + ' KB'
      : (size / 1048576).toFixed(2) + ' MB';

  const ext    = filename.split('.').pop().toLowerCase();
  const icon   = {
    md: '📝', txt: '📄', js: '📜', ts: '📜', jsx: '📜', tsx: '📜',
    py: '🐍', cs: '📜', cpp: '📜', java: '📜', html: '🌐', css: '🎨',
    json: '📋', csv: '📊', sql: '🗄', xml: '📋', sh: '💻',
    docx: '📘', xlsx: '📗',
  }[ext] ?? '📎';

  const card   = document.createElement('a');
  card.href     = url;
  card.download = filename;
  card.className = 'file-card';
  card.innerHTML = `
    <span class="file-card-icon">${icon}</span>
    <span class="file-card-info">
      <span class="file-card-name">${escHtml(filename)}</span>
      <span class="file-card-meta">${escHtml(kb)}</span>
    </span>
    <span class="file-card-dl">
      <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
        <path d="M8 2v8M4 7l4 4 4-4M2 13h12" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </span>`;

  // Attach below the active bubble, or directly to chat if no bubble yet
  const target = bubble ?? chatEl;
  target.appendChild(card);
  scrollToBottom();
}

// ── File attachment state ──────────────────────────────────────────────────────
let pendingFiles  = [];   // File objects queued, not yet uploaded
let attachedFiles = [];   // { file_id, filename } confirmed uploaded this conv

window.hasPendingFile     = () => pendingFiles.length > 0;
window.getPendingFileName = () =>
  pendingFiles.length === 0 ? null
  : pendingFiles.length === 1 ? pendingFiles[0].name
  : `${pendingFiles.length} files`;

window.uploadPendingFile = async (convId) => {
  if (!pendingFiles.length) return false;
  let allOk = true;
  for (const file of pendingFiles) {
    const form = new FormData();
    form.append("file", file);
    form.append("conv_id", convId);
    try {
      const res  = await fetch("/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) { showError(data.error || `Upload failed: ${file.name}`); allOk = false; continue; }
      attachedFiles.push({ file_id: data.file_id, filename: data.filename });
    } catch (err) {
      showError(`Upload failed: ${file.name} — ${err.message}`);
      allOk = false;
    }
  }
  pendingFiles = [];
  clearPendingChips();
  renderAttachedChips();
  return allOk;
};

function clearPendingChips() {
  document.querySelectorAll(".file-chip.pending").forEach(c => c.remove());
}

function renderAttachedChips() {
  let bar = document.getElementById("attached-files-bar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "attached-files-bar";
    inputEl.parentElement.insertBefore(bar, inputEl);
  }
  bar.innerHTML = attachedFiles.map(f => `
    <span class="file-chip" data-id="${escHtml(f.file_id)}">
      📎 ${escHtml(f.filename)}
      <button class="file-chip-remove" title="Remove">✕</button>
    </span>`).join("");
  bar.querySelectorAll(".file-chip-remove").forEach(btn => {
    btn.addEventListener("click", async () => {
      const chip   = btn.closest(".file-chip");
      const fileId = chip.dataset.id;
      await fetch(`/upload/${fileId}`, { method: "DELETE" });
      attachedFiles = attachedFiles.filter(f => f.file_id !== fileId);
      renderAttachedChips();
    });
  });
}

// Wire the attach button to a hidden file input
(function wireAttachButton() {
  if (!attachBtn) return;
  const fileInput = document.createElement("input");
  fileInput.type   = "file";
  fileInput.accept = [
    ".txt",".md",".py",".js",".ts",".jsx",".tsx",".cs",".cpp",".c",
    ".java",".json",".yaml",".yml",".sh",".sql",".xml",".csv",
    ".toml",".html",".css",".pdf",".docx",".xlsx",
  ].join(",");
  fileInput.style.display = "none";
  document.body.appendChild(fileInput);

  fileInput.multiple = true;
  attachBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const chosen = Array.from(fileInput.files);
    if (!chosen.length) return;
    fileInput.value = "";

    // Deduplicate by name against already-queued files
    const existing = new Set(pendingFiles.map(f => f.name));
    const fresh    = chosen.filter(f => !existing.has(f.name));
    pendingFiles.push(...fresh);

    // Render a chip for each newly added file
    fresh.forEach(file => {
      const chip     = document.createElement("span");
      chip.className = "file-chip pending";
      chip.dataset.pendingName = file.name;
      chip.innerHTML = `📎 ${escHtml(file.name)} <button class="file-chip-remove" title="Cancel">✕</button>`;
      chip.querySelector("button").addEventListener("click", () => {
        pendingFiles = pendingFiles.filter(f => f.name !== file.name);
        chip.remove();
      });
      inputEl.parentElement.insertBefore(chip, inputEl);
    });
  });
})();


// ── Clear all attachment state and UI ────────────────────────────────────────
function clearAllAttachments() {
  pendingFiles  = [];
  attachedFiles = [];
  // Clear every possible bar element
  ['attachment-bar', 'attached-files-bar'].forEach(id => {
    const bar = document.getElementById(id);
    if (bar) { bar.innerHTML = ''; bar.style.display = 'none'; }
  });
  // Nuclear option — remove ALL chip elements regardless of where they ended up
  document.querySelectorAll('.file-chip').forEach(c => c.remove());
}

// ── Thinking bubble ───────────────────────────────────────────────────────────
let _thinkStart   = null;
let _thinkBubble  = null;
let _thinkDotTimer = null;

function showThinkingBubble() {
  removeThinkingBubble();
  _thinkStart = Date.now();

  const row = document.createElement('div');
  row.className = 'msg-row ai';
  row.id        = 'thinking-row';
  row.innerHTML = `
    <div class="avatar ai">MyAI</div>
    <div class="bubble thinking-bubble">
      <span class="thinking-text">Thinking</span><span class="thinking-dots"></span>
    </div>`;
  chatEl.appendChild(row);
  scrollToBottom();

  _thinkBubble = row;

  // Animate the trailing dots: . .. ...
  let dotCount = 0;
  const dotsEl = row.querySelector('.thinking-dots');
  _thinkDotTimer = setInterval(() => {
    dotCount = (dotCount + 1) % 4;
    dotsEl.textContent = '.'.repeat(dotCount);
  }, 400);
}

function resolveThinkingBubble() {
  if (!_thinkBubble) return;
  clearInterval(_thinkDotTimer);

  const elapsed = _thinkStart ? ((Date.now() - _thinkStart) / 1000).toFixed(1) : '?';
  const bubble  = _thinkBubble.querySelector('.bubble');
  if (bubble) {
    bubble.classList.remove('thinking-bubble');
    bubble.classList.add('thought-bubble');
    bubble.innerHTML = `
      <svg class="thought-icon" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
        <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.25"/>
        <path d="M5.5 8.5l1.5 1.5 3-3" stroke="currentColor" stroke-width="1.25"
          stroke-linecap="round" stroke-linejoin="round" fill="none"/>
      </svg>
      Thought for ${elapsed}s`;

    // Fade out after 3 seconds
    setTimeout(() => {
      if (_thinkBubble) {
        _thinkBubble.classList.add('thought-fade');
        setTimeout(() => {
          _thinkBubble?.remove();
          _thinkBubble  = null;
          _thinkStart   = null;
        }, 600);
      }
    }, 3000);
  }
}

function removeThinkingBubble() {
  clearInterval(_thinkDotTimer);
  _thinkBubble?.remove();
  _thinkBubble  = null;
  _thinkStart   = null;
  _thinkDotTimer = null;
}

// ── SSE streaming ─────────────────────────────────────────────────────────────
async function sendPrompt(promptText) {
  let bubble = null;
  showThinkingBubble();

  const response = await fetch('/prompt', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      prompt:  promptText,
      name:    userName,
      conv_id: currentConvId,
    }),
  });

  if (!response.ok) {
    showError(`Server error ${response.status}`);
    return;
  }

  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer    = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    let eventType = null;

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith('data: ') && eventType) {
        let payload;
        try { payload = JSON.parse(line.slice(6)); } catch { continue; }

        // ── New conversation ID assigned by server ──
        if (eventType === 'conv_id') {
          currentConvId = payload.conv_id;
          const item = makeConvItem({ conv_id: currentConvId, title: 'New conversation' });
          const firstHeading = convListEl.querySelector('.conv-section-label');
          if (firstHeading && firstHeading.textContent === 'Today') {
            firstHeading.after(item);
          } else {
            const heading = document.createElement('div');
            heading.className   = 'conv-section-label';
            heading.textContent = 'Today';
            convListEl.prepend(item);
            convListEl.prepend(heading);
          }
          setActiveConv(currentConvId);
        }

        // if (eventType === 'status') {
        // //   showStatus(payload.message);
        // }

        if (eventType === 'token') {
          resolveThinkingBubble();
          removeStatus();
          if (!bubble) bubble = createAiBubble();
          // Accumulate raw text; show plain while streaming
          bubble.dataset.raw = (bubble.dataset.raw || '') + payload.text;
          bubble.textContent  = bubble.dataset.raw;
          scrollToBottom();
        }

        if (eventType === 'ctx') {
          updateCtxBar(payload);
        }

        if (eventType === 'done') {
          resolveThinkingBubble();
          removeStatus();
          if (bubble) {
            bubble.classList.remove('streaming');
            // Render full Markdown now that the response is complete
            bubble.innerHTML = renderMarkdown(bubble.dataset.raw || '');
            delete bubble.dataset.raw;
            highlightAll(bubble);
          }
          if (payload.truncated) {
            const warn = document.createElement('div');
            warn.className   = 'error-bubble';
            warn.textContent = '⚠ Response cut off — context window full.';
            chatEl.appendChild(warn);
          }
        }

        if (eventType === 'file') {
          renderFileCard(payload, bubble);
        }

        if (eventType === 'title') {
          updateConvTitle(payload.conv_id, payload.title);
        }

        if (eventType === 'error') {
          resolveThinkingBubble();
          showError(payload.message);
          if (bubble) bubble.classList.remove('streaming');
        }

        eventType = null;
      }
    }
  }
}

// ── Main send handler (upload-aware) ─────────────────────────────────────────
async function handleSend() {
  const text = inputEl.value.trim();
  if (!text && !window.hasPendingFile?.()) return;

  const pendingFilename = window.getPendingFileName?.() ?? null;

  inputEl.value        = '';
  inputEl.style.height = 'auto';

  appendUserBubble(text, pendingFilename);
  setLocked(true);

  try {
    if (window.hasPendingFile?.()) {
      const uploaded = await window.uploadPendingFile(currentConvId);
      if (!uploaded) showError('File upload failed — sending message without attachment.');
    }
    if (text) await sendPrompt(text);
  } catch (err) {
    showError(err.message || 'Network error');
  } finally {
    setLocked(false);
    clearAllAttachments();
    inputEl.focus();
  }
}

// ── Initialise ────────────────────────────────────────────────────────────────
(async () => {
  showEmptyState();
  await loadConversations();
  inputEl.focus();
})();