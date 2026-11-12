'use strict';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { project: null, files: [], dirty: false, pending: false, phase: 'request',
  selectedRun: null, selectedProposal: null, document: 'diff', epoch: 0, projects: [] };
const statusNames = { draft: '入力済み', running: '処理中', completed: '完了', review: '要確認', failed: '失敗', interrupted: '中断' };
const stageNames = { queued: '開始待ち', extracting: '変更内容を抽出中', searching: '仕様書を検索中',
  assessing: '修正必要性を判断中', generating: '修正案を生成中', finished: '完了', awaiting_confirmation: '確認待ち' };

function node(tag, className = '', text = '') {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function icon(name) {
  const element = document.createElement('i');
  element.dataset.lucide = name;
  return element;
}

function refreshIcons() {
  if (globalThis.lucide) lucide.createIcons({ attrs: { 'aria-hidden': 'true' } });
}

function notice(message = '') {
  $('#notice').textContent = message;
  $('#notice').hidden = !message;
}

function errorText(data) {
  if (Array.isArray(data.detail)) return data.detail.map((item) => item.msg).join(' / ');
  return data.detail || '処理に失敗しました。接続を確認してください。';
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (response.status === 204) return null;
  const data = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(errorText(data));
  return data;
}

function jsonOptions(method, body) {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
}

function running() {
  return state.project?.runs.some((run) => run.status === 'running') || false;
}

async function action(operation) {
  if (state.pending) return;
  state.pending = true;
  notice();
  controls();
  try { await operation(); }
  catch (error) { notice(error.message); }
  finally { state.pending = false; controls(); refreshIcons(); }
}

function controls() {
  const busy = state.pending || running();
  $('#extract-button').disabled = busy;
  $('#mask-button').disabled = busy;
  $$('#file-list button').forEach((button) => { button.disabled = state.pending; });
  $('#change-fields').disabled = busy;
  for (const selector of ['#save-changes', '#generate-button', '#add-change', '#delete-project']) {
    $(selector).disabled = busy;
  }
  $('#request-text').readOnly = Boolean(state.project) || state.pending;
  $('#files').disabled = Boolean(state.project) || state.pending;
  $('#attach-label').hidden = Boolean(state.project);
  $$('[data-phase]').forEach((button) => {
    button.disabled = !state.project && button.dataset.phase !== 'request';
    if (button.dataset.phase === state.phase) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
  $('#dirty-state').textContent = state.dirty ? '未保存の変更あり' : '';
}

function phase(name) {
  state.phase = name;
  for (const item of ['request', 'changes', 'results']) $(`#${item}-panel`).hidden = item !== name;
  controls();
}

function dateLabel(value) {
  return new Intl.DateTimeFormat('ja-JP', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value));
}

async function loadHistory() {
  state.projects = await api('/api/projects');
  const list = $('#history');
  list.replaceChildren();
  if (!state.projects.length) list.append(node('p', 'history-empty', '案件はまだありません。'));
  for (const project of state.projects) {
    const button = node('button', `history-item${state.project?.id === project.id ? ' active' : ''}`);
    button.type = 'button';
    button.append(node('span', 'history-title', project.title));
    const meta = node('span', 'history-meta');
    meta.append(node('span', '', dateLabel(project.created_at)), node('span', '', statusNames[project.status]));
    button.append(meta);
    button.addEventListener('click', () => action(() => openProject(project.id)));
    list.append(button);
  }
}

function abandonEdits() {
  return !state.dirty || confirm('未保存の変更を破棄しますか？');
}

async function openProject(projectId) {
  if (!abandonEdits()) return;
  clearMaskPreview();
  state.epoch += 1;
  state.project = await api(`/api/projects/${projectId}`);
  state.dirty = false;
  state.selectedRun = null;
  state.selectedProposal = null;
  $('#sidebar').classList.remove('open');
  $('#history-toggle').setAttribute('aria-expanded', 'false');
  renderProject();
  const latest = state.project.runs.at(-1);
  phase(latest?.kind === 'generate' ? 'results' : state.project.changes.length || latest ? 'changes' : 'request');
  await loadHistory();
  if (latest?.status === 'running') watch(latest.id, projectId, state.epoch);
}

function resetProject() {
  if (state.pending || !abandonEdits()) return;
  state.epoch += 1;
  Object.assign(state, { project: null, files: [], dirty: false, selectedRun: null, selectedProposal: null });
  $('#request-form').reset();
  clearMaskPreview();
  $('#delete-project').hidden = true;
  $('#request-heading').textContent = '変更要求';
  $('#request-meta').textContent = '';
  $('#extract-button span').textContent = '変更内容を抽出';
  $('#attachment-warnings').replaceChildren();
  $('#change-list').replaceChildren();
  renderFiles();
  notice();
  phase('request');
  loadHistory().catch((error) => notice(error.message));
}

function renderProject() {
  const project = state.project;
  $('#request-text').value = project.request;
  $('#request-meta').textContent = `作成 ${dateLabel(project.created_at)}`;
  $('#delete-project').hidden = false;
  $('#extract-button span').textContent = '再抽出';
  $('#revision').textContent = `版 ${project.revision} / ${project.changes.length} 件`;
  renderFiles();
  renderChanges();
  renderResults();
  controls();
  refreshIcons();
}

function renderFiles() {
  const list = $('#file-list');
  list.replaceChildren();
  const files = state.project ? state.project.attachments : state.files;
  files.forEach((file, index) => {
    const row = node('li');
    row.append(icon('file-text'), node('span', 'file-name', file.filename || file.name));
    if (file.size !== undefined) row.append(node('span', 'file-size', `${Math.ceil(file.size / 1024)} KB`));
    if (!state.project) {
      const remove = node('button', 'icon-button');
      remove.type = 'button';
      remove.title = remove.ariaLabel = '添付を削除';
      remove.append(icon('x'));
      remove.addEventListener('click', () => {
        state.files.splice(index, 1);
        state.dirty = true;
        clearMaskPreview();
        renderFiles();
      });
      row.append(remove);
    }
    list.append(row);
  });
  const warnings = [...new Set((state.project?.attachments || []).flatMap((file) => file.warnings))];
  $('#attachment-warnings').replaceChildren(...warnings.map((text) => node('p', '', text)));
  refreshIcons();
}

function markDirty() {
  state.dirty = true;
  controls();
}

function appendChange(change) {
  const fragment = $('#change-template').content.cloneNode(true);
  const row = fragment.querySelector('.change-item');
  row.dataset.id = change.id;
  for (const field of ['summary', 'before', 'after', 'sources', 'unknowns']) {
    const value = change[field];
    row.querySelector(`[data-field="${field}"]`).value = Array.isArray(value) ? value.join('\n') : value || '';
  }
  row.querySelector('.remove-change').addEventListener('click', () => { row.remove(); renumber(); markDirty(); });
  row.addEventListener('input', markDirty);
  $('#change-list').append(fragment);
  renumber();
}

function renumber() {
  $$('.change-index').forEach((item, index) => { item.textContent = String(index + 1).padStart(2, '0'); });
}

function renderChanges() {
  $('#change-list').replaceChildren();
  state.project.changes.forEach(appendChange);
  const last = state.project.runs.at(-1);
  const extracting = last?.kind === 'extract' && last.status === 'running';
  const status = $('#extraction-status');
  status.hidden = !extracting && state.project.changes.length > 0;
  status.replaceChildren();
  if (extracting) {
    const spinner = icon('loader-circle');
    spinner.classList.add('spinning');
    status.append(spinner, node('span', '', '変更内容を抽出中'));
  } else if (!state.project.changes.length) {
    status.textContent = last?.kind === 'extract' && last.status === 'completed'
      ? 'このコントローラーの役割に該当する変更は見つかりませんでした。'
      : '変更内容はまだありません。';
  }
  if (last?.kind === 'extract' && ['failed', 'interrupted'].includes(last.status)) {
    status.hidden = false;
    status.replaceChildren(...last.issues.map((message) => node('p', '', message)));
    const retry = node('button', 'button secondary', '再実行');
    retry.type = 'button';
    retry.addEventListener('click', () => action(() => retryRun(last.id)));
    status.append(retry);
  }
}

function collectChanges() {
  return $$('.change-item').map((row) => {
    const get = (field) => row.querySelector(`[data-field="${field}"]`).value.trim();
    return { id: row.dataset.id, summary: get('summary'), before: get('before') || null, after: get('after') || null,
      sources: get('sources').split('\n').filter(Boolean), unknowns: get('unknowns').split('\n').filter(Boolean) };
  });
}

async function saveChanges() {
  const changes = collectChanges();
  if (!changes.length) throw new Error('変更項目を1件以上入力してください。');
  if (!$('#changes-form').reportValidity()) throw new Error('変更概要を入力してください。');
  const project = await api(`/api/projects/${state.project.id}/changes`, jsonOptions('PUT', { revision: state.project.revision, changes }));
  state.project = project;
  state.dirty = false;
  renderProject();
  await loadHistory();
}

async function startRun(kind) {
  const endpoint = kind === 'extract' ? 'extractions' : 'runs';
  const projectId = state.project.id;
  const run = await api(`/api/projects/${projectId}/${endpoint}`, jsonOptions('POST', { revision: state.project.revision }));
  state.project.runs.push(run);
  state.selectedRun = kind === 'generate' ? run.id : null;
  state.selectedProposal = null;
  renderProject();
  phase(kind === 'extract' ? 'changes' : 'results');
  watch(run.id, projectId, ++state.epoch);
}

async function retryRun(runId) {
  const run = await api(`/api/runs/${runId}/retry`, { method: 'POST' });
  const index = state.project.runs.findIndex((item) => item.id === runId);
  state.project.runs[index] = run;
  renderProject();
  watch(run.id, state.project.id, ++state.epoch);
}

async function watch(runId, projectId, epoch) {
  try {
    const result = await api(`/api/runs/${runId}`);
    if (state.epoch !== epoch || state.project?.id !== projectId) return;
    const index = state.project.runs.findIndex((item) => item.id === runId);
    state.project.runs[index] = result.run;
    if (result.run.status !== 'running') {
      const completed = await api(`/api/projects/${projectId}`);
      if (state.epoch !== epoch) return;
      state.project = completed;
      renderProject();
      await loadHistory();
      return;
    }
    if (result.run.kind === 'generate') renderResults();
    controls();
    refreshIcons();
    setTimeout(() => watch(runId, projectId, epoch), 1200);
  } catch (error) {
    if (state.epoch === epoch) {
      notice(`進捗を取得できません: ${error.message}`);
      setTimeout(() => watch(runId, projectId, epoch), 4000);
    }
  }
}

function selectedRun() {
  const runs = state.project?.runs.filter((run) => run.kind === 'generate') || [];
  return runs.find((run) => run.id === state.selectedRun) || runs.at(-1);
}

function renderResults() {
  const runs = state.project.runs.filter((run) => run.kind === 'generate');
  const run = selectedRun();
  const select = $('#run-select');
  select.replaceChildren(...[...runs].reverse().map((item) => new Option(`版 ${item.revision} · ${dateLabel(item.created_at)} · ${statusNames[item.status]}`, item.id)));
  select.hidden = !runs.length;
  if (run) select.value = run.id;
  $('#progress').hidden = !run;
  $('#run-issues').replaceChildren();
  $('#proposal-area').hidden = !run || !Object.keys(run.proposals).length;
  $('#result-empty').hidden = Boolean(run && (Object.keys(run.proposals).length || run.status === 'running'));
  $('#candidate-results').replaceChildren();
  if (!run) return;
  renderProgress(run);
  renderIssues(run);
  const proposals = Object.values(run.proposals);
  const proposalSelect = $('#proposal-select');
  proposalSelect.replaceChildren(...proposals.map((item) => new Option(item.file_path, item.id)));
  if (proposals.some((item) => item.id === state.selectedProposal)) proposalSelect.value = state.selectedProposal;
  renderDocument();
  renderCandidates(run);
}

function renderProgress(run) {
  const progress = $('#progress');
  const heading = node('div', `progress-heading ${run.status}`);
  heading.append(icon(run.status === 'running' ? 'loader-circle' : run.status === 'completed' ? 'circle-check' : 'circle-alert'));
  heading.append(node('span', '', run.status === 'running' ? stageNames[run.stage] || '処理中' : statusNames[run.status]));
  const count = Object.values(run.assessments).reduce((total, items) => total + Object.keys(items).length, 0);
  const counts = `${run.changes.length} 変更 / ${count} 件の判断 / ${Object.keys(run.proposals).length} 件の修正案`;
  progress.replaceChildren(heading, node('div', 'progress-counts', counts));
}

function renderIssues(run) {
  const issues = $('#run-issues');
  if (run.revision !== state.project.revision) issues.append(node('p', '', 'この結果は過去の版です。変更内容は更新されています。'));
  issues.append(...run.issues.map((message) => node('p', '', message)));
  if (['review', 'failed', 'interrupted'].includes(run.status) && run.revision === state.project.revision) {
    const retry = node('button', 'button secondary');
    retry.append(icon('rotate-cw'), node('span', '', '未完了の処理を再実行'));
    retry.disabled = running();
    retry.addEventListener('click', () => action(() => retryRun(run.id)));
    issues.append(retry);
  }
}

function renderDocument() {
  const run = selectedRun();
  const proposalId = $('#proposal-select').value;
  const proposal = run && Object.values(run.proposals).find((item) => item.id === proposalId);
  if (!proposal) return;
  state.selectedProposal = proposal.id;
  $('#proposal-summary').textContent = proposal.summary;
  for (const kind of ['markdown', 'diff']) $(`#download-${kind}`).href = `/api/projects/${state.project.id}/proposals/${proposal.id}/download?format=${kind}`;
  const content = $('#document-content');
  content.replaceChildren();
  const lines = proposal[state.document].split('\n');
  const fragment = document.createDocumentFragment();
  for (const line of lines) {
    const style = state.document !== 'diff' ? '' : line.startsWith('@@') ? 'diff-header' :
      line.startsWith('+') && !line.startsWith('+++') ? 'diff-add' : line.startsWith('-') && !line.startsWith('---') ? 'diff-remove' : '';
    fragment.append(node('span', `code-line ${style}`, line || ' '));
  }
  content.append(fragment);
  $$('[data-document]').forEach((button) => {
    const active = button.dataset.document === state.document;
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
  });
  content.setAttribute('aria-labelledby', `tab-${state.document}`);
}

function renderCandidates(run) {
  const container = $('#candidate-results');
  if (!Object.keys(run.candidates).length) return;
  container.append(node('h2', '', '選定結果'));
  for (const change of run.changes) {
    if (!(change.id in run.candidates)) continue;
    const group = node('details', 'candidate-group');
    group.append(node('summary', '', change.summary));
    const scroll = node('div', 'table-scroll');
    const table = node('table');
    const head = node('thead');
    const header = node('tr');
    ['仕様書', '修正必要性', '検索関連度', '選定'].forEach((label) => header.append(node('th', '', label)));
    head.append(header);
    const body = node('tbody');
    const candidates = [...run.candidates[change.id]].sort((left, right) => (run.assessments[change.id]?.[right.document_id]?.score || 0) - (run.assessments[change.id]?.[left.document_id]?.score || 0));
    for (const item of candidates) {
      const chosen = run.selections[change.id] === item.document_id;
      const score = run.assessments[change.id]?.[item.document_id]?.score;
      const row = node('tr', chosen ? 'selected' : '');
      row.append(node('td', '', item.file_path), node('td', '', score === undefined ? '未評価' : `${(score * 100).toFixed(1)}%`),
        node('td', '', item.max_score.toFixed(3)), node('td', 'selected-mark', chosen ? '対象' : ''));
      body.append(row);
    }
    table.append(head, body);
    scroll.append(table);
    group.append(scroll);
    container.append(group);
  }
}

function requestData() {
  const data = new FormData();
  data.append('request', $('#request-text').value);
  state.files.forEach((file) => data.append('files', file));
  return data;
}

function clearMaskPreview() {
  $('#mask-preview').hidden = true;
  $('#mask-status').textContent = '';
  for (const id of ['mask-request', 'mask-attachments', 'mask-roles', 'mask-instruction', 'mask-message', 'mask-schema']) {
    $(`#${id}`).replaceChildren();
  }
}

function renderMaskPreview(preview) {
  const data = JSON.parse(preview.message);
  $('#mask-request').textContent = data.request || '（変更要求の本文なし）';
  $('#mask-roles').textContent = data.target_roles;
  $('#mask-instruction').textContent = preview.system_prompt;
  $('#mask-message').textContent = JSON.stringify(data, null, 2);
  $('#mask-schema').textContent = JSON.stringify(preview.response_schema, null, 2);
  for (const [index, attachment] of data.attachments.entries()) {
    const heading = node('h3', '', `添付 ${index + 1}：${attachment.filename}`);
    const text = node('pre', 'masked-text', attachment.text);
    text.tabIndex = 0;
    text.setAttribute('aria-label', `添付 ${index + 1} のマスク後の本文`);
    $('#mask-attachments').append(heading, text);
  }
  $('#mask-preview').hidden = false;
  $('#mask-status').textContent = 'マスクが完了しました。抽出用 LLM には送信していません。';
  $('#mask-preview-heading').focus();
}

async function previewMask() {
  clearMaskPreview();
  $('#mask-status').textContent = 'マスクを実行中です。';
  $('#mask-button span').textContent = 'マスク中…';
  try {
    const preview = state.project
      ? await api(`/api/projects/${state.project.id}/extraction-previews`, jsonOptions('POST', { revision: state.project.revision }))
      : await api('/api/extraction-previews', { method: 'POST', body: requestData() });
    renderMaskPreview(preview);
  } catch (error) {
    clearMaskPreview();
    throw error;
  } finally {
    $('#mask-button span').textContent = 'マスクのみ実行・確認';
  }
}

$('#mask-button').addEventListener('click', () => action(previewMask));
$('#close-mask-preview').addEventListener('click', () => {
  clearMaskPreview();
  $('#mask-button').focus();
});
$('#request-form').addEventListener('submit', (event) => {
  event.preventDefault();
  action(async () => {
    if (state.project && !confirm('変更内容を再抽出しますか？現在の変更内容は置き換わります。')) return;
    if (!state.project) {
      state.project = await api('/api/projects', { method: 'POST', body: requestData() });
      state.files = [];
      state.dirty = false;
      await loadHistory();
    }
    clearMaskPreview();
    await startRun('extract');
  });
});

$('#changes-form').addEventListener('submit', (event) => {
  event.preventDefault();
  action(async () => { await saveChanges(); await startRun('generate'); });
});
$('#save-changes').addEventListener('click', () => action(saveChanges));
$('#add-change').addEventListener('click', () => {
  appendChange({ id: crypto.randomUUID(), summary: '' });
  markDirty();
  refreshIcons();
  $$('.change-item').at(-1).querySelector('input').focus();
});
$('#files').addEventListener('change', (event) => {
  clearMaskPreview();
  state.files.push(...event.target.files);
  state.dirty = true;
  event.target.value = '';
  renderFiles();
});
$('#request-text').addEventListener('input', () => {
  clearMaskPreview();
  if (!state.project) state.dirty = true;
});
$('#new-project').addEventListener('click', resetProject);
$('#delete-project').addEventListener('click', () => action(async () => {
  if (!confirm('この案件と保存済みの資料・修正案を削除しますか？')) return;
  await api(`/api/projects/${state.project.id}`, { method: 'DELETE' });
  state.dirty = false;
  state.pending = false;
  resetProject();
}));
$$('[data-phase]').forEach((button) => button.addEventListener('click', () => phase(button.dataset.phase)));
$('#run-select').addEventListener('change', (event) => {
  state.selectedRun = event.target.value;
  state.selectedProposal = null;
  renderResults();
  refreshIcons();
});
$('#proposal-select').addEventListener('change', renderDocument);
$$('[data-document]').forEach((button) => button.addEventListener('click', () => {
  state.document = button.dataset.document;
  renderDocument();
}));
$('#document-tabs').addEventListener('keydown', (event) => {
  const tabs = $$('[data-document]');
  const index = tabs.indexOf(document.activeElement);
  if (index < 0 || !['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
    (index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
  tabs[next].click();
  tabs[next].focus();
});
$('#history-toggle').addEventListener('click', () => {
  const open = $('#sidebar').classList.toggle('open');
  $('#history-toggle').setAttribute('aria-expanded', String(open));
});
window.addEventListener('beforeunload', (event) => {
  if (state.dirty) { event.preventDefault(); event.returnValue = ''; }
});

async function initialize() {
  refreshIcons();
  controls();
  try {
    const health = await api('/api/health');
    $('#connection').textContent = health.ai_configured ? 'AI 設定済み' : 'AI 未設定';
    $('#connection').classList.add(health.ai_configured ? 'ready' : 'unconfigured');
    await loadHistory();
  } catch (error) { notice(error.message); }
}

initialize();
