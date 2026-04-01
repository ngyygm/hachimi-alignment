#!/usr/bin/env python3
"""
Lyric Annotation Tool — Flask Web Interface
支持断点续传、多标注员、实时保存。

Usage:
    python3 annotate.py [--port 5000] [--annotator A1]

Annotations saved to: annotation_results/{annotator}_annotations.json
"""
import json, os, sys, time, argparse
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_from_directory

# ================================================================
# Config
# ================================================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_FILE = BASE_DIR / "scripts" / "annotation" / "annotation_data.json"
RESULTS_DIR = BASE_DIR / "annotation_results"
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# Load annotation data
with open(DATA_FILE, "r", encoding="utf-8") as f:
    DATA = json.load(f)

ANNOTATOR = "A1"  # default

# ================================================================
# HTML Template — Single-file app
# ================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>歌词标注工具 — Lyric Annotation</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --border: #334155;
    --text: #e2e8f0; --dim: #94a3b8; --accent: #38bdf8;
    --green: #34d399; --orange: #fb923c; --purple: #a78bfa;
    --red: #f87171; --pink: #f472b6;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

  .header {
    background: var(--card); border-bottom: 2px solid var(--border);
    padding: 16px 24px; display: flex; justify-content: space-between; align-items: center;
    position: sticky; top: 0; z-index: 100;
  }
  .header h1 { font-size: 18px; font-weight: 600; }
  .progress-bar { display: flex; align-items: center; gap: 12px; }
  .progress-track { width: 200px; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .progress-fill { height: 100%; background: var(--green); transition: width 0.3s; }
  .progress-text { font-size: 13px; color: var(--dim); min-width: 80px; }

  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }

  /* Cards */
  .lyrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .lyric-card {
    background: var(--card); border-radius: 12px; padding: 20px;
    border: 1px solid var(--border); transition: border-color 0.2s;
  }
  .lyric-card:hover { border-color: var(--accent); }
  .lyric-card h3 {
    font-size: 14px; font-weight: 600; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
  }
  .badge {
    font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 500;
  }
  .badge-orig { background: #10b98120; color: var(--green); }
  .badge-para { background: #8b5cf620; color: var(--purple); }
  .badge-homo { background: #f9731620; color: var(--orange); }
  .badge-hach { background: #f472b620; color: var(--pink); }
  .lyric-text {
    font-size: 15px; line-height: 2; color: var(--text);
    max-height: 240px; overflow-y: auto; padding-right: 8px;
    white-space: pre-wrap; word-break: break-all;
  }
  .lyric-text::-webkit-scrollbar { width: 4px; }
  .lyric-text::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .song-name { font-size: 12px; color: var(--dim); margin-bottom: 4px; }

  /* Rating */
  .rating-section {
    background: var(--card); border-radius: 12px; padding: 24px;
    border: 1px solid var(--border); margin-bottom: 20px;
  }
  .rating-section h3 { font-size: 16px; margin-bottom: 16px; }
  .rating-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 0; border-bottom: 1px solid var(--border);
  }
  .rating-row:last-child { border-bottom: none; }
  .rating-label { font-size: 14px; color: var(--dim); flex: 1; }
  .rating-label strong { color: var(--text); }
  .stars { display: flex; gap: 4px; }
  .star {
    width: 36px; height: 36px; border-radius: 8px; border: 2px solid var(--border);
    background: transparent; cursor: pointer; font-size: 16px;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s; color: var(--dim);
  }
  .star:hover { border-color: var(--accent); color: var(--accent); transform: scale(1.1); }
  .star.active { background: var(--accent); color: var(--bg); border-color: var(--accent); }
  .star-desc { font-size: 11px; color: var(--dim); text-align: center; margin-top: 2px; }

  /* Comparison */
  .comparison {
    background: var(--card); border-radius: 12px; padding: 24px;
    border: 1px solid var(--border); margin-bottom: 20px;
  }
  .comparison h3 { margin-bottom: 16px; }
  .comp-row { display: flex; gap: 8px; margin-bottom: 8px; }
  .comp-btn {
    flex: 1; padding: 10px; border-radius: 8px; border: 2px solid var(--border);
    background: transparent; color: var(--dim); cursor: pointer; font-size: 13px;
    transition: all 0.15s; text-align: center;
  }
  .comp-btn:hover { border-color: var(--accent); }
  .comp-btn.selected { background: var(--accent); color: var(--bg); border-color: var(--accent); font-weight: 600; }

  /* Buttons */
  .actions { display: flex; gap: 12px; justify-content: space-between; align-items: center; }
  .btn {
    padding: 12px 24px; border-radius: 10px; border: none; cursor: pointer;
    font-size: 14px; font-weight: 600; transition: all 0.15s;
  }
  .btn-primary { background: var(--accent); color: var(--bg); }
  .btn-primary:hover { filter: brightness(1.1); transform: translateY(-1px); }
  .btn-secondary { background: var(--border); color: var(--text); }
  .btn-secondary:hover { background: #475569; }
  .btn-danger { background: #ef444440; color: var(--red); border: 1px solid #ef444440; }
  .btn-danger:hover { background: #ef444460; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Start screen */
  .start-screen {
    max-width: 500px; margin: 80px auto; text-align: center;
  }
  .start-screen h1 { font-size: 28px; margin-bottom: 12px; }
  .start-screen p { color: var(--dim); margin-bottom: 32px; line-height: 1.8; }
  .start-card {
    background: var(--card); border-radius: 16px; padding: 32px;
    border: 1px solid var(--border); text-align: left;
  }
  .start-card label { display: block; font-size: 13px; color: var(--dim); margin-bottom: 6px; }
  .start-card input {
    width: 100%; padding: 10px 14px; border-radius: 8px; border: 2px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 15px; margin-bottom: 16px;
  }
  .start-card input:focus { outline: none; border-color: var(--accent); }

  /* Status indicator */
  .status { display: flex; gap: 16px; font-size: 13px; color: var(--dim); }
  .status span { display: flex; align-items: center; gap: 4px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-green { background: var(--green); }
  .dot-orange { background: var(--orange); }

  /* Toast */
  .toast {
    position: fixed; bottom: 24px; right: 24px; padding: 12px 20px;
    border-radius: 10px; font-size: 14px; font-weight: 500;
    transform: translateY(100px); opacity: 0; transition: all 0.3s;
    z-index: 200;
  }
  .toast.show { transform: translateY(0); opacity: 1; }
  .toast-success { background: #10b981; color: white; }
  .toast-info { background: var(--accent); color: var(--bg); }
</style>
</head>
<body>

<!-- Toast notification -->
<div id="toast" class="toast"></div>

<!-- Start Screen -->
<div id="startScreen" class="start-screen">
  <h1>歌词标注工具</h1>
  <p>Lyric Annotation Tool<br>对比原歌词、复述、同音字、哈基米版本，评价语义保持度和自然度</p>
  <div class="start-card">
    <label>标注员编号 (Annotator ID)</label>
    <input id="annotatorInput" placeholder="e.g., A1, A2, A3..." value="A1">
    <label>每轮标注数量</label>
    <input id="batchSizeInput" type="number" value="20" min="5" max="50">
    <button class="btn btn-primary" onclick="startAnnotation()" style="width:100%;margin-top:8px;">
      开始标注 / 继续标注
    </button>
    <p style="margin-top:12px;font-size:12px;color:var(--dim);text-align:center;">
      支持断点续传 — 已标注的数据不会丢失
    </p>
  </div>
</div>

<!-- Annotation Screen -->
<div id="annotScreen" style="display:none;">
  <div class="header">
    <div>
      <h1>歌词标注 — <span id="annotatorName"></span></h1>
      <div class="status">
        <span><div class="dot dot-green"></div> 自动保存</span>
        <span id="songInfo"></span>
      </div>
    </div>
    <div class="progress-bar">
      <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
      <div class="progress-text" id="progressText">0/0</div>
    </div>
  </div>

  <div class="container">
    <div class="song-name" id="currentSong"></div>

    <!-- Lyrics display -->
    <div class="lyrics-grid">
      <div class="lyric-card">
        <h3><span class="badge badge-orig">C0 原歌词</span> Original Lyrics</h3>
        <div class="lyric-text" id="originalText"></div>
      </div>
      <div class="lyric-card">
        <h3><span class="badge badge-para">C8 复述</span> Paraphrase</h3>
        <div class="lyric-text" id="paraphraseText"></div>
      </div>
      <div class="lyric-card">
        <h3><span class="badge badge-homo">同音字</span> Homophone</h3>
        <div class="lyric-text" id="homophoneText"></div>
      </div>
      <div class="lyric-card">
        <h3><span class="badge badge-hach">哈基米</span> Hachimi</h3>
        <div class="lyric-text" id="hachimiText"></div>
      </div>
    </div>

    <!-- Rating: Paraphrase -->
    <div class="rating-section">
      <h3>复述 (Paraphrase) 评价</h3>
      <div class="rating-row">
        <div class="rating-label"><strong>语义保持度</strong><br>复述是否保持了原歌词的意思？</div>
        <div class="stars" data-name="para_meaning">
          <div class="star" data-val="1" onclick="setRating(this)" title="完全不同">1</div>
          <div class="star" data-val="2" onclick="setRating(this)" title="大部分丢失">2</div>
          <div class="star" data-val="3" onclick="setRating(this)" title="部分保持">3</div>
          <div class="star" data-val="4" onclick="setRating(this)" title="大部分保持">4</div>
          <div class="star" data-val="5" onclick="setRating(this)" title="完全保持">5</div>
        </div>
      </div>
      <div class="rating-row">
        <div class="rating-label"><strong>流畅自然度</strong><br>复述读起来是否自然通顺？</div>
        <div class="stars" data-name="para_natural">
          <div class="star" data-val="1" onclick="setRating(this)">1</div>
          <div class="star" data-val="2" onclick="setRating(this)">2</div>
          <div class="star" data-val="3" onclick="setRating(this)">3</div>
          <div class="star" data-val="4" onclick="setRating(this)">4</div>
          <div class="star" data-val="5" onclick="setRating(this)">5</div>
        </div>
      </div>
      <div class="rating-row">
        <div class="rating-label"><strong>可唱性</strong><br>复述是否适合唱出来？</div>
        <div class="stars" data-name="para_singable">
          <div class="star" data-val="1" onclick="setRating(this)">1</div>
          <div class="star" data-val="2" onclick="setRating(this)">2</div>
          <div class="star" data-val="3" onclick="setRating(this)">3</div>
          <div class="star" data-val="4" onclick="setRating(this)">4</div>
          <div class="star" data-val="5" onclick="setRating(this)">5</div>
        </div>
      </div>
    </div>

    <!-- Rating: Homophone -->
    <div class="rating-section">
      <h3>同音字 (Homophone) 评价</h3>
      <div class="rating-row">
        <div class="rating-label"><strong>语音相似度</strong><br>同音字版本听起来是否和原歌词一样？</div>
        <div class="stars" data-name="homo_phonology">
          <div class="star" data-val="1" onclick="setRating(this)">1</div>
          <div class="star" data-val="2" onclick="setRating(this)">2</div>
          <div class="star" data-val="3" onclick="setRating(this)">3</div>
          <div class="star" data-val="4" onclick="setRating(this)">4</div>
          <div class="star" data-val="5" onclick="setRating(this)">5</div>
        </div>
      </div>
      <div class="rating-row">
        <div class="rating-label"><strong>可读性</strong><br>同音字版本能否被读懂？</div>
        <div class="stars" data-name="homo_readable">
          <div class="star" data-val="1" onclick="setRating(this)">1</div>
          <div class="star" data-val="2" onclick="setRating(this)">2</div>
          <div class="star" data-val="3" onclick="setRating(this)">3</div>
          <div class="star" data-val="4" onclick="setRating(this)">4</div>
          <div class="star" data-val="5" onclick="setRating(this)">5</div>
        </div>
      </div>
      <div class="rating-row">
        <div class="rating-label"><strong>语义丢失程度</strong><br>同音字替换后，意思丢失了多少？</div>
        <div class="stars" data-name="homo_meaning_loss">
          <div class="star" data-val="1" onclick="setRating(this)" title="没丢">1</div>
          <div class="star" data-val="2" onclick="setRating(this)" title="丢了一点">2</div>
          <div class="star" data-val="3" onclick="setRating(this)" title="丢了一半">3</div>
          <div class="star" data-val="4" onclick="setRating(this)" title="丢了大部分">4</div>
          <div class="star" data-val="5" onclick="setRating(this)" title="完全不知所云">5</div>
        </div>
      </div>
    </div>

    <!-- Comparison -->
    <div class="comparison">
      <h3>整体对比</h3>
      <p style="font-size:13px;color:var(--dim);margin-bottom:12px;">哪个版本最适合唱？选择一个</p>
      <div class="comp-row">
        <button class="comp-btn" data-name="best_sing" data-val="original" onclick="setComp(this)">原歌词 C0</button>
        <button class="comp-btn" data-name="best_sing" data-val="paraphrase" onclick="setComp(this)">复述 C8</button>
        <button class="comp-btn" data-name="best_sing" data-val="homophone" onclick="setComp(this)">同音字</button>
        <button class="comp-btn" data-name="best_sing" data-val="hachimi" onclick="setComp(this)">哈基米 C1</button>
      </div>
      <p style="font-size:13px;color:var(--dim);margin:16px 0 12px;">哪个版本最有意义？</p>
      <div class="comp-row">
        <button class="comp-btn" data-name="most_meaningful" data-val="original" onclick="setComp(this)">原歌词</button>
        <button class="comp-btn" data-name="most_meaningful" data-val="paraphrase" onclick="setComp(this)">复述</button>
        <button class="comp-btn" data-name="most_meaningful" data-val="homophone" onclick="setComp(this)">同音字</button>
        <button class="comp-btn" data-name="most_meaningful" data-val="hachimi" onclick="setComp(this)">哈基米</button>
      </div>
    </div>

    <!-- Note -->
    <div class="rating-section">
      <h3>备注（可选）</h3>
      <textarea id="noteInput" style="width:100%;height:60px;background:var(--bg);color:var(--text);border:2px solid var(--border);border-radius:8px;padding:10px;font-size:14px;resize:vertical;" placeholder="任何关于这首歌的观察..."></textarea>
    </div>

    <!-- Actions -->
    <div class="actions">
      <button class="btn btn-secondary" id="prevBtn" onclick="prevItem()">← 上一首</button>
      <span style="color:var(--dim);font-size:13px;" id="saveStatus"></span>
      <button class="btn btn-primary" id="nextBtn" onclick="nextItem()">下一首 →</button>
    </div>
  </div>
</div>

<script>
// ================================================================
// State
// ================================================================
let items = [];
let annotations = {};
let annotator = 'A1';
let currentIndex = 0;
let batchSize = 20;
let currentRatings = {};

// ================================================================
// API calls
// ================================================================
async function api(url, data) {
  const resp = await fetch(url, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });
  return resp.json();
}

// ================================================================
// Start
// ================================================================
async function startAnnotation() {
  annotator = document.getElementById('annotatorInput').value.trim() || 'A1';
  batchSize = parseInt(document.getElementById('batchSizeInput').value) || 20;

  const res = await api('/api/start', {annotator, batch_size: batchSize});
  if (res.error) { alert(res.error); return; }

  items = res.items;
  annotations = res.annotations;
  currentIndex = res.next_index;

  document.getElementById('startScreen').style.display = 'none';
  document.getElementById('annotScreen').style.display = 'block';
  document.getElementById('annotatorName').textContent = annotator;

  showItem(currentIndex);
  updateProgress();
}

// ================================================================
// Display
// ================================================================
function showItem(idx) {
  if (idx < 0 || idx >= items.length) return;
  currentIndex = idx;
  const item = items[idx];

  document.getElementById('currentSong').textContent = `#${idx+1} / ${items.length} — ${item.song_name}`;
  document.getElementById('originalText').textContent = item.original || item.full_original;
  document.getElementById('paraphraseText').textContent = item.paraphrase || item.full_paraphrase || '(无复述)';
  document.getElementById('homophoneText').textContent = item.homophone || item.full_homophone || '(无同音字)';
  document.getElementById('hachimiText').textContent = item.hachimi || item.full_hachimi || '(无哈基米)';

  // Restore existing ratings
  currentRatings = annotations[item.id] ? JSON.parse(JSON.stringify(annotations[item.id])) : {};
  restoreRatings();

  // Update buttons
  document.getElementById('prevBtn').disabled = (idx === 0);
  document.getElementById('nextBtn').textContent = idx === items.length - 1 ? '完成 ✓' : '下一首 →';
  document.getElementById('noteInput').value = currentRatings.note || '';
  document.getElementById('saveStatus').textContent = '';
}

function restoreRatings() {
  // Clear all
  document.querySelectorAll('.star').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.comp-btn').forEach(b => b.classList.remove('selected'));

  // Restore stars
  for (const [name, val] of Object.entries(currentRatings)) {
    if (name === 'note' || name === 'best_sing' || name === 'most_meaningful') continue;
    const container = document.querySelector(`.stars[data-name="${name}"]`);
    if (container) {
      const star = container.querySelector(`.star[data-val="${val}"]`);
      if (star) star.classList.add('active');
    }
  }

  // Restore comparisons
  if (currentRatings.best_sing) {
    const btn = document.querySelector(`.comp-btn[data-name="best_sing"][data-val="${currentRatings.best_sing}"]`);
    if (btn) btn.classList.add('selected');
  }
  if (currentRatings.most_meaningful) {
    const btn = document.querySelector(`.comp-btn[data-name="most_meaningful"][data-val="${currentRatings.most_meaningful}"]`);
    if (btn) btn.classList.add('selected');
  }
}

// ================================================================
// Rating handlers
// ================================================================
function setRating(el) {
  const container = el.closest('.stars');
  const name = container.dataset.name;
  const val = parseInt(el.dataset.val);

  container.querySelectorAll('.star').forEach(s => s.classList.remove('active'));
  el.classList.add('active');
  currentRatings[name] = val;
  autoSave();
}

function setComp(el) {
  const name = el.dataset.name;
  const val = el.dataset.val;

  document.querySelectorAll(`.comp-btn[data-name="${name}"]`).forEach(b => b.classList.remove('selected'));
  el.classList.add('selected');
  currentRatings[name] = val;
  autoSave();
}

// ================================================================
// Navigation
// ================================================================
function prevItem() {
  if (currentIndex > 0) {
    saveCurrent();
    showItem(currentIndex - 1);
  }
}

async function nextItem() {
  saveCurrent();

  if (currentIndex >= items.length - 1) {
    // Last item — finish
    await api('/api/save', {annotator, annotations});
    showToast('标注完成！感谢你的工作 🎉', 'success');
    setTimeout(() => location.reload(), 2000);
    return;
  }

  showItem(currentIndex + 1);
  updateProgress();
}

function saveCurrent() {
  currentRatings.note = document.getElementById('noteInput').value;
  if (Object.keys(currentRatings).length > 1) { // has ratings beyond just note
    annotations[items[currentIndex].id] = {...currentRatings};
  }
}

async function autoSave() {
  saveCurrent();
  await api('/api/save', {annotator, annotations});
  const status = document.getElementById('saveStatus');
  status.textContent = '✓ 已保存';
  setTimeout(() => status.textContent = '', 1500);
  updateProgress();
}

function updateProgress() {
  const done = Object.keys(annotations).length;
  const total = items.length;
  const pct = total > 0 ? (done / total * 100) : 0;
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent = `${done}/${total} (${pct.toFixed(0)}%)`;
}

function showToast(msg, type='info') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast toast-${type} show`;
  setTimeout(() => t.classList.remove('show'), 2500);
}
</script>
</body>
</html>
"""

# ================================================================
# Routes
# ================================================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/start', methods=['POST'])
def api_start():
    global ANNOTATOR
    data = request.json
    annotator = data.get('annotator', 'A1')
    batch_size = data.get('batch_size', 20)
    ANNOTATOR = annotator

    # Load existing annotations (resume support)
    result_file = RESULTS_DIR / f"{annotator}_annotations.json"
    existing = {}
    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            saved = json.load(f)
            existing = saved.get('annotations', {})
        print(f"Loaded {len(existing)} existing annotations for {annotator}")

    # Select batch (prioritize unannotated items)
    all_items = DATA['items']
    annotated_ids = set(existing.keys())
    unannotated = [item for item in all_items if str(item['id']) not in annotated_ids]

    # Take batch_size items, prioritizing unannotated
    batch = unannotated[:batch_size]
    if len(batch) < batch_size:
        # Fill with annotated items for review
        remaining = [item for item in all_items if str(item['id']) in annotated_ids]
        batch.extend(remaining[:batch_size - len(batch)])

    if not batch:
        batch = all_items[:batch_size]

    # Find next unannotated index
    next_idx = 0
    for i, item in enumerate(batch):
        if str(item['id']) not in annotated_ids:
            next_idx = i
            break

    return jsonify({
        'items': batch,
        'annotations': existing,
        'next_index': next_idx,
        'total': len(all_items),
        'annotated': len(annotated_ids),
    })

@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.json
    annotator = data.get('annotator', ANNOTATOR)
    annotations = data.get('annotations', {})

    result_file = RESULTS_DIR / f"{annotator}_annotations.json"
    result = {
        'annotator': annotator,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_annotated': len(annotations),
        'annotations': annotations,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(annotations)} annotations for {annotator}")
    return jsonify({'status': 'ok', 'n_saved': len(annotations)})

# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lyric Annotation Tool')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--annotator', default='A1')
    args = parser.parse_args()

    ANNOTATOR = args.annotator
    print(f"\n{'='*50}")
    print(f"  Lyric Annotation Tool")
    print(f"  http://localhost:{args.port}")
    print(f"  Annotator: {args.annotator}")
    print(f"  Data: {len(DATA['items'])} songs")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*50}\n")

    app.run(host='0.0.0.0', port=args.port, debug=False)
