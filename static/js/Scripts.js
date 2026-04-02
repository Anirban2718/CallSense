
/* ═══════════════════════════════════════════════════════════════════════════
   CallSense Dashboard — Client-side JS
═══════════════════════════════════════════════════════════════════════════ */

let scoreRingChart, textEmotionChart, voiceEmotionChart;
const TIER_COLORS = {
  excellent: '#3ddc84', good: '#3ddc84',
  acceptable: '#f5a623', poor: '#ff4d6d', critical: '#ff4d6d',
};

// ── Chart.js global defaults ──────────────────────────────────────────────
Chart.defaults.color = '#626878';
Chart.defaults.borderColor = '#252932';
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 11;

// ── Score ring ────────────────────────────────────────────────────────────
function initScoreRing(score, tier) {
  const ctx = document.getElementById('scoreRingChart').getContext('2d');
  if (scoreRingChart) scoreRingChart.destroy();
  const color = TIER_COLORS[tier] || '#00d4c8';
  scoreRingChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [score, 100 - score],
        backgroundColor: [color, '#1a1d24'],
        borderWidth: 0,
        circumference: 280,
        rotation: -140,
      }],
    },
    options: {
      cutout: '78%',
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { animateRotate: true, duration: 800 },
    },
  });
}

// ── Emotion bar chart ─────────────────────────────────────────────────────
function renderEmotionChart(canvasId, scores, existingChart) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (existingChart) existingChart.destroy();

  const labels = Object.keys(scores);
  const values = labels.map(k => Math.round(scores[k] * 100));

  const EMOTION_COLORS = {
    joy: '#3ddc84', happy: '#3ddc84', excited: '#3ddc84',
    calm: '#00d4c8', surprise: '#9d7fff',
    neutral: '#626878',
    sadness: '#6fa8dc', sad: '#6fa8dc', fear: '#f5a623', fearful: '#f5a623',
    disgust: '#ff4d6d', angry: '#ff4d6d', anger: '#ff4d6d',
    frustrated: '#ff8c42',
  };
  const colors = labels.map(l => EMOTION_COLORS[l] || '#626878');

  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets: [{
        data: values,
        backgroundColor: colors.map(c => c + '99'),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: {
          beginAtZero: true, max: 100,
          grid: { color: '#252932' },
          ticks: { callback: v => v + '%' },
        },
        y: { grid: { display: false } },
      },
      animation: { duration: 600 },
    },
  });
  return chart;
}

// ── Waveform ──────────────────────────────────────────────────────────────
function renderWaveform(data) {
  const wrap = document.getElementById('waveformWrap');
  wrap.innerHTML = '';
  const maxH = 56;
  data.forEach(v => {
    const bar = document.createElement('div');
    bar.className = 'waveform-bar';
    bar.style.height = Math.max(2, v * maxH) + 'px';
    wrap.appendChild(bar);
  });
}

// ── Sub-score bars ────────────────────────────────────────────────────────
function renderSubScores(subScores, weights) {
  const row = document.getElementById('subScoreRow');
  row.innerHTML = '';
  const labels = {
    text_sentiment: 'Text Sentiment',
    text_emotion:   'Text Emotion',
    audio_emotion:  'Voice Emotion',
    audio_prosody:  'Audio Prosody',
  };
  Object.entries(subScores).forEach(([k, v]) => {
    const color = v >= 75 ? 'var(--green)' : v >= 55 ? 'var(--amber)' : 'var(--red)';
    row.innerHTML += `
      <div class="sub-score-item">
        <div class="sub-score-header">
          <span class="sub-score-label">${labels[k] || k} <span style="color:var(--text-mute)">(${Math.round((weights[k]||0)*100)}%)</span></span>
          <span class="sub-score-val" style="color:${color}">${v.toFixed(1)}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${v}%; background:${color};"></div>
        </div>
      </div>`;
  });
}

// ── Timeline ──────────────────────────────────────────────────────────────
function renderTimeline(segments) {
  const tl = document.getElementById('sentimentTimeline');
  tl.innerHTML = '';
  segments.forEach(seg => {
    const t = seg.start;
    const mins = String(Math.floor(t/60)).padStart(2,'0');
    const secs = String(Math.floor(t%60)).padStart(2,'0');
    const sp = (seg.speaker||'?').toLowerCase();
    tl.innerHTML += `
      <div class="tl-row">
        <div class="tl-time">${mins}:${secs}</div>
        <div class="tl-speaker ${sp}">${seg.speaker||'?'}</div>
        <div class="tl-text">${(seg.text||'').slice(0,80)}</div>
        <div class="tl-sentiment ${seg.sentiment||'neutral'}">${(seg.sentiment||'neutral').toUpperCase()}</div>
      </div>`;
  });
}

// ── Main render function ──────────────────────────────────────────────────
function renderResult(data) {
  const csat   = data.csat;
  const sent   = data.sentiment;
  const audio  = data.audio_features;
  const ve     = data.voice_emotion;
  const tr     = data.transcription;

  // Show results panel
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('loader').classList.remove('active');
  document.getElementById('results').style.display = 'flex';

  // Score ring
  const score = csat.csat_score;
  const tier  = csat.tier;
  initScoreRing(score, tier);
  document.getElementById('scoreBig').textContent = Math.round(score);
  document.getElementById('scoreBig').style.color = TIER_COLORS[tier] || '#00d4c8';
  document.getElementById('scoreTier').textContent = tier.charAt(0).toUpperCase() + tier.slice(1);
  document.getElementById('scoreTier').style.color = TIER_COLORS[tier] || '#00d4c8';
  document.getElementById('scoreGrade').textContent = csat.grade;
  document.getElementById('scoreGrade').style.color = TIER_COLORS[tier] || '#00d4c8';
  document.getElementById('scoreSummaryText').textContent = csat.summary || data.summary || '';

  // Alert
  const alertEl = document.getElementById('alertBanner');
  if (csat.alert_level !== 'none' && csat.alert_reasons.length) {
    alertEl.style.display = 'flex';
    alertEl.className = `alert-banner ${csat.alert_level}`;
    document.getElementById('alertIcon').textContent = csat.alert_level === 'critical' ? '🚨' : '⚠️';
    document.getElementById('alertTitle').textContent = csat.alert_level === 'critical' ? 'CRITICAL ALERT' : 'WARNING';
    document.getElementById('alertReasons').innerHTML = csat.alert_reasons.map(r => `• ${r}`).join('<br>');
  } else {
    alertEl.style.display = 'none';
  }

  // Stat tiles
  const dur = tr.duration_seconds;
  const m = Math.floor(dur/60), s = Math.floor(dur%60);
  document.getElementById('statDuration').textContent = `${m}m ${s}s`;
  document.getElementById('statWords').textContent = `${tr.word_count} words · ${tr.language?.toUpperCase()}`;
  document.getElementById('statSentiment').textContent = sent.sentiment_label.charAt(0).toUpperCase() + sent.sentiment_label.slice(1);
  document.getElementById('statPolarity').textContent = `Polarity: ${sent.sentiment_polarity > 0 ? '+' : ''}${sent.sentiment_polarity.toFixed(2)}`;
  document.getElementById('statVoiceEmotion').textContent = ve.dominant_emotion.charAt(0).toUpperCase() + ve.dominant_emotion.slice(1);
  document.getElementById('statVoiceConf').textContent = `Confidence: ${Math.round(ve.confidence*100)}%`;
  const sr = audio.prosody?.speaking_rate_wpm || 0;
  document.getElementById('statSpeakRate').textContent = `${Math.round(sr)} wpm`;
  document.getElementById('statSilence').textContent = `Silence ratio: ${Math.round((audio.prosody?.silence_ratio||0)*100)}%`;

  // Sub-scores
  renderSubScores(csat.sub_scores, csat.sub_weights);

  // Waveform
  if (data.waveform) renderWaveform(data.waveform);

  // Audio features
  document.getElementById('apPitch').textContent = `${audio.pitch?.mean_hz?.toFixed(0) || '—'} Hz`;
  document.getElementById('apTempo').textContent = `${audio.tempo_bpm?.toFixed(0) || '—'} BPM`;
  document.getElementById('apCentroid').textContent = `${audio.spectral_centroid_hz?.toFixed(0) || '—'} Hz`;
  document.getElementById('apProsody').textContent = `${((audio.prosody?.prosody_score||0)*100).toFixed(0)}%`;

  // Emotion charts
  textEmotionChart  = renderEmotionChart('textEmotionChart', sent.emotion_scores, textEmotionChart);
  voiceEmotionChart = renderEmotionChart('voiceEmotionChart', ve.emotion_scores, voiceEmotionChart);

  // Timeline
  renderTimeline(sent.segment_sentiments || []);

  // Keywords
  const negChips = document.getElementById('negChips');
  const posChips = document.getElementById('posChips');
  negChips.innerHTML = '';
  posChips.innerHTML = '';
  (sent.negativity_indicators || []).forEach(kw => {
    negChips.innerHTML += `<span class="chip neg">${kw}</span>`;
  });
  (sent.key_phrases || []).filter(k => !sent.negativity_indicators?.includes(k)).forEach(kw => {
    posChips.innerHTML += `<span class="chip pos">${kw}</span>`;
  });
  if (!negChips.innerHTML) negChips.innerHTML = '<span style="font-size:.75rem;color:var(--text-mute)">None detected</span>';
  if (!posChips.innerHTML) posChips.innerHTML = '<span style="font-size:.75rem;color:var(--text-mute)">None detected</span>';

  // Summary
  document.getElementById('callSummary').textContent = data.summary || '—';

  // Transcript
  const tb = document.getElementById('transcriptBox');
  tb.innerHTML = '';
  (tr.segments || []).forEach(seg => {
    const sp = (seg.speaker || 'Unknown').toLowerCase();
    const t = seg.start;
    const mm = String(Math.floor(t/60)).padStart(2,'0');
    const ss = String(Math.floor(t%60)).padStart(2,'0');
    tb.innerHTML += `
      <div class="seg">
        <div class="seg-meta">
          <div class="seg-speaker ${sp}">${seg.speaker || '?'}</div>
          <div style="font-size:.65rem">${mm}:${ss}</div>
        </div>
        <div class="seg-text">${seg.text || ''}</div>
      </div>`;
  });

  // Recommendations
  const rl = document.getElementById('recList');
  rl.innerHTML = '';
  (csat.recommendations || []).forEach((r, i) => {
    rl.innerHTML += `<div class="rec-item ${i < 2 && csat.alert_level !== 'none' ? 'priority' : ''}">${r}</div>`;
  });
  if (!csat.recommendations?.length) rl.innerHTML = '<div style="font-size:.8rem;color:var(--text-mute)">No action items.</div>';

  // Agent
  const as = csat.agent_score;
  if (as != null) {
    document.getElementById('agentScore').textContent = Math.round(as);
    document.getElementById('agentBar').style.width = as + '%';
    document.getElementById('agentBar').style.background = as >= 75 ? 'var(--purple)' : as >= 50 ? 'var(--amber)' : 'var(--red)';
  }
  const af = document.getElementById('agentFlags');
  af.innerHTML = '';
  (csat.agent_flags || []).forEach(f => { af.innerHTML += `<div class="agent-flag">${f}</div>`; });
}

// ── Delete call ───────────────────────────────────────────────────────────
async function deleteCall(event, callId) {
  event.stopPropagation();   // don't trigger loadResult
  if (!confirm('Delete this call record?')) return;
  try {
    await fetch('/results/' + callId, { method: 'DELETE' });
    // If it was currently displayed, go back to empty state
    const activeItem = document.querySelector('.call-item.active');
    if (activeItem && activeItem.querySelector(`[onclick*="${callId}"]`)) {
      showEmpty();
    }
    await refreshCallList(null);
  } catch (err) {
    alert('Delete failed: ' + err.message);
  }
}

// ── Upload handler ────────────────────────────────────────────────────────
async function handleUpload(input) {
  const file = input.files[0];
  if (!file) return;
  showLoader(file.name);

  const form = new FormData();
  form.append('file', file);

  const steps = ['Loading Whisper model…', 'Transcribing audio…', 'Analyzing sentiment…', 'Extracting audio features…', 'Detecting voice emotion…', 'Computing CSAT score…'];
  let si = 0;
  const stepInterval = setInterval(() => {
    document.getElementById('loaderStep').textContent = steps[Math.min(si++, steps.length-1)];
  }, 4000);

  try {
    const res = await fetch('/analyze', { method: 'POST', body: form });
    clearInterval(stepInterval);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Analysis failed');
    renderResult(data);
    await refreshCallList(data.call_id);
  } catch (err) {
    clearInterval(stepInterval);
    alert('Error: ' + err.message);
    showEmpty();
  }
}

// ── Demo ──────────────────────────────────────────────────────────────────
async function loadDemo() {
  showLoader('Demo Call');
  document.getElementById('loaderStep').textContent = 'Generating synthetic call data…';
  try {
    const res  = await fetch('/demo');
    const data = await res.json();
    renderResult(data);
    await refreshCallList(data.call_id);
  } catch (err) {
    alert('Demo failed: ' + err.message);
    showEmpty();
  }
}

// ── Load stored result ────────────────────────────────────────────────────
async function loadResult(callId) {
  showLoader('Loading…');
  const res  = await fetch('/results/' + callId);
  const data = await res.json();
  renderResult(data);
}

// ── Sidebar call list ─────────────────────────────────────────────────────
async function refreshCallList(activeId) {
  try {
    const res   = await fetch('/calls');
    const calls = await res.json();
    const list  = document.getElementById('callList');
    if (!calls.length) {
      list.innerHTML = '<div style="color:var(--text-mute); font-size:.78rem; padding:8px 4px;">No calls yet.</div>';
      return;
    }
    list.innerHTML = '';
    calls.forEach(c => {
      const score = c.csat_score != null ? Math.round(c.csat_score) : '—';
      const tier  = c.tier || 'neutral';
      const name  = c.filename || c.call_id;
      const dur   = c.duration ? (() => { const m=Math.floor(c.duration/60),s=Math.floor(c.duration%60); return `${m}m ${s}s`; })() : '';
      const item  = document.createElement('div');
      item.className = 'call-item' + (c.call_id === activeId ? ' active' : '');
      item.innerHTML = `
        <div class="call-dot ${tier}"></div>
        <div class="call-info">
          <div class="call-name" title="${name}">${name}</div>
          <div class="call-meta">${dur} ${c.alert_level !== 'none' ? '· ⚠' : ''}</div>
        </div>
        <div class="call-score" style="color:${TIER_COLORS[tier]||'var(--text-dim)'}">${score}</div>
        <button class="call-delete" title="Delete" onclick="deleteCall(event, '${c.call_id}')">✕</button>`;
      item.onclick = () => loadResult(c.call_id);
      list.appendChild(item);
    });
  } catch (_) {}
}

// ── UI helpers ────────────────────────────────────────────────────────────
function showLoader(name) {
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('results').style.display = 'none';
  document.getElementById('loaderStep').textContent = `Processing: ${name}`;
  document.getElementById('loader').classList.add('active');
}
function showEmpty() {
  document.getElementById('emptyState').style.display = 'flex';
  document.getElementById('results').style.display = 'none';
  document.getElementById('loader').classList.remove('active');
}

// ── Drag & drop ───────────────────────────────────────────────────────────
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    const fakeInput = { files: [file] };
    handleUpload(fakeInput);
  }
});

// ── Init ──────────────────────────────────────────────────────────────────
refreshCallList(null);
