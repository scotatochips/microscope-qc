/* ═══════════════════════════════════════════════════════════════
   MicroScope QC — Frontend Logic
   ═══════════════════════════════════════════════════════════════ */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
  fileInput:    $('#fileInput'),
  dropZone:     $('#dropZone'),
  uploadFrame:  $('.upload-frame'),
  uploadContent:$('#uploadContent'),
  uploadPreview:$('#uploadPreview'),
  previewImg:   $('#previewImg'),
  previewName:  $('#previewName'),
  previewDim:   $('#previewDim'),
  browseBtn:    $('#browseBtn'),
  clearBtn:     $('#clearBtn'),
  analyzeBtn:   $('#analyzeBtn'),

  hero:         $('#hero'),
  loading:      $('#loadingSection'),
  results:      $('#resultsSection'),
  newAnalysis:  $('#newAnalysisBtn'),
};

let selectedFile = null;

// ════════════════════════════════════════════
// FILE SELECTION
// ════════════════════════════════════════════
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    alert('Please select a valid image file.');
    return;
  }
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    els.previewImg.src = e.target.result;
    els.previewImg.onload = () => {
      els.previewDim.textContent = `${els.previewImg.naturalWidth} × ${els.previewImg.naturalHeight} px · ${(file.size/1024).toFixed(0)} KB`;
    };
    els.previewName.textContent = file.name;
    els.uploadContent.classList.add('hidden');
    els.uploadPreview.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  selectedFile = null;
  els.fileInput.value = '';
  els.uploadContent.classList.remove('hidden');
  els.uploadPreview.classList.add('hidden');
}

// Click handlers
els.browseBtn.addEventListener('click', () => els.fileInput.click());
els.uploadFrame.addEventListener('click', (e) => {
  if (e.target.closest('.upload-preview')) return;
  els.fileInput.click();
});
els.clearBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  resetUpload();
});

els.fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// Drag & drop
['dragover', 'dragenter'].forEach(ev =>
  els.uploadFrame.addEventListener(ev, (e) => {
    e.preventDefault();
    els.uploadFrame.classList.add('dragging');
  })
);
['dragleave', 'drop'].forEach(ev =>
  els.uploadFrame.addEventListener(ev, (e) => {
    e.preventDefault();
    els.uploadFrame.classList.remove('dragging');
  })
);
els.uploadFrame.addEventListener('drop', (e) => {
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});


// ════════════════════════════════════════════
// ANALYSIS
// ════════════════════════════════════════════
els.analyzeBtn.addEventListener('click', async (e) => {
  e.stopPropagation();
  if (!selectedFile) return;
  await runAnalysis(selectedFile);
});

els.newAnalysis.addEventListener('click', () => {
  els.results.classList.add('hidden');
  els.hero.classList.remove('hidden');
  resetUpload();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});


async function runAnalysis(file) {
  els.hero.classList.add('hidden');
  els.results.classList.add('hidden');
  els.loading.classList.remove('hidden');

  // Animate loading steps
  const steps = $$('.loading-step');
  steps.forEach(s => s.classList.remove('active', 'done'));
  let stepIdx = 0;
  steps[0].classList.add('active');
  const stepTimer = setInterval(() => {
    if (stepIdx < steps.length - 1) {
      steps[stepIdx].classList.remove('active');
      steps[stepIdx].classList.add('done');
      stepIdx++;
      steps[stepIdx].classList.add('active');
    }
  }, 600);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Analysis failed');
    }
    const data = await res.json();
    clearInterval(stepTimer);
    steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });

    // Brief pause so user sees completion
    await new Promise(r => setTimeout(r, 300));

    els.loading.classList.add('hidden');
    els.results.classList.remove('hidden');
    renderResults(data);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (err) {
    clearInterval(stepTimer);
    alert('Error: ' + err.message);
    els.loading.classList.add('hidden');
    els.hero.classList.remove('hidden');
  }
}


// ════════════════════════════════════════════
// RENDER RESULTS
// ════════════════════════════════════════════
function scoreColor(s) {
  if (s >= 75) return '#00e5a0';
  if (s >= 45) return '#ffb800';
  return '#ff3d5a';
}

function severityClass(sev) {
  return { ok: 'ok', warning: 'warning', critical: 'critical' }[sev] || 'ok';
}

function renderResults(data) {
  // ── Score banner ──
  const score = data.overall_score;
  const scoreNum = $('#overallScore');
  animateNumber(scoreNum, 0, score, 1200);

  const lbl = $('#scoreLabel');
  lbl.textContent = data.label;
  lbl.classList.remove('good', 'bad');
  lbl.classList.add(data.label.startsWith('GOOD') ? 'good' : 'bad');

  // Animated arc
  const arc = $('#meterArc');
  const circumference = 628.3;
  const offset = circumference - (score / 100) * circumference;
  arc.style.transition = 'stroke-dashoffset 1.4s cubic-bezier(.16,1,.3,1)';
  setTimeout(() => { arc.style.strokeDashoffset = offset; }, 100);

  // Tick marks
  const ticks = $('#meterTicks');
  ticks.innerHTML = '';
  for (let i = 0; i < 60; i++) {
    const angle = (i / 60) * 360 - 90;
    const rad = angle * Math.PI / 180;
    const r1 = 110, r2 = i % 5 === 0 ? 117 : 114;
    const x1 = 120 + Math.cos(rad) * r1;
    const y1 = 120 + Math.sin(rad) * r1;
    const x2 = 120 + Math.cos(rad) * r2;
    const y2 = 120 + Math.sin(rad) * r2;
    ticks.innerHTML += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="rgba(255,255,255,${i % 5 === 0 ? 0.3 : 0.1})" stroke-width="1"/>`;
  }

  // Image profile
  $('#infoDim').textContent = `${data.image_info.width} × ${data.image_info.height} px`;
  $('#infoIssues').textContent = data.summary.length;
  const criticalCount = Object.values(data.metrics).filter(m => m.severity === 'critical').length;
  $('#infoCritical').textContent = criticalCount;
  $('#infoRec').textContent = score >= 60 ? 'PROCEED' : 'REJECT';
  $('#infoRec').style.color = score >= 60 ? '#00e5a0' : '#ff3d5a';


  // ── Metric cards ──
  ['blur', 'lighting', 'noise', 'density'].forEach(key => {
    const m = data.metrics[key];
    const card = $(`#card${key.charAt(0).toUpperCase() + key.slice(1)}`);
    card.classList.remove('severity-ok', 'severity-warning', 'severity-critical');
    card.classList.add(`severity-${m.severity}`);

    const numEl = card.querySelector('.ms-num');
    animateNumber(numEl, 0, m.score, 1000);

    const fill = card.querySelector('.metric-fill');
    setTimeout(() => {
      fill.style.width = m.score + '%';
      fill.style.background = scoreColor(m.score);
    }, 100);

    const sev = card.querySelector('.metric-sev');
    sev.textContent = m.severity.toUpperCase();
    sev.className = `metric-sev ${m.severity}`;

    // Raw values
    const raw = card.querySelector('.metric-raw');
    raw.innerHTML = Object.entries(m.raw).map(([k, v]) =>
      `<div class="raw-row"><span>${k.replace(/_/g, ' ')}</span><span>${v}</span></div>`
    ).join('');

    // Issues
    const issues = card.querySelector('.metric-issues');
    if (m.issues.length === 0) {
      issues.innerHTML = `<div class="metric-issue ok">No issues</div>`;
    } else {
      issues.innerHTML = m.issues.map(i =>
        `<div class="metric-issue ${m.severity === 'critical' ? 'critical' : ''}">${i}</div>`
      ).join('');
    }
  });


  // ── Visualizations ──
  $('#annotatedImg').src     = data.images.annotated || data.images.original;
  $('#heatmapImg').src       = data.images.heatmap   || data.images.original;
  $('#compareOriginal').src  = data.images.original;
  $('#compareAnnotated').src = data.images.annotated || data.images.original;

  // Histogram
  drawHistogram(data.histogram);

  // ── Issues list ──
  const issuesList = $('#issuesList');
  if (data.summary.length === 0) {
    issuesList.innerHTML = `<li class="no-issues">✓ NO ISSUES DETECTED · IMAGE PASSES ALL QUALITY CHECKS</li>`;
  } else {
    const items = [];
    Object.entries(data.metrics).forEach(([cat, m]) => {
      m.issues.forEach(text => {
        items.push({ category: cat.toUpperCase(), severity: m.severity, text });
      });
    });
    issuesList.innerHTML = items.map(it => `
      <li class="issue-item">
        <span class="issue-cat ${it.severity}">${it.category}</span>
        <span class="issue-text">${it.text}</span>
      </li>
    `).join('');
  }
}


// ════════════════════════════════════════════
// HISTOGRAM RENDERING
// ════════════════════════════════════════════
function drawHistogram(histData) {
  const svg = $('#histogramSvg');
  const W = 600, H = 240, PAD = 8;

  const colors = { Red: '#ff3d5a', Green: '#00e5a0', Blue: '#0099ff' };

  // Find max across all channels for normalisation
  let maxVal = 0;
  Object.values(histData).forEach(arr => {
    arr.forEach(v => { if (v > maxVal) maxVal = v; });
  });

  let svgContent = '';

  // grid lines
  for (let i = 0; i <= 4; i++) {
    const y = PAD + (H - 2*PAD) * (i / 4);
    svgContent += `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="#1c2330" stroke-width="0.5"/>`;
  }
  for (let i = 0; i <= 8; i++) {
    const x = (W * i / 8);
    svgContent += `<line x1="${x}" y1="${PAD}" x2="${x}" y2="${H-PAD}" stroke="#1c2330" stroke-width="0.5"/>`;
  }

  // channel paths (256 bins → SVG path)
  Object.entries(histData).forEach(([channel, data]) => {
    const colour = colors[channel];
    const points = data.map((v, i) => {
      const x = (i / 255) * W;
      const y = H - PAD - (v / maxVal) * (H - 2 * PAD);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    // filled area
    const areaPath = `M 0,${H-PAD} L ` + points.join(' L ') + ` L ${W},${H-PAD} Z`;
    svgContent += `<path d="${areaPath}" fill="${colour}" opacity="0.12"/>`;
    // line
    const linePath = 'M ' + points.join(' L ');
    svgContent += `<path d="${linePath}" fill="none" stroke="${colour}" stroke-width="1.2" opacity="0.85"/>`;
  });

  svg.innerHTML = svgContent;
}


// ════════════════════════════════════════════
// VIZ TABS
// ════════════════════════════════════════════
$$('.viz-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    $$('.viz-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    $$('.viz-content').forEach(c => c.classList.remove('active'));
    $(`.viz-content[data-content="${target}"]`).classList.add('active');
  });
});


// ════════════════════════════════════════════
// NUMBER ANIMATION
// ════════════════════════════════════════════
function animateNumber(el, from, to, duration = 1000) {
  const start = performance.now();
  const ease = (t) => 1 - Math.pow(1 - t, 3);  // easeOutCubic
  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const value = from + (to - from) * ease(progress);
    el.textContent = Math.round(value);
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = Math.round(to);
  }
  requestAnimationFrame(tick);
}


// ════════════════════════════════════════════
// API HEALTH PING
// ════════════════════════════════════════════
fetch('/api/health').then(r => {
  if (!r.ok) throw new Error('offline');
}).catch(() => {
  const status = $('#apiStatus');
  status.innerHTML = '<span class="pulse" style="background:#ff3d5a;box-shadow:0 0 8px #ff3d5a"></span> SYSTEM OFFLINE';
  status.style.color = '#ff3d5a';
  status.style.borderColor = 'rgba(255,61,90,0.25)';
  status.style.background = 'rgba(255,61,90,0.05)';
});
