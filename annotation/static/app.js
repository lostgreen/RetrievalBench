(function(){
  function $(sel, root){ return (root||document).querySelector(sel); }
  function $all(sel, root){ return Array.from((root||document).querySelectorAll(sel)); }

  function onShotPeek(e){
    const card = e.target.closest('.shot');
    const details = $('.details', card);
    const framesDiv = $('.frames', details);
    details.classList.toggle('hidden');
    if (!details.classList.contains('hidden') && framesDiv.childElementCount === 0){
      const start = parseInt(card.dataset.start, 10);
      const end = parseInt(card.dataset.end, 10);
      const url = window.ANNO.frames_api(start, end) + '?k=6';
      fetch(url).then(r=>r.json()).then(data => {
        framesDiv.innerHTML = '';
        const urls = data.urls || [];
        const idxs = data.frames || [];
        urls.forEach((u, i) => {
          const wrap = document.createElement('div');
          wrap.className = 'crop-container';
          const img = document.createElement('img');
          img.src = u;
          img.setAttribute('data-frame', (idxs[i] != null ? idxs[i] : ''));
          wrap.appendChild(img);
          framesDiv.appendChild(wrap);
        });
      });
    }
  }

  function collectGT(){
    let selected = $all('.shot')
      .filter(card => $('.select-shot', card))
      .filter(card => $('.select-shot', card).checked)
      .map(card => parseInt(card.dataset.shotId || card.getAttribute('data-shot-id'), 10));
    if ((selected.length === 0) && window.ANNO && Array.isArray(window.ANNO.selected_shots)){
      selected = window.ANNO.selected_shots;
    }
    const ansEl = $('input[name="answer"]:checked');
    const answer = ansEl ? ansEl.value : null;
    const notes = $('#notes').value || '';
    // basic action log: peeked shots when frames loaded
    const actions = [];
    $all('.shot').forEach(card => {
      const details = $('.details', card);
      const framesDiv = $('.frames', card);
      const visible = details ? !details.classList.contains('hidden') : true;
      if (framesDiv && visible && framesDiv.childElementCount > 0){
        actions.push({act:'peek_shot', args:{shot_id: parseInt(card.getAttribute('data-shot-id'),10)}});
      }
    });
    // proposed crops from dataset
    const proposed_crops = [];
    $all('.shot').forEach(card => {
      $all('.frames .crop-container', card).forEach(container => {
        const rect = $('.crop-rect', container);
        if (rect){
          const img = $('img', container);
          const fw = img.clientWidth, fh = img.clientHeight;
          const x1 = parseFloat(rect.style.left)/fw;
          const y1 = parseFloat(rect.style.top)/fh;
          const w = parseFloat(rect.style.width), h = parseFloat(rect.style.height);
          const x2 = (parseFloat(rect.style.left)+w)/fw;
          const y2 = (parseFloat(rect.style.top)+h)/fh;
          const frame = parseInt(img.getAttribute('data-frame')||'0', 10);
          const shot_id = parseInt(card.getAttribute('data-shot-id'),10);
          proposed_crops.push({shot_id, frame, bbox:[x1,y1,x2,y2]});
          actions.push({act:'request_hd_crop', args:{shot_id, frame, bbox:[x1,y1,x2,y2]}});
        }
      });
      // also add manual crops
      $all('.proposed .badge', card).forEach(b => {
        const frame = parseInt(b.getAttribute('data-frame')||'0',10);
        const bbox = (b.getAttribute('data-bbox')||'').split(',').map(parseFloat);
        const shot_id = parseInt(card.getAttribute('data-shot-id'),10);
        proposed_crops.push({shot_id, frame, bbox: bbox});
        actions.push({act:'request_hd_crop', args:{shot_id, frame, bbox: bbox}});
      });
    });
    return { selected_shots: selected, answer: answer, notes: notes, actions: actions, proposed_crops: proposed_crops };
  }

  function saveGT(){
    const payload = collectGT();
    fetch(window.ANNO.save_url, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    }).then(r=>r.json()).then(data => {
      alert('Saved to: ' + (data.path || 'OK'));
    }).catch(err => {
      alert('Save failed: ' + err);
    });
  }

  document.addEventListener('click', function(e){
    if (e.target.matches('.peek')){
      onShotPeek(e);
    }
    if (e.target.closest && e.target.closest('#saveBtn')){
      saveGT();
    }
    if (e.target.matches('#stage2Btn')){
      const maxFrames = Math.max(1, parseInt($('#stage2MaxFrames').value||'128', 10));
      const selectedIds = $all('.shot')
        .filter(card => $('.select-shot', card))
        .filter(card => $('.select-shot', card).checked)
        .map(card => parseInt(card.getAttribute('data-shot-id'),10));
      if (selectedIds.length === 0){
        alert('Select at least one shot for Stage 2 simulation.');
        return;
      }
      const qs = new URLSearchParams({ shots: selectedIds.join(','), max: String(maxFrames) });
      const base = window.ANNO && window.ANNO.video_id ? ('/simulate/' + window.ANNO.video_id) : window.location.pathname.replace('/annotate/', '/simulate/');
      window.location.href = base + '?' + qs.toString();
    }
    if (e.target.matches('.add-crop')){
      const card = e.target.closest('.shot');
      const frameEl = $('.crop-frame', card);
      const bboxEl = $('.crop-bbox', card);
      const frame = parseInt(frameEl.value||card.getAttribute('data-rep'),10);
      const bbox = (bboxEl.value||'0.25,0.25,0.75,0.75').split(',').map(parseFloat);
      const proposedDiv = $('.proposed', card);
      const tag = document.createElement('span');
      tag.className = 'badge';
      tag.textContent = `crop f=${frame} bbox=[${bbox.map(x=>x.toFixed(2)).join(',')}]`;
      tag.setAttribute('data-frame', String(frame));
      tag.setAttribute('data-bbox', bbox.join(','));
      proposedDiv.appendChild(tag);
    }
  });

  // interactive crop via drag on frame thumbnail
  let dragState = null;
  document.addEventListener('mousedown', function(e){
    const img = e.target.closest('.frames img');
    if (!img) return;
    const container = img.parentElement;
    const rect = container.getBoundingClientRect();
    dragState = {
      container: container,
      startX: e.clientX - rect.left,
      startY: e.clientY - rect.top,
      rectEl: null,
    };
    e.preventDefault();
  });
  document.addEventListener('mousemove', function(e){
    if (!dragState) return;
    const {container, startX, startY} = dragState;
    const rect = container.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const y = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
    const left = Math.min(startX, x);
    const top = Math.min(startY, y);
    const width = Math.abs(x - startX);
    const height = Math.abs(y - startY);
    let r = dragState.rectEl;
    if (!r){
      r = document.createElement('div');
      r.className = 'crop-rect';
      container.appendChild(r);
      dragState.rectEl = r;
    }
    r.style.left = left + 'px';
    r.style.top = top + 'px';
    r.style.width = width + 'px';
    r.style.height = height + 'px';
  });
  document.addEventListener('mouseup', function(e){
    if (!dragState) return;
    // finalize selection; keep rectEl in DOM for save
    dragState = null;
  });
})();
