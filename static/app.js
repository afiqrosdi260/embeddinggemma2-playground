/* EmbedLab: intentionally transparent about demonstration versus real inference. */
(() => {
  'use strict';
  const examples = [
    ['What are embeddings?', 'An embedding is a numerical representation of the meaning of an input. Similar content often maps to nearby vectors. Embeddings support semantic search, recommendation, clustering, and retrieval for RAG systems.'],
    ['How retrieval-augmented generation works', 'RAG retrieves relevant passages from a document collection, passes the evidence to a generative language model, and asks it to answer with citations. The embedding model performs retrieval, not the generation of the answer.'],
    ['Vector search versus keyword search', 'Keyword search matches exact terms. Vector search compares the semantic meaning of a question and indexed content. A query about automobile maintenance may retrieve a passage about taking care of a car even when the wording differs.'],
    ['EmbeddingGemma 2 model', 'EmbeddingGemma 2 is a 740-million-parameter multimodal embedding model from Google DeepMind. It maps text, code, images, audio, and video into a shared 768-dimensional vector space. Smaller Matryoshka dimensions include 512, 256, and 128.'],
    ['Cross-modal retrieval', 'You can embed a photograph of a drilling rig and retrieve it with a text search such as offshore drilling equipment. Images and text are compared in the same embedding space when indexed by the same multimodal model.'],
    ['API capabilities and semantic discovery', 'An API capability registry describes verified resources, endpoints, authentication, field schemas, and supported operations. Embedding search can help locate relevant capability descriptions, but it must not replace deterministic API validation.'],
    ['Geospatial well data', 'An oil well dataset can include latitude, longitude, field name, depth, operator and status. Spatial analysis requires actual coordinates and GIS functions. Semantic embeddings alone cannot calculate geographic distance or perform geospatial joins.'],
    ['Bahasa Melayu and multilingual search', 'Carian semantik boleh mencari dokumen berdasarkan maksud ayat dan bukan hanya padanan perkataan. Model embedding pelbagai bahasa boleh membantu pencarian silang bahasa Melayu dan bahasa Inggeris.'],
    ['Precision and dimension trade-offs', 'EmbeddingGemma 2 supports full 768-dimensional vectors and truncated 512, 256, and 128-dimensional vectors. After truncation, vectors must be L2-normalized again. Smaller vectors save memory and storage but may reduce retrieval quality.']
  ];
  const $ = (id) => document.getElementById(id);
  const state = {
    backend: false, modelState: 'unloaded', dimension: 768, docs: examples.map(([title, content], i) => ({
      id: `demo-${i}`, title, kind: 'text', preview: content, content, chunks: 1
    })), latest: null, busy: false
  };
  let poller = null;
  let toastTimer = null;

  const node = (tag, cls, content) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (content !== undefined) e.textContent = content;
    return e;
  };
  function toast(message) {
    const t = $('toast'); t.textContent = message; t.classList.add('visible');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('visible'), 3400);
  }
  async function api(path, options={}) {
    const response = await fetch(path, { ...options, headers: options.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' } });
    let result;
    try { result = await response.json(); } catch { result = {}; }
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : `Server error (${response.status})`);
    return result;
  }
  const isReal = () => state.backend && state.modelState === 'ready';

  function renderStatus() {
    const real = isReal(); const loading = ['loading','indexing'].includes(state.modelState);
    const label = real ? 'Real inference' : state.backend ? (loading ? 'Initializing model…' : 'Demo · lexical search') : 'Demo · browser only';
    $('side-mode').replaceChildren(node('span', `dot ${real ? '' : 'amber'}`), node('span', '', label));
    $('top-mode').replaceChildren(node('span', `dot ${real ? '' : 'amber'}`), document.createTextNode(real ? ' Real model' : ' Demo mode'));
    $('top-mode').classList.toggle('real', real);
    $('chat-disclaimer').textContent = real
      ? 'Real cosine similarity from EmbeddingGemma 2. Retrieved passages are not generated answers.'
      : 'Demo mode: results use basic word overlap, NOT actual EmbeddingGemma embeddings.';
    const btn = $('load-model');
    if (!state.backend) {btn.textContent = 'Run local backend ↗'; btn.disabled = true; btn.title = 'Start: uvicorn server:app --reload';}
    else if (real) {btn.textContent = 'Model ready ✓'; btn.disabled = true;}
    else if (loading) {btn.textContent = state.modelState === 'loading' ? 'Downloading / loading…' : 'Indexing sources…'; btn.disabled = true;}
    else {btn.textContent = state.modelState === 'error' ? 'Retry load ↗' : 'Load real model ↗'; btn.disabled = false;}
    $('document-count').textContent = String(state.docs.length);
    $('library-count').textContent = `${state.docs.length} ITEMS`;
  }
  async function refreshDocuments() {
    if (!state.backend) return;
    const data = await api('/api/documents'); state.docs = data.documents;
    renderDocs(); renderStatus();
  }
  async function checkBackend() {
    try {
      const response = await fetch('/api/status', {signal: AbortSignal.timeout(2500)});
      if (!response.ok) throw Error('API unavailable');
      const result = await response.json();
      state.backend = true; state.modelState = result.state;
      await refreshDocuments();
      if (state.modelState === 'error' && result.error) toast(`Model error: ${result.error.slice(0, 110)}`);
      return result;
    } catch {
      state.backend = false; state.modelState = 'unloaded'; renderStatus();
      return null;
    }
  }
  async function loadModel() {
    if (!state.backend || isReal()) return;
    state.modelState = 'loading'; renderStatus();
    if (!poller) poller = setInterval(async () => {
      const result = await checkBackend();
      if (!result || !['loading', 'indexing'].includes(result.state)) {
        clearInterval(poller); poller = null;
      }
    }, 2600);
    try {
      await api('/api/model/load', {method:'POST'});
      state.modelState = 'ready';
      await refreshDocuments();
      toast('EmbeddingGemma 2 loaded. Real embeddings enabled.');
    } catch (err) {
      toast(`Unable to load model: ${err.message}`);
      await checkBackend();
    } finally {
      if (poller) {clearInterval(poller); poller = null;}
      renderStatus();
    }
  }
  function changeView(view) {
    document.querySelectorAll('.view').forEach(e => e.classList.toggle('active', e.id === `view-${view}`));
    document.querySelectorAll('.nav-item').forEach(e => e.classList.toggle('active', e.dataset.view === view));
    const titles = {playground:'Playground', similarity:'Similarity lab', experiments:'Experiments', library:'Knowledge base', guide:'Model guide'};
    $('view-breadcrumb').textContent = titles[view] || 'Playground';
    if (view === 'experiments') refreshExperiments();
    window.scrollTo({top:0, behavior:'smooth'});
  }
  // Demonstration scoring is deliberately lexical and NEVER presented as a vector score.
  const stop = new Set('the a an and or to of is are with what how do does about can you me your in on by for it we this that from as be which between some into my at'.split(' '));
  const aliases = {automobile:'car', vehicle:'car', cars:'car', repairs:'maintenance', repairing:'maintenance', service:'maintenance', servicing:'maintenance', retrieval:'search', searching:'search', finding:'search', find:'search', locate:'search', queries:'query', questions:'query', documents:'document', data:'document', api:'endpoint', apis:'endpoint', gis:'geospatial', spatial:'geospatial', well:'geospatial', wells:'geospatial', vector:'embedding', vectors:'embedding', embeddings:'embedding', embedd:'embedding', rag:'generation', ai:'model', ml:'model', model:'model', meaning:'semantic', similar:'similarity', maksud:'semantik', carian:'semantik', mencari:'semantik'};
  function tokens(s) {return new Set((s.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(w => w.length > 1 && !stop.has(w)).map(w => aliases[w] || w));}
  function lexical(a,b) {
    const aa=tokens(a), bb=tokens(b); if (!aa.size || !bb.size) return 0;
    const common = [...aa].filter(w => bb.has(w)).length;
    return common / Math.sqrt(aa.size * bb.size);
  }
  function demoSearch(query) {
    const k=Number($('top-k').value);
    const results = state.docs.map(d => ({id:d.id,title:d.title,kind:d.kind,passage:d.content || d.preview,score:Number(lexical(query, `${d.title} ${d.content || d.preview}`).toFixed(4)), chunk:1})).sort((a,b)=>b.score-a.score).slice(0,k);
    return {mode:'demo', dimension:state.dimension, results, indexed_chunks:state.docs.length, vector_preview:null};
  }
  function renderInspector(data) {
    const results=data.results || [];
    $('result-count').textContent = `${results.length} SHOWN`;
    $('inspector-state').classList.add('found');
    $('inspector-state').innerHTML = '<i></i> RANKED';
    const r=$('inspector-results'); r.replaceChildren();
    for (const item of results) {
      const wrap=node('div','result-item'); const top=node('div','result-top');
      top.append(node('span','result-name',item.title),node('span','result-score',item.score.toFixed(3)));
      const track=node('div','score-track'); const fill=node('div');
      fill.style.width = `${Math.max(0, Math.min(100,item.score*100))}%`; track.append(fill);
      wrap.append(top,track); r.append(wrap);
    }
    $('vector-bars').replaceChildren();
    if (data.mode==='real' && Array.isArray(data.vector_preview)) {
      $('vector-label').textContent = `First ${data.vector_preview.length} / ${data.dimension} dimensions`;
      for (const x of data.vector_preview) {
        const bar=node('div',`vector-bar ${x<0 ? 'neg' : ''}`);
        bar.style.height=`${Math.min(100,Math.max(6,Math.abs(x)*420))}%`;
        bar.title=Number(x).toFixed(4);
        $('vector-bars').append(bar);
      }
    } else {
      $('vector-label').textContent = 'Not available in demo';
      $('vector-bars').append(node('span','vector-placeholder','No embedding vectors are calculated in lexical demo mode.'));
    }
    $('score-hint').textContent = data.mode === 'real'
      ? 'Scores are raw cosine similarities, not probability estimates or verified facts.'
      : 'These are approximate keyword-overlap scores. They are NOT EmbeddingGemma cosine similarities.';
  }
  function mediaNode(item) {
    const kind=item.kind; const element=node(kind==='image'?'img':kind==='video'?'video':'audio');
    element.src=item.media_url;
    if(kind==='image'){element.alt=item.title;element.loading='lazy';}
    else {element.controls=true;element.preload='none';if(kind==='video') element.playsInline=true;}
    return element;
  }

  const experimentIcons={text:'▤',code:'⌘',image:'▧',pdf:'▣',audio:'♫',voice:'♬',video:'▶'};
  let experiments=[]; let chosenExperiment='text';let experimentBusy=false;
  function experimentStatus(message){$('experiment-status').textContent=message;}
  async function refreshExperiments(){
    if(!state.backend){experimentStatus('Start the FastAPI backend to use built-in experiments.');return;}
    try { const response=await api('/api/experiments'); experiments=response.experiments;
      const cards=$('experiments-cards');cards.replaceChildren();
      for(const ex of experiments){
        const b=node('button','experiment-card'+(ex.id===chosenExperiment?' selected':''));b.type='button';
        b.append(node('span','experiment-type', `${experimentIcons[ex.id]||'✳'}  ${ex.id.toUpperCase()} · ${ex.sample_count} SAMPLES`),node('strong','',ex.title),node('small','',ex.description));
        b.addEventListener('click',()=>{chosenExperiment=ex.id;renderExperiment();});cards.append(b);
      }
      renderExperiment();
    }catch(err){experimentStatus(`Experiment catalog unavailable: ${err.message}`);}
  }
  function renderExperiment(){
    const ex=experiments.find(e=>e.id===chosenExperiment);if(!ex)return;
    document.querySelectorAll('.experiment-card').forEach(b=>b.classList.toggle('selected',b.querySelector('.experiment-type')?.textContent.includes(ex.id.toUpperCase()+' ·')));
    $('experiment-title').textContent=ex.title; $('experiment-description').textContent=ex.description;
    $('experiment-query').textContent=ex.query;$('experiment-expected').textContent=ex.expected_key;
    $('experiment-lesson').textContent=ex.lesson;
    $('experiment-loaded').textContent=ex.loaded?'SAMPLES READY':'NOT LOADED';
    $('experiment-run').disabled=!ex.loaded || !isReal();
    $('experiment-compare').disabled=!ex.loaded || !isReal();
    $('experiment-results').replaceChildren();$('experiment-dimensions').replaceChildren();
    experimentStatus(ex.loaded?isReal()?'Samples ready. Run the real-model experiment.':'Samples ready. Click Load real model in the header before measuring embeddings.':'Step 1: Load the isolated sample collection. This does not require the model yet.');
  }
  async function experimentAction(action){
    if(experimentBusy)return;
    const ex=experiments.find(e=>e.id===chosenExperiment);if(!ex)return;
    experimentBusy=true;for(const id of ['experiment-load','experiment-run','experiment-compare'])$(id).disabled=true;
    experimentStatus(action==='load'?'Generating and indexing samples…':action==='run'?'Calculating real embeddings and ranking sources…':'Comparing rankings at 128 / 256 / 512 / 768 dimensions…');
    try{
      if(action==='load'){
        await api(`/api/experiments/${ex.id}/load`,{method:'POST'});
        await refreshDocuments();await refreshExperiments();
        experimentStatus('Sample collection loaded. In real mode, run the query. Media sample generation depends on local encoders/codecs.');
      } else if(action==='run'){
        const data=await api(`/api/experiments/${ex.id}/run`,{method:'POST',body:JSON.stringify({dimension:Number($('experiment-dim').value),top_k:Math.min(10,ex.sample_count)})});
        const area=$('experiment-results');area.replaceChildren();
        const metrics=node('div','experiment-metrics');
        for(const [label,value] of [['Expected rank',data.rank??'Not found'],['Hit@1',data.hit_at_1?'Yes':'No'],['Hit@3',data.hit_at_3?'Yes':'No'],['Reciprocal rank',data.reciprocal_rank]]){
          const el=node('span','experiment-metric');el.append(node('b','',`${label}: `),document.createTextNode(String(value)));metrics.append(el);
        }area.append(metrics);
        for (const [i,hit] of data.results.entries()) {
          const box=node('article','experiment-hit'+(i+1===data.rank?' expected':''));
          const head=node('header');head.append(node('strong','',`#${i+1} · ${hit.title}`),node('span','',`Cosine ${hit.score.toFixed(3)}`));box.append(head,node('p','',hit.passage||'Media input'));
          if(hit.media_url)box.append(mediaNode(hit));area.append(box);
        }
        experimentStatus(`Real model · ${data.dimension}d · task ${data.task} · ${data.corpus_size} isolated sources. These are measured retrieval results, not generated answers.`);
      }else{
        const data=await api(`/api/experiments/${ex.id}/dimensions`);
        const area=$('experiment-dimensions');area.replaceChildren(node('strong','', 'Embedding dimension comparison'));
        for(const line of data.dimensions)area.append(node('div','experiment-hit',`${line.dimension}d · expected rank ${line.rank??'—'} · Hit@1 ${line.hit_at_1?'Yes':'No'} · top: ${line.top_title??'none'}`));
        experimentStatus('Dimensions compared using one query embedding, truncated and re-normalized at each selected width.');
      }
    }catch(err){experimentStatus(`${action} failed: ${err.message}`);}
    finally{experimentBusy=false;const current=experiments.find(e=>e.id===chosenExperiment);$('experiment-load').disabled=false;$('experiment-run').disabled=!current?.loaded||!isReal();$('experiment-compare').disabled=!current?.loaded||!isReal();}
  }
  function renderReply(query, data) {
    const wrap=node('div','message bot');
    const top=node('div','response-top');
    const avatar=node('div','bot-avatar','✳');
    const badge=node('span', `response-tag ${data.mode === 'real' ? 'real' : ''}`, data.mode === 'real' ? 'REAL EMBEDDINGS' : 'LEXICAL DEMO');
    top.append(avatar,node('strong','', 'EmbedLab'),badge); wrap.append(top);
    const matched=data.results.filter(i=>i.score>0);
    let description;
    if (!matched.length) description='No positive matches found in the current sources. Add more relevant material and try again.';
    else description=`I ranked ${data.results.length} nearby source passages for your question. Here ${matched.length===1?'is the highest matching source':'are the closest matches'} — these are retrieved excerpts, not an LLM-generated answer.`;
    wrap.append(node('div','response-intro',description));
    for (const item of matched.slice(0,3)) {
      const card=node('div','source-card'); const h=node('div','source-card-head');
      h.append(node('span','source-card-title',item.title),node('span','source-score',item.score.toFixed(3)));
      card.append(h,node('p','',item.passage || 'Media match'));
      if (item.media_url) {card.append(mediaNode(item));}
      wrap.append(card);
    }
    wrap.append(node('div','response-note',data.mode==='real' ? 'Source ranking: normalized vector dot product · No generative LLM' : 'Demonstration ranking: word overlap · NOT actual embedding inference'));
    $('messages').append(wrap); $('messages').scrollTop=$('messages').scrollHeight;
  }
  async function chat(query) {
    query=(query||'').trim(); if (!query || state.busy) return;
    const welcome=$('messages').querySelector('.welcome-message'); if (welcome) welcome.remove();
    const usr=node('div','message user'); usr.append(node('div','user-bubble',query)); $('messages').append(usr);
    $('messages').scrollTop=$('messages').scrollHeight;
    state.busy=true; $('send-button').disabled=true; $('chat-text').value='';
    try {
      const data=isReal() ? await api('/api/search',{method:'POST',body:JSON.stringify({query,dimension:state.dimension,top_k:Number($('top-k').value)})}) : demoSearch(query);
      state.latest=data; renderReply(query,data); renderInspector(data);
    } catch(e) { toast(e.message); $('messages').append(node('div','response-intro',`Search error: ${e.message}`)); }
    finally {state.busy=false; $('send-button').disabled=false; $('chat-text').focus();}
  }
  function renderDocs() {
    $('document-list').replaceChildren();
    for (const d of state.docs) {
      const row=node('div','doc-item');
      const typ=node('div',`doc-type ${d.kind!=='text'?'media':''}`, ({text:'▤',image:'▧',audio:'♫',video:'▶'})[d.kind]||'▤');
      const info=node('div','doc-detail'); info.append(node('strong','',d.title),node('span','',`${d.kind.toUpperCase()} · ${d.chunks || 1} chunk(s) · ${d.preview||''}`));
      const remove=node('button','doc-delete','×'); remove.type='button';remove.title=`Delete ${d.title}`;
      remove.addEventListener('click',()=>removeDoc(d.id));
      row.append(typ,info,remove); $('document-list').append(row);
    }
    renderStatus();
  }
  async function addText() {
    const title=$('doc-title').value.trim(), content=$('doc-content').value.trim();
    if (!title||!content) {toast('Enter a title and text first.');return;}
    try {
      if(state.backend) await api('/api/documents/text',{method:'POST',body:JSON.stringify({title,content})});
      else state.docs.unshift({id:`local-${Date.now()}`,title,content,preview:content.slice(0,175),kind:'text',chunks:1});
      if(state.backend) await refreshDocuments(); else renderDocs();
      $('doc-title').value='';$('doc-content').value='';toast('Text source added to the knowledge base.');
      $('library-message').textContent='Source added. Try a semantic query in Playground.';
    } catch (err) {toast(err.message);}
  }
  async function uploadFile(file) {
    if(!file) return;
    if(!state.backend) {
      if (/\.(txt|md)$/i.test(file.name)) {
        const content=await file.text();$('doc-title').value=file.name;$('doc-content').value=content;toast('Loaded text into editor. Click Add text source.');
      } else toast('Start the local backend for PDFs, images, audio, and video uploads.');
      return;
    }
    if (file.size > 15*1024*1024) {toast('File exceeds the 15 MB prototype limit.'); return;}
    const fd=new FormData();fd.append('file',file);
    $('library-message').textContent=`Processing ${file.name}…`;
    try{await api('/api/documents/upload',{method:'POST',body:fd});await refreshDocuments();toast(`${file.name} added.`);$('library-message').textContent='File added. Search for related content in Playground.';}
    catch(e){toast(e.message);$('library-message').textContent=`Upload failed: ${e.message}`;}
  }
  async function removeDoc(id) {
    try{
      if(state.backend) {await api(`/api/documents/${encodeURIComponent(id)}`,{method:'DELETE'});await refreshDocuments();}
      else {state.docs=state.docs.filter(d=>d.id!==id);renderDocs();}
      toast('Source removed.');
    } catch(e){toast(e.message);}
  }
  async function compare() {
    const a=$('similarity-a').value.trim(),b=$('similarity-b').value.trim(),dimension=Number($('similarity-dim').value);
    if(!a||!b){toast('Enter both texts first.');return;}
    $('compare-button').disabled=true;
    try {
      const result=isReal() ? await api('/api/similarity',{method:'POST',body:JSON.stringify({a,b,dimension})}) : {mode:'demo',score:lexical(a,b),dimension};
      $('sim-score').textContent=result.score.toFixed(3);
      $('sim-score-label').textContent=result.mode==='real'?'Cosine similarity':'Lexical overlap (demo)';
      $('sim-progress-fill').style.width=`${Math.max(0,Math.min(100,result.score*100))}%`;
      $('sim-status').textContent=result.mode==='real'
       ? `Actual ${dimension}d normalized EmbeddingGemma 2 embeddings. Negative values are possible.`
       : 'This is only shared-word overlap. Load the local model to compare semantic meaning.';
    } catch(e){toast(e.message)} finally{$('compare-button').disabled=false;}
  }
  function init() {
    document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>changeView(b.dataset.view)));
    document.querySelectorAll('.suggestion').forEach(b=>b.addEventListener('click',()=>chat(b.dataset.prompt)));
    $('chat-form').addEventListener('submit',e=>{e.preventDefault();chat($('chat-text').value)});
    $('chat-text').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();chat($('chat-text').value)}});
    $('clear-chat').addEventListener('click',()=>{
      $('messages').replaceChildren();
      const first=node('div','welcome-message');first.append(node('div','welcome-icon','✳'),node('h2','','Start a new search'),node('p','','Type a question below to retrieve the most relevant content from your knowledge base.'));$('messages').append(first);
      $('inspector-state').classList.remove('found');$('inspector-state').innerHTML='<i></i> IDLE';
      $('inspector-results').replaceChildren(node('div','inspector-empty','Search something to see ranked results.'));
      $('vector-label').textContent='Not available';$('vector-bars').replaceChildren(node('span','vector-placeholder','Real inference produces numeric embedding values here.'));
      $('result-count').textContent='—';
    });
    document.querySelectorAll('[data-dim]').forEach(b=>b.addEventListener('click',()=>{
      state.dimension=Number(b.dataset.dim);
      document.querySelectorAll('[data-dim]').forEach(x=>x.classList.toggle('selected',x===b));
      if(state.latest) toast('Dimension changed. Run the query again to update scores.');
    }));
    $('load-model').addEventListener('click',loadModel);
    $('add-source-shortcut').addEventListener('click',()=>changeView('library'));
    $('learn-more').addEventListener('click',()=>changeView('guide'));
    $('add-text').addEventListener('click',addText);
    $('file-upload').addEventListener('change',e=>{const file=e.target.files?.[0];uploadFile(file);e.target.value='';});
    $('compare-button').addEventListener('click',compare);
    $('experiment-load').addEventListener('click',()=>experimentAction('load'));
    $('experiment-run').addEventListener('click',()=>experimentAction('run'));
    $('experiment-compare').addEventListener('click',()=>experimentAction('compare')); 
    renderDocs();checkBackend().then(()=>refreshExperiments());
  }
  init();
})();