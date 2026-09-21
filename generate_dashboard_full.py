#!/usr/bin/env python3
"""
Generate a self-contained, offline dashboard for the 3-class comparison:
LLM prompt-based classifier vs NRC-lexicon sentiment+emotion, on a balanced
stratified sample pulled from the whole file (rating 4-5 POSITIVE / 3 NEUTRAL /
1-2 NEGATIVE).

Reads the enriched per-review file (predictions_nrc.jsonl) and bakes everything
into ONE HTML file. No CDN / server / network required.
"""
import json
import os
from collections import Counter

IN_FILE = "predictions_nrc.jsonl"
OUT_FILE = "dashboard.html"

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

# ---- Load enriched records -----------------------------------------------------
rows = []
with open(IN_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
n = len(rows)

# ---- Confusion (truth x LLM pred), per-class P/R/F1, macro-F1 ------------------
cm = Counter((r["label"], r["pred"]) for r in rows)
per_class = {}
macro_sum = 0.0
for c in CLASSES:
    tp = cm[(c, c)]
    col = sum(cm[(a, c)] for a in CLASSES)
    row = sum(cm[(c, p)] for p in CLASSES)
    prec = tp / col if col else 0.0
    rec = tp / row if row else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    per_class[c] = {"count": row, "precision": round(prec * 100, 1),
                    "recall": round(rec * 100, 1), "f1": round(f1 * 100, 1)}
    macro_sum += f1
llm_acc = round(100.0 * sum(cm[(c, c)] for c in CLASSES) / n, 1)
macro_f1 = round(100.0 * macro_sum / len(CLASSES), 1)

# ---- NRC vs ground truth (decided only) + LLM-vs-NRC agreement -----------------
decided_rows = [r for r in rows if r["nrc_sentiment"] in ("POSITIVE", "NEGATIVE")]
nrc_decided = len(decided_rows)
nrc_correct = sum(1 for r in decided_rows if r["nrc_sentiment"] == r["label"])
nrc_acc = round(100.0 * nrc_correct / nrc_decided, 1) if nrc_decided else 0.0

agree_all = sum(1 for r in rows if r["pred"] == r["nrc_sentiment"])
agree_decided = sum(1 for r in decided_rows if r["pred"] == r["nrc_sentiment"])
agree_pct_all = round(100.0 * agree_all / n, 1)
agree_pct_dec = round(100.0 * agree_decided / nrc_decided, 1) if nrc_decided else 0.0
neutral_count = n - nrc_decided

sent_dist = Counter(r["nrc_sentiment"] for r in rows)
pred_dist = Counter(r["pred"] for r in rows)
emo_raw = Counter(r["nrc_primary_emotion"] for r in rows)
emo_dist = {("none" if k is None else k): v for k, v in emo_raw.items()}

# ---- Per-review rows for the table ---------------------------------------------
table = []
for i, r in enumerate(rows):
    nr = r["nrc_sentiment"]
    decided = nr in ("POSITIVE", "NEGATIVE")
    table.append({
        "i": i + 1,
        "title": r.get("title") or "",
        "text": r.get("text") or "",
        "label": r["label"],
        "pred": r["pred"],
        "ok": r["pred"] == r["label"],
        "nrc": nr,
        "emo": r.get("nrc_primary_emotion"),
        "decided": decided,
        "agree": decided and nr == r["pred"],
    })
data_json = json.dumps(table)

baked = {
    "%%TITLE%%": "Gift Card Review Classifier — 3-class LLM vs NRC Dashboard",
    "%%N%%": str(n),
    "%%DATA%%": data_json,
    "%%CLASSES%%": json.dumps(CLASSES),
    "%%CM%%": json.dumps({a: {p: cm[(a, p)] for p in CLASSES} for a in CLASSES}),
    "%%EMOS%%": json.dumps(EMOTIONS),
    "%%STATS%%": json.dumps({
        "n": n,
        "llm_acc": llm_acc, "macro_f1": macro_f1, "per_class": per_class,
        "nrc_acc": nrc_acc, "nrc_decided": nrc_decided, "nrc_correct": nrc_correct,
        "agree_all": agree_all, "agree_pct_all": agree_pct_all,
        "agree_decided": agree_decided, "agree_pct_dec": agree_pct_dec,
        "neutral": neutral_count,
        "pred_dist": dict(pred_dist), "sent_dist": dict(sent_dist),
        "emo_dist": emo_dist,
    }),
}

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<style>
  :root{--green:#1a8f5c;--greenbg:#e7f6ee;--red:#c0433a;--redbg:#fbecea;--amber:#b98a00;--amberbg:#fdf3d7;
        --ink:#1c2333;--mut:#5c6472;--line:#e4e7ee;--bg:#f5f7fb;--card:#fff;}
  *{box-sizing:border-box}
  body{margin:0;padding:32px;background:var(--bg);color:var(--ink);
       font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:22px 0 8px}h3{font-size:14px;margin:14px 0 4px}
  .sub{color:var(--mut);font-size:13px;margin-bottom:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .kpi .num{font-size:26px;font-weight:650;letter-spacing:-.5px}
  .kpi .lbl{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:860px){.grid2{grid-template-columns:1fr}}
  table.cm{border-collapse:separate;border-spacing:8px;width:100%;table-layout:fixed}
  table.cm td{text-align:center;border-radius:10px;padding:12px 6px;overflow:hidden}
  .cm .lbl{color:var(--mut);font-size:11px;margin-top:4px}.cm .big{font-size:22px;font-weight:700}
  table.cm .cok{background:var(--greenbg)}table.cm .cbad{background:var(--redbg)}
  .labelfooter{text-align:center;color:var(--mut);font-size:12px;margin-top:10px}
  .classview .ant{background:#eef2ff}
  .cv-row{margin:16px 0}
  .cv-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
  .cv-head b{font-size:15px}.cv-head .cv-meta{color:var(--mut);font-size:12px}
  .seg{display:flex;height:38px;border-radius:9px;overflow:hidden;border:1px solid var(--line)}
  .seg .s{display:flex;align-items:center;justify-content:center;gap:6px;font-size:12px;color:#fff;white-space:nowrap;overflow:hidden}
  .seg .s.okseg{background:var(--green);min-width:70px}
  .seg .s.badseg-w{background:#8a93a6}.seg .s.badseg-m{background:#b8c2d4}
  .seg .s.badseg-b{background:#c08681}.seg .s.badseg-r{background:#c0433a}
  .seg .s .tick{font-weight:800}
  .cv-desc{font-size:13px;color:var(--mut);margin-top:5px}
  .cv-desc b.g{color:var(--green)}.cv-desc b.x{color:var(--red)}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 2px;font-size:12px;color:var(--mut)}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .dot{width:12px;height:12px;border-radius:3px;display:inline-block}
  .barwrap{margin:9px 0}.barrow{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);margin-bottom:3px}
  .bar{height:16px;border-radius:5px;background:#eef0f5;overflow:hidden}
  .bar>span{display:block;height:100%;background:var(--red)}.bar>span.bok{background:var(--green)}
  .pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;font-weight:600;white-space:nowrap}
  .p-pos{background:#e8f4fd;color:#1467a8}.p-neu{background:#f0ecdc;color:#8a7230}
  .p-neg{background:#fdeeee;color:#b03a2e}.ok-pill{background:var(--greenbg);color:var(--green)}
  .bad-pill{background:var(--redbg);color:var(--red)}.nu-pill{background:#f0f0f4;color:#6a7080}
  .ag-pill{background:#eef2ff;color:#3f51b5}.di-pill{background:#fdeeee;color:#b03a2e}
  table.rows{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
  table.rows th,table.rows td{text-align:left;padding:8px 9px;border-bottom:1px solid var(--line);vertical-align:top}
  table.rows th{position:sticky;top:0;background:#fbfcfe;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
  table.rows tr.ok{background:#fff}table.rows tr.bad{background:var(--redbg)}
  tr.neurow{background:var(--amberbg)}
  .title{font-weight:550;max-width:190px}.text{color:var(--mut);max-width:330px}
  tr.bad .title,tr.bad .text{color:#7d2a22}
  .filters button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 11px;cursor:pointer;
                  font-size:13px;color:var(--mut);margin-right:5px;margin-bottom:4px}
  .filters button.active{background:#1c2333;color:#fff;border-color:#1c2333}.filters .cnt{opacity:.6}
  .toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between}
  .searchbox{position:relative;flex:1 1 220px;max-width:350px}
  .searchbox input{width:100%;padding:8px 12px 8px 30px;border:1px solid var(--line);border-radius:8px;font-size:13px;color:var(--ink);background:#fff}
  .searchbox input:focus{outline:none;border-color:#1c2333}
  .searchbox .mag{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--mut);font-size:13px;pointer-events:none}
  .livecount{font-size:13px;color:var(--mut);white-space:nowrap}.livecount b{color:var(--ink)}
  select.emofilter{border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:13px;color:var(--ink);background:#fff}
  .memo{color:var(--mut);font-size:13px}.method{font-size:13px;color:var(--mut)}
  .foot{margin-top:28px;color:var(--mut);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
</style>
</head>
<body>
<h1>Gift Card Review Classifier — LLM vs NRC Lexicon (3-class)</h1>
<div class="sub">Same balanced sample, two methods. Labels: <b>rating 4&ndash;5 &rarr; POSITIVE</b>, <b>3 &rarr; NEUTRAL</b>, <b>1&ndash;2 &rarr; NEGATIVE</b>.
 Sample pulled as a balanced, stratified random draw from the whole file (fixed seed; &asymp;equal per class), not the first rows in order.
 Both methods use only title + body text; the rating only supplies ground truth.</div>

<div class="kpis">
  <div class="card kpi"><div class="num" style="color:var(--green)">%%STAT_LLMACC%%%</div><div class="lbl">LLM accuracy</div></div>
  <div class="card kpi"><div class="num" style="color:var(--green)">%%STAT_MACRO%%%</div><div class="lbl">LLM macro-F1</div></div>
  <div class="card kpi"><div class="num" style="color:var(--amber)">%%STAT_NRCACC%%%</div><div class="lbl">NRC accuracy</div></div>
  <div class="card kpi"><div class="num" style="color:var(--amber)">%%STAT_AGREE%%%</div><div class="lbl">LLM&thinsp;=&thinsp;NRC agree</div></div>
  <div class="card kpi"><div class="num">%%STAT_NEU%%</div><div class="lbl">NRC neutral/tie</div></div>
  <div class="card kpi"><div class="num">%%N%%</div><div class="lbl">Reviews</div></div>
</div>

<div class="grid2" style="margin-top:16px">
  <div class="card">
    <h2 style="margin-top:0">Confusion matrix &mdash; LLM vs ground truth</h2>
    <table class="cm" id="cm"></table>
    <div class="labelfooter">Columns = predicted · Rows = actual · green = correct · red = error</div>
    <h3>Per-class performance (LLM)</h3>
    <p class="method" id="perclass"></p>
  </div>
  <div class="card">
    <h2 style="margin-top:0">NRC primary emotion (highest summed score)</h2>
    <div class="bars" id="emobars"></div>
    <p class="method" style="margin-top:10px"><b>NRC output:</b> LLM predictions
      %%STAT_PRED%% · NRC sentiment %%STAT_NRC%%.</p>
  </div>
</div>

<div class="card classview" style="margin-top:16px">
  <h2 style="margin-top:0">Correct answer vs what the LLM predicted — by class</h2>
  <p class="method">For each true class, the bar shows how the model answered: the green slice is the correct call
     (matching the rating band), the other slices are where the model misplaced those reviews.</p>
  <div class="legend">
    <span><span class="dot" style="background:var(--green)"></span>predicted correctly (green)</span>
    <span><span class="dot" style="background:#c08681"></span>predicted POSITIVE</span>
    <span><span class="dot" style="background:#b8c2d4"></span>predicted NEUTRAL</span>
    <span><span class="dot" style="background:#c0433a"></span>predicted NEGATIVE</span>
  </div>
  <div id="classview"></div>
</div>

<div class="grid2" style="margin-top:16px">
  <div class="card">
    <h2 style="margin-top:0">Method agreement</h2>
    <p class="method" id="agreeline"></p>
    <h3>Ground-truth agreement</h3>
    <p class="method" id="gtruth"></p>
  </div>
  <div class="card">
    <h2 style="margin-top:0">NRC sentiment vs ground truth</h2>
    <div id="nrcgtr"></div>
    <p class="method" id="nrcgtdetail"></p>
  </div>
</div>

<div class="card" style="margin-top:16px">
  <h2 style="margin-top:0">Per-Review Detail <span class="memo">(%%N%% rows; red = LLM missed ground truth, amber = NRC neutral)</span></h2>
  <div class="filters">
    <div class="toolbar">
      <div class="filters" id="filters"></div>
      <div class="searchbox"><span class="mag">&#9906;</span>
        <input id="search" type="text" placeholder="Search title or text&hellip;" autocomplete="off"></div>
      <div class="toolbar">
        <select id="emofilter" class="emofilter"><option value="">All emotions</option></select>
        <div class="livecount">Showing <b id="shown">0</b> of <span id="total">0</span></div>
      </div>
    </div>
  </div>
  <div style="overflow:auto; max-height:620px; margin-top:10px">
    <table class="rows" id="rows">
      <thead><tr>
        <th>#</th><th>Title</th><th>Body (clipped)</th><th>Truth</th>
        <th>LLM</th><th>NRC</th><th>Primary emotion</th><th>LLM verdict</th><th>Methods</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<div class="foot">
  <b>Methodology</b> &mdash; <b>LLM:</b> reasoning model (<code>cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit</code>) told to answer
  exactly POSITIVE/NEUTRAL/NEGATIVE from title+text, weighting body over title on conflict. <b>NRC:</b> token lookup in the
  NRC Emotion Lexicon v0.92 &mdash; sentiment = sign of summed positive-vs-negative flags (tie/none &rarr; neutral);
  primary emotion = argmax of the 8 emotion flags. Balanced stratified sample: each class (rating &ge;4 / ==3 / &le;2) is
  bucketed, shuffled (fixed seed 42), &asymp;50 taken per class, re-shuffled. This page is fully self-contained.
</div>

<script>
const DATA=%%DATA%%;
const CLASSES=%%CLASSES%%;
const CM=%%CM%%;
const LN=%%STATS%%;
const EMOS=%%EMOS%%;

// ---------- 3x3 confusion matrix ----------
function initCM(){
  const head=CLASSES.map(c=>'<td class="lbl" style="font-weight:600">'+c+'</td>').join('');
  let h='<tr><td></td>'+head+'</tr>';
  let maxc=0;
  CLASSES.forEach(a=>CLASSES.forEach(p=>{ if(CM[a][p]>maxc)maxc=CM[a][p]; }));
  CLASSES.forEach(a=>{
    const rowtot=CLASSES.reduce((s,p)=>s+CM[a][p],0)||1;
    h+='<tr><td class="lbl" style="font-weight:600;white-space:nowrap">'+a+'</td>';
    CLASSES.forEach(p=>{
      const v=CM[a][p], ok=(a===p), intensity=(v/maxc);
      const a1 = ok ? `rgba(26,143,92,${0.12+0.5*intensity})` : `rgba(192,67,58,${0.08+0.5*intensity})`;
      h+='<td class="'+(ok?'cok':'cbad')+'" style="background:'+a1+';vertical-align:top">'
        +'<div class="big">'+v+'</div><div class="lbl">'+Math.round(100*v/rowtot)+'% of row</div></td>';
    });
    h+='</tr>';
  });
  document.getElementById('cm').innerHTML='<tbody>'+h+'</tbody>';
}

// ---------- emotion bars ----------
function initEmoBars(){
  let h='';
  EMOS.forEach(e=>{
    const cnt=LN.emo_dist[e]||0, w=Math.max(1,Math.round(100*cnt/LN.n));
    h+='<div class="barwrap"><div class="barrow"><span>'+e+'</span><span>'+cnt+'</span></div>'
       +'<div class="bar"><span class="bok" style="width:'+w+'%"></span></div></div>';
  });
  const none=LN.emo_dist['none']||0;
  h+='<div class="barwrap"><div class="barrow"><span>none (no lexicon hits)</span><span>'+none+'</span></div>'
     +'<div class="bar"><span class="bok" style="width:'+Math.max(1,Math.round(100*none/LN.n))+'%"></span></div></div>';
  document.getElementById('emobars').innerHTML=h;
}

// ---------- correct vs predicted, by class ----------
const WRONGCLS={POSITIVE:'badseg-b',NEUTRAL:'badseg-m',NEGATIVE:'badseg-r'};
const SHORT={POSITIVE:'POS',NEUTRAL:'NEU',NEGATIVE:'NEG'};
function initClassView(){
  const box=document.getElementById('classview'); let h='';
  CLASSES.forEach(c=>{
    const row=CLASSES.reduce((s,p)=>s+CM[c][p],0)||1;
    const cnt=LN.per_class[c]; const ok=CM[c][c];
    const order=[c].concat(CLASSES.filter(p=>p!==c));
    let segs='';
    order.forEach(p=>{
      const v=CM[c][p]; if(v<=0)return;
      const cls=(p===c)?'okseg':WRONGCLS[p];
      const mark=(p===c)?' <span class="tick">&#10003;</span>':'';
      segs+='<div class="s '+cls+'" style="flex-grow:'+v+'">'+v+' '+SHORT[p]+mark+'</div>';
    });
    // descriptive sentence
    const tot=row; const pct=Math.round(100*ok/tot);
    let who=CLASSES.filter(p=>p!==c&&CM[c][p]>0)
                .map(p=>'<b class="x">'+CM[c][p]+' called '+SHORT[p]+'</b>').join(', ');
    if(CLASSES.filter(p=>p!==c&&CM[c][p]>0).length===0) who='<b class="g">all of them correctly</b>';
    else if(CLASSES.filter(p=>p!==c&&CM[c][p]>0).length===1) who=who;
    else who=who.replace(/, ([^,]*)$/,' and $1');
    h+='<div class="cv-row">'
      +'<div class="cv-head"><b>'+c+'</b><span class="cv-meta">'+tot+' true reviews &middot; '
        +pct+'% correct &middot; F1 '+cnt.f1+'% &middot; recall '+cnt.recall+'%</span></div>'
      +'<div class="seg">'+segs+'</div>'
      +'<div class="cv-desc">Of '+tot+' true <b>'+c.toLowerCase()+'</b> reviews the model called '
        +'<b class="g">'+ok+' right ('+pct+'%)</b> &mdash; the rest: '+who+'.</div>'
      +'</div>';
  });
  box.innerHTML=h;
}

// ---------- text panels ----------
function initText(){
  let pc='';
  CLASSES.forEach(c=>{
    const p=LN.per_class[c];
    pc+='<span style="display:inline-block;margin-right:14px"><b>'+c+'</b> (n='+p.count+') &mdash; exact '+
        p.recall+'% &middot; prec '+p.precision+'% &middot; F1 '+p.f1+'%</span>';
  });
  document.getElementById('perclass').innerHTML=pc;

  const dec=LN.nrc_decided, aggD=LN.agree_decided;
  document.getElementById('agreeline').innerHTML =
    'All reviews: LLM and NRC agree on <b>'+LN.agree_all+'/'+LN.n+' ('+LN.agree_pct_all+'%)</b> when a strict '+
    'label match (incl. both-neutral) is required. Looking only at the <b>'+dec+'</b> reviews where NRC took a side, '+
    'they agree on '+aggD+' ('+LN.agree_pct_dec+'%). NRC left <b>'+LN.neutral+'</b> undecided (tie or no lexicon hits). '+
    'NRC is permissive-positive ('+LN.sent_dist.POSITIVE+' pos vs '+LN.sent_dist.NEGATIVE+' neg on a rating-balanced batch) '+
    'because neutral product words (card, gift, order, work) sit in its positive list.';

  document.getElementById('gtruth').innerHTML =
    'LLM matches the rating break on <b>'+LN.llm_acc+'%</b> of '+LN.n+' (macro-F1 '+LN.macro_f1+'%) &mdash; NEUTRAL (rating 3) '+
    'is the hardest to hit, as expected, since a 3-star review is usually mixed. NRC matches on '+LN.nrc_acc+'% of its '+
    LN.nrc_decided+' decided reviews.';

  document.getElementById('nrcgtdetail').innerHTML =
    'NRC got it right on <b>'+LN.nrc_acc+'%</b> of its decided reviews (vs '+LN.llm_acc+'% for the LLM) &mdash; the raw '+
    'word-count method is markedly weaker at recovering sentiment here. Its primary-emotion read is dominated by '+
    '<b>anticipation</b> ('+(LN.emo_dist.anticipation||0)+'), which tracks the gift/order/shipping vocabulary rather '+
    'than true sentiment.';
}

// ---------- NRC vs ground truth bars ----------
function initNrcGt(){
  let tp2=0,tn2=0,fp2=0,fn2=0;
  DATA.forEach(r=>{
    if(!r.decided)return;
    if(r.nrc===r.label){ r.nrc==='POSITIVE'?tp2++:tn2++; } else { r.nrc==='POSITIVE'?fp2++:fn2++; }
  });
  const rows=[['NRC matched rating',tp2+tn2],['NRC did not match',fp2+fn2]];
  const tot=rows[0][1]+rows[1][1]||1;
  let h='';
  rows.forEach(r=>{
    const w=Math.max(1,Math.round(100*r[1]/tot));
    h+='<div class="barwrap"><div class="barrow"><span>'+r[0]+'</span><span>'+r[1]+'</span></div>'
       +'<div class="bar"><span class="'+(r[0].indexOf('matched')===0?'bok':'')+'" style="width:'+w+'%"></span></div></div>';
  });
  document.getElementById('nrcgtr').innerHTML=h;
}

// ---------- table ----------
let filt='all',q='',emo='';
function matches(r){
  if(filt==='ok'&&!r.ok)return false;
  if(filt==='err'&&r.ok)return false;
  if(filt==='tPOS'&&r.label!=='POSITIVE')return false;
  if(filt==='tNEU'&&r.label!=='NEUTRAL')return false;
  if(filt==='tNEG'&&r.label!=='NEGATIVE')return false;
  if(filt==='agr'&&!r.agree)return false;
  if(filt==='dis'&&!(r.decided&&!r.agree))return false;   // NRC decided AND disagrees
  if(filt==='neu'&&r.decided)return false;
  if(emo && r.emo!==emo)return false;
  if(q){ const hay=(r.title+' '+r.text).toLowerCase(); if(!hay.includes(q))return false; }
  return true;
}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
const pillc={POSITIVE:'p-pos',NEUTRAL:'p-neu',NEGATIVE:'p-neg'};
function render(rows){
  const tb=document.getElementById('tbody'); tb.innerHTML='';
  rows.forEach(r=>{
    const tr=document.createElement('tr');
    tr.className = r.ok?'ok':'bad'; if(!r.decided)tr.className+=' neurow';
    const text=r.text.length>140?r.text.slice(0,140)+'…':(r.text||'(no body text)');
    const nrcCell=r.decided
      ?'<span class="pill '+pillc[r.nrc]+'">'+r.nrc+'</span>'
      :'<span class="pill nu-pill">neutral</span>';
    const met=r.decided
      ?(r.agree?'<span class="pill ag-pill">agree</span>':'<span class="pill di-pill">differ</span>')
      :'<span class="pill nu-pill">—</span>';
    const emoDisp=r.emo?r.emo:'<span style="color:var(--mut)">—</span>';
    tr.innerHTML='<td>'+r.i+'</td>'
      +'<td class="title">'+esc(r.title||'(no title)')+'</td>'
      +'<td class="text">'+esc(text)+'</td>'
      +'<td><span class="pill '+pillc[r.label]+'">'+r.label+'</span></td>'
      +'<td><span class="pill '+pillc[r.pred]+'">'+r.pred+'</span></td>'
      +'<td>'+nrcCell+'</td>'
      +'<td>'+emoDisp+'</td>'
      +'<td><span class="pill '+(r.ok?'ok-pill':'bad-pill')+'">'+(r.ok?'Correct':'WRONG')+'</span></td>'
      +'<td>'+met+'</td>';
    tb.appendChild(tr);
  });
  document.getElementById('shown').textContent=rows.length;
}
function applyFilter(){ render(DATA.filter(matches)); }

function initFilters(){
  const c={all:DATA.length,ok:0,err:0,tPOS:0,tNEU:0,tNEG:0,agr:0,dis:0,neu:0};
  DATA.forEach(r=>{
    if(r.ok)c.ok++; else c.err++;
    c['t'+r.label]++;
    if(r.decided&&r.agree)c.agr++;
    if(r.decided&&!r.agree)c.dis++;
    if(!r.decided)c.neu++;
  });
  const defs=[['all','All',c.all],['ok','Correct',c.ok],['err','Wrong',c.err],
    ['tPOS','POSITIVE',c.tPOS],['tNEU','NEUTRAL',c.tNEU],['tNEG','NEGATIVE',c.tNEG],
    ['dis','Differ',c.dis],['agr','Agree',c.agr],['neu','NRC neutral',c.neu]];
  const box=document.getElementById('filters');
  defs.forEach(d=>{
    const b=document.createElement('button');
    b.innerHTML=d[1]+' <span class="cnt">'+d[2]+'</span>';
    b.className=d[0]==='all'?'active':'';
    b.onclick=()=>{filt=d[0];box.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');applyFilter();};
    box.appendChild(b);
  });
  document.getElementById('total').textContent=DATA.length;
  document.getElementById('shown').textContent=DATA.length;
  const sel=document.getElementById('emofilter');
  EMOS.forEach(e=>{const o=document.createElement('option');o.value=e;o.textContent=e;sel.appendChild(o);});
  sel.addEventListener('change',()=>{emo=sel.value||'';applyFilter();});
  document.getElementById('search').addEventListener('input',e=>{q=e.target.value.trim().toLowerCase();applyFilter();});
}

initCM(); initClassView(); initEmoBars(); initText(); initNrcGt(); initFilters(); render(DATA);
</script>
</body>
</html>
"""

out = TEMPLATE
repl = {
    "%%TITLE%%": baked["%%TITLE%%"],
    "%%N%%": baked["%%N%%"],
    "%%DATA%%": baked["%%DATA%%"],
    "%%CLASSES%%": baked["%%CLASSES%%"],
    "%%CM%%": baked["%%CM%%"],
    "%%EMOS%%": baked["%%EMOS%%"],
    "%%STATS%%": baked["%%STATS%%"],
    "%%STAT_LLMACC%%": str(llm_acc),
    "%%STAT_MACRO%%": str(macro_f1),
    "%%STAT_NRCACC%%": str(nrc_acc),
    "%%STAT_AGREE%%": str(agree_pct_dec),
    "%%STAT_NEU%%": str(neutral_count),
    "%%STAT_PRED%%": " ".join(f"{k} {v}" for k, v in pred_dist.items()),
    "%%STAT_NRC%%": " ".join(f"{k} {v}" for k, v in sent_dist.items()),
}
for k, v in repl.items():
    out = out.replace(k, v)

with open(OUT_FILE, "w") as f:
    f.write(out)

print(f"Wrote {OUT_FILE}: {os.path.getsize(OUT_FILE):,} bytes, {n} reviews")
print(f"LLM acc={llm_acc}% macro-F1={macro_f1}%  NRC acc={nrc_acc}% (decided {nrc_decided})  "
      f"agree all {agree_all}/{n} ({agree_pct_all}%), decided {agree_decided}/{nrc_decided} ({agree_pct_dec}%), neutral={neutral_count}")
print(f"per_class={ {k:{'n':v['count'],'f1':v['f1']} for k,v in per_class.items()} }")
leftover = [k for k in repl if k in out]
print("leftover placeholders:", leftover)
