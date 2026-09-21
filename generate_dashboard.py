#!/usr/bin/env python3
"""
Generate a clean, minimal, self-contained dashboard (single HTML, offline).

Shows: headline numbers, easy per-class "correct vs predicted" bars, a small
LLM confusion matrix, the LLM-vs-NRC comparison, primary emotion, and an
explorable per-review table. Input: predictions_nrc.jsonl.
"""
import json
import os
from collections import Counter

IN_FILE = "predictions_nrc.jsonl"
OUT_FILE = "dashboard.html"

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

rows = []
with open(IN_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
n = len(rows)

# ---- metrics ----
cm = Counter((r["label"], r["pred"]) for r in rows)
llm_acc = round(100.0 * sum(cm[(c, c)] for c in CLASSES) / n, 1)
per_class = {}
mf1 = 0.0
for c in CLASSES:
    tp = cm[(c, c)]
    col = sum(cm[(a, c)] for a in CLASSES)
    row = sum(cm[(c, p)] for p in CLASSES)
    pr = tp / col if col else 0; rc = tp / row if row else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
    per_class[c] = {"count": row, "recall": round(rc * 100), "f1": round(f1 * 100)}
    mf1 += f1
macro_f1 = round(100.0 * mf1 / len(CLASSES), 1)

dec = [r for r in rows if r["nrc_sentiment"] in ("POSITIVE", "NEGATIVE")]
nrc_dec = len(dec)
nrc_acc = round(100.0 * sum(1 for r in dec if r["nrc_sentiment"] == r["label"]) / nrc_dec, 1) if nrc_dec else 0.0
agree_dec = sum(1 for r in dec if r["nrc_sentiment"] == r["pred"])
agree_pct = round(100.0 * agree_dec / nrc_dec, 1) if nrc_dec else 0.0
neutral = n - nrc_dec
emo_dist = dict(Counter(r["nrc_primary_emotion"] for r in rows))
emo_dist = {("none" if k is None else k): v for k, v in emo_dist.items()}

table = []
for i, r in enumerate(rows):
    nr = r["nrc_sentiment"]
    decided = nr in ("POSITIVE", "NEGATIVE")
    table.append({
        "i": i + 1, "title": r.get("title") or "", "text": r.get("text") or "",
        "label": r["label"], "pred": r["pred"], "ok": r["pred"] == r["label"],
        "nrc": nr, "emo": r.get("nrc_primary_emotion"),
        "decided": decided, "agree": decided and nr == r["pred"],
    })

S = {
    "%%N%%": str(n),
    "%%LLMACC%%": str(llm_acc),
    "%%MACRO%%": str(macro_f1),
    "%%NRCACC%%": str(nrc_acc),
    "%%AGREE%%": str(agree_pct),
    "%%NEU%%": str(neutral),
    "%%DATA%%": json.dumps(table),
    "%%CLASSES%%": json.dumps(CLASSES),
    "%%CM%%": json.dumps({a: {p: cm[(a, p)] for p in CLASSES} for a in CLASSES}),
    "%%EMOS%%": json.dumps(EMOTIONS),
    "%%STATS%%": json.dumps({
        "n": n, "per_class": per_class,
        "nrc_acc": nrc_acc, "nrc_dec": nrc_dec, "agree_pct": agree_pct,
        "neutral": neutral, "emo_dist": emo_dist,
        "pred_dist": dict(Counter(r["pred"] for r in rows)),
        "sent_dist": dict(Counter(r["nrc_sentiment"] for r in rows)),
    }),
}

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gift Card Sentiment — LLM vs NRC</title>
<style>
  :root{--green:#15945c;--g-bg:#e7f6ee;--red:#c0473b;--r-bg:#fbeae8;--gray:#f0f1f5;
        --ink:#20242c;--mut:#6a707d;--line:#e4e7ee;--bg:#fff;--accent:#2f6fae}
  *{box-sizing:border-box}
  body{margin:0;padding:28px;background:#f6f8fb;color:var(--ink);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
  h1{font-size:21px;margin:0}
  .sub{color:var(--mut);font-size:12.5px;margin:4px 0 18px}
  .card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px}
  h2{font-size:14px;margin:0 0 12px;font-weight:650}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
  .kpi .num{font-size:26px;font-weight:700;letter-spacing:-.5px}
  .kpi .lbl{font-size:11.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
  .two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  @media(max-width:820px){.two{grid-template-columns:1fr}}
  /* correct-vs-predicted */
  .cv-row{margin:12px 0}
  .cv-head{display:flex;justify-content:space-between;margin-bottom:4px;font-size:13px}
  .cv-head .meta{color:var(--mut);font-size:11.5px}
  .seg{display:flex;height:34px;border-radius:8px;overflow:hidden}
  .seg .s{display:flex;align-items:center;justify-content:center;gap:5px;color:#fff;font-size:11.5px;white-space:nowrap}
  .seg .ok{background:var(--green)}
  .seg .w-pos{background:#b3805a}.seg .w-neu{background:#9aa3b5}.seg .w-neg{background:#c0473b}
  .cv-desc{color:var(--mut);font-size:12.5px;margin-top:4px}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--mut);margin:10px 0 2px}
  .legend i{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:5px;vertical-align:-1px}
  /* confusion */
  table.cm{border-collapse:separate;border-spacing:6px;width:100%;table-layout:fixed;text-align:center}
  table.cm th,table.cm td.lab{font-size:10px;color:var(--mut);font-weight:600;text-transform:uppercase}
  table.cm td{font-size:15px;font-weight:650;border-radius:6px;padding:7px 0}
  table.cm td.diag{background:var(--g-bg);color:var(--green)}
  table.cm td.off{background:var(--r-bg);color:var(--red)}
  /* emotion chips */
  .emo{display:flex;flex-wrap:wrap;gap:6px}
  .chip{background:var(--gray);border-radius:99px;padding:2px 9px;font-size:11.5px;color:var(--mut)}
  .chip b{color:var(--ink)}
  /* table */
  .filters{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
  .filters button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:4px 10px;cursor:pointer;
                  font-size:12.5px;color:var(--mut)}
  .filters button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
  .filters .cnt{opacity:.55}
  input#q{border:1px solid var(--line);border-radius:7px;padding:4px 9px;font-size:12.5px;min-width:180px}
  .live{margin-left:auto;font-size:12.5px;color:var(--mut)}.live b{color:var(--ink)}
  table.rows{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}
  table.rows th{text-align:left;font-size:10px;text-transform:uppercase;color:var(--mut);letter-spacing:.4px;
                padding:6px 8px;border-bottom:1px solid var(--line)}
  table.rows td{padding:6px 8px;border-bottom:1px solid var(--gray);vertical-align:top}
  table.rows tr.err{background:var(--r-bg)}
  .pill{display:inline-block;padding:0 7px;border-radius:99px;font-size:10.5px;font-weight:600}
  .p-pos{background:#e8f3fb;color:#1f6398}.p-neu{background:#f0ecdc;color:#8a7230}.p-neg{background:#fbeae8;color:#b03a2e}
  .okp{background:var(--g-bg);color:var(--green)}.badp{background:var(--r-bg);color:var(--red)}.nup{background:var(--gray);color:#6a7080}
  .wrap{overflow:auto;max-height:560px}
  .foot{margin-top:18px;color:var(--mut);font-size:11.5px}
</style>
</head>
<body>
<h1>Gift Card Reviews — Sentiment: LLM vs NRC</h1>
<div class="sub">150 reviews (50 POSITIVE / 50 NEUTRAL / 50 NEGATIVE) &middot; title + text only &middot; rating used only as ground truth
  (4&ndash;5 POS, 3 NEU, 1&ndash;2 NEG) &middot; balanced stratified sample (seed 42)</div>

<div class="kpis">
  <div class="card kpi"><div class="num" style="color:var(--green)">%%LLMACC%%%</div><div class="lbl">LLM accuracy</div></div>
  <div class="card kpi"><div class="num" style="color:var(--green)">%%MACRO%%%</div><div class="lbl">LLM macro-F1</div></div>
  <div class="card kpi"><div class="num" style="color:var(--accent)">%%NRCACC%%%</div><div class="lbl">NRC accuracy</div></div>
  <div class="card kpi"><div class="num" style="color:var(--accent)">%%AGREE%%%</div><div class="lbl">LLM = NRC agree</div></div>
  <div class="card kpi"><div class="num">%%N%%</div><div class="lbl">Reviews</div></div>
</div>

<div class="card" style="margin-top:12px">
  <h2>Correct answer vs what the LLM predicted</h2>
  <div class="legend">
    <span><i style="background:var(--green)"></i>predicted correctly</span>
    <span><i style="background:#b3805a"></i>→ POSITIVE</span>
    <span><i style="background:#9aa3b5"></i>→ NEUTRAL</span>
    <span><i style="background:#c0473b"></i>→ NEGATIVE</span>
  </div>
  <div id="classview"></div>
</div>

<div class="two" style="margin-top:12px">
  <div class="card">
    <h2>LLM confusion (truth × predicted)</h2>
    <table class="cm" id="cm"></table>
  </div>
  <div class="card">
    <h2>Primary emotion (NRC lexicon)</h2>
    <div class="emo" id="emochips"></div>
    <p class="sub" style="margin-top:12px">NRC left <b>%%NEU%%</b> reviews undecided; it agrees with the LLM on
      <b>%%AGREE%%%</b> of decided ones.</p>
  </div>
</div>

<div class="card" style="margin-top:12px">
  <h2>Reviews</h2>
  <div class="filters" id="filters"></div>
  <div class="wrap">
    <table class="rows" id="rows">
      <thead><tr><th>#</th><th>Title</th><th>Body</th><th>Truth</th><th>LLM</th><th>NRC</th><th>Emotion</th><th>Verdict</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<div class="foot">LLM = reasoning model told to answer POSITIVE/NEUTRAL/NEGATIVE from title+text. NRC = word lookup (v0.92):
  sentiment = sign of positive-vs-negative hits, emotion = most-hit of the 8. Prediction file <code>predictions_nrc.jsonl</code>.
  The full/verbose dashboard is saved as <code>dashboard_full.html</code> for comparison.</div>

<script>
const DATA=%%DATA%%;const CLASSES=%%CLASSES%%;const CM=%%CM%%;const LN=%%STATS%%;const EMOS=%%EMOS%%;

// per-class correct-vs-predicted
const WRONGCLS={POSITIVE:'w-pos',NEUTRAL:'w-neu',NEGATIVE:'w-neg'};
const SHORT={POSITIVE:'POS',NEUTRAL:'NEU',NEGATIVE:'NEG'};
function classview(){
  let h='';
  CLASSES.forEach(c=>{
    const tot=CLASSES.reduce((s,p)=>s+CM[c][p],0)||1, ok=CM[c][c];
    const order=[c].concat(CLASSES.filter(p=>p!==c));
    let segs='';
    order.forEach(p=>{
      const v=CM[c][p]; if(v<=0)return;
      segs+='<div class="s '+(p===c?'ok':WRONGCLS[p])+'" style="flex-grow:'+v+'">'+v+' '+SHORT[p]
           +(p===c?' &#10003;':'')+'</div>';
    });
    const pct=Math.round(100*ok/tot);
    const mis=CLASSES.filter(p=>p!==c&&CM[c][p]>0)
      .map(p=>CM[c][p]+' called '+SHORT[p]).join(', ');
    h+='<div class="cv-row">'
      +'<div class="cv-head"><b>'+c+'</b><span class="meta">'+tot+' reviews · '+pct+'% correct · F1 '+LN.per_class[c].f1+'%</span></div>'
      +'<div class="seg">'+segs+'</div>'
      +'<div class="cv-desc">'+ok+' right ('+pct+'%)'+(mis?' — '+mis+' wrong.':' — all correct.')+'</div>'
      +'</div>';
  });
  document.getElementById('classview').innerHTML=h;
}

function cmx(){
  let h='<tr><th></th>'+CLASSES.map(c=>'<th>'+SHORT[c]+'</th>').join('')+'</tr>';
  CLASSES.forEach(a=>{
    h+='<tr><td class="lab">'+SHORT[a]+'</td>';
    CLASSES.forEach(p=>{
      const v=CM[a][p];
      h+='<td class="'+(a===p?'diag':'off')+'">'+v+'</td>';
    });
    h+='</tr>';
  });
  document.getElementById('cm').innerHTML=h;
}

function emochips(){
  const picks=[...EMOS.filter(e=>LN.emo_dist[e]>0), 'none_'];
  let h='';
  picks.forEach(e=>{
    const key=e==='none_'?'none':e;
    h+='<span class="chip"><b>'+key+'</b> '+LN.emo_dist[key]+'</span>';
  });
  document.getElementById('emochips').innerHTML=h;
}

// table
let filt='all',query='',emo='';
function matches(r){
  if(filt==='ok'&&!r.ok)return false;
  if(filt==='err'&&r.ok)return false;
  if(filt==='agr'&&!r.agree)return false;
  if(filt==='dis'&&!(r.decided&&!r.agree))return false;
  if(filt==='neu'&&r.decided)return false;
  if(filt.indexOf('t')===0&&r.label!==filt.slice(1))return false;
  if(emo&&r.emo!==emo)return false;
  if(query&&(r.title+' '+r.text).toLowerCase().indexOf(query)<0)return false;
  return true;
}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function render(){
  const rows=DATA.filter(matches), tb=document.getElementById('tbody'); tb.innerHTML='';
  rows.forEach(r=>{
    const tr=document.createElement('tr'); tr.className=r.ok?'':'err';
    const text=r.text.length>120?r.text.slice(0,120)+'…':(r.text||'(no body)');
    const pill=c=>'<span class="pill p-'+c[0].toLowerCase()+'">'+c+'</span>';
    const nrc=r.decided?pill(r.nrc):'<span class="pill nup">neutral</span>';
    const verdict=r.ok?'<span class="pill okp">correct</span>':'<span class="pill badp">wrong</span>';
    tr.innerHTML='<td>'+r.i+'</td><td><b>'+esc(r.title||'(no title)')+'</b></td>'
      +'<td>'+esc(text)+'</td><td>'+pill(r.label)+'</td><td>'+pill(r.pred)+'</td>'
      +'<td>'+nrc+'</td><td>'+(r.emo?r.emo:'—')+'</td><td>'+verdict+'</td>';
    tb.appendChild(tr);
  });
  document.getElementById('shown').textContent=rows.length;
}
function filters(){
  const c={all:DATA.length,ok:0,err:0,agr:0,dis:0,neu:0,
    tPOSITIVE:0,tNEUTRAL:0,tNEGATIVE:0};
  DATA.forEach(r=>{r.ok?c.ok++:c.err++;
    if(r.decided){r.agree?c.agr++:c.dis++;}else c.neu++;
    c['t'+r.label]++;
  });
  const defs=[['all','All',c.all],['ok','Correct',c.ok],['err','Wrong',c.err],
    ['tPOSITIVE','POS',c.tPOSITIVE],['tNEUTRAL','NEU',c.tNEUTRAL],['tNEGATIVE','NEG',c.tNEGATIVE],
    ['dis','Differ',c.dis],['agr','Agree',c.agr],['neu','NRC neutral',c.neu]];
  const box=document.getElementById('filters');
  defs.forEach(d=>{
    const b=document.createElement('button');
    b.innerHTML=d[1]+' <span class="cnt">'+d[2]+'</span>';
    b.onclick=()=>{filt=d[0];box.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');render();};
    box.appendChild(b);
  });
  const sel=document.createElement('select');
  sel.innerHTML='<option value="">all emotions</option>'+EMOS.map(e=>'<option>'+e+'</option>').join('');
  sel.onchange=()=>{emo=sel.value;render();};
  box.appendChild(sel);
  const qinput=document.createElement('input');
  qinput.placeholder='search title/text';qinput.type='text';
  qinput.oninput=()=>{query=qinput.value.trim().toLowerCase();render();};
  box.appendChild(qinput);
  const live=document.createElement('span');
  live.className='live';live.innerHTML='<b id="shown">'+DATA.length+'</b> / '+DATA.length;
  box.appendChild(live);
}
classview();cmx();emochips();filters();render();</script>
</body>
</html>
"""

out = TEMPLATE
for k, v in S.items():
    out = out.replace(k, v)

with open(OUT_FILE, "w") as f:
    f.write(out)
print(f"Wrote {OUT_FILE}: {os.path.getsize(OUT_FILE):,} bytes, {n} reviews")
