# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The live HTML dashboard: one self-contained HTML5 page, rewritten every tick, that
reloads itself every 2 minutes and keeps the scroll position and open sections. No external scripts. Two variants from
the same code (write): the local file, whose priority box calls the monitor's local service on 127.0.0.1 (webserve),
and the published copy, whose box only posts requests to the web host beside it; nothing else uses the network.

The page follows a PR through the system, top to bottom:
1. a sticky header: monitor state, develop's status word (report.develop_status), section navigation; on the admin
   page, while the monitor is in standby, its top line (standby_line);
2. Needs you: what only the user can do, including PRs ready for the user's approval and the in-depth reviews waiting
   for it (on the admin dashboard, with their Approve and Decline: decide_buttons; the published copy has each
   decision as the team sees it, report.need_ask); on the admin dashboard only,
   then the "Tickets waiting for you" card (rkmon_lib/tickets.py) and the "Not sent" card (not_sent_card);
3. KPI tiles ("Issues avoided" on the admin page only);
4. the pipeline: review queue → in review (every review in flight by its part: report.in_review) → needs work → needs
   approval → mergeable candidates (passed CI, can merge now; "+N waiting on CI" counted only) → merged today;
5. reviews (in flight, next up, backlog and throughput charts), right after the pipeline;
6. the merge boxes:
   a. Needs platform approval, then Needs rocke-lib approval: one box per code-owner team (config order), each in
      merge-plan order; then Needs any approval: a general approval, no code-owner team's (report.any_approvals);
   b. the Mergeable Candidates: the PRs that passed CI and can merge now, and each gate's state (in landing order;
      their batch, auto-merge), above the merge plan;
   c. the merge plan: the "Request PR priority" box and the priority requests, the
      order and its conflict-free waves, a conflict grid with textual (✗) and ABI (Ⓐ) conflicts and its table view, the
      ABI lanes in plan order, and the rebase advice, collapsed;
   d. insights from recent PRs (collapsed: our 🔴/🟠 findings by cause with what prevents each, open develop issues by
      cause, the merge-conflict hot files and CI time per PR of the last 7 days, the latest insights report), and
      develop health (the status word and why, CI, sanity, incidents, develop issues by kind, each by
      its label);
7. the admin page only (the LOCAL file; the published copy has none of them, nor their links):
   a. the scoreboard: per PR author, what lands and the rework it causes for others, over 7, 30 or 90 days
      (scoreboard_card; scoreboard.py);
   b. review spend: today and this week so far, the last 7 days, this week's outliers (spend.card_data; review workload
      only, no experiment spend), and the model guard's WARN (spend.guard);
   c. today's log, the last card.

Follows the dataviz method: reference palette slots 1–3 (validated all-pairs, light and dark: worst adjacent CVD ΔE
9.2 / 9.4, normal vision 27.6 / 26.5; light slot 3 is 2.74:1 on the surface, so its chart labels the latest day and has
a table view), light and dark values both declared, 2px lines, columns ≤ 24px with 4px rounded caps and 2px gaps,
hairline grids, status colors only with an icon and a label, hover tooltips set through textContent (titles are
untrusted), and a table view under every chart. Slot 1 is cleared reviews, slot 2 arrivals, slot 3 merges; on the
scoreboard, slot 3 is PRs merged and slot 2 rework for others (that pair validated on its own: CVD ΔE 9.2 light, 9.4
dark).
"""

from __future__ import annotations

import datetime as dt
import html
import json
import pathlib
import re
import urllib.parse
from collections import Counter

from . import (
    abi,
    ci_times,
    config,
    conflicts,
    daylog,
    derived,
    guard,
    idle,
    insights,
    merges,
    mergeplan,
    priority,
    publish,
    registry,
    report,
    scoreboard,
    spend,
    webserve,
    whole_file,
)
from .breakage import (
    RED,
    STABILITY,
    entry_kind,
    entry_label,
    issue_repository,
    kind_label,
)
from .checks import FIX_LABEL
from .plan import gap_label
from .queue import FAIL, PASS, USER_PRIORITY, active_holds, gate_review
from .snapshot import ci_verdict
from .store import minutes_since, now
from .queue import fixes_develop
from . import paths

E = html.escape
REFRESH_S = 120

CSS = """
.viz-root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series-1:#2a78d6;--series-2:#eb6834;
--series-3:#1baf7a;--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;
--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--page);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
min-height:100vh}
a{color:inherit}
header.top{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--page) 90%,transparent);
backdrop-filter:blur(8px);border-bottom:1px solid var(--border);padding:10px 24px 8px}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;max-width:1500px;margin:0 auto}
h1{font-size:18px;margin:0 10px 0 0}
.meta{color:var(--muted);font-size:12px;margin-left:auto}
nav{display:flex;flex-wrap:wrap;gap:2px;max-width:1500px;margin:6px auto 0}
nav a{color:var(--text-secondary);text-decoration:none;font-size:13px;padding:2px 10px;border-radius:999px;
border:1px solid transparent}
nav a:hover,nav a:focus-visible{border-color:var(--border);color:var(--text-primary)}
nav a .c{color:var(--muted);font-variant-numeric:tabular-nums}
main{padding:16px 24px 40px;display:grid;gap:16px;max-width:1500px;margin:0 auto}
section{scroll-margin-top:100px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:14px 16px;min-width:0}
h2{font-size:15px;margin:0 0 10px;display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
h2 .c{color:var(--text-secondary);font-weight:500;font-size:13px}
h3{font-size:13px;margin:14px 0 6px;color:var(--text-secondary)}
.sub{color:var(--text-secondary);font-size:13px}.muted{color:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile .label{color:var(--text-secondary);font-size:12px}
.tile .value{font-size:28px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums;display:flex;
align-items:center;gap:10px}
.tile .note{color:var(--muted);font-size:12px;margin-top:2px}
.flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px}
.stage{position:relative;background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:12px;
border-top:3px solid var(--series-1);min-width:0}
.stage:not(:last-child)::after{content:"\\2192";position:absolute;right:-13px;top:18px;color:var(--axis);font-size:14px}
.stage .n{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.1}
.stage .lbl{font-weight:600;margin-top:2px}.stage .hint{color:var(--muted);font-size:12px;min-height:34px}
.stage .parts{color:var(--text-secondary);font-size:12px;margin:2px 0 4px}
@media (max-width:1150px){.flow{grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}.stage::after{display:none}}
.chips{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin-top:8px}
.chip{font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--text-primary);text-decoration:none;
border:1px solid var(--border);border-radius:6px;padding:0 6px;background:var(--page);white-space:nowrap}
a.chip:hover,a.chip:focus-visible{border-color:var(--series-1)}
.chip.chip-prio{border-color:var(--good);box-shadow:inset 0 0 0 1px var(--good)}
.chip-ready-early{margin-left:4px;font-size:10px;font-weight:700;color:var(--good);text-transform:uppercase}
.side{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:10px;color:var(--text-secondary);font-size:13px}
ul.left{margin:0;padding:0;list-style:none}ul.left li{margin:1px 0}ul.left .who{color:var(--text-secondary)}
ul.left li.you .what,ul.left li.you .who{font-weight:600;color:var(--text-primary)}
.needs{border-left:4px solid var(--serious)}.needs.ok{border-left-color:var(--good)}
.needs.crit{border-left-color:var(--critical)}
.needs ol{margin:4px 0 0 20px;padding:0}.needs li{margin:3px 0}
.standby{max-width:1500px;margin:6px auto 0;padding:3px 10px;border-left:4px solid var(--warning);border-radius:6px;
background:var(--surface-1);font-size:13px}
pre.unsent{margin:4px 0 0;max-height:12em;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.pill{display:inline-flex;gap:6px;align-items:center;border:1px solid var(--border);border-radius:999px;
padding:1px 10px;font-size:13px;background:var(--surface-1);white-space:nowrap}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}
.st{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;color:var(--text-secondary);font-weight:600;border-bottom:1px solid var(--axis);padding:4px 6px;
white-space:nowrap}
td{border-bottom:1px solid var(--grid);padding:4px 6px;vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.nw{white-space:nowrap}
.wrap{overflow-x:auto}
td.g,th.g{text-align:center;width:28px;padding:4px 2px;font-weight:700}
.g.pass{color:var(--good)}.g.fail{color:var(--critical)}.g.pending{color:var(--muted);font-weight:400}
.g.care{color:var(--warning)}.g.abi,.g .abi{color:var(--serious)}
table.grid{width:auto;border-collapse:separate;border-spacing:2px}
table.grid td.g.two{font-size:11px;letter-spacing:-1px}
table.grid td.g{width:24px;min-width:24px;height:24px;padding:0;border:1px solid var(--grid);border-radius:4px}
table.grid td.g.diag{background:var(--grid)}
table.grid td.g[tabindex]:hover,table.grid td.g[tabindex]:focus-visible{outline:2px solid var(--series-1);
outline-offset:-1px}
table.grid th{border:0;padding:0 2px}table.grid th.rowh{text-align:right;padding-right:6px}
table.grid th.v{writing-mode:vertical-rl;transform:rotate(180deg);font:500 11px ui-monospace,SFMono-Regular,Consolas,
monospace;color:var(--text-secondary);height:52px}
details{margin-top:8px}summary{cursor:pointer;color:var(--text-secondary);font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--text-secondary);font-size:12px;margin-bottom:4px}
.key{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);
border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12px;box-shadow:0 2px 10px rgba(0,0,0,.15);
display:none;z-index:9}
#tip b{font-size:14px}
svg text{fill:var(--muted);font-size:11px}svg .val{fill:var(--text-primary);font-weight:600}
footer{color:var(--muted);font-size:12px;max-width:1500px;margin:0 auto;padding:0 24px 24px}
.prio{margin:0 0 12px;padding:10px 12px;border:1px solid var(--border);border-radius:10px;background:var(--page)}
.prio.live{border-color:var(--series-1)}
.prio-row{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px}.prio label{font-weight:600}
.prio input{font:inherit;width:11ch;padding:2px 8px;border:1px solid var(--axis);border-radius:6px;
background:var(--surface-1);color:var(--text-primary)}
.prio-ask:not([hidden]){display:inline-flex;flex-wrap:wrap;align-items:center;gap:6px 10px}
.prio #prio-who{width:16ch}.prio #prio-why{width:34ch}
.prio button,button.prio-act{font:inherit;font-weight:600;padding:2px 12px;border:1px solid var(--series-1);
border-radius:6px;background:var(--surface-1);color:var(--text-primary);cursor:pointer}
button.prio-act{font-size:12px;padding:0 8px}
.prio button:hover:not(:disabled),.prio button:focus-visible,button.prio-act:hover:not(:disabled),
button.prio-act:focus-visible{background:color-mix(in srgb,var(--series-1) 14%,var(--surface-1))}
.prio :disabled,button.prio-act:disabled{opacity:.55;cursor:not-allowed}.prio p{margin:6px 0 0}
button.tk-act{font:inherit;font-weight:600;font-size:12px;padding:0 8px;border:1px solid var(--series-1);
border-radius:6px;background:var(--surface-1);color:var(--text-primary);cursor:pointer}
button.tk-act:hover:not(:disabled),button.tk-act:focus-visible{background:color-mix(in srgb,var(--series-1) 14%,
var(--surface-1))}button.tk-act:disabled{opacity:.55;cursor:not-allowed}
.sb-switch{display:flex;gap:6px;align-items:center;margin:0 0 10px}
.sb-win{font:inherit;font-size:13px;padding:1px 12px;border:1px solid var(--border);border-radius:999px;
background:var(--surface-1);color:var(--text-secondary);cursor:pointer}
.sb-win[aria-pressed='true']{border-color:var(--series-1);color:var(--text-primary);font-weight:600;
background:color-mix(in srgb,var(--series-1) 14%,var(--surface-1))}
.sb-win:focus-visible,.sb-sort:focus-visible{outline:2px solid var(--series-1);outline-offset:1px}
.sb-sort{font:inherit;font-weight:600;color:inherit;background:none;border:0;padding:0;cursor:pointer;text-align:inherit}
th[aria-sort='descending'] .sb-sort::after{content:' \\25BE'}th[aria-sort='ascending'] .sb-sort::after{content:' \\25B4'}
svg.sb-chart{max-width:760px;display:block}h3 .c{color:var(--muted);font-weight:400}
td.sb-tip{cursor:help;text-decoration:underline dotted var(--axis);text-underline-offset:3px}
"""

JS = (
    """
const tip=document.getElementById('tip');
function show(e,rows){tip.replaceChildren();for(const [k,v,c] of rows){const d=document.createElement('div');
if(c){const s=document.createElement('span');s.className='key';s.style.background=c;d.appendChild(s);}
const b=document.createElement('b');b.textContent=v;d.appendChild(b);d.appendChild(document.createTextNode(' '+k));
tip.appendChild(d);}tip.style.display='block';const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);
tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('[data-tip]').forEach(el=>{const rows=JSON.parse(el.dataset.tip);
el.addEventListener('pointermove',e=>show(e,rows));el.addEventListener('pointerleave',hide);
el.addEventListener('focus',()=>{const r=el.getBoundingClientRect();show({clientX:r.right,clientY:r.top},rows);});
el.addEventListener('blur',hide);});
document.querySelectorAll('[data-line]').forEach(ov=>{const pts=JSON.parse(ov.dataset.line);
const cross=document.getElementById(ov.dataset.cross);const svg=ov.ownerSVGElement;
ov.addEventListener('pointermove',e=>{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;
const q=p.matrixTransform(svg.getScreenCTM().inverse());let best=pts[0];for(const t of pts){if(Math.abs(t[0]-q.x)<
Math.abs(best[0]-q.x))best=t;}cross.setAttribute('x1',best[0]);cross.setAttribute('x2',best[0]);
cross.style.display='';show(e,[[best[2],String(best[3]),getComputedStyle(svg).getPropertyValue('--series-1')]]);});
ov.addEventListener('pointerleave',()=>{cross.style.display='none';hide();});});
const KEY='rkmon-dash';
try{const s=JSON.parse(sessionStorage.getItem(KEY)||'{}');for(const id of s.open||[]){const d=document.getElementById(id);
if(d)d.open=true;}if(s.y)scrollTo(0,s.y);}catch(e){}
let left=%d;const cd=document.getElementById('countdown');
setInterval(()=>{left--;if(cd)cd.textContent=Math.floor(left/60)+':'+String(left%%60).padStart(2,'0');if(left>0)return;
if(window.rkmonBusy&&rkmonBusy()){left=15;return;}
try{sessionStorage.setItem(KEY,JSON.stringify({y:scrollY,open:[...document.querySelectorAll('details[open][id]')]
.map(d=>d.id)}));}catch(e){}location.reload();},1000);
"""
    % REFRESH_S
)

# The page's two variants, rendered by the same code: LOCAL, the admin dashboard (webdash.write's path, which the
# monitor's service serves), and PUBLISHED, the copy publish.html pushes to the team web host. Only the local page
# contacts the monitor's service; the published copy never does (Chrome's Local Network Access would prompt or block
# every viewer on other teams). The local page is yours only; the published copy is for every reader.
LOCAL, PUBLISHED = "local", "published"
PUBLISHED_FILE = "published.html"  # in state/: the copy publish.html pushes
PAGE_MODES = {LOCAL: 0o600, PUBLISHED: 0o644}  # each variant's file permissions

# The "Request PR priority" box (priority_box; other teams' clicks wait for the user's OK). The local page, as the
# service serves it, pings the service on load: where it answers, a direct prioritize, and Approve / Decline on the
# pending requests and on the Needs-you lines decided on the page (decide_buttons; a line's answer shows beside it),
# every request with the browser's access token; otherwise the box stays off and its note says why (opened from disk,
# the page links to the served one). The published copy only asks: a request with a name and a reason, posted to the
# web host beside the page (a relative URL: same origin). The reload waits while a request is in flight or a field is
# being typed in. Every message from a service is set through textContent. An answer that is not JSON says its HTTP
# status (a plain `python3 -m http.server` answers a POST with an HTML 501).
_PRIO_COMMON = """
(()=>{const f=document.getElementById('prio');if(!f)return;const $=id=>document.getElementById(id);
const inp=$('prio-n'),who=$('prio-who'),why=$('prio-why'),btn=$('prio-go'),out=$('prio-status');
const me=f.dataset.me,KEEP='rkmon-prio',ICON={ok:'✓',bad:'✗',wait:'…'},LEVEL={ok:'good',bad:'critical',wait:'muted'};
let busy=false,how='';
function mark(el,k,text){const d=document.createElement('span');d.className='dot';
d.style.background='var(--'+LEVEL[k]+')';el.replaceChildren(d,document.createTextNode(' '+ICON[k]+' '+text));}
function say(k,text){mark(out,k,text);
if(k!=='wait')try{sessionStorage.setItem(KEEP,JSON.stringify({k,text,t:Date.now()}));}catch(e){}}
window.rkmonBusy=()=>busy||(f.contains(document.activeElement)&&!!document.activeElement.value);
function post(url,data){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Rkmon':'1'},
body:JSON.stringify(data),cache:'no-store'}).then(r=>r.json().catch(()=>({ok:false,status:r.status})));}
function text(j,none){return j.message||(j.status?'The server answered HTTP '+j.status+' without a message.':none);}
function go(m,text){how=m;f.classList.add('live');inp.disabled=btn.disabled=false;out.textContent=text;
try{const s=JSON.parse(sessionStorage.getItem(KEEP)||'null');if(s&&Date.now()-s.t<600000)say(s.k,s.text);}catch(e){}}
function number(){const n=parseInt(inp.value,10);if(n>0)return n;say('bad','Enter a PR number.');return 0;}
"""
_PRIO_LOCAL = (
    """
const port=f.dataset.port,HOME=f.dataset.home||'http://"""
    + webserve.HOST
    + """:'+port+'/',served=location.protocol!=='file:',
base=location.origin,DOWN='No answer from the monitor service: is tools/rkweb.py running?';
if(!served){const a=document.createElement('a');a.href=HOME;a.textContent=HOME;out.replaceChildren(
document.createTextNode('This page, opened from a file, does not reach the monitor service: open the admin dashboard at '),
a,document.createTextNode(' (it asks for the access token) for the box and the Approve and Decline buttons.'));}
if(served){const stop=new AbortController();setTimeout(()=>stop.abort(),3000);
fetch(base+'/ping',{headers:{'X-Rkmon':'1'},cache:'no-store',signal:stop.signal}).then(r=>r.ok?r.json():null)
.then(j=>{if(!j||!j.ok)throw new Error('no answer');document.querySelectorAll('[data-local-only]').forEach(e=>e.hidden=false);
go('local','Live: '+j.me+"'s monitor service answers here.");}).catch(()=>say('bad',DOWN));}
f.addEventListener('submit',e=>{e.preventDefault();if(busy||how!=='local')return;const n=number();if(!n)return;
busy=true;btn.disabled=true;say('wait','Requesting priority for #'+n+' (if a tick is running, this waits for it to end)');
post(base+'/priority',{pr:n}).then(j=>{say(j.ok?'ok':'bad',text(j,'No message.'));if(j.ok)inp.value='';})
.catch(()=>say('bad',DOWN)).finally(()=>{busy=false;btn.disabled=false;});});
document.querySelectorAll('button.prio-act').forEach(b=>b.addEventListener('click',()=>{if(how!=='local'||busy)return;
const row=b.closest('tr,li'),st=row.querySelector('.prio-st'),line=row.tagName==='LI';
const tell=line?((k,m)=>mark(st,k,m)):say;busy=true;b.disabled=true;
post(base+'/'+b.dataset.route+'-'+b.dataset.act,{pr:+b.dataset.pr}).then(j=>{
tell(j.ok?'ok':'bad',text(j,'No message.'));if(!j.ok)return;
if(!line)st.textContent=b.dataset.act==='approve'?'approved':'declined';
b.parentNode.replaceChildren();}).catch(()=>tell('bad',DOWN)).finally(()=>{busy=false;b.disabled=false;});}));
const tko=document.getElementById('tk-status');function tsay(k,text){if(tko)mark(tko,k,text);}
document.querySelectorAll('button.tk-act').forEach(b=>b.addEventListener('click',()=>{if(how!=='local'||busy)return;
busy=true;b.disabled=true;const st=b.closest('tr').querySelector('.tk-st');
tsay('wait',b.dataset.act==='approve'?'Filing it in JIRA (this waits for a running tick)':'Declining it');
post(base+'/ticket-'+b.dataset.act,{id:b.dataset.id}).then(j=>{tsay(j.ok?'ok':'bad',text(j,'No message.'));
if(j.row)st.textContent=j.row;if(!j.ok)return;if(b.dataset.act==='decline')st.textContent='declined';
b.parentNode.replaceChildren();}).catch(()=>tsay('bad',DOWN)).finally(()=>{busy=false;b.disabled=false;});}));})();
"""
)
_PRIO_PUBLISHED = """
f.querySelectorAll('.prio-ask').forEach(e=>e.hidden=false);who.disabled=why.disabled=false;
try{who.value=localStorage.getItem('rkmon-who')||'';}catch(e){}
go('ask','Ask for a PR to be reviewed and merged first: '+me+' approves or declines each request.');
f.addEventListener('submit',e=>{e.preventDefault();if(busy)return;const n=number();if(!n)return;
const w=who.value.trim(),y=why.value.trim();if(!w||!y){say('bad','Enter your name and why it should go first.');return;}
busy=true;btn.disabled=true;say('wait','Sending #'+n+'…');
post('priority-request',{pr:n,who:w,why:y}).then(j=>{if(!j.ok){say('bad',j.message||(j.status?'The web host '
+'answered HTTP '+j.status+': it does not take requests yet. Tell '+me+'.':'Not accepted.'));return;}
inp.value='';why.value='';try{localStorage.setItem('rkmon-who',w);}catch(e){}
say('ok','Requested: '+me+' will review it. The list below shows it from the next tick (within 10 min).');})
.catch(()=>say('bad','No answer from the web host: try again later.')).finally(()=>{busy=false;btn.disabled=false;});
});})();
"""
PRIO_JS = {LOCAL: _PRIO_COMMON + _PRIO_LOCAL, PUBLISHED: _PRIO_COMMON + _PRIO_PUBLISHED}

# The scoreboard's window switch and its sortable table (admin page only). The chosen window and sort survive the reload
# (sessionStorage). Sorting reads each cell's data-v: numbers numerically, the author by name.
SCORE_JS = """
(()=>{const card=document.getElementById('scoreboard');if(!card)return;const KEEP='rkmon-score';
let keep={};try{keep=JSON.parse(sessionStorage.getItem(KEEP)||'{}');}catch(e){}
function save(){try{sessionStorage.setItem(KEEP,JSON.stringify(keep));}catch(e){}}
function pick(days){card.querySelectorAll('.sb-win').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.days===days)));
card.querySelectorAll('.sb-panel').forEach(p=>p.hidden=p.dataset.days!==days);keep.days=days;save();}
card.querySelectorAll('.sb-win').forEach(b=>b.addEventListener('click',()=>pick(b.dataset.days)));
function sort(table,col,dir){const body=table.tBodies[0];const rows=[...body.rows];const num=col>0;
rows.sort((a,b)=>{const x=a.cells[col].dataset.v,y=b.cells[col].dataset.v;
const c=num?(+x)-(+y):x.localeCompare(y,undefined,{sensitivity:'base'});
return (dir==='ascending'?c:-c)||a.cells[0].dataset.v.localeCompare(b.cells[0].dataset.v);});
rows.forEach(r=>body.appendChild(r));table.querySelectorAll('thead th').forEach((th,i)=>{
if(i===col)th.setAttribute('aria-sort',dir);else th.removeAttribute('aria-sort');});}
card.querySelectorAll('table.sb-table').forEach(t=>t.querySelectorAll('thead th').forEach((th,i)=>{
const b=th.querySelector('.sb-sort');if(!b)return;b.addEventListener('click',()=>{
const dir=th.getAttribute('aria-sort')==='descending'?'ascending':'descending';
card.querySelectorAll('table.sb-table').forEach(o=>sort(o,i,dir));keep.col=i;keep.dir=dir;save();});}));
if(keep.days&&card.querySelector(".sb-win[data-days='"+keep.days+"']"))pick(keep.days);
if(keep.col!=null)card.querySelectorAll('table.sb-table').forEach(t=>sort(t,keep.col,keep.dir));})();
"""


def _tip(rows: list[tuple]) -> str:
    return E(json.dumps(rows, ensure_ascii=False), quote=True)


def _nice(v: float) -> int:
    for step in (5, 10, 20, 25, 50, 100, 200, 500, 1000):
        if v <= step * 4:
            return step
    return 2000


def backlog_series(store, hours: int = 168) -> list[tuple[str, int]]:
    return [
        (m["ts"], m["review_backlog"])
        for m in store.read_jsonl("metrics.jsonl")
        if "review_backlog" in m and minutes_since(m["ts"]) <= hours * 60
    ]


def daily_series(store, days: int = 7) -> list[tuple[str, int, int]]:
    """Per UTC day: cleared (posts + reviews kept unposted + clears without a post: rebase- or merge-only) and arrived
    (new PRs + pushes)."""
    today = dt.datetime.now(dt.timezone.utc).date()
    keys = [(today - dt.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    posts = Counter(r["ts"][:10] for r in store.ledger() if r["kind"] == "post")
    kept = Counter(
        r["ts"][:10] for r in daylog.entries(store) if r["kind"] == daylog.UNPOSTED
    )
    equiv = Counter(ts[:10] for ts in report.rebase_clears(store).values())
    ev = store.read_jsonl("events.jsonl")
    arrived = Counter(
        e["ts"][:10] for e in ev if e.get("kind") in ("NEW_PR", "PUSH") and e.get("ts")
    )
    return [(k, posts[k] + kept[k] + equiv[k], arrived[k]) for k in keys]


def sparkline(
    points: list[tuple[str, int]], w: int = 120, h: int = 32, color: str = "--series-1"
) -> str:
    """A single-series trend for a KPI tile: 2px line, end dot, no axes (the tile's number is the label)."""
    if len(points) < 2:
        return ""
    vals = [v for _, v in points]
    lo, span = min(vals), max(max(vals) - min(vals), 1)
    xs = [2 + i * (w - 8) / (len(vals) - 1) for i in range(len(vals))]
    ys = [h - 4 - (v - lo) * (h - 8) / span for v in vals]
    d = " ".join(
        f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
        for i, (x, y) in enumerate(zip(xs, ys))
    )
    return (
        f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' aria-hidden='true'><path d='{d}' fill='none' "
        f"stroke='var({color})' stroke-width='2' stroke-linejoin='round' stroke-linecap='round'/>"
        f"<circle cx='{xs[-1]:.1f}' cy='{ys[-1]:.1f}' r='3' fill='var({color})'/></svg>"
    )


def line_chart(points: list[tuple[str, int]], title: str) -> str:
    w, h, ml, mr, mt, mb = 560, 210, 40, 46, 12, 26
    if len(points) < 2:
        return f"<p class='sub'>{E(title)}: collecting data (one point per tick).</p>"
    t0 = dt.datetime.strptime(points[0][0], "%Y-%m-%dT%H:%M:%SZ")
    xs = [
        (dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") - t0).total_seconds()
        for t, _ in points
    ]
    span = max(xs[-1], 1.0)
    step = _nice(max(v for _, v in points))
    top = max(step, ((max(v for _, v in points) + step - 1) // step) * step)
    X = lambda x: ml + (w - ml - mr) * x / span
    Y = lambda v: mt + (h - mt - mb) * (1 - v / top)
    out = [
        f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' aria-label='{E(title)}'>"
    ]
    for g in range(0, top + 1, step):
        out.append(
            f"<line x1='{ml}' x2='{w - mr}' y1='{Y(g):.1f}' y2='{Y(g):.1f}' stroke='var(--grid)' stroke-width='1'/>"
            f"<text x='{ml - 6}' y='{Y(g) + 4:.1f}' text-anchor='end'>{g}</text>"
        )
    for frac in (0, 0.5, 1):
        t = t0 + dt.timedelta(seconds=span * frac)
        out.append(
            f"<text x='{X(span * frac):.1f}' y='{h - 6}' text-anchor='middle'>{t:%m-%d %H:%M}</text>"
        )
    pts = [(X(x), Y(v)) for x, (_, v) in zip(xs, points)]
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts)
    )
    base = Y(0)
    out.append(
        f"<path d='{path} L{pts[-1][0]:.1f},{base:.1f} L{pts[0][0]:.1f},{base:.1f} Z' fill='var(--series-1)' "
        "fill-opacity='.10' stroke='none'/>"
    )
    out.append(
        f"<path d='{path}' fill='none' stroke='var(--series-1)' stroke-width='2' stroke-linejoin='round' "
        "stroke-linecap='round'/>"
    )
    ex, ey = pts[-1]
    out.append(
        f"<circle cx='{ex:.1f}' cy='{ey:.1f}' r='4' fill='var(--series-1)' stroke='var(--surface-1)' "
        f"stroke-width='2'/><text class='val' x='{ex + 8:.1f}' y='{ey + 4:.1f}'>{points[-1][1]}</text>"
    )
    data = [
        [round(x, 1), round(y, 1), f"{t[5:16].replace('T', ' ')}Z", v]
        for (x, y), (t, v) in zip(pts, points)
    ]
    out.append(
        f"<line id='cross-bl' y1='{mt}' y2='{base:.1f}' stroke='var(--axis)' stroke-width='1' style='display:none'/>"
    )
    out.append(
        f"<rect x='{ml}' y='{mt}' width='{w - ml - mr}' height='{h - mt - mb}' fill='transparent' "
        f"data-cross='cross-bl' data-line='{E(json.dumps(data), quote=True)}'/></svg>"
    )
    rows = "".join(
        f"<tr><td>{E(t)}</td><td class='num'>{v}</td></tr>" for t, v in points[-48:]
    )
    out.append(
        f"<details id='d-backlog'><summary>Table view (last {min(48, len(points))} ticks)</summary><table><tr>"
        f"<th>Tick (UTC)</th><th class='num'>Waiting</th></tr>{rows}</table></details>"
    )
    return "".join(out)


def _bar(x: float, v: int, bw: float, Y, color: str) -> str:
    """One column anchored to the baseline, with a 4px rounded data end."""
    y0, y1 = Y(0), Y(v)
    if v == 0:
        return ""
    r = min(4, (y0 - y1))
    return (
        f"<path d='M{x:.1f},{y0:.1f} L{x:.1f},{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} "
        f"L{x + bw - r:.1f},{y1:.1f} Q{x + bw:.1f},{y1:.1f} {x + bw:.1f},{y1 + r:.1f} L{x + bw:.1f},{y0:.1f} Z' "
        f"fill='var({color})'/>"
    )


def column_chart(days: list[tuple[str, int, int]]) -> str:
    w, h, ml, mr, mt, mb, bw, gap = 560, 210, 40, 12, 18, 26, 18, 2
    step = _nice(max([c for _, c, _ in days] + [a for _, _, a in days] + [1]))
    top = max(
        step,
        (
            (max([c for _, c, _ in days] + [a for _, _, a in days] + [1]) + step - 1)
            // step
        )
        * step,
    )
    Y = lambda v: mt + (h - mt - mb) * (1 - v / top)
    band = (w - ml - mr) / len(days)

    def col(x: float, v: int, color: str) -> str:
        return _bar(x, v, bw, Y, color)

    out = [
        "<div class='legend'><span><span class='key' style='background:var(--series-1)'></span>Cleared (posted + "
        "kept unposted + rebase- or merge-only)</span><span><span class='key' style='background:var(--series-2)'>"
        "</span>Arrived (new PRs + "
        f"pushes)</span></div><svg viewBox='0 0 {w} {h}' width='100%' role='img' aria-label='Cleared vs arrived per day'>"
    ]
    for g in range(0, top + 1, step):
        out.append(
            f"<line x1='{ml}' x2='{w - mr}' y1='{Y(g):.1f}' y2='{Y(g):.1f}' stroke='var(--grid)' stroke-width='1'/>"
            f"<text x='{ml - 6}' y='{Y(g) + 4:.1f}' text-anchor='end'>{g}</text>"
        )
    for i, (d, c, a) in enumerate(days):
        x0 = ml + band * i + (band - 2 * bw - gap) / 2
        out.append(col(x0, c, "--series-1") + col(x0 + bw + gap, a, "--series-2"))
        out.append(
            f"<text x='{ml + band * i + band / 2:.1f}' y='{h - 6}' text-anchor='middle'>{d[5:]}</text>"
        )
        out.append(
            f"<rect x='{ml + band * i:.1f}' y='{mt}' width='{band:.1f}' height='{h - mt - mb}' fill='transparent' "
            f"tabindex='0' data-tip='{_tip([['cleared', str(c), 'var(--series-1)'], ['arrived', str(a), 'var(--series-2)'], ['day', d, None]])}'/>"
        )
        if i == len(days) - 1:  # label only the latest day
            out.append(
                f"<text class='val' x='{x0 + bw / 2:.1f}' y='{Y(c) - 5:.1f}' text-anchor='middle'>{c}</text>"
                f"<text class='val' x='{x0 + bw * 1.5 + gap:.1f}' y='{Y(a) - 5:.1f}' text-anchor='middle'>{a}</text>"
            )
    out.append(
        f"<line x1='{ml}' x2='{w - mr}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='var(--axis)' stroke-width='1'/></svg>"
    )
    rows = "".join(
        f"<tr><td>{d}</td><td class='num'>{c}</td><td class='num'>{a}</td></tr>"
        for d, c, a in days
    )
    out.append(
        "<details id='d-daily'><summary>Table view</summary><table><tr><th>Day (UTC)</th><th class='num'>Cleared"
        f"</th><th class='num'>Arrived</th></tr>{rows}</table></details>"
    )
    return "".join(out)


def merged_chart(cfg: dict, st: dict, days: list[tuple[str, int, list[int]]]) -> str:
    """rocKE PRs merged into develop per UTC day. One series, so no legend box: the
    heading names it. Slot 3 (aqua) is 2.74:1 on the light surface, so the latest day is labelled and a table view lists
    every day's PRs."""
    w, h, ml, mr, mt, mb, bw = 560, 180, 40, 12, 18, 26, 20
    peak = max([c for _, c, _ in days] + [1])
    step = _nice(peak)
    top = max(step, ((peak + step - 1) // step) * step)
    Y = lambda v: mt + (h - mt - mb) * (1 - v / top)
    band = (w - ml - mr) / len(days)
    out = [
        f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' aria-label='rocKE PRs merged per day'>"
    ]
    for g in range(0, top + 1, step):
        out.append(
            f"<line x1='{ml}' x2='{w - mr}' y1='{Y(g):.1f}' y2='{Y(g):.1f}' stroke='var(--grid)' stroke-width='1'/>"
            f"<text x='{ml - 6}' y='{Y(g) + 4:.1f}' text-anchor='end'>{g}</text>"
        )
    for i, (d, c, prs) in enumerate(days):
        out.append(_bar(ml + band * i + (band - bw) / 2, c, bw, Y, "--series-3"))
        if (
            len(days) - 1 - i
        ) % 2 == 0:  # every other day, ending today: the dates never collide
            out.append(
                f"<text x='{ml + band * i + band / 2:.1f}' y='{h - 6}' text-anchor='middle'>{d[5:]}</text>"
            )
        listed = " ".join(f"#{n}" for n in prs[:8]) + (" …" if len(prs) > 8 else "")
        tip = [["merged", str(c), "var(--series-3)"], ["day", d, None]] + (
            [["PRs", listed, None]] if prs else []
        )
        out.append(
            f"<rect x='{ml + band * i:.1f}' y='{mt}' width='{band:.1f}' height='{h - mt - mb}' "
            f"fill='transparent' tabindex='0' data-tip='{_tip(tip)}'/>"
        )
    last = days[-1][1]
    out.append(
        f"<text class='val' x='{ml + band * (len(days) - 1) + band / 2:.1f}' y='{Y(last) - 5:.1f}' "
        f"text-anchor='middle'>{last}</text>"
    )
    out.append(
        f"<line x1='{ml}' x2='{w - mr}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='var(--axis)' stroke-width='1'/></svg>"
    )
    rows = "".join(
        f"<tr><td class='nw'>{d}</td><td class='num'>{c}</td>"
        f"<td>{''.join(_chip(cfg, st, n) for n in prs) or '—'}</td></tr>"
        for d, c, prs in reversed(days)
    )
    out.append(
        "<details id='d-merged'><summary>Table view (newest first)</summary><table><tr><th>Day (UTC)</th>"
        f"<th class='num'>Merged</th><th>PRs</th></tr>{rows}</table></details>"
    )
    return "".join(out)


# The merge-queue gates in plain words: gate, name, what it checks, passes when, who fixes.
GATES = (
    (
        "G1",
        "Our review",
        "Our automated review of the PR's current version",
        "our verdict at this exact head is 🟢 or 🟡 (a rebase-only push keeps it)",
        "the author fixes our 🔴/🟠 items; we re-review every substantive push",
    ),
    (
        "G2",
        "Approvals",
        "GitHub approvals, code-owner rules and review threads",
        "the required code owners approved, nobody requests changes, every review thread is resolved, and GitHub does not "
        "block the merge",
        "reviewers and code owners; the author answers and resolves threads",
    ),
    (
        "G3",
        "CI",
        "GitHub CI on the PR's head",
        "every check is green",
        "the author, unless the failure is develop's (our review says which)",
    ),
    (
        "G4",
        "Fresh merge",
        "Does the PR merge cleanly into today's develop?",
        "no conflicts",
        "the author rebases",
    ),
    (
        "G5",
        "Goldens",
        "Stored expected outputs (representative golden, differential golden, parity) after merging "
        "with today's develop",
        "nothing changes unexpectedly",
        "the author re-blesses on current develop",
    ),
    (
        "G6",
        "Build and engines",
        "Native build, ctest, C++/Python byte identity of the touched kernel families and the "
        "PR's own tests, on that merge",
        "everything passes",
        "the author",
    ),
    (
        "G7",
        "Compile per target",
        "Every touched kernel configuration, compiled with ROCm clang for its own GPU target",
        "nothing that develop compiles is newly rejected",
        "the author",
    ),
    (
        "G8",
        "Dependencies and labels",
        "Not stacked on another open PR, no hold label (DO NOT MERGE, WIP, on hold, "
        "Not ready), queue not frozen",
        "nothing keeps it back",
        "the author removes the label; the base PR lands; develop gets " "fixed",
    ),
    (
        "G9",
        "ABI lane",
        "If the PR bumps an ABI counter: is it next in line, claiming develop + 1?",
        "it is the lane head with the right number",
        "the author renumbers when it is their turn",
    ),
)


# --- small pieces -----------------------------------------------------------------------------------------------------

STATUS = {"good": "✓", "warning": "…", "serious": "!", "critical": "✗"}
CI_STATUS = {
    "green": ("good", "green"),
    "success": ("good", "green"),
    "red": ("critical", "red"),
    "failure": ("critical", "red"),
    "red-other": ("serious", "red outside rocKE"),
    "infra": ("warning", "infra failure, tests never ran"),
}


def _status(level: str, text: str, icon: str | None = None) -> str:
    """A status always ships with an icon and a label, never color alone."""
    return (
        f"<span class='st'><span class='dot' style='background:var(--{level})'></span>"
        f"{STATUS[level] if icon is None else icon} {E(text)}</span>"
    )


def _pill(level: str, text: str, icon: str | None = None) -> str:
    return (
        f"<span class='pill'><span class='dot' style='background:var(--{level})'></span>"
        f"{STATUS[level] if icon is None else icon} {E(text)}</span>"
    )


def _ci(v: str) -> tuple[str, str]:
    return CI_STATUS.get(v, ("warning", v))


def _chip(cfg: dict, st: dict, n: int, extra: str = "", ready: bool = False) -> str:
    """A PR's chip. The user's priorities wear a green border wherever they show. `ready`: a top priority ready to merge before the waves ahead of it (mergeplan.ready_early) gets a small marker."""
    rec = st["prs"].get(str(n)) or {}
    title = f"{rec.get('title', '')} ({rec.get('author', '?')})"
    prio = str(n) in (st.get("user_priority") or {}) or any(
        w.startswith(USER_PRIORITY)
        for w in (st.get("develop_fixes") or {}).get(str(n)) or []
    )
    mark = "<span class='chip-ready-early'>ready</span>" if ready else ""
    tip = (
        title
        + (" · your priority" if prio else "")
        + (" · ready to merge now, before its wave: your call" if ready else "")
    )
    return (
        f"<a class='chip{' chip-prio' if prio else ''}' href='{derived.github_address(cfg['repo'])}/pull/{n}' "
        f"title='{E(tip, quote=True)}'>#{n}{E(extra)}{mark}</a>"
    )


def _issue(cfg: dict, n: int) -> str:
    """A chip for develop issue n, linked to it in the develop issues' repository."""
    where = derived.github_address(issue_repository(cfg))
    return f"<a class='chip' href='{where}/issues/{n}'>#{n}</a>"


def _md(s: str) -> str:
    """Inline markdown from the insights report: escaped text, then **bold** and `code`."""
    return re.sub(
        r"`([^`]+)`", r"<code>\1</code>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", E(s))
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def conflict_files_html(cfg: dict, st: dict, hot: dict) -> str:
    """The files PRs conflicted on in the last 7 days, with the develop PRs that last
    changed each (the collector's cached `git log`), then by directory. Paths are untrusted: escaped.
    """
    if not hot["files"]:
        return (
            f"<p class='sub'>No merge conflict recorded in the last {conflicts.DAYS} days "
            f"({E(conflicts.tracked(hot))}).</p>"
        )
    by = ((st.get("conflicts") or {}).get("offenders") or {}).get("by") or {}
    rows = "".join(
        f"<tr><td><code title='{E(f['path'], quote=True)}'>{E(f['short'])}</code></td>"
        f"<td class='num'>{len(f['prs'])}</td><td class='num'>{f['episodes']}</td>"
        f"<td>{_chips(cfg, st, f['prs'], 8)}</td>"
        f"<td>{_chips(cfg, st, by.get(f['path']) or [], 3) or '<span class=muted>—</span>'}</td></tr>"
        for f in hot["files"]
    )
    drows = "".join(
        f"<tr><td><code>{E(d['dir'])}/</code></td><td class='num'>{d['prs']}</td>"
        f"<td class='num'>{d['files']}</td><td class='num'>{d['episodes']}</td></tr>"
        for d in hot["dirs"]
    )
    return (
        f"<div class='wrap'><table><tr><th>File</th><th class='num'>PRs</th><th class='num'>Episodes</th>"
        f"<th>Which PRs</th><th>Last changed on develop by</th></tr>{rows}</table></div>"
        f"<div class='wrap'><table><tr><th>Directory</th><th class='num'>PRs</th><th class='num'>Files</th>"
        f"<th class='num'>Episodes</th></tr>{drows}</table></div>"
        f"<p class='sub'>Live: open PRs GitHub reports conflicting, merged with develop here. Resolved: develop "
        f"merges in which the author resolved a conflict ({hot['merges_conflicted']} of {hot['merges']} develop "
        f"merges on PR branches); a rebase is seen only while it conflicts. An episode is one PR head conflicting "
        f"on the file, or one merge resolving it. "
        f"Tracked {E(conflicts.tracked(hot).removeprefix('tracked '))}.</p>"
    )


def ci_times_html(cfg: dict, st: dict, s: dict) -> str:
    """CI time per PR for the last 7 days: push → CI result, the runner queue wait, by
    workflow, the slowest PRs, jobs and runner labels. Workflow, job and runner names are escaped.
    """
    if not s["overall"]:
        return (
            f"<p class='sub'>No finished CI recorded for heads pushed in the last {ci_times.DAYS} days yet (heads "
            "are read hourly).</p>"
        )
    o, w = s["overall"], s["job_wait"]
    h, m = ci_times.hours, ci_times.minutes
    line = (
        f"Push → CI result: median {h(o['median'])} · mean {h(o['mean'])} · p90 {h(o['p90'])} · n={o['n']} heads"
        + (
            f" · runner queue wait median {m(w['median'])}, p90 {m(w['p90'])} ({w['n']} jobs)"
            if w
            else ""
        )
        + f" · cut short by a newer push or a cancel: {s['cut']} of {s['heads']} heads "
        f"({100 * s['cut_share']:.0f}%) · {s['reruns']} finished heads needed a re-run attempt"
    )
    wrows = "".join(
        f"<tr><td>{E(x['name'])}</td><td class='num'>{ci_times.dur(x['median_s'])}</td>"
        f"<td class='num'>{ci_times.dur(x['p90_s'])}</td><td class='num'>{x['n']}</td></tr>"
        for x in s["workflows"]
    )
    prows = "".join(
        f"<tr><td class='nw'>{_chip(cfg, st, p['pr'])}</td>"
        f"<td class='nw'>{E((st['prs'].get(str(p['pr'])) or {}).get('author', ''))}</td>"
        f"<td class='num'>{ci_times.dur(p['mean_s'])}</td><td class='num'>{ci_times.dur(p['max_s'])}</td>"
        f"<td class='num'>{p['n']} / {p['heads']}</td></tr>"
        for p in s["prs"]
    )
    jrows = "".join(
        f"<tr><td>{E(j['name'][:110])}</td><td class='num'>{m(j['median_s'])}</td>"
        f"<td class='num'>{j['n']}</td></tr>"
        for j in s["jobs"]
    )
    lrows = "".join(
        f"<tr><td>{E(x['label'])}</td><td class='num'>{m(x['median_s'])}</td>"
        f"<td class='num'>{m(x['p90_s'])}</td><td class='num'>{x['n']}</td></tr>"
        for x in s["labels"]
    )
    return (
        f"<p class='sub'>{E(line)}</p><div class='grid2'><div class='wrap'><table><tr><th>Workflow</th>"
        f"<th class='num'>Median</th><th class='num'>p90</th><th class='num'>Runs</th></tr>{wrows}</table></div>"
        f"<div class='wrap'><table><tr><th>Slowest PRs</th><th>Author</th><th class='num'>Mean</th>"
        f"<th class='num'>Longest</th><th class='num' title='finished / all heads'>Heads</th></tr>{prows}"
        "</table></div></div>"
        + (
            f"<div class='grid2'><div class='wrap'><table><tr><th>Slowest jobs</th><th class='num'>Median</th>"
            f"<th class='num'>Runs</th></tr>{jrows}</table></div><div class='wrap'><table><tr><th>Runner label</th>"
            f"<th class='num'>Queue wait median</th><th class='num'>p90</th><th class='num'>Jobs</th></tr>{lrows}"
            "</table></div></div>"
            if jrows or lrows
            else ""
        )
        + "<p class='sub'>Push → result: the first workflow run created to the last one completed, re-run "
        "attempts included (pull_request runs; bots, Copilot reviews and later re-triggers left out). The times "
        "count every head whose CI ran to the end, also when a newer push came meanwhile; a cancelled CI counts "
        "only in the cut-short share. Jobs: the latest attempt of each head's two longest workflows.</p>"
    )


def insights_card(cfg: dict, st: dict, store) -> str:
    """Insights from recent PRs: a one-line summary, then a collapsed section with our
    recent 🔴/🟠 findings by cause and what prevents each, the open rocKE develop issues by cause, the merge-conflict hot
    files and CI time per PR of the last 7 days, and the tables of the newest insights
    report. Finding and issue texts are untrusted: escaped, and shown only as chip tooltips.
    """
    fs = insights.findings(store)
    causes = insights.by_cause(fs)
    fixed = sum(f["fixed"] for f in fs)
    end = dt.datetime.now(dt.timezone.utc)
    hot = conflicts.hot_files(
        store,
        end - dt.timedelta(days=conflicts.DAYS),
        end,
        rocke_path=paths.repository_folder(cfg),
    )
    ci = ci_times.summary(store, end - dt.timedelta(days=ci_times.DAYS), end)
    head = f"Last {insights.DAYS} days: {_plural(len(fs), '🔴/🟠 finding')} on {_plural(len({f['pr'] for f in fs}), 'PR')}" f" · {fixed} confirmed fixed before merge" + (
        " · most common: " + ", ".join(f"{r['cause']} ({r['n']})" for r in causes[:3])
        if causes
        else ""
    ) + (
        f" · CI median {ci_times.hours(ci['overall']['median'])} push→result"
        if ci["overall"]
        else ""
    ) + (
        f" · top conflict file: {hot['files'][0]['path'].rsplit('/', 1)[-1]} "
        f"({_plural(len(hot['files'][0]['prs']), 'PR')})"
        if hot["files"]
        else ""
    )
    url = derived.github_address(cfg["repo"])
    issues_url = derived.github_address(issue_repository(cfg))

    def pr_chips(prs: dict) -> str:
        return (
            "<div class='chips'>"
            + "".join(
                f"<a class='chip' href='{url}/pull/{n}' title='{E(' · '.join(t), quote=True)}'>#{n}</a>"
                for n, t in sorted(prs.items())
            )
            + "</div>"
        )

    frows = "".join(
        f"<tr><td>{E(r['cause'])}</td><td class='num'>{r['red']}</td><td class='num'>{r['orange']}</td>"
        f"<td>{pr_chips(r['prs'])}</td><td class='num'>{r['fixed']} of {r['n']}</td>"
        f"<td class='num'>{r['last7']} ({r['prior7']})</td><td>{_md(r['practice'])}</td></tr>"
        for r in causes
    )
    dev = insights.develop(cfg, st)
    drows = "".join(
        f"<tr><td>{E(r['cause'])}</td><td><div class='chips'>"
        + "".join(
            f"<a class='chip' href='{issues_url}/issues/{i['n']}' title='{E(i['title'], quote=True)}'>"
            f"#{i['n']}</a>"
            for i in r["issues"]
        )
        + f"</div></td><td class='num'>{sum(entry_kind(cfg, i) == RED for i in r['issues'])}</td>"
        f"<td class='num'>{sum(entry_kind(cfg, i) == STABILITY for i in r['issues'])}</td></tr>"
        for r in dev
    )
    rep = insights.latest_report(cfg)
    tables = insights.report_tables(rep) if rep else []
    rhtml = "".join(
        f"<h3>{_md(h)}</h3><div class='wrap'><table><tr>"
        + "".join(f"<th>{_md(c)}</th>" for c in rows[0])
        + "</tr>"
        + "".join(
            "<tr>" + "".join(f"<td>{_md(c)}</td>" for c in r) + "</tr>"
            for r in rows[1:]
        )
        + "</table></div>"
        for h, rows in tables
        if rows
    )
    return (
        f"<p class='sub'>{E(head)}</p><details id='d-insights'><summary>Show the insights</summary>"
        "<h3>Our 🔴/🟠 findings by cause <span class='c'>hover a PR for its findings</span></h3>"
        + (
            f"<div class='wrap'><table><tr><th>Cause</th><th class='num' title='blocking'>🔴</th>"
            f"<th class='num' title='should-fix'>🟠</th><th>PRs</th><th class='num'>Fixed before merge</th>"
            f"<th class='num'>Last 7 d (prior 7)</th><th>What prevents it</th></tr>{frows}</table></div>"
            "<p class='sub'>A finding's cause comes from keywords in its title and impact (first match wins), so "
            "each finding counts once. Fixed: a later review of ours confirmed it fixed.</p>"
            if causes
            else f"<p class='sub'>No 🔴/🟠 findings posted in the last {insights.DAYS} days.</p>"
        )
        + "<h3>Open rocKE develop issues by cause</h3>"
        + (
            f"<table><tr><th>Cause</th><th>Issues</th><th class='num'>CI red ({E(kind_label(cfg, RED))})</th>"
            f"<th class='num'>CI does not see it ({E(kind_label(cfg, STABILITY))})</th></tr>{drows}</table>"
            if dev
            else "<p class='sub'>None open.</p>"
        )
        + f"<h3>Merge-conflict hot files <span class='c'>last {conflicts.DAYS} days, most PRs first</span></h3>"
        + conflict_files_html(cfg, st, hot)
        + f"<h3>CI time per PR <span class='c'>heads pushed in the last {ci_times.DAYS} days</span></h3>"
        + ci_times_html(cfg, st, ci)
        + pr_stats_html(cfg, st, store)
        + (
            f"<details id='d-insights-report'><summary>From the latest insights report ({E(rep.name)})</summary>"
            f"{rhtml}<p class='sub'>Full report: {E(str(rep))}</p></details>"
            if rep
            else ""
        )
        + "</details>"
    )


def pr_stats_html(cfg: dict, st: dict, store) -> str:
    """Where the time goes, per PR (insights.pr_statistics): each category's
    overall figures and its top PRs, and the analyst's reading when it is on and passed its check
    (rkmon_lib/analyst.py). The runner labels are in the CI section above. Advisory: a failure shows as a line.
    """
    from . import analyst

    try:
        stats = insights.pr_statistics(cfg, st, store)
    except Exception as exc:  # noqa: BLE001 - the page never breaks on the statistics
        return f"<p class='sub'>Per-PR statistics not computed: {E(str(exc)[:160])}</p>"
    got = analyst.reading(st)
    titles = {c["key"]: c["title"] for c in stats["categories"]}
    out = [
        f"<h3>Where the time goes, per PR <span class='c'>last {stats['days']} days, the top "
        f"{insights.TOP} of each</span></h3>"
    ]
    if got:
        items = "".join(
            f"<li>{E(titles.get(r['category'], r['category']))}"
            + (f" {_chip(cfg, st, r['pr'])}" if r.get("pr") else "")
            + f": {E(r['note'])}</li>"
            for r in got["ranking"]
        )
        out.append(
            f"<p class='sub'><b>The analyst's reading</b> ({E(str(got.get('model')))}, {E(str(got.get('at')))}; "
            f"every number in it is in the statistics): {E(got['summary'])}</p><ul>{items}</ul>"
        )
    cells = []
    for c in stats["categories"]:
        overall = ", ".join(
            f"{k.replace('_', ' ')} {v}" for k, v in c["overall"].items()
        )
        rows = "".join(
            f"<tr><td class='nw'>{_chip(cfg, st, r['pr'])}</td><td class='num'>{r['value']}</td>"
            f"<td>{E(r['detail'])}</td></tr>"
            for r in c["top"]
        )
        cells.append(
            f"<div class='wrap'><p class='sub'><b>{E(c['title'])}</b>: {E(c['what'])}. Overall: "
            f"{E(overall)}.</p>"
            + (
                f"<table><tr><th>PR</th><th class='num'>Value</th><th>Detail</th></tr>{rows}</table>"
                if rows
                else "<p class='sub'>None in the window.</p>"
            )
            + "</div>"
        )
    return "".join(out) + "<div class='grid2'>" + "".join(cells) + "</div>"


def usage_tiles(cfg: dict, store) -> str:
    """The models' tokens over the last 7 days (rkmon_lib/usage.py; tokens only, never a cost): Claude's per day and per
    review, the reviews posted per day, and the reviewer model's per day. Advisory: a failure shows as a line.
    """
    from . import usage

    try:
        t = usage.tile(cfg, store)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the page never breaks on the usage records
        return f"<p class='sub'>Token usage not read: {E(str(exc)[:160])}</p>"
    days = len(t["days"]) or 1
    return (
        "<div class='kpis' role='group' aria-label='Token usage'>"
        + "".join(
            [
                _tile(
                    "Claude tokens a review (7 d)",
                    f"{t['per_review']:,}" if t["per_review"] is not None else "—",
                    f"{t['claude'] // days:,} a day · the orchestrator's session and its subagents",
                    sparkline([(d, c) for d, c, _p, _m in t["days"]]),
                ),
                _tile(
                    "Reviews posted a day (7 d)",
                    f"{t['posted'] / days:.1f}",
                    f"{t['posted']} in 7 days",
                    sparkline(
                        [(d, p) for d, _c, p, _m in t["days"]], color="--series-3"
                    ),
                ),
                _tile(
                    "Reviewer model tokens a day (7 d)",
                    f"{t['model'] // days:,}",
                    "drafts, the tick triage and the other roles on its endpoint",
                    sparkline(
                        [(d, m) for d, _c, _p, m in t["days"]], color="--series-2"
                    ),
                ),
            ]
        )
        + "</div>"
    )


def _chips(
    cfg: dict,
    st: dict,
    ns: list,
    limit: int = 12,
    extra=None,
    label: str = "",
    ready=frozenset(),
) -> str:
    """PR chips in one row; `label` (plain text) leads the row; the PRs in `ready` get the ready marker (_chip)."""
    out = "".join(
        _chip(cfg, st, n, (extra or {}).get(n, ""), n in ready) for n in ns[:limit]
    )
    more = (
        f"<span class='chip muted'>+{len(ns) - limit}</span>" if len(ns) > limit else ""
    )
    lead = f"<span class='muted'>{E(label)}</span>" if label else ""
    return f"<div class='chips'>{lead}{out}{more}</div>" if ns else ""


def _tile(label: str, value, note: str = "", spark: str = "", tip: str = "") -> str:
    title = f" title='{E(tip, quote=True)}'" if tip else ""
    return (
        f"<div class='card tile'{title}><div class='label'>{E(label)}</div><div class='value'>{E(str(value))}{spark}"
        f"</div><div class='note'>{E(note)}</div></div>"
    )


def _gs(r: dict, k: str) -> str:
    """A gate's state from a full queue row (`gates`) or a stored compact row (`gs`)."""
    if "gates" in r:
        return (r["gates"].get(k) or ["pending"])[0]
    return (r.get("gs") or {}).get(k, "pending")


def _gt(r: dict, k: str) -> str:
    return (r.get("gates") or {}).get(k, [None, ""])[1] or _gs(r, k)


def _ago(ts: str | None) -> str:
    return report._ago_min(minutes_since(ts)) if ts else "?"


# --- the pipeline ------------------------------------------------------------------------------------------------------


def pipeline(cfg: dict, st: dict, store, plan: list[dict], q: list[dict]) -> list[dict]:
    """Each open PR in the stage it waits in, left to right. In review counts every review in flight by its part
    (report.in_review: the same rows as `rk review-status`); a stage's `parts` and `more` are its count lines.
    Mergeable Candidates are the PRs that passed CI and can merge now (report.mergeable); an approved candidate with a
    clean review whose CI still runs, fails on known infrastructure or fails on develop too is only counted there
    ("+N waiting on CI"), and one whose CI fails on its own goes to Needs Work."""
    flight = report.in_review(cfg, st, store)
    in_review = [r["pr"] for r in flight]
    review_q = [
        p["pr"]
        for p in plan
        if p["kind"] in report.REVIEW_KINDS and p.get("pr") and p["pr"] not in in_review
    ]
    m = report.mergeable(st, q)
    ready, work = {r["n"] for r in m["rows"]}, set(m["work"])
    author, approval, cand = [], [], []
    for r in q:
        rec = st["prs"].get(str(r["n"])) or {}
        if rec.get("status") != "open" or r["n"] in in_review or r["n"] in review_q:
            continue
        g1 = gate_review(cfg, rec)
        if r["n"] in ready:
            cand.append(r["n"])
        elif r.get("stage") == "candidate":
            if r["n"] in work:
                author.append(r["n"])
        elif g1[0] == FAIL and g1[1].startswith("our verdict"):
            author.append(r["n"])
        elif g1[0] == PASS and _gs(r, "G2") != PASS:
            approval.append(r["n"])
    order = [r["n"] for r in m["rows"]]
    cand.sort(key=order.index)  # landing order
    merged = merges.series(st, store, 1)[-1][
        2
    ]  # the tile's and the chart's source: develop merges only
    return [
        {
            "id": "review",
            "label": "Review queue",
            "hint": "follow-ups and first rounds waiting to start",
            "prs": review_q,
        },
        {
            "id": "inreview",
            "label": "In review",
            "hint": report.in_review_hint(cfg),
            "prs": in_review,
            "parts": report.in_review_counts(flight),
            "extra": {r["pr"]: f" {r['part']}" for r in flight},
        },
        {
            "id": "author",
            "label": "Needs Work",
            "hint": "our 🔴/🟠 stands at head, or CI fails because of the PR: waiting for a fix",
            "prs": author,
        },
        {
            "id": "approval",
            "label": "Needs Approval",
            "hint": "our review is clean at head; approvals missing",
            "prs": approval,
        },
        {
            "id": "candidate",
            "label": "Mergeable Candidates",
            "hint": "passed CI and can merge now: batched and proposed in landing order",
            "prs": cand,
            "more": (report.not_mergeable_line(m), report.not_mergeable_detail(m)),
        },
        {
            "id": "merged",
            "label": "Merged today",
            "hint": "landed on develop (UTC day)",
            "prs": merged,
        },
    ]


# --- the merge plan ----------------------------------------------------------------------------------------------------

GRID_PRS = 20  # PRs in the conflict grid, in plan order


def _grid_cell(mp: dict, a: int, b: int) -> str:
    """One cell of the conflict grid: ✗ textual conflict on develop, Ⓐ ABI conflict (both bump one ABI counter: the
    later renumbers; its own mark in the grid), both marks
    when both hold, ◆ care, ? not checked yet, · clean. Marks with something to say are focusable and carry their
    files or reasons as a tooltip (escaped: paths are untrusted)."""
    if a == b:
        return "<td class='g diag'></td>"
    e = mp["edges"].get(f"{min(a, b)}-{max(a, b)}") or {}
    pair = f"#{a} × #{b}"
    marks, tip = [], []
    if "x" in e:
        files = e["x"]
        marks.append(("?", "pending") if files is None else ("✗", "fail"))
        tip.append(
            [
                pair,
                (
                    "? not checked yet: counts as a conflict"
                    if files is None
                    else "✗ conflict on develop"
                ),
                None,
            ]
        )
        if files:
            tip.append(
                ["", mergeplan.some([f.rsplit("/", 1)[-1] for f in files], 4), None]
            )
    if e.get("abi"):
        marks.append(("Ⓐ", "abi"))
        tip += [
            ["" if tip else pair, f"Ⓐ {mergeplan.ABI_LABEL[lane]}", None]
            for lane in e["abi"]
        ]
    if not marks and "care" in e:
        marks.append(("◆", "care"))
        tip.append([pair, "◆ care: no conflict", None])
    if not marks:
        return "<td class='g pending'>·</td>"
    if e.get("care"):
        tip.append(["", "; ".join(e["care"])[:200], None])
    glyphs = marks[0][0] + "".join(
        f"<span class='{kind}'>{glyph}</span>" for glyph, kind in marks[1:]
    )
    return (
        f"<td class='g {marks[0][1]}{' two' if len(marks) > 1 else ''}' tabindex='0' data-tip='{_tip(tip)}'>"
        f"{glyphs}</td>"
    )


def decide_buttons(route: str, n: int) -> str:
    """The admin dashboard's Approve and Decline of a decision on PR n: PRIO_JS posts a click to the admin service's
    /<route>-approve or /<route>-decline (a pending priority request's, or a Needs-you line's:
    report.DECIDED_ON_THE_PAGE). Their holder is data-local-only: they show once the page reaches the service.
    """
    return " ".join(
        f"<button type='button' class='prio-act' data-route='{route}' data-act='{act}' data-pr='{n}'>{label}</button>"
        for act, label in (("approve", "Approve"), ("decline", "Decline"))
    )


def priority_box(cfg: dict, st: dict, variant: str = PUBLISHED) -> str:
    """The "Request PR priority" box at the top of the merge plan (a PR number and a button: the PR joins the priority
    list, for its reviews and its merge, unless it is there already), and the priority requests other teams made (their
    clicks wait for the user's OK). It starts off, with a note, until PRIO_JS turns it on. The LOCAL variant (the admin
    dashboard) prioritizes through the local service and has Approve / Decline on the list; only it names the service's
    port and address, and it carries no secret. The PUBLISHED variant asks: a PR number, a name and a reason, for the
    web host. Requesters' names and reasons are untrusted: escaped.
    """
    me = E(config.operator_login(cfg) or "the monitor owner")
    local = variant == LOCAL
    items = priority.shown(st)
    rows = "".join(
        f"<tr><td class='nw'>{_chip(cfg, st, it['pr'])}</td><td>{E(priority.requesters(it))}</td>"
        f"<td>{E(' / '.join([it['why']] + [a['why'] for a in it.get('also') or []]))}</td>"
        f"<td class='nw'>{E(_ago(it['at']))} ago</td><td class='prio-st'>{E(priority.status_text(it))}</td>"
        + (
            "<td class='nw' data-local-only hidden>"
            + (
                decide_buttons("priority", it["pr"])
                if it["status"] == "pending"
                else ""
            )
            + "</td>"
            if local
            else ""
        )
        + "</tr>"
        for it in items
    )
    pend = sum(it["status"] == "pending" for it in items)
    home = webserve.home_url(cfg)
    service = (
        f" data-port='{webserve.port(cfg)}' data-home='{E(home)}'" if local else ""
    )
    ask = (
        ""
        if local
        else "<span class='prio-ask' hidden><label for='prio-who'>Your name</label>"
        f"<input id='prio-who' type='text' maxlength='{priority.WHO_MAX}' placeholder='who asks' disabled>"
        f"<label for='prio-why'>Why</label><input id='prio-why' type='text' maxlength='{priority.WHY_MAX}' "
        "placeholder='why it should go first' disabled></span>"
    )
    note = (
        f"Priority requests work here while {me}'s monitor service runs (tools/rkweb.py): the admin dashboard is "
        f"{E(home)}."
        if local
        else f"Ask for a PR to be reviewed and merged first: {me} approves or declines each request."
    )
    return (
        f"<form id='prio' class='prio'{service} data-me='{me}' autocomplete='off' aria-label='Request PR priority'>"
        "<div class='prio-row'><label for='prio-n'>PR number</label>"
        f"<input id='prio-n' type='number' min='1' max='{webserve.PR_MAX - 1}' step='1' placeholder='e.g. 101' "
        f"required disabled>{ask}<button id='prio-go' type='submit' disabled>Request PR priority</button></div>"
        "<p class='sub'>A priority PR is reviewed first and merged first (from the next tick, wave 1 of the plan "
        "unless it conflicts with a higher priority), "
        f"as <code>rk prioritize</code> does; a PR already in the priority queue stays as it is. {me} approves or "
        "declines each request.</p>"
        f"<p id='prio-status' class='sub' role='status' aria-live='polite'>{note}</p>"
        f"<h3>Priority requests <span class='c'>{pend} pending · {me} approves or declines</span></h3>"
        + (
            "<div class='wrap'><table id='prio-list'><tr><th>PR</th><th>Requested by</th><th>Why</th><th>When</th>"
            "<th>Status</th>"
            + ("<th data-local-only hidden>Decide</th>" if local else "")
            + f"</tr>{rows}</table>"
            "</div>"
            if items
            else "<p class='sub'>None right now.</p>"
        )
        + "</form>"
    )


def standby_line() -> str:
    """The admin dashboard's top line while the monitor is in standby: the doctor's standby row
    (doctor.standby_checks), what it found and its fix; "" while standby is off."""
    # a call-time import: the doctor reads much the page does not need
    from . import doctor

    return "".join(
        f"<p class='standby' role='status'>{_status('warning', f'{c.name}: {c.found}', '⚠')} · "
        f"{E(FIX_LABEL + str(c.fix))}</p>"
        for c in doctor.standby_checks()
    )


def tickets_shown(cfg: dict, st: dict) -> bool:
    """The admin dashboard shows the Tickets card while found-issue-tickets is on, or any proposed ticket is on it."""
    from . import features, tickets

    return features.on(cfg, tickets.FEATURE) or bool(tickets.shown(st))


def tickets_card(cfg: dict, st: dict) -> str:
    """The "Tickets waiting for you" card (built like the priority-request card), on the admin dashboard only: one row
    per proposed ticket (a ticket waiting for your Approve), with its title, type, priority, assignee,
    one-line summary, a link to its full text, its state, and Approve / Decline once the service answers (the priority
    box's ping). Approve files it through the atlassian CLI (its key then shows in the row), Decline drops it; one action
    per proposed ticket. While ticket creation does not send (prototype, or the monitor in standby: tickets.held_words),
    the card says that Approve keeps the proposed ticket as an unsent item instead, listed under "Not sent"
    (not_sent_card). Every value is escaped: the monitor's session writes the proposed tickets, from what it found.
    """
    from . import features, tickets

    if not tickets_shown(cfg, st):
        return ""
    me = E(config.operator_login(cfg) or "the monitor owner")
    home = webserve.home_url(cfg)
    items = tickets.shown(st)

    def state(it: dict) -> str:
        text = E(tickets.status_text(cfg, it))
        if it["status"] == "filed" and it.get("url", "").startswith(
            ("https://", "http://")
        ):
            text = (
                f"filed as <a href='{E(it['url'])}' target='_blank' rel='noopener noreferrer'>{E(it['key'])}</a>"
                + (f"; {E(it['note'])}" if it.get("note") else "")
            )
        return text

    rows = "".join(
        f"<tr><td>{E(it['title'])}</td><td class='nw'>{E(it['type'])}</td><td class='nw'>{E(it['priority'] or '—')}</td>"
        f"<td class='nw'>{E(it['assignee'] or 'unassigned')}</td><td>{E(it['summary'])}</td>"
        f"<td class='nw'><a href='{E(home)}ticket-draft?id={E(it['id'])}' target='_blank' rel='noopener noreferrer'>"
        "full text</a></td>"
        f"<td class='tk-st'>{state(it)}</td><td class='nw' data-local-only hidden>"
        + (
            f"<button type='button' class='tk-act' data-act='approve' data-id='{E(it['id'])}'>Approve</button> "
            f"<button type='button' class='tk-act' data-act='decline' data-id='{E(it['id'])}'>Decline</button>"
            if it["status"] == "waiting"
            else ""
        )
        + "</td></tr>"
        for it in items
    )
    n = len(tickets.waiting(st))
    notes = [
        "Proposed tickets, for problems the monitor found along the way (not breakages, not review findings): Approve "
        "files the ticket in JIRA through the atlassian CLI, and its key shows here; Decline drops it. One action per "
        "proposed ticket."
    ]
    gone = (
        features.missing(cfg, tickets.FEATURE)
        if features.on(cfg, tickets.FEATURE)
        else ["out-of-scope issue tickets are off"]
    )
    held = tickets.held_words(cfg)
    if gone:
        notes.append(
            "Approve files nothing while it lacks: " + E("; ".join(gone)) + "."
        )
    elif held:
        notes.append(
            f"Approve files nothing now, because {E(held)}. It keeps the proposed ticket as an unsent item, listed "
            "under <a href='#not-sent'>Not sent</a>."
        )
    return (
        f"<section id='tickets' class='card'><h2>Tickets waiting for you <span class='c'>{n} waiting · {me} "
        "approves or declines each</span></h2>"
        + "".join(f"<p class='sub'>{x}</p>" for x in notes)
        + "<p id='tk-status' class='sub' role='status' aria-live='polite'></p>"
        + (
            "<div class='wrap'><table id='tk-list'><tr><th>Title</th><th>Type</th><th>Priority</th><th>Assignee</th>"
            f"<th>Summary</th><th>Text</th><th>Status</th><th data-local-only hidden>Decide</th></tr>{rows}</table>"
            "</div>"
            if items
            else "<p class='sub'>None right now.</p>"
        )
        + "</section>"
    )


# The "Not sent" card lists the newest this many unsent items; the state folder keeps every one.
UNSENT_SHOWN = 100


def unsent_file(cfg: dict, store, item: dict) -> pathlib.Path | None:
    """The file that holds unsent ITEM's content, which the "Not sent" card links to and the admin dashboard's service
    serves (webserve.UNSENT_ROUTE): a review kept unposted, a Markdown file in the outputs folder (post.posted_dir), or
    the team dashboard's page, the published copy in the state folder (PUBLISHED_FILE); None for any other item, a file
    elsewhere, a link, or a file that is gone."""
    from . import post, state_store

    if not item.get("file"):
        return None
    p = pathlib.Path(str(item["file"]))
    if item.get("type") == state_store.REVIEW:
        ours = p.suffix == ".md" and p.parent == post.posted_dir(cfg)
    else:
        ours = item.get("type") == state_store.PAGE and p == store.dir / PUBLISHED_FILE
    return p if ours and not p.is_symlink() and p.is_file() else None


def team_page(cfg: dict, store) -> pathlib.Path | None:
    """The team dashboard's page as the last tick made it, the published copy in the state folder (PUBLISHED_FILE),
    while the team dashboard is on and does not send (features.held: in prototype, or the monitor in standby), in
    either delivery, "publish" or "serve": it is made after every tick and shown on the admin dashboard,
    so the "Not sent" card links it and the admin dashboard's service serves it (unsent_named), whether an unsent item
    was kept for it or not (served here, none is). None otherwise, for a link, and before a tick has made it.
    """
    from . import features

    p = store.dir / PUBLISHED_FILE
    shown = features.enabled(cfg, "dashboards.team") and features.held(
        cfg, "dashboards.team"
    )
    return p if shown and not p.is_symlink() and p.is_file() else None


def unsent_named(cfg: dict, store, name: str) -> tuple[pathlib.Path, str] | None:
    """(the file, its item's type) of the newest unsent item whose file (unsent_file) is named NAME, else of the team
    dashboard's page while it does not send (team_page), as the page item's type; None when there is neither.
    """
    from . import state_store

    for item in reversed(state_store.unsent_items(store.dir)):
        p = unsent_file(cfg, store, item)
        if p is not None and p.name == name:
            return p, item["type"]
    page = team_page(cfg, store)
    return (page, state_store.PAGE) if page is not None and page.name == name else None


def not_sent_shown(cfg: dict, items: list) -> bool:
    """The admin dashboard shows the "Not sent" card while any unsent item ITEMS is kept, or while one may be: the
    monitor in standby, or a feature that sends (features.SENDS) on and in prototype."""
    from . import features

    return (
        bool(items)
        or features.standby()
        or any(
            features.enabled(cfg, f) and not features.live(cfg, f)
            for f in features.SENDS
        )
    )


def not_sent_card(cfg: dict, store, items: list) -> str:
    """The "Not sent" card, on the admin dashboard only, while not_sent_shown: the unsent items ITEMS
    (state_store.unsent_items: what a feature in prototype, or the monitor in standby, kept instead of sending), newest
    first, the UNSENT_SHOWN newest, each with when it was kept, its type, what it was for, why (post.why_kept: its
    feature in prototype, or the monitor in standby) and its content as it would have gone out; an item whose content
    is a file (a review kept unposted, the team dashboard's page) links to it (unsent_file). While the team dashboard
    does not send, a note links its page as the last tick made it, in either delivery (team_page). Every value is
    escaped: the monitor wrote them from what it read."""
    from . import features, post, state_store

    if not not_sent_shown(cfg, items):
        return ""
    link = webserve.home_url(cfg) + webserve.UNSENT_ROUTE.lstrip("/") + "?file="

    def when(it: dict) -> str:
        try:
            return f"{_ago(it.get('at'))} ago"
        except (TypeError, ValueError):
            return str(it.get("at"))

    def file_link(p: pathlib.Path) -> str:
        return (
            f"<a href='{E(link + urllib.parse.quote(p.name))}' target='_blank' rel='noopener noreferrer'>"
            f"{E(p.name)}</a>"
        )

    def content(it: dict) -> str:
        p = unsent_file(cfg, store, it)
        out = (
            file_link(p)
            if p is not None
            else E(pathlib.PurePath(str(it.get("file") or "")).name)
        )
        text = str(it.get("content") or "")
        return out + (f"<pre class='unsent'>{E(text)}</pre>" if text else "")

    rows = "".join(
        f"<tr><td class='nw'>{E(when(it))}</td><td class='nw'>{E(str(it.get('type') or ''))}</td>"
        f"<td>{E(str(it.get('for') or ''))}</td>"
        f"<td>{E(post.why_kept(it.get('feature'), it.get('why')))}</td><td>{content(it)}</td></tr>"
        for it in items[::-1][:UNSENT_SHOWN]
    )
    notes = [
        "What a feature in prototype, or the monitor in standby, would have sent, kept on this host instead, newest "
        "first. None is ever sent later: going live sends what comes up from then on, and you can send an item's "
        "content yourself. Each is kept for the package's retention, then dropped."
    ]
    page = team_page(cfg, store)
    if page is not None:
        why = post.why_kept("dashboards.team", features.held(cfg, "dashboards.team"))
        notes.append(
            f"The team dashboard's page, made after every tick ({E(why)}): {file_link(page)}."
        )
    if len(items) > UNSENT_SHOWN:
        notes.append(
            f"The {UNSENT_SHOWN} newest of {len(items)}; the rest are in "
            f"{E(str(state_store.unsent_path(store.dir)))}, or its older copy beside it."
        )
    return (
        f"<section id='not-sent' class='card'><h2>Not sent <span class='c'>{len(items)} kept · never sent later"
        "</span></h2>"
        + "".join(f"<p class='sub'>{x}</p>" for x in notes)
        + (
            "<div class='wrap'><table id='unsent-list'><tr><th>When</th><th>Type</th><th>For</th><th>Why kept</th>"
            f"<th>Content</th></tr>{rows}</table></div>"
            if items
            else "<p class='sub'>None right now.</p>"
        )
        + "</section>"
    )


def plan_card(cfg: dict, st: dict, variant: str = PUBLISHED) -> str:
    """The merge plan (which PRs can land together without a conflict, and the merge order that saves the most rebases):
    the order and its waves with who each landing makes rebase and who acts,
    auto-merge that may land early, a conflict grid of the first GRID_PRS planned PRs (status glyphs with a legend, hover
    or focus a mark for its files or reasons, and a table view) with the ABI conflicts as their own mark, the ABI lanes
    in plan order, and every planned PR's rebase advice, collapsed. It opens
    with the "Request PR priority" box of the page's variant, whatever the plan's state (priority_box).
    """
    mp = st.get("merge_plan")
    head = (
        "<section id='plan' class='card'><h2>Merge plan <span class='c'>which PRs can land together without a "
        "conflict, in what order, and who rebases when</span></h2>"
        + priority_box(cfg, st, variant)
    )
    if not mp:
        return (
            head
            + "<p class='sub'>Not computed yet: the next tick computes it.</p></section>"
        )
    if mp.get("unavailable"):
        return head + (
            f"<p class='sub'>Not computed this tick: {E(mp['unavailable'])}. The queue orders and batches as "
            "before (head-to-head pairs).</p></section>"
        )
    nodes, dc = mp["nodes"], mp["dev_conflicts"]

    def extra(ns: list[int]) -> dict:
        return {
            n: f" {mergeplan.plan_cls(mp, n)}"
            + (" ✓" if (nodes.get(str(n)) or {}).get("go") else "")
            for n in ns
        }

    def later(ns: list[int]) -> str:
        if not ns:  # only PRs that merge with develop after a rebase: not known
            return "<span class='muted'>—</span>"
        xs = sorted(
            {
                x
                for n in ns
                for x in (nodes.get(str(n)) or {}).get("forces", [])
                if x not in ns
            },
            key=lambda x: mergeplan.plan_rank(mp, x),
        )
        return _chips(cfg, st, xs, 12, extra(xs)) or "<span class='muted'>none</span>"

    early = {
        int(k) for k in mp.get("ready_early") or {}
    }  # ready now, before the waves ahead (the user's call)

    def row(label: str, prs: list[int], tags: dict, rebase: str) -> str:
        return (
            f"<tr><td class='nw'>{E(label)}</td><td>{_chips(cfg, st, prs, 30, tags, ready=early)}</td>"
            f"<td>{rebase}</td><td>{E(report.plan_needs(cfg, st, prs)) or '—'}</td></tr>"
        )

    # A top priority that conflicts with develop ends its wave, "after its rebase", instead of sinking into "after a
    # rebase" with everyone else.
    wrows = "".join(
        row(
            label,
            ns + rb,
            {
                **extra(ns),
                **{n: f" {mergeplan.plan_cls(mp, n)}, after its rebase" for n in rb},
            },
            later(ns),
        )
        for label, ns, rb in mergeplan.wave_rows(mp)
    )
    last = mergeplan.rebase_last(mp)
    if last:
        wrows += row("after a rebase", last, extra(last), later([]))
    nx, nabi, ncare = report.plan_counts(mp)
    line = (
        f"On develop {mp['develop'][:10]}: {_plural(len(mp['scope']), 'near-merge PR')}, {len(nodes)} merge cleanly "
        f"with develop, {len(dc)} conflict with it (they rebase first)"
        + (f", {len(mp['unknown'])} not checked yet" if mp["unknown"] else "")
        + f". Between the clean ones: {_plural(nx, 'textual conflict')}, {_plural(nabi, 'ABI conflict')}, {ncare} "
        "more pairs to take care with"
        + (
            f"; {_plural(len(mp['unions']), 'union conflict')} (PRs that conflict only together)"
            if mp.get("unions")
            else ""
        )
        + "."
        + (
            f" Partial: {mp['partial']['pairs']} pairs and {mp['partial']['prs']} PRs not checked yet; they count as "
            "conflicts."
            + (
                f" {mergeplan.unchecked(mp).capitalize()}."
                if mergeplan.unchecked(mp)
                else ""
            )
            if mp.get("partial")
            else ""
        )
        + f" {mp['merges']} merges this tick ({mp['secs']} s)."
        + (f" {mergeplan.made_by_line(mp)}" if mergeplan.made_by_line(mp) else "")
    )
    am = mp.get("automerge") or {}
    amh = "".join(
        f"<li>{_status('serious', 'ahead of the plan', '!') + ' ' if a.get('alert') else ''}"
        f"{_chip(cfg, st, int(k))} {E(a['note'])}</li>"
        for k, a in sorted(am.items(), key=lambda kv: int(kv[0]))
    )
    order = [n for w in mp["waves"] for n in w][:GRID_PRS]
    grid = (
        "<tr><th></th>"
        + "".join(
            f"<th class='v' title='{E((st['prs'].get(str(n)) or {}).get('title', ''), quote=True)}'>"
            f"{n}</th>"
            for n in order
        )
        + "</tr>"
        + "".join(
            f"<tr><th class='rowh'>{_chip(cfg, st, a, ' ' + mergeplan.plan_cls(mp, a))}</th>"
            + "".join(_grid_cell(mp, a, b) for b in order)
            + "</tr>"
            for a in order
        )
    )

    def pair_text(e: dict) -> str:
        parts = (
            [
                (
                    "✗ " + mergeplan.some([f.rsplit("/", 1)[-1] for f in e["x"]], 4)
                    if e["x"]
                    else "? not checked yet"
                )
            ]
            if "x" in e
            else []
        ) + [f"Ⓐ {mergeplan.ABI_LABEL[lane]}" for lane in e.get("abi") or []]
        return " · ".join(
            parts + (["◆ " + "; ".join(e["care"])] if e.get("care") else [])
        )

    prow = "".join(
        f"<tr><td class='nw'>#{E(k.replace('-', ' × #'))}</td><td>{E(pair_text(e))}</td></tr>"
        for k, e in sorted(
            mp["edges"].items(), key=lambda kv: tuple(map(int, kv[0].split("-")))
        )
    )
    # The ABI lanes in plan order (the lanes follow the plan), every open claim, clean or not.
    lanes = "".join(
        f"<p class='sub'><b class='g abi'>Ⓐ</b> {E(lane)} lane, {E(abi.DEFINES[lane])} (develop "
        f"{E(str(L['develop']))}): "
        + " → ".join(
            "+".join(_chip(cfg, st, m) for m in g["members"]) for g in L["groups"]
        )
        + "</p>"
        for lane, L in (st.get("abi_lanes") or {}).items()
        if lane in abi.DEFINES and L["groups"]
    )
    advice = sorted(
        mp["advice"],
        key=lambda k: ((nodes.get(k) or dc.get(k) or {}).get("rank", 10**6), int(k)),
    )
    arows = "".join(
        f"<tr><td class='nw'>{_chip(cfg, st, int(k))}</td><td class='nw'>{E(mergeplan.plan_cls(mp, int(k)))}</td>"
        f"<td class='nw'>{E(mergeplan.place(st, int(k)) or '—')}</td><td>{E(mp['advice'][k])}</td><td>"
        + (
            E(
                mergeplan.some(
                    [f"#{x}" for x in (nodes.get(k) or {}).get("care", [])], 6
                )
            )
            or "<span class='muted'>—</span>"
        )
        + "</td></tr>"
        for k in advice
    )
    # No list of the priorities above the table: it would repeat the table.
    return (
        head
        + f"<p class='sub'>{E(line)}</p>"
        + "<div class='wrap'><table><tr><th>Order</th><th>PRs (class; ✓ nothing ahead conflicts: it lands as soon as "
        f"it is ready)</th><th>Landing them makes these rebase</th><th>Whose turn</th></tr>{wrows}</table></div>"
        + f"<p class='sub'>{E(mergeplan.RULE)} Classes: T0 top priority, R0 ready, R1 only "
        "approvals or threads (or a CI run) left, R3 approved with fixes or our follow-up left, R4 further out. "
        "Textual (✗) and ABI (Ⓐ) conflicts order the plan; only textual ones hold a PR; care is shown.</p>"
        + (f"<h3>Auto-merge on</h3><ul class='left'>{amh}</ul>" if amh else "")
        + f"<h3>Conflicts between the first {len(order)} <span class='c'>hover or focus a mark</span></h3>"
        "<div class='legend'><span><b class='g fail'>✗</b> conflicts on today's develop</span><span><b class='g abi'>"
        "Ⓐ</b> ABI: both bump ROCKE_ABI_VERSION (or ROCKE_RECIPE_ABI): the later renumbers to develop + 1, even "
        "when git merges them cleanly</span><span><b class='g care'>◆</b> care: no conflict, but the same golden "
        "shard, DG family, ABI family string, CMake list or test-category file, or new parity indices</span>"
        "<span><b class='g pending'>·</b> clean</span><span><b class='g pending'>?</b> not checked yet</span></div>"
        + f"<div class='wrap'><table class='grid' aria-label='Conflict grid'>{grid}</table></div>"
        + f"<details id='d-plan-pairs'><summary>Table view: every conflict and care pair ({len(mp['edges'])})</summary>"
        f"<table><tr><th>Pair</th><th>What</th></tr>{prow}</table></details>"
        + (
            f"<h3>ABI lanes <span class='c'>in plan order: each later claim renumbers to develop + 1 once the one "
            f"before it has landed</span></h3>{lanes}"
            if lanes
            else ""
        )
        # Collapsed by default; once opened it stays open across reloads.
        + f"<details id='d-plan-advice'><summary>Rebase advice ({_plural(len(advice), 'PR')})</summary>"
        "<div class='wrap'><table><tr><th>PR</th><th>Class</th><th>Place</th><th>Advice</th>"
        f"<th>Care with</th></tr>{arows}</table></div></details></section>"
    )


# --- the page ----------------------------------------------------------------------------------------------------------


def render(
    cfg: dict,
    st: dict,
    store,
    plan: list[dict],
    q: list[dict],
    board: dict,
    variant: str = PUBLISHED,
) -> str:
    """The page, as the LOCAL file (the admin dashboard) or the PUBLISHED copy (the default: it never contacts
    127.0.0.1). Only the LOCAL file, the admin page, has the scoreboard, review spend and today's log (ADMIN_CARDS),
    last.
    """
    admin = variant == LOCAL
    d = st["develop"]
    ci = ci_verdict(d["ci"].get(d.get("sha"), {})) if d.get("sha") else "unknown"
    tp, holds = board["throughput"], active_holds(st)
    # One approval box per code-owner team (the first keeps the old anchor), then "Needs any approval".
    teams = board["teams"] if "teams" in board else report.team_approvals(cfg, st)
    anyone = board["any"] if "any" in board else report.any_approvals(cfg, st, teams)
    boxes = [
        ("approvals" if i == 0 else f"approvals-{t}", t, teams[t])
        for i, t in enumerate(teams)
    ] + [("approvals-any", report.ANY, anyone)]
    mq = report.mergeable(st, q)
    mrows_src = mq[
        "rows"
    ]  # the Mergeable Candidates card: PRs that passed CI and can merge now
    needs = board["needs"]
    today = now()[:10]
    log = daylog.entries(store, today)
    fu = sum(1 for p in plan if p["kind"] == "FOLLOW_UP")
    first = sum(1 for p in plan if p["kind"] == "FIRST_ROUND")
    # The same sources as the cleared-vs-arrived chart (ledger, day log and journal), so tiles and chart always agree.
    posted = sum(
        1 for r in store.ledger() if r["kind"] == "post" and r["ts"][:10] == today
    )
    unposted = sum(1 for r in log if r["kind"] == daylog.UNPOSTED)
    carried = [
        how for ts, how in report.equivalent_clears(store).values() if ts[:10] == today
    ]
    merged_only = carried.count(
        report.MERGE_ONLY
    )  # merge-only heads carried with clean checks, no post
    rebase = len(carried) - merged_only
    gapped = report.gap_waiting(
        plan
    )  # follow-ups waiting for the gap after our last review of their PR
    gap_word = gap_label(cfg)
    saves = daylog.saves_since(store, 7) if admin else {}
    landed = merges.series(st, store)
    week = sum(c for _, c, _ in landed[-7:])
    incs = [
        x
        for x in guard.incidents(store)
        if x.get("status") == "open" and not x.get("rehearsal")
    ]
    pm = sorted(
        (st.get("post_merge") or {}).get("open") or [],
        key=lambda i: (not i.get("rocke"), -i["n"]),
    )
    pm_rocke = [i for i in pm if i.get("rocke")]
    risks = st.get("automerge_risks") or {}
    stages = pipeline(cfg, st, store, plan, q)
    ready = [r for r in q if r.get("position")]
    sc = report.stage_counts(q, st)
    txt = _ci(ci)[1]
    ds = report.develop_status(
        cfg, st, store
    )  # one word from the latest completed develop state

    o = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<noscript><meta http-equiv='refresh' content='{REFRESH_S}'></noscript>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'><title>rocKE monitor</title>",
        f"<style>{CSS}</style></head><body><div class='viz-root'><div id='tip' role='status'></div>",
    ]

    # 1. header
    pills = [
        _pill("good", "ON", "▶") if st["enabled"] else _pill("serious", "OFF", "■"),
        _pill("good", report.modes(cfg, " · "), ""),
    ]
    if st.get("frozen"):
        pills.append(
            _pill("critical", f"FROZEN: {st['frozen'].get('reason', '')}", "■")
        )
    pills.append(_pill(ds["level"], f"develop {ds['word']}", ds["light"]))
    if pm_rocke:
        pills.append(
            _pill(
                "critical",
                f"{len(pm_rocke)} rocKE breakage{'s' if len(pm_rocke) > 1 else ''} open",
            )
        )
    mp = st.get("merge_plan") or {}
    from . import state_store
    from . import tickets as tickets_mod

    card = variant == LOCAL and tickets_shown(cfg, st)
    unsent = state_store.unsent_items(store.dir) if admin else []
    kept = admin and not_sent_shown(cfg, unsent)
    nav = (
        [("needs", "Needs you", len(needs))]
        + ([("tickets", "Tickets", len(tickets_mod.waiting(st)))] if card else [])
        + ([("not-sent", "Not sent", len(unsent))] if kept else [])
        + [("flow", "Pipeline", None), ("reviews", "Reviews", fu + first)]
        + [(sid, f"Needs {report.team_label(t)}", len(rows)) for sid, t, rows in boxes]
        + [
            ("merge", "Mergeable Candidates", len(mrows_src)),
            (
                "plan",
                "Merge plan",
                len(mp["waves"]) if mp.get("waves") is not None else None,
            ),
            ("insights", "Insights", None),
        ]
        + [("develop", "Develop", len(pm_rocke))]
        + (
            [
                ("scoreboard", "Scoreboard", None),
                ("spend", "Review spend", None),
                ("log", "Today's log", len(log)),
            ]
            if admin
            else []
        )
    )
    at = now()
    every = cfg.get("tick_interval_min", registry.default("tick_interval_min"))
    o.append(
        "<header class='top'><div class='bar'><h1>rocKE monitor</h1>"
        + "".join(pills)
        + f"<span class='meta'>updated {E(at[11:16])}Z · checked {checked_span(at)} · ticks every {every:g} min · "
        f"reload in <span id='countdown'>{REFRESH_S // 60}:00</span></span></div>"
        + (standby_line() if admin else "")
        + "<nav aria-label='Sections'>"
        + "".join(
            f"<a href='#{i}'>{E(t)}"
            + (f" <span class='c'>{c}</span>" if c is not None else "")
            + "</a>"
            for i, t, c in nav
        )
        + "</nav></header><main>"
    )

    # 2. needs you
    crit = any(
        t.startswith(("Critical", "Auto-merge", "Develop has been broken"))
        for t in needs
    )
    # On the admin dashboard, a decision taken on its line (report.DECIDED_ON_THE_PAGE) has its Approve and Decline
    # there, and the service's answer beside them.
    decided = {
        a["text"]: f" <span class='nw' data-local-only hidden>"
        f"{decide_buttons(report.DECIDED_ON_THE_PAGE[a['kind']], a['pr'])}</span> <span class='prio-st'></span>"
        for a in board.get("asks") or []
        if admin and a["kind"] in report.DECIDED_ON_THE_PAGE
    }
    # The published copy shows each decision as the team sees it (report.need_ask's team).
    if not admin:
        team = {a["text"]: a["team"] for a in board.get("asks") or []}
        needs = [team.get(t, t) for t in needs]
    o.append(
        f"<section id='needs' class='card needs{' crit' if crit else '' if needs else ' ok'}' aria-live='polite'>"
        "<h2>"
        + ("⚠ Needs you" if needs else "✓ Nothing needs you right now")
        + (f" <span class='c'>{len(needs)}</span>" if needs else "")
        + "</h2>"
        + (
            "<ol>"
            + "".join(f"<li>{E(t)}{decided.get(t, '')}</li>" for t in needs)
            + "</ol>"
            if needs
            else ""
        )
        + "</section>"
    )

    # 2'. the tickets waiting for the user, then what was not sent: the admin dashboard only
    if variant == LOCAL:
        o.append(tickets_card(cfg, st))
        o.append(not_sent_card(cfg, store, unsent))

    # 3. KPI tiles
    o.append(
        "<section class='kpis' aria-label='Key numbers'>"
        + "".join(
            [
                _tile(
                    "Reviews waiting",
                    fu + first,
                    f"{fu} follow-ups ({len(gapped)} wait for the {gap_word} gap) · {first} "
                    f"first rounds · oldest {report._ago_min(tp['oldest_min'])}",
                    sparkline(backlog_series(store, 24)),
                ),
                _tile(
                    "Cleared today",
                    posted + unposted + rebase + merged_only,
                    f"{posted} posted · "
                    + (f"{unposted} kept unposted · " if unposted else "")
                    + f"{rebase} rebase-only · {merged_only} merge-only · trend {tp['trend']}",
                ),
                _tile(
                    "Merged today (rocKE)",
                    landed[-1][1],
                    f"last 7 days {week} ({week / 7:.1f}/day) · last "
                    f"{len(landed)} days {sum(c for _, c, _ in landed)}",
                    sparkline([(d, c) for d, c, _ in landed], color="--series-3"),
                ),
                _tile(
                    "Needs approval",
                    " / ".join(str(len(rows)) for _, _, rows in boxes),
                    " / ".join(report.team_label(t) for _, t, _ in boxes)
                    + " · anyone in the team can approve; any: any approver",
                ),
                _tile(
                    "Merge queue",
                    f"{len(ready)} / {sc['candidate']}",
                    f"ready / candidates · {sc['analysed']} analysed · "
                    f"{len(holds)} held",
                ),
                _tile(
                    "Develop",
                    f"{ds['light']} {ds['word']}",
                    report.develop_status_line(ds),
                    tip="; ".join(
                        ds["why"]
                        + [
                            f"{len(incs)} incidents · rocKE issues: "
                            f"{sum(1 for i in pm_rocke if entry_kind(cfg, i) == RED)} {kind_label(cfg, RED)} "
                            f"(CI red), {sum(1 for i in pm_rocke if entry_kind(cfg, i) == STABILITY)} "
                            f"{kind_label(cfg, STABILITY)}"
                        ]
                    ),
                ),
            ]
            + (
                [  # the admin page only
                    _tile(
                        "Issues avoided (7 d)",
                        sum(saves.values()),
                        ", ".join(
                            f"{report.save_label(cfg, k)} {v}"
                            for k, v in sorted(saves.items())
                        )
                        or "none recorded yet",
                    )
                ]
                if admin
                else []
            )
        )
        + "</section>"
    )

    # 4. the pipeline
    extra = {}
    for s in stages:
        if s["id"] in ("author", "approval", "candidate"):
            for n in s["prs"]:
                v = ((st["prs"].get(str(n)) or {}).get("ours") or {}).get("verdict")
                if v and v != "?":
                    extra[n] = f" {v}"

    def count_lines(s: dict) -> str:
        """A stage's count lines: In review's parts, the Mergeable Candidates' "+N waiting on CI" (each PR and why
        as its tooltip)."""
        out = f"<div class='parts'>{E(s['parts'])}</div>" if s.get("parts") else ""
        line, why = s.get("more") or ("", "")
        return out + (
            f"<div class='parts' title='{E(why, quote=True)}'>{E(line)}</div>"
            if line
            else ""
        )

    label_held = [
        r["n"]
        for r in q
        if any("out of the queue until removed" in b for b in r.get("blockers") or [])
    ]
    drafts = sum(
        1 for r in st["prs"].values() if r.get("status") == "open" and r.get("draft")
    )
    o.append(
        "<section id='flow' class='card'><h2>Pipeline <span class='c'>each open PR in the stage it waits in</span>"
        "</h2><div class='flow'>"
        + "".join(
            f"<div class='stage' id='st-{s['id']}'><div class='n'>{len(s['prs'])}</div><div class='lbl'>"
            f"{E(s['label'])}</div><div class='hint'>{E(s['hint'])}</div>{count_lines(s)}"
            f"{_chips(cfg, st, s['prs'], 10, s.get('extra') or extra)}</div>"
            for s in stages
        )
        + "</div><div class='side'>"
        + f"<span>Held: {', '.join(_chip(cfg, st, int(k)) for k in sorted(holds, key=int)) or 'none'}</span>"
        + f"<span>On a hold label: {''.join(_chip(cfg, st, n) for n in label_held) or 'none'}</span>"
        + f"<span>Auto-merge risks: {''.join(_chip(cfg, st, int(k)) for k in sorted(risks, key=int)) or 'none'}</span>"
        + f"<span>Drafts: {drafts}</span></div></section>"
    )

    # 5. reviews and throughput, right after the pipeline
    from .review import pipes, waiting_on_verifier

    waiting_verify = waiting_on_verifier(st)
    cur = (
        "; ".join(
            f"{_chip(cfg, st, pipe['pr'])} {E(pipe['kind'])}: "
            f"{E(report.pipe_part(pipe, store))} "
            f"(started {E(_ago((pipe.get('worker') or {}).get('started')))} ago)"
            for pipe in pipes(st)
        )
        or "idle"
    )
    gap_at = {p["pr"]: p["gap_until"][11:16] for p in gapped}

    def kind_cell(p: dict) -> str:
        """The item's kind; a merge-only push's (its checks decide whether anything is posted), and until when a
        follow-up waits for the gap."""
        return (
            E(p["kind"].replace("_", "-").lower())
            + (
                " <span class='muted'>(merge-only)</span>"
                if p.get("merge_only")
                else ""
            )
            + (
                f" <span class='muted'>(gap: after {E(gap_at[p['pr']])}Z)</span>"
                if p["pr"] in gap_at
                else ""
            )
        )

    nxt = "".join(
        f"<tr><td class='nw'>{_chip(cfg, st, p['pr'])}</td><td class='nw'>{kind_cell(p)}</td>"
        f"<td class='nw'>{E((st['prs'].get(str(p['pr'])) or {}).get('author', ''))}</td>"
        f"<td>{E((st['prs'].get(str(p['pr'])) or {}).get('title', '')[:90])}</td></tr>"
        for p in [p for p in plan if p["kind"] in report.REVIEW_KINDS][:12]
    )
    done = "".join(
        f"<tr><td class='nw'>{E(r['ts'][11:16])}Z</td><td class='nw'>{_chip(cfg, st, r['pr']) if r.get('pr') else ''}</td>"
        f"<td>{E(r.get('verdict', ''))}</td><td>{E(r['text'][:140])}</td></tr>"
        for r in log
        if r["kind"] in ("review", daylog.UNPOSTED, "rebase")
    )
    handed = "".join(
        _chip(cfg, st, r["pr"], " verifying" if report.verifying(cfg, r) else "")
        for r in waiting_verify
    )
    o.append(
        "<section id='reviews' class='grid2'><div class='card'><h2>Reviews</h2>"
        f"<p>In flight: <b>{cur}</b><br>Handed to my verification: "
        f"{handed or 'none'}<br>"
        # One follow-up review per PR per gap; a merge-only head gets one only when its checks flag something.
        f"Waiting for the {E(gap_word)} gap after our last review: "
        f"{''.join(_chip(cfg, st, p['pr']) for p in gapped) or 'none'}<br>"
        f"<span class='sub'>Last 24 h: {tp['posts'] + tp['equiv']} cleared vs {tp['new'] + tp['pushes']} arrived; "
        f"{tp['merge_only']} heads carried as merge-only (clean checks, no post); trend {E(tp['trend'])}</span></p>"
        f"<h3>Next up</h3><table><tr><th>PR</th><th>Kind</th><th>Author</th><th>Title</th></tr>{nxt}</table>"
        # Collapsed by default; once opened it stays open across reloads.
        f"<details id='d-cleared'><summary>Cleared today ({posted + unposted + rebase + merged_only})</summary><table>"
        "<tr><th>Time</th>"
        f"<th>PR</th><th>Verdict</th><th>Detail</th></tr>{done}</table></details></div>"
        "<div class='card'><h2>Review backlog <span class='c'>reviews waiting, last 7 days</span></h2>"
        + line_chart(backlog_series(store), "Review backlog")
        + "<h2 style='margin-top:16px'>Cleared vs arrived <span class='c'>per UTC day</span></h2>"
        + column_chart(daily_series(store))
        + f"<h2 style='margin-top:16px'>rocKE PRs merged <span class='c'>into develop, per UTC day, last "
        f"{len(landed)} days</span></h2>"
        + merged_chart(cfg, st, landed)
        + "</div></section>"
    )

    # 6a. each code-owner team's approval ("Needs platform approval", then "Needs rocke-lib approval" underneath), then
    # "Needs any approval": a general approval, any approver will do
    def approval_box(sid: str, team: str, rows: list[dict]) -> str:
        arows = "".join(
            f"<tr><td class='nw'>{_chip(cfg, st, a['n'])}</td><td>{E(a['verdict'])}</td>"
            f"<td class='nw'>{E(a['author'])}</td><td>{E(a['title'][:90])}</td><td>{E(a['why'])}</td>"
            f"<td class='nw'>{_status(_ci(a['ci'])[0], a['ci'])}</td>"
            f"<td>{E('; '.join(a['also'])) or '<span class=muted>nothing</span>'}</td>"
            f"<td class='nw'>{E(a.get('place') or '—')}</td><td class='nw'>{E(_ago(a['since']))}</td></tr>"
            for a in rows
        )
        who = (
            "a general approval is what is missing, no code-owner team's: any approver will do"
            if team == report.ANY
            else f"a {E(team)} code-owner approval is what is missing: anyone in {E(team)} can approve"
        )
        return (
            f"<section id='{sid}' class='card'><h2>Needs {E(report.team_label(team))} approval <span class='c'>no "
            "more reviews needed: our review is current at head and clean, every review thread is resolved, and "
            f"{who}. In merge-plan order</span></h2>"
            + (
                "<div class='wrap'><table><tr><th>PR</th><th>Ours</th><th>Author</th><th>Title</th><th>Why</th>"
                f"<th>CI</th><th>Also needs</th><th>Merge plan</th><th>Verdict age</th></tr>{arows}</table></div>"
                if rows
                else "<p class='sub'>None right now.</p>"
            )
            + "</section>"
        )

    o += [approval_box(sid, t, rows) for sid, t, rows in boxes]

    # 6b. the Mergeable Candidates: the PRs that passed CI and can merge now, and their gates; the ABI lanes are in the
    # merge plan, and a PR's batch shows in its stage. Above the merge plan.
    G = report.GATES
    names = {
        "G1": "our review",
        "G2": "approvals",
        "G3": "CI",
        "G4": "fresh merge",
        "G5": "goldens",
        "G6": "build, byte identity, tests",
        "G7": "compile per target",
        "G8": "dependencies and labels",
        "G9": "ABI lane",
    }
    glyph = {PASS: "✓", FAIL: "✗"}

    def am(n: int) -> str:
        s = report.automerge_state(st, n)
        if not s:
            return "<span class='muted'>—</span>"
        return _status(s[0], s[1], None if s[0] == "critical" else "▶")

    mrows = "".join(
        f"<tr><td class='num'>{r.get('position') or ''}</td><td class='nw'>{E(report.stage_text(st, r))}</td>"
        f"<td class='nw'>{_chip(cfg, st, r['n'])}</td><td class='nw'>{E((st['prs'].get(str(r['n'])) or {}).get('author', ''))}</td>"
        f"<td class='num'>{(st['prs'].get(str(r['n'])) or {}).get('size', '')}</td>"
        + "".join(
            f"<td class='g {_gs(r, k)}' tabindex='0' data-tip='{_tip([[names[k], k, None], [_gt(r, k)[:120], _gs(r, k), None]])}'>"
            f"{glyph.get(_gs(r, k), '…')}</td>"
            for k in G
        )
        + f"<td class='nw'>{am(r['n'])}</td><td>{E(((r.get('blockers') or [''])[0])[:110])}</td></tr>"
        for r in mrows_src
    )
    explain = "".join(
        f"<tr><td class='nw'><b>{g}</b> {E(name)}</td><td>{E(what)}</td><td>{E(passes)}</td>"
        f"<td>{E(who)}</td></tr>"
        for g, name, what, passes, who in GATES
    )
    more = report.not_mergeable_line(mq)
    o.append(
        "<section id='merge' class='card'><h2>Mergeable Candidates <span class='c'>the PRs that passed CI and can "
        "merge now, and their gates, in landing order</span></h2>"
        "<details id='d-gates'><summary>What the gates mean (G1–G9)</summary><p class='sub'>A <b>candidate</b> "
        "passes G1, G2 and G8: its merge analysis (G5–G7) starts early, while CI may still run. A PR is "
        "<b>ready</b> when G1–G4, G8 and G9 pass; it is then batched, checked on today's develop (G5–G7) and "
        "proposed to you, or armed if you cleared it. G5–G7 always run on the PR merged with today's develop, "
        "and stay valid while develop's rocKE tree does not change.</p><div class='wrap'><table><tr><th>Gate</th>"
        f"<th>What it checks</th><th>Passes when</th><th>Who fixes a failure</th></tr>{explain}</table></div>"
        "</details>"
        + (
            "<div class='legend'><span>✓ pass</span><span>✗ fail</span><span>… pending</span><span>hover a mark "
            "for the gate's detail</span></div><div class='wrap'><table><tr><th class='num'>#</th><th>Stage</th>"
            "<th>PR</th><th>Author</th><th class='num'>Lines</th>"
            + "".join(
                f"<th class='g' title='{E(names[k], quote=True)}'>{k}</th>" for k in G
            )
            + f"<th>Auto-merge</th><th>First blocker</th></tr>{mrows}</table></div>"
            if mrows_src
            else "<p class='sub'>No PR has passed CI and can merge now.</p>"
        )
        + (
            f"<p class='sub'>Not mergeable yet: {E(more)} ({E(report.not_mergeable_detail(mq))})</p>"
            if more
            else ""
        )
        + "</section>"
    )

    # 6c. the merge plan
    o.append(plan_card(cfg, st, variant))

    # 6d. insights from recent PRs, collapsed by default
    o.append(
        "<section id='insights' class='card'><h2>Insights from recent PRs <span class='c'>what keeps going wrong, "
        "and what prevents it</span></h2>" + insights_card(cfg, st, store)
        # the models' tokens, on the local page only: never published
        + (usage_tiles(cfg, store) if variant == LOCAL else "") + "</section>"
    )

    # 6e. develop health
    g = st.get("guard") or {}
    run = g.get("running")
    lg = d.get("last_green")
    links = {
        i.get("issue"): i.get("id") for i in guard.incidents(store) if i.get("issue")
    }
    irows = "".join(
        f"<tr><td class='nw'>{E(x.get('id', ''))}</td><td class='nw'>{E(_ago(x['start']))}</td>"
        f"<td class='nw'>{_chip(cfg, st, x['culprit']) if x.get('culprit') else 'unknown'}</td>"
        f"<td class='nw'>{_issue(cfg, x['issue']) if x.get('issue') else '<span class=muted>none yet</span>'}</td>"
        f"<td>{E((x.get('summary') or '')[:140])}</td></tr>"
        for x in incs
    )
    prows = "".join(
        f"<tr><td class='nw'>{_issue(cfg, i['n'])}</td><td class='nw'>{'rocKE' if i.get('rocke') else ''}</td>"
        f"<td class='nw'>{E(entry_label(i) + (' (CI red)' if entry_kind(cfg, i) == RED else ''))}</td>"
        f"<td>{E(i['title'][:110])}</td><td class='nw'>{E(i['author'])}</td>"
        f"<td class='nw'>{E(', '.join(i.get('assignees') or []) or '—')}</td><td class='nw'>{E(_ago(i['created']))}</td>"
        f"<td class='nw'>{E(links.get(i['n']) or '')}</td></tr>"
        for i in pm
    )
    o.append(
        "<section id='develop' class='card'><h2>Develop health</h2>"
        f"<p>{_status(ds['level'], ds['word'] + ': ' + report.develop_status_line(ds), ds['light'])}"
        + (
            f"<br><span class='sub'>{E('; '.join(ds['why']))}</span>"
            if ds["why"]
            else ""
        )
        + f"<br>develop head {E((d.get('sha') or '?')[:10])} CI {E(txt)} · last green "
        f"{E((lg or {}).get('sha', 'none')[:10])} · last sanity green {E((g.get('last_sanity_green') or 'none')[:10])}"
        + (f" · sanity running on {E(run['sha'][:10])}" if run else "")
        + "</p>"
        + (
            f"<h3>Open incidents</h3><table><tr><th>Incident</th><th>Open for</th><th>Culprit</th><th>Issue</th>"
            f"<th>Summary</th></tr>{irows}</table>"
            if incs
            else "<p class='sub'>No open incident.</p>"
        )
        + (
            "<h3>Top priority: PRs fixing develop, and PRs you prioritized (reviewed and merged first)</h3>"
            "<ul class='left'>"
            + "".join(
                f"<li>{_chip(cfg, st, int(k))} {'fixes ' if fixes_develop(why) else ''}{E(', '.join(why))}: "
                + (
                    E(
                        "; ".join(
                            f"{b['what']} → {b['who']}"
                            for b in report.merge_blockers(cfg, st, st["prs"][k])
                        )
                    )
                    or "nothing left that we can see"
                )
                + "</li>"
                for k, why in sorted(
                    (st.get("develop_fixes") or {}).items(), key=lambda kv: int(kv[0])
                )
                if k in st["prs"]
            )
            + "</ul>"
            if st.get("develop_fixes")
            else ""
        )
        + f"<h3>Develop issues (the repo's breakage tracker: {E(kind_label(cfg, RED))} when CI is red, "
        f"{E(kind_label(cfg, STABILITY))} when CI does not see it; rocKE first)</h3>"
        + (
            f"<div class='wrap'><table><tr><th>Issue</th><th></th><th>Kind</th><th>Title</th><th>Filed by</th>"
            f"<th>Assignees</th><th>Age</th><th>Our incident</th></tr>{prows}</table></div>"
            if pm
            else "<p class='sub'>None open"
            + ("" if (st.get("post_merge") or {}).get("at") else " (not read yet)")
            + ".</p>"
        )
        + "</section>"
    )

    # 7. the admin page's own cards (ADMIN_CARDS): the scoreboard, review spend, and today's log last
    if admin:
        o.append(scoreboard_card(cfg, scoreboard.load(store)))
        o.append(spend_card(spend.load_card(store)))
        o.append(log_card(cfg, st, q, log, today))

    o.append(
        "</main><footer>Generated by rkmon from GitHub reads, local gate jobs and the reviews. Gate "
        "marks: ✓ pass · ✗ fail · … pending. Status colors always come with an icon and a label. "
        f"Monitor {'ON' if st['enabled'] else 'OFF'} · {E(report.modes(cfg, ' · '))}.</footer>"
    )
    o.append(
        f"<script>{JS}{PRIO_JS[variant]}{SCORE_JS if admin else ''}</script></div></body></html>"
    )
    return "".join(o)


# The cards only the admin page (the LOCAL file) shows, in page order; today's log is the last card.
ADMIN_CARDS = ("scoreboard", "spend", "log")


def log_card(cfg: dict, st: dict, q: list[dict], log: list[dict], today: str) -> str:
    """Today's log (the daily log's entries of the UTC day) and the PRs not in the merge queue yet."""
    waiting = [
        r
        for r in q
        if not r.get("position") and r.get("stage") not in ("candidate", "held")
    ]
    blockers = "".join(
        f"<tr><td class='nw'>{_chip(cfg, st, r['n'])}</td>"
        f"<td>{E((r.get('blockers') or ['pending: ' + ', '.join(r.get('pending', []))])[0][:140])}</td></tr>"
        for r in waiting
    )
    rows = "".join(
        f"<tr><td class='nw'>{E(r['ts'][11:16])}Z</td>"
        f"<td class='nw'>{E(r['kind'] + (' (' + r['save'] + ')' if r['kind'] == 'save' else ''))}</td>"
        f"<td class='nw'>{_chip(cfg, st, r['pr']) if r.get('pr') else '—'}</td><td>{E(r['text'][:170])}</td></tr>"
        for r in log
    )
    return (
        f"<section id='log' class='card'><h2>Today's log <span class='c'>{E(today)} UTC</span></h2>"
        f"<p class='sub'>{E(daylog.summary(log))}</p><details id='d-log'><summary>Events ({len(log)})</summary>"
        f"<table><tr><th>Time</th><th>What</th><th>PR</th><th>Detail</th></tr>{rows}</table></details>"
        f"<details id='d-notready'><summary>Not in the merge queue yet ({len(waiting)} PRs, first blocker)"
        f"</summary><table><tr><th>PR</th><th>Blocker</th></tr>{blockers}</table></details></section>"
    )


# --- the scoreboard (admin page only) -----------------------------------------------------------------------------------

SB_COLS = (  # (key, heading, what it counts: the heading's tooltip)
    ("author", "Author", "the PR author"),
    ("merged", "Merged", "their in-scope PRs that merged into develop"),
    (
        "rework",
        "Rework for others",
        "rebases forced plus re-reviews caused: extra work their merged PRs caused "
        "other authors (hover a number for its parts)",
    ),
    (
        "develop_break",
        "Develop breaks",
        "develop breaks filed (of either kind) that name their PR "
        "as the one that broke develop",
    ),
    (
        "landed_ahead",
        "Landed ahead of plan",
        "their PRs that merged while an AUTOMERGE_JUMP or PRIORITY_READY_EARLY "
        "applied, or against the merge plan's order",
    ),
    (
        "merged_open_items",
        "Merged with open review items",
        "their PRs that merged over our open 🔴/🟠 review "
        "(MERGED_OVER_RED) or while held (HELD_MERGED)",
    ),
    ("followup", "Follow-ups", "follow-up tickets filed for their merged PRs"),
)


def sb_chart(trend: list[dict], days: int) -> str:
    """Grouped columns per bucket (a day for 7 days, else an ISO week): PRs merged (slot 3) and rework for others
    (slot 2), the latest bucket labelled, a tooltip per bucket and a table view."""
    w, h, ml, mr, mt, mb, gap = 560, 190, 40, 12, 18, 26, 2
    peak = max([b["merged"] for b in trend] + [b["rework"] for b in trend] + [1])
    step = _nice(peak)
    top = max(step, ((peak + step - 1) // step) * step)
    Y = lambda v: mt + (h - mt - mb) * (1 - v / top)  # noqa: E731
    band = (w - ml - mr) / max(len(trend), 1)
    bw = max(4.0, min(18.0, (band - gap - 8) / 2))
    unit = "day" if days <= 7 else "week of"
    out = [
        f"<div class='legend'><span><span class='key' style='background:var(--series-3)'></span>PRs merged</span>"
        "<span><span class='key' style='background:var(--series-2)'></span>Rework for others (rebases forced + "
        f"re-reviews caused)</span></div><svg class='sb-chart' viewBox='0 0 {w} {h}' width='100%' role='img' "
        f"aria-label='PRs merged vs rework for others, last {days} days'>"
    ]
    for g in range(0, top + 1, step):
        out.append(
            f"<line x1='{ml}' x2='{w - mr}' y1='{Y(g):.1f}' y2='{Y(g):.1f}' stroke='var(--grid)' "
            f"stroke-width='1'/><text x='{ml - 6}' y='{Y(g) + 4:.1f}' text-anchor='end'>{g}</text>"
        )
    every = 1 if len(trend) <= 8 else 2
    for i, b in enumerate(trend):
        x0 = ml + band * i + (band - 2 * bw - gap) / 2
        out.append(
            _bar(x0, b["merged"], bw, Y, "--series-3")
            + _bar(x0 + bw + gap, b["rework"], bw, Y, "--series-2")
        )
        if (len(trend) - 1 - i) % every == 0:
            out.append(
                f"<text x='{ml + band * i + band / 2:.1f}' y='{h - 6}' text-anchor='middle'>{E(b['label'])}</text>"
            )
        tip = [
            ["merged", str(b["merged"]), "var(--series-3)"],
            ["rework for others", str(b["rework"]), "var(--series-2)"],
            [unit, b["key"], None],
        ]
        out.append(
            f"<rect x='{ml + band * i:.1f}' y='{mt}' width='{band:.1f}' height='{h - mt - mb}' fill='transparent' "
            f"tabindex='0' data-tip='{_tip(tip)}'/>"
        )
        if i == len(trend) - 1:  # label only the latest bucket
            out.append(
                f"<text class='val' x='{x0 + bw / 2:.1f}' y='{Y(b['merged']) - 5:.1f}' text-anchor='middle'>"
                f"{b['merged']}</text><text class='val' x='{x0 + bw * 1.5 + gap:.1f}' y='{Y(b['rework']) - 5:.1f}' "
                f"text-anchor='middle'>{b['rework']}</text>"
            )
    out.append(
        f"<line x1='{ml}' x2='{w - mr}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='var(--axis)' stroke-width='1'/>"
        "</svg>"
    )
    rows = "".join(
        f"<tr><td class='nw'>{E(b['key'])}</td><td class='num'>{b['merged']}</td>"
        f"<td class='num'>{b['rework']}</td></tr>"
        for b in trend
    )
    out.append(
        f"<details id='d-score-trend-{days}'><summary>Table view</summary><table><tr><th>"
        f"{'Day' if days <= 7 else 'Week of (Monday)'} (UTC)</th><th class='num'>Merged</th>"
        f"<th class='num'>Rework for others</th></tr>{rows}</table></details>"
    )
    return "".join(out)


def sb_table(s: dict) -> str:
    """One row per author, sorted by Rework for others (the headings sort it again), and the team's totals."""
    num, down = " class='num'", " aria-sort='descending'"
    heads = "".join(
        f"<th{'' if k == 'author' else num}{down if k == 'rework' else ''}>"
        f"<button type='button' class='sb-sort' title='{E(what, quote=True)}'>{E(label)}</button></th>"
        for k, label, what in SB_COLS
    )

    def cell(r: dict, k: str) -> str:
        if k == "author":
            return f"<td class='nw' data-v='{E(r['author'], quote=True)}'>{E(r['author'])}</td>"
        if k == "rework":
            tip = [
                ["rebases forced", str(r["rebase_forced"]), None],
                ["re-reviews caused", str(r["rereview_caused"]), None],
            ]
            return (
                f"<td class='num sb-tip' data-v='{r['rework']}' tabindex='0' data-tip='{_tip(tip)}'>"
                f"{r['rework']}</td>"
            )
        return f"<td class='num' data-v='{r[k]}'>{r[k]}</td>"

    body = "".join(
        "<tr>" + "".join(cell(r, k) for k, _, _ in SB_COLS) + "</tr>" for r in s["rows"]
    )
    t = s["totals"]
    total = (
        f"Team: {t.get('merged', 0)} merged · {t.get('rework', 0)} rework for others ({t.get('rebase_forced', 0)} "
        f"rebases forced + {t.get('rereview_caused', 0)} re-reviews caused) · {t.get('develop_break', 0)} develop "
        f"breaks · {t.get('landed_ahead', 0)} landed ahead of plan · {t.get('merged_open_items', 0)} merged with "
        f"open review items · {t.get('followup', 0)} follow-ups · {len(s['rows'])} authors"
    )
    if not s["rows"]:
        return "<p class='sub'>No activity recorded in this window.</p>"
    return (
        f"<div class='wrap'><table class='sb-table'><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table>"
        f"</div><p class='sub'>{E(total)}</p>"
    )


def scoreboard_card(
    cfg: dict, events: list[dict], end: dt.datetime | None = None
) -> str:
    """The scoreboard (admin page only): a window switch (7, 30 or 90 days; 30 first), then per window the trend chart
    and the per-author table. Built from state/scoreboard_events.jsonl alone (scoreboard.load).
    """
    end = end or dt.datetime.now(dt.timezone.utc)
    head = (
        "<section id='scoreboard' class='card'><h2>Scoreboard <span class='c'>per PR author: what lands, and the "
        "extra work it causes for other authors · admin dashboard only</span></h2>"
    )
    if not events:
        return head + (
            "<p class='sub'>Nothing recorded yet: each full tick records the last days, and "
            "<code>rk scoreboard-backfill</code> the history.</p></section>"
        )
    switch = (
        "<div class='sb-switch' role='group' aria-label='Window'>"
        + "".join(
            f"<button type='button' class='sb-win' data-days='{d}' aria-pressed='{str(d == scoreboard.DEFAULT_DAYS).lower()}'>"
            f"{d} days</button>"
            for d in scoreboard.WINDOWS
        )
        + "</div>"
    )
    panels = []
    for d in scoreboard.WINDOWS:
        s = scoreboard.summary(events, d, end)
        panels.append(
            f"<div class='sb-panel' data-days='{d}'{'' if d == scoreboard.DEFAULT_DAYS else ' hidden'}>"
            f"<h3>PRs merged vs rework for others <span class='c'>· the team, per "
            f"{'day' if d <= 7 else 'ISO week'}, {E(s['start'][:10])} to {E(s['end'][:10])} UTC"
            + (f"; recorded from {E(s['from'][:10])}" if s["from"] > s["start"] else "")
            + "</span></h3>"
            + sb_chart(s["trend"], d)
            + "<h3>By author <span class='c'>· sorted by Rework for others; click a "
            "heading to sort</span></h3>" + sb_table(s) + "</div>"
        )
    rules = "".join(
        f"<li><b>{E(label)}</b>: {E(what)}</li>" for _, label, what in SB_COLS[1:]
    )
    return (
        head
        + switch
        + "".join(panels)
        + "<details id='d-score-rules'><summary>How each column is counted</summary><ul class='left'>"
        + rules
        + "</ul><p class='sub'>Attribution is conservative: when the records do not single out one PR, nothing is "
        "counted (the rules: rkmon_lib/scoreboard.py).</p></details></section>"
    )


def spend_card(card: dict | None) -> str:
    """The review spend card: today and this week so far, the last 7 days, this week's outliers (spend.card_data), and
    the model guard's WARN when a call ran on another model family than its pass asked for (spend.guard).
    """
    head = "<section id='spend' class='card'><h2>Review spend <span class='c'>review workload, UTC</span></h2>"
    if not card:
        return head + "<p class='sub'>Not computed yet (rk spend).</p></section>"
    cols = "".join(f"<th class='num'>{E(c)}</th>" for c in spend.CARD_COLS)

    def row(label: str, r: dict) -> str:
        return (
            f"<tr><td class='nw'>{E(label)}</td>"
            + "".join(f"<td class='num'>{E(c)}</td>" for c in spend.card_cells(r))
            + "</tr>"
        )

    top = row(f"Today ({card['today']['day']})", card["today"]) + row(
        f"This week ({card['week']['week']})", card["week"]
    )
    days = "".join(row(r["day"], r) for r in card["days"])
    found = spend.guard_text(card.get("off_family") or [])
    warn = (
        f"<p>{_status('serious', 'WARN', '!')} <b>Model guard:</b> {E(found)}.</p>"
        if found
        else ""
    )
    return (
        head + warn + f"<table><tr><th></th>{cols}</tr>{top}</table>"
        f"<h3>Last 7 days</h3><table><tr><th>Day</th>{cols}</tr>{days}</table>"
        f"<p><b>Outliers this week:</b> {E(spend.outliers_text(card))}</p>"
        f"<p class='sub'>{E(spend.card_note(card))}</p></section>"
    )


def local_path(cfg: dict):
    """Where the LOCAL page is written (`paths.html`: rocke_monitor.html in the outputs folder by default); the monitor
    service serves it at its root too (webserve.PAGE_ROUTES)."""
    import pathlib

    return pathlib.Path(paths.get(cfg, "html"))


def write(
    cfg: dict, st: dict, store, plan: list[dict], q: list[dict], board: dict
) -> str:
    """Both variants of the page, from the same code: the LOCAL file at paths.html, yours only (the monitor's service
    serves it, every request with the access token), and the PUBLISHED copy in state/, which publish.html pushes to the
    team web host: it never contacts 127.0.0.1. Neither carries a secret.
    """
    path = local_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each page whole, so `rk serve` and a browser never read half of one.
    whole_file.write(
        path, render(cfg, st, store, plan, q, board, LOCAL), PAGE_MODES[LOCAL]
    )
    pub = store.dir / PUBLISHED_FILE
    whole_file.write(
        pub, render(cfg, st, store, plan, q, board, PUBLISHED), PAGE_MODES[PUBLISHED]
    )
    # Not in a mirror, not while stopped (publish.blocked); a change of the outcome is journaled.
    why = publish.team(cfg, st, store, str(pub))
    if why:
        print(f"(html dashboard not published: {why})")
    return str(path)


# --- the "checked" stamp (idle.py): every tick, idle ones too, writes its time into both pages ------------------------

CHECKED = re.compile(r"<span id='checked'>[^<]*</span>")


def checked_span(t: str) -> str:
    return f"<span id='checked'>{E(t[11:16])}Z</span>"


@idle.stamp("html dashboard")
def _html_checked(cfg: dict, store, t: str) -> list[str]:
    """Both pages get the tick's time; the published copy is pushed to the web host again."""
    pub = store.dir / PUBLISHED_FILE
    for path, variant in ((local_path(cfg), LOCAL), (pub, PUBLISHED)):
        if path.exists():
            whole_file.write(
                path,
                CHECKED.sub(checked_span(t), path.read_text(encoding="utf-8"), count=1),
                PAGE_MODES[variant],
            )
    try:
        st = store.load()
    except (OSError, ValueError):
        st = None
    why = publish.team(cfg, st, store, str(pub)) if pub.exists() else None
    return [f"(html dashboard not published: {why})"] if why else []
