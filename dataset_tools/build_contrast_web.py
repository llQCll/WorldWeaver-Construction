from __future__ import annotations

import html
import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "dataset_runs_contrast" / "run_20260715_202726"
ROOT_ASSETS_DIR = BASE_DIR / "topic_root_assets"
WEB_DIR = RUN_DIR / "web"


def slug_title(slug: str) -> str:
    return slug.replace("_", " ").title()


def safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative_web_tree(tree: dict[str, Any]) -> dict[str, Any]:
    web_tree = copy.deepcopy(tree)
    web_tree["root_image"] = "images/n0.png"
    for node in web_tree.get("nodes", []):
        node["image"] = f"images/{node['node_id']}.png"
    for edge in web_tree.get("edges", []):
        edge["target_image"] = f"images/{edge['target_node']}.png"
    return web_tree


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def css() -> str:
    return """
:root{--ink:#17202a;--muted:#667085;--line:#d6dde7;--blue:#174ea6;--green:#16805c;--red:#b13b3b;--dot:#e11d48}
*{box-sizing:border-box}body{margin:0;font-family:Aptos,Segoe UI,Calibri,Arial,sans-serif;color:var(--ink);background:#eef2f7}
header.hero{padding:26px 34px;background:#fff;border-bottom:1px solid var(--line)}h1{margin:0 0 8px;font-size:30px}h2{margin:28px 0 14px;font-size:22px}h3{margin:0 0 10px;font-size:16px}h4{margin:14px 0 8px;font-size:13px;text-transform:uppercase;color:var(--muted)}
main{padding:24px 34px 60px}.muted{color:var(--muted);line-height:1.55}a{color:var(--blue)}nav a{display:inline-block;margin-right:10px;padding:8px 11px;border:1px solid var(--line);border-radius:8px;color:#174ea6;text-decoration:none;background:#f8fbff}nav a.active{background:#174ea6;color:#fff;border-color:#174ea6}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px;box-shadow:0 8px 22px rgba(28,42,64,.06)}
.topic-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px}.topic-card{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 8px 22px rgba(28,42,64,.06)}.topic-card img{width:100%;aspect-ratio:1/1;object-fit:cover;background:#111827;display:block}.topic-card div{padding:13px}.topic-card h3{font-size:16px}.topic-card p{font-size:13px;color:#667085;line-height:1.45}
.metric{display:grid;grid-template-columns:1fr 54px;gap:10px;align-items:center;margin:7px 0;font-size:13px}.metric i{grid-column:1/3;display:block;height:8px;border-radius:99px;background:#edf1f6;overflow:hidden}.metric em{display:block;height:100%;background:linear-gradient(90deg,#29a37a,#1b66d2)}
.tree-stage{position:relative;height:2200px;min-width:1900px;background:#f8fafc;border:1px solid var(--line);border-radius:10px;overflow:auto}.tree-stage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.tree-stage line{stroke:#8aa1bd;stroke-width:.32;vector-effect:non-scaling-stroke}
.node-card{position:absolute;transform:translateX(-50%);width:340px;background:#fff;border:1px solid #cbd5e1;border-radius:9px;padding:8px;box-shadow:0 8px 18px rgba(15,23,42,.09)}.node-card header{display:flex;justify-content:space-between;margin-bottom:6px}.node-card small{color:var(--muted)}
.image-wrap{position:relative;aspect-ratio:1/1;overflow:hidden;border-radius:7px;background:#111827;cursor:zoom-in}.image-wrap img{width:100%;height:100%;object-fit:contain;display:block;background:#111827}.hotspot{position:absolute;border:1.5px solid rgba(255,204,51,.75);background:rgba(255,204,51,.08);pointer-events:none}.hotspot span{position:absolute;left:0;top:100%;background:#111827;color:#fff;font-size:12px;padding:3px 5px;white-space:nowrap;max-width:280px;overflow:hidden;text-overflow:ellipsis}.hotspot i{position:absolute;left:50%;top:50%;width:14px;height:14px;transform:translate(-50%,-50%);border-radius:50%;background:var(--dot);border:2px solid #fff;box-shadow:0 0 0 3px rgba(225,29,72,.35),0 1px 4px rgba(0,0,0,.35)}
.node-status{font-size:13px;color:#475467;line-height:1.35;min-height:58px}.edge-detail,.state-card{background:#fff;border:1px solid var(--line);border-radius:9px;padding:16px;margin-bottom:14px}.state-grid{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:16px}
.chip{display:inline-flex;gap:7px;align-items:center;padding:5px 8px;border-radius:999px;background:#edf4ff;color:#174ea6;font-size:12px;margin:2px 4px 2px 0}.chip.pos{background:#eaf8f1;color:var(--green)}.chip.neg{background:#fff0f0;color:var(--red)}
.modal{position:fixed;inset:0;display:none;background:rgba(15,23,42,.86);z-index:9999;padding:24px;overflow:auto}.modal.open{display:block}.modal-card{background:#fff;border-radius:10px;max-width:1280px;margin:0 auto;padding:18px}.modal-top{display:grid;grid-template-columns:minmax(420px,560px) 1fr;gap:18px;align-items:start}.close{position:fixed;right:26px;top:18px;border:1px solid rgba(255,255,255,.3);background:rgba(255,255,255,.12);color:#fff;border-radius:8px;padding:9px 12px;cursor:pointer;font-size:14px}.pre{white-space:pre-wrap;background:#f8fafc;border:1px solid var(--line);padding:10px;border-radius:8px;overflow:auto;max-height:440px}.story-lines{padding-left:18px;line-height:1.55}.story-lines li{margin-bottom:10px}
"""


def js() -> str:
    return """
const DATA=JSON.parse(document.getElementById('treeData').textContent);const nodeById=Object.fromEntries(DATA.tree.nodes.map(n=>[n.node_id,n]));const edgesBySource={};const edgeByTarget={};DATA.tree.edges.forEach(e=>{(edgesBySource[e.source_node] ||= []).push(e);edgeByTarget[e.target_node]=e;});
function renderedImageRect(wrap){const img=wrap.querySelector('img');const w=wrap.clientWidth,h=wrap.clientHeight,nw=img.naturalWidth||w,nh=img.naturalHeight||h;const ia=nw/nh,ba=w/h;let rw,rh,ox,oy;if(ia>ba){rw=w;rh=w/ia;ox=0;oy=(h-rh)/2;}else{rh=h;rw=h*ia;oy=0;ox=(w-rw)/2;}return{rw,rh,ox,oy};}
function positionOverlays(wrap){if(!wrap)return;const r=renderedImageRect(wrap);wrap.querySelectorAll('.hotspot').forEach(h=>{const x1=Number(h.dataset.x1||0),y1=Number(h.dataset.y1||0),x2=Number(h.dataset.x2||x1),y2=Number(h.dataset.y2||y1);h.style.left=(r.ox+x1*r.rw)+'px';h.style.top=(r.oy+y1*r.rh)+'px';h.style.width=Math.max(2,(x2-x1)*r.rw)+'px';h.style.height=Math.max(2,(y2-y1)*r.rh)+'px';});}
function positionAllOverlays(){document.querySelectorAll('.image-wrap').forEach(positionOverlays)}window.addEventListener('resize',positionAllOverlays);window.addEventListener('load',positionAllOverlays);document.addEventListener('click',e=>{const w=e.target.closest('.image-wrap[data-node-id]');if(w)openNodeModal(w.dataset.nodeId,w.dataset.mode);});
function metrics(obj){if(!obj)return '<p class="muted">No values.</p>';return Object.entries(obj).map(([k,v])=>`<div class="metric"><span>${k.replaceAll('_',' ')}</span><b>${Number(v).toFixed(3)}</b><i><em style="width:${Math.max(0,Math.min(100,Number(v)*100))}%"></em></i></div>`).join('');}
function chips(obj){if(!obj)return '<span class="muted">none</span>';return Object.entries(obj).map(([k,v])=>`<span class="chip ${Number(v)>=0?'pos':'neg'}">${k.replaceAll('_',' ')} ${Number(v).toFixed(3)}</span>`).join('');}
function imgBlock(id){const markers=(edgesBySource[id]||[]).map(e=>{const b=e.grounding?.target_box||[];if(b.length!==4)return '';return `<div class="hotspot" data-x1="${b[0]}" data-y1="${b[1]}" data-x2="${b[2]}" data-y2="${b[3]}"><span>${e.branch_label||e.edge_id}</span><i></i></div>`;}).join('');return `<div class="image-wrap"><img src="images/${id}.png" onload="positionOverlays(this.closest('.image-wrap'))" />${markers}</div>`;}
function openNodeModal(id,mode){const node=nodeById[id],incoming=edgeByTarget[id],outgoing=edgesBySource[id]||[];let html=`<div class="modal-top"><div>${imgBlock(id)}</div><div><h2>${id}</h2><p>${node.story_state?.narrative_status||''}</p><h4>Affective State</h4>${metrics(node.affective_state)}<h4>Current Composite State</h4>${metrics(node.current_state)}</div></div>`;if(mode==='intent'){html+='<h2>Outgoing Intent Data</h2>'+outgoing.map(e=>`<article class="edge-detail"><h3>${e.edge_id}: ${e.branch_label||''}</h3><p><b>Closed-domain intent:</b> <span class="chip">${e.closed_intent || 'missing'}</span></p><p><b>Intent ranking:</b> ${(e.intent_ranking||[]).map(i=>`${i.intent} (${i.score})`).join(' | ')}</p><p><b>Natural-language intent:</b> ${e.grounding?.target_caption||e.slots?.narrative_goal||''}</p><p><b>Fixed slots:</b></p><pre class="pre">${JSON.stringify(e.slots||{},null,2)}</pre></article>`).join('');}else if(mode==='profile'){html+='<h2>Incoming Profile Modeling Data</h2>';html+=incoming?`<article class="edge-detail"><h3>${incoming.edge_id}</h3><p><b>Expected affect:</b> ${chips(incoming.expected_affective_delta)}</p><p><b>Observed affect:</b> ${chips(incoming.observed_affective_delta)}</p><p><b>Final affect:</b> ${chips(incoming.final_affective_delta)}</p><p><b>Expected profile:</b> ${chips(incoming.expected_profile_delta)}</p><p><b>Observed profile:</b> ${chips(incoming.observed_profile_delta)}</p><p><b>Final profile:</b> ${chips(incoming.final_profile_delta)}</p><h4>Generation prompt for this node</h4><pre class="pre">${incoming.generation_prompt||''}</pre><h4>Calibration</h4><pre class="pre">${JSON.stringify(incoming.delta_alignment||incoming.calibration||{},null,2)}</pre></article>`:`<article class="edge-detail"><h3>Root node</h3><p class="muted">Root image is topic-only and shared before personalization starts.</p><pre class="pre">${DATA.tree.root_prompt||''}</pre></article>`;}else{html+='<h2>Generation Prompt for This Image</h2>';html+=incoming?`<pre class="pre">${incoming.generation_prompt||''}</pre>`:`<pre class="pre">${DATA.tree.root_prompt||''}</pre>`;html+='<h2>Story Branches</h2><ul>'+outgoing.map(e=>`<li><b>${e.target_node}</b>: ${e.branch_label||''}<br>${e.slots?.narrative_goal||''}<details><summary>Prompt for target image</summary><pre class="pre">${e.generation_prompt||''}</pre></details></li>`).join('')+'</ul>';}document.getElementById('modalBody').innerHTML=html;document.getElementById('nodeModal').classList.add('open');requestAnimationFrame(positionAllOverlays);}
function closeModal(){document.getElementById('nodeModal').classList.remove('open')}document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});
"""


def truncate(text: str, n: int = 132) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def metric_html(obj: dict[str, Any]) -> str:
    return "".join(
        f'<div class="metric"><span>{html.escape(k.replace("_", " "))}</span><b>{float(v):.3f}</b><i><em style="width:{max(0,min(100,float(v)*100))}%"></em></i></div>'
        for k, v in (obj or {}).items()
    )


def scale_summary_html(tree: dict[str, Any]) -> str:
    metadata = tree.get("scale_metadata", {})
    if not metadata:
        return '<p class="muted">Scale metadata is not available in this run.</p>'
    blocks = []
    for key in ["stable_profile", "affective_state"]:
        item = metadata.get(key, {})
        dims = item.get("dimensions", {})
        rows = "".join(
            f'<li><b>{html.escape(dim.replace("_", " "))}</b>: {html.escape(spec.get("definition", ""))}</li>'
            for dim, spec in dims.items()
        )
        blocks.append(
            f'<h4>{html.escape(item.get("name", key))}</h4>'
            f'<p class="muted">{html.escape(item.get("scale", ""))}</p>'
            f'<p class="muted">{html.escape(item.get("primary_source", ""))}</p>'
            f'<ul class="story-lines">{rows}</ul>'
        )
    return "".join(blocks)


def paths_html(tree: dict[str, Any]) -> str:
    edges = {edge["target_node"]: edge for edge in tree.get("edges", [])}
    lines = []
    for path in tree.get("sampled_paths", []):
        labels = []
        for node in path[1:]:
            edge = edges.get(node)
            if edge:
                labels.append(edge.get("branch_label", edge.get("edge_id", "")))
        lines.append(f"<li><b>{html.escape(' -> '.join(path))}</b><br><span>{html.escape(' / '.join(labels))}</span></li>")
    return "<ul class=\"story-lines\">" + "".join(lines) + "</ul>"


def tree_stage_html(tree: dict[str, Any], mode: str) -> str:
    edges_by_source: dict[str, list[dict[str, Any]]] = {}
    for edge in tree.get("edges", []):
        edges_by_source.setdefault(edge["source_node"], []).append(edge)
    node_by_id = {node["node_id"]: node for node in tree.get("nodes", [])}
    children = {source: [edge["target_node"] for edge in edges] for source, edges in edges_by_source.items()}
    max_depth = max((int(node.get("depth", 0)) for node in tree.get("nodes", [])), default=0)
    leaves: list[str] = []

    def visit(node_id: str) -> None:
        node_children = children.get(node_id, [])
        if not node_children:
            leaves.append(node_id)
            return
        for child in node_children:
            visit(child)

    visit("n0")
    if not leaves:
        leaves = ["n0"]
    leaf_x = {node_id: (index + 1) * 100 / (len(leaves) + 1) for index, node_id in enumerate(leaves)}
    positions: dict[str, tuple[float, float]] = {}

    def place(node_id: str) -> float:
        if node_id in positions:
            return positions[node_id][0]
        node = node_by_id[node_id]
        node_children = children.get(node_id, [])
        if not node_children:
            x = leaf_x.get(node_id, 50.0)
        else:
            child_xs = [place(child) for child in node_children if child in node_by_id]
            x = sum(child_xs) / len(child_xs) if child_xs else 50.0
        y = 4 + int(node.get("depth", 0)) * (86 / max(1, max_depth))
        positions[node_id] = (x, y)
        return x

    place("n0")
    stage_height = max(2200, 430 * (max_depth + 1))
    stage_width = max(1900, 360 * max(5, len(leaves)))
    lines = '<svg viewBox="0 0 100 100" preserveAspectRatio="none">'
    for edge in tree.get("edges", []):
        source = edge.get("source_node")
        target = edge.get("target_node")
        if source in positions and target in positions:
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            lines += f'<line x1="{x1:.3f}" y1="{min(98, y1 + 9):.3f}" x2="{x2:.3f}" y2="{max(0, y2 - 1):.3f}" />'
    lines += "</svg>"
    for node_id, (left, top) in sorted(positions.items(), key=lambda item: (node_by_id[item[0]].get("depth", 0), item[0])):
        node = node_by_id.get(node_id)
        if not node:
            continue
        markers = ""
        for edge in edges_by_source.get(node_id, []):
            box = edge.get("grounding", {}).get("target_box", [])
            if len(box) == 4:
                label = html.escape(edge.get("branch_label", edge.get("edge_id", "")))
                markers += f'<div class="hotspot" data-x1="{box[0]}" data-y1="{box[1]}" data-x2="{box[2]}" data-y2="{box[3]}"><span>{label}</span><i></i></div>'
        status = html.escape(truncate(node.get("story_state", {}).get("narrative_status", "")))
        lines += f'''<section class="node-card" id="{node_id}" style="left:{left}%;top:{top}%;">
<header><strong>{node_id}</strong><small>depth {node.get("depth", "")}</small></header>
<div class="image-wrap" data-node-id="{node_id}" data-mode="{mode}"><img src="images/{node_id}.png" onload="positionOverlays(this.closest('.image-wrap'))" />{markers}</div>
<p class="node-status">{status}</p></section>'''
    return f'<section class="tree-stage" style="height:{stage_height}px;min-width:{stage_width}px">{lines}</section>'


def write_tree_pages(tree_path: Path, rel_root: str) -> None:
    user_dir = tree_path.parent
    tree = read_json(tree_path)
    root_asset_path = user_dir / "root_asset.json"
    if root_asset_path.exists():
        root_asset = read_json(root_asset_path)
        tree.setdefault("root_prompt", root_asset.get("root_prompt", ""))
        tree.setdefault("synopsis", root_asset.get("synopsis", ""))
    page_dir = WEB_DIR / "users" / f"user_{tree['user_id']}" / tree["topic_id"]
    page_dir.mkdir(parents=True, exist_ok=True)
    img_dir = page_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for node in tree.get("nodes", []):
        src = Path(node["image"])
        if src.exists():
            copy_file(src, img_dir / f"{node['node_id']}.png")
    profile_summary = tree.get("nodes", [{}])[0].get("profile_summary", "")
    world = tree.get("world_bible", {})
    nav = f'<nav><a href="{rel_root}index.html">Topic Gallery</a><a href="index.html">Main Story</a><a href="intent_task.html">Intent Recognition Task</a><a href="profile_modeling.html">Generation-assisted User Modeling</a></nav>'
    header = f'<header class="hero"><h1>{html.escape(tree["tree_id"])}</h1><p class="muted">User <b>{tree["user_id"]}</b> | Topic <b>{tree["topic_id"]}</b></p>{nav}</header>'
    data = safe_json({"tree": relative_web_tree(tree)})

    def shell(title: str, body: str, mode: str) -> str:
        return f'<!doctype html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>{html.escape(title)}</title><style>{css()}</style></head><body>{header}<main>{body}{tree_stage_html(tree, mode)}</main><div class="modal" id="nodeModal"><button class="close" onclick="closeModal()">Close</button><div class="modal-card" id="modalBody"></div></div><script type="application/json" id="treeData">{data}</script><script>{js()}</script></body></html>'

    story_body = f'''<section class="grid"><div class="panel"><h2>Story Overview</h2><p><b>Premise:</b> {html.escape(world.get("premise",""))}</p><p><b>Visual style:</b> {html.escape(world.get("visual_style",""))}</p><h4>Branch Story Lines</h4>{paths_html(tree)}</div><div class="panel"><h2>User + Generation Prompt</h2><p><b>One-sentence profile:</b> {html.escape(profile_summary)}</p><h4>Long-term Interactive Motivation Profile</h4>{metric_html(tree["nodes"][0].get("stable_profile", {}))}<h4>Root prompt</h4><p class="muted">{html.escape(truncate(tree.get("root_prompt",""), 680))}</p></div></section><h2>Story Image Branch Tree</h2>'''
    (page_dir / "index.html").write_text(shell("Main Story", story_body, "main"), encoding="utf-8")

    intent_body = '<section class="panel"><h2>Intent Recognition Task</h2><p class="muted">Click any node. The modal shows outgoing grounded regions, closed-domain labels, semi-open ranking, fixed slots, and natural-language intent.</p></section><h2>Story Image Branch Tree</h2>'
    (page_dir / "intent_task.html").write_text(shell("Intent Task", intent_body, "intent"), encoding="utf-8")

    profile_body = '<section class="grid"><div class="panel"><h2>Initial User Profile</h2><p class="muted">' + html.escape(profile_summary) + '</p><h4>Long-term Interactive Motivation Profile</h4>' + metric_html(tree["nodes"][0].get("stable_profile", {})) + '</div><div class="panel"><h2>Scale Definition</h2>' + scale_summary_html(tree) + '</div></section><section class="panel"><h2>Generation-assisted Modeling</h2><p class="muted">Click a node to inspect incoming affect/profile deltas, observed calibration, final deltas, and the prompt that produced the image stimulus.</p></section><h2>Story Image Branch Tree</h2>'
    (page_dir / "profile_modeling.html").write_text(shell("Profile Modeling", profile_body, "profile"), encoding="utf-8")


def build_gallery() -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    topic_cards = []
    root_img_dir = WEB_DIR / "topic_roots"
    root_img_dir.mkdir(exist_ok=True)
    topic_dirs = sorted([p for p in ROOT_ASSETS_DIR.iterdir() if p.is_dir() and ".backup" not in p.name]) if ROOT_ASSETS_DIR.exists() else []
    for topic_dir in topic_dirs:
        asset_path = topic_dir / "root_asset.json"
        if asset_path.exists():
            asset = read_json(asset_path)
        else:
            asset = {"topic_id": topic_dir.name, "topic": slug_title(topic_dir.name), "synopsis": ""}
        src = topic_dir / "root.png"
        dst = root_img_dir / f"{topic_dir.name}.png"
        if src.exists():
            copy_file(src, dst)
        topic_cards.append(f'''<article class="topic-card"><img src="topic_roots/{topic_dir.name}.png"/><div><h3>{html.escape(asset.get("topic", slug_title(topic_dir.name)))}</h3><p>{html.escape(asset.get("synopsis",""))}</p></div></article>''')

    experiment_links = []
    for tree_path in sorted(RUN_DIR.glob("users/user_*/*/tree.json")):
        tree = read_json(tree_path)
        rel = f'users/user_{tree["user_id"]}/{tree["topic_id"]}/index.html'
        experiment_links.append(f'<li><a href="{rel}"><b>{html.escape(tree["tree_id"])}</b></a> | user {tree["user_id"]} | {tree["topic_id"]}</li>')

    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>Contrast Dataset Web</title><style>{css()}</style></head><body><header class="hero"><h1>Contrast Dataset Web</h1><p class="muted">Topic root gallery and task pages for the current contrast run.</p></header><main><section class="panel"><h2>Experiment Groups</h2><ul class="story-lines">{''.join(experiment_links)}</ul></section><h2>12 Topic Root Images</h2><section class="topic-grid">{''.join(topic_cards)}</section></main></body></html>'''
    (WEB_DIR / "index.html").write_text(body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local HTML pages for a generated comic tree dataset run.")
    parser.add_argument("--run-dir", default=str(RUN_DIR), help="Run directory containing manifest.jsonl and users/*/*/tree.json.")
    parser.add_argument("--root-assets-dir", default=str(ROOT_ASSETS_DIR), help="Optional topic root asset gallery directory.")
    parser.add_argument("--web-dir", default="", help="Output web directory. Defaults to <run-dir>/web.")
    return parser.parse_args()


def main() -> None:
    global RUN_DIR, ROOT_ASSETS_DIR, WEB_DIR
    args = parse_args()
    RUN_DIR = Path(args.run_dir)
    if not RUN_DIR.is_absolute():
        RUN_DIR = (BASE_DIR / RUN_DIR).resolve()
    ROOT_ASSETS_DIR = Path(args.root_assets_dir)
    if not ROOT_ASSETS_DIR.is_absolute():
        ROOT_ASSETS_DIR = (BASE_DIR / ROOT_ASSETS_DIR).resolve()
    WEB_DIR = Path(args.web_dir) if args.web_dir else RUN_DIR / "web"
    if not WEB_DIR.is_absolute():
        WEB_DIR = (BASE_DIR / WEB_DIR).resolve()
    if WEB_DIR.exists():
        shutil.rmtree(WEB_DIR)
    WEB_DIR.mkdir(parents=True)
    for tree_path in sorted(RUN_DIR.glob("users/user_*/*/tree.json")):
        write_tree_pages(tree_path, rel_root="../../../")
    build_gallery()
    print(f"Wrote web pages to {WEB_DIR}")


if __name__ == "__main__":
    main()
