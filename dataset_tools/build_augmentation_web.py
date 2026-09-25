from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


INTENT_ORDER = ["interact", "branch_out", "follow", "reveal", "zoom_in", "reframe"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def safe(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def copy_asset(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)


def make_thumbnail(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return
    try:
        from PIL import Image, ImageOps

        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            image.thumbnail((720, 720), Image.Resampling.LANCZOS)
            image.convert("RGB").save(target, "JPEG", quality=74, optimize=True)
    except (ImportError, OSError):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-vf",
             "scale=720:-2:force_original_aspect_ratio=decrease", "-frames:v", "1", "-q:v", "6", str(target)],
            check=True,
        )


def source_tree_for(record: dict[str, Any], source_root: Path) -> Path:
    path = source_root / str(record["source_tree_relative"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing source tree for {record['augmentation_id']}: {path}")
    return path.resolve()


def node_image_for(tree_path: Path, node: dict[str, Any]) -> Path:
    node_id = str(node["node_id"])
    image = Path(str(node.get("image", f"images/{node_id}.png")))
    path = image if image.is_absolute() else tree_path.parent / image
    if not path.is_file():
        path = tree_path.parent / "images" / f"{node_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing image for node {node_id}: {tree_path}")
    return path.resolve()


def source_image_for(record: dict[str, Any], source_root: Path) -> Path:
    tree_path = source_tree_for(record, source_root)
    tree = read_json(tree_path)
    node_id = str(record["source_node_id"])
    node = next(node for node in tree["nodes"] if str(node["node_id"]) == node_id)
    return node_image_for(tree_path, node)


def public_tree(record: dict[str, Any], source_root: Path, web_dir: Path) -> dict[str, Any]:
    augmentation_id = str(record["augmentation_id"])
    tree_path = source_tree_for(record, source_root)
    source_tree = read_json(tree_path)
    asset_dir = web_dir / "assets" / "augmentation-trees" / augmentation_id
    nodes: list[dict[str, Any]] = []
    for node in source_tree["nodes"]:
        node_id = str(node["node_id"])
        image_target = asset_dir / f"{node_id}.jpg"
        make_thumbnail(node_image_for(tree_path, node), image_target)
        story = node.get("story_state", {})
        nodes.append({
            "node_id": node_id,
            "depth": int(node.get("depth", 0)),
            "image": image_target.relative_to(web_dir).as_posix(),
            "location": str(story.get("location", "")),
            "narrative_status": str(story.get("narrative_status", "")),
            "added": False,
        })
    target_node = record["target_node"]
    target_id = str(target_node["node_id"])
    target_image = Path(str(target_node["image"]))
    target_asset = asset_dir / f"{target_id}.jpg"
    make_thumbnail(target_image, target_asset)
    target_story = target_node.get("story_state", {})
    nodes.append({
        "node_id": target_id,
        "depth": int(target_node.get("depth", 0)),
        "image": target_asset.relative_to(web_dir).as_posix(),
        "location": str(target_story.get("location", "")),
        "narrative_status": str(target_story.get("narrative_status", "")),
        "added": True,
    })
    edges = [{
        "source": str(edge["source_node"]),
        "target": str(edge["target_node"]),
        "intent": str(edge.get("closed_intent", "")),
        "label": str(edge.get("branch_label", "")),
        "added": False,
    } for edge in source_tree["edges"]]
    added_edge = record["edge"]
    edges.append({
        "source": str(added_edge["source_node"]),
        "target": str(added_edge["target_node"]),
        "intent": str(added_edge.get("closed_intent", record["target_intent"])),
        "label": str(added_edge.get("branch_label", "")),
        "added": True,
    })
    return {
        "augmentation_id": augmentation_id,
        "source_dataset": str(record["source_dataset"]),
        "user_id": str(record["user_id"]),
        "tree_id": str(record["tree_id"]),
        "topic_id": str(record["topic_id"]),
        "source_node_id": str(record["source_node_id"]),
        "target_node_id": target_id,
        "target_intent": str(record["target_intent"]),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def public_record(record: dict[str, Any], source_root: Path, web_dir: Path) -> dict[str, Any]:
    augmentation_id = str(record["augmentation_id"])
    asset_dir = web_dir / "assets" / "augmentation" / augmentation_id
    source_image = source_image_for(record, source_root)
    target_image = Path(str(record["target_node"]["image"]))
    if not target_image.is_file():
        raise FileNotFoundError(f"Missing target image for {augmentation_id}")
    files = {
        "source_full": asset_dir / "source.png",
        "target_full": asset_dir / "target.png",
        "source_thumb": asset_dir / "source-thumb.jpg",
        "target_thumb": asset_dir / "target-thumb.jpg",
    }
    copy_asset(source_image, files["source_full"])
    copy_asset(target_image, files["target_full"])
    make_thumbnail(source_image, files["source_thumb"])
    make_thumbnail(target_image, files["target_thumb"])
    edge = record["edge"]
    alignment = edge["delta_alignment"]
    calibration = edge["calibration"]
    result = {
        "augmentation_id": augmentation_id,
        "source_dataset": str(record["source_dataset"]),
        "user_id": str(record["user_id"]),
        "tree_id": str(record["tree_id"]),
        "topic_id": str(record["topic_id"]),
        "cohort_intent": str(record["cohort_intent"]),
        "target_intent": str(record["target_intent"]),
        "source_node_id": str(record["source_node_id"]),
        "target_node_id": str(record["target_node"]["node_id"]),
        "natural_language_intent": str(edge["natural_language_intent"]),
        "slots": edge["slots"],
        "target_label": str(edge.get("grounding", {}).get("target_label", "")),
        "node_affordance": float(record["affordance_attempts"][-1]["target_intent_score"]),
        "image_intent_alignment": float(alignment["image_intent_alignment"]),
        "continuity_alignment": float(alignment["continuity_alignment"]),
        "observed_intent": str(calibration["observed_closed_intent"]),
        "image_attempts": len(record.get("image_attempts", [])),
        "used_backup_node": len(record.get("affordance_attempts", [])) > 1,
        "tree_view_id": augmentation_id,
    }
    result.update({key: path.relative_to(web_dir).as_posix() for key, path in files.items()})
    return result


def css() -> str:
    return """
:root{--ink:#17211f;--muted:#68736f;--paper:#f7f9f8;--line:#d7dedb;--green:#176c52;--soft:#e9f5ef;--amber:#a45b14;--amber-soft:#fff3df;--blue:#1f5fa8;--shadow:0 12px 30px rgba(23,33,31,.08)}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);font-family:Aptos,"Segoe UI","Noto Sans SC",Arial,sans-serif}.topbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:18px;min-height:58px;padding:8px clamp(18px,4vw,56px);color:#fff;background:#17211f;border-bottom:1px solid #34413d}.brand{margin-right:auto;font-weight:800;text-decoration:none}.nav{display:flex;gap:7px;flex-wrap:wrap}.nav a{padding:8px 11px;border:1px solid #46534f;border-radius:5px;color:#e5ece9;font-size:12px;text-decoration:none}.nav a:hover,.nav a.active{background:#2b3935}
.hero{padding:36px clamp(18px,5vw,72px) 28px;background:#fff;border-bottom:1px solid var(--line)}.eyebrow{margin:0 0 8px;color:var(--green);font-size:12px;font-weight:800;text-transform:uppercase}.hero h1{margin:0;max-width:900px;font-family:Georgia,"Noto Serif SC",serif;font-size:clamp(30px,4vw,48px);letter-spacing:0}.lead{max-width:860px;margin:12px 0 0;color:var(--muted);font-size:15px;line-height:1.7}.stats{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:1px;max-width:1100px;margin-top:26px;background:var(--line);border:1px solid var(--line)}.stat{padding:15px;background:#fbfcfc}.stat b{display:block;color:var(--green);font-family:Georgia,serif;font-size:27px}.stat span{display:block;margin-top:4px;color:var(--muted);font-size:11px}
.filters{position:sticky;top:58px;z-index:15;display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px clamp(18px,5vw,72px);background:rgba(247,249,248,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}.search{width:min(320px,100%);height:38px;padding:0 11px;border:1px solid #bfcac6;border-radius:5px;background:#fff;font:inherit}.filter-group{display:flex;gap:5px;flex-wrap:wrap}.filter-button{height:34px;padding:0 10px;border:1px solid #bdc9c5;border-radius:5px;color:#3f4b47;background:#fff;font-size:12px;cursor:pointer}.filter-button.active{color:#fff;background:var(--green);border-color:var(--green)}.shown{margin-left:auto;color:var(--muted);font-size:12px}
.main{padding:22px clamp(18px,5vw,72px) 80px}.distribution{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 25px}.count-pill{display:flex;align-items:center;gap:7px;padding:8px 10px;background:#fff;border:1px solid var(--line);border-radius:5px;font-size:12px}.count-pill b{color:var(--green)}
.user-section{padding:24px 0 30px;border-top:2px solid #a8bbb4}.user-section:first-of-type{border-top:0}.user-header{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin-bottom:14px}.user-header h2{margin:0;font-family:Georgia,"Noto Serif SC",serif;font-size:25px;letter-spacing:0}.user-meta{margin-top:5px;color:var(--muted);font-size:12px}.badges{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.badge{display:inline-flex;align-items:center;min-height:27px;padding:4px 8px;border:1px solid #b8d8c9;border-radius:4px;color:var(--green);background:var(--soft);font-size:11px;font-weight:800}.badge.new{color:#fff;background:var(--green);border-color:var(--green)}.badge.v2{color:var(--amber);background:var(--amber-soft);border-color:#edc994}
.branch-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:14px}.branch{min-width:0;overflow:hidden;background:#fff;border:2px solid #73a992;border-radius:6px;box-shadow:var(--shadow)}.branch-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;padding:12px 13px;border-bottom:1px solid var(--line)}.branch-head h3{margin:7px 0 0;font-size:15px;overflow-wrap:anywhere}.branch-head small{display:block;margin-top:4px;color:var(--muted)}.intent{padding:5px 8px;border-radius:4px;color:#fff;background:var(--blue);font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}.panel{position:relative;margin:0;background:#111}.panel img{display:block;width:100%;aspect-ratio:3/2;object-fit:cover;cursor:zoom-in}.panel figcaption{position:absolute;left:7px;bottom:7px;padding:4px 6px;color:#fff;background:rgba(13,20,18,.82);border-radius:3px;font-size:10px}.panel.new-panel{outline:3px solid #4fa37f;outline-offset:-3px}
.branch-body{padding:13px}.intent-text{margin:0 0 12px;font-size:13px;line-height:1.55}.node-line{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:11px;color:var(--muted);font:11px ui-monospace,SFMono-Regular,Consolas,monospace}.node-line strong{color:var(--green)}.quality{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:11px}.quality div{padding:7px;background:#f2f6f4;border:1px solid #dce6e2}.quality b{display:block;font-size:14px}.quality span{color:var(--muted);font-size:9px}.slots{margin:0;padding:9px 10px;background:#f8faf9;border-left:3px solid #9eb8ad;font-size:11px;line-height:1.5}.slots dt{float:left;clear:left;margin-right:5px;color:var(--muted)}.slots dd{margin:0 0 4px;overflow-wrap:anywhere}.tree-tag{margin:0 0 10px;color:var(--green);font-size:11px;font-weight:800}.branch-actions{display:flex;justify-content:flex-end;margin-top:12px}.tree-button{display:inline-flex;align-items:center;min-height:36px;padding:7px 11px;color:#fff;background:var(--green);border:1px solid var(--green);border-radius:4px;font-size:12px;font-weight:800;text-decoration:none}.tree-button:hover{background:#0f5942}.empty{display:none;padding:30px;text-align:center;color:var(--muted);border:1px dashed #aebbb6}.empty.visible{display:block}.user-section.hidden,.branch.hidden{display:none}
.method-band{padding:28px clamp(18px,5vw,72px);background:#eef3f1;border-bottom:1px solid var(--line)}.method-inner{max-width:1500px;margin:0 auto}.section-title{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:15px}.section-title h2{margin:0;font-family:Georgia,"Noto Serif SC",serif;font-size:25px;letter-spacing:0}.section-title p{max-width:680px;margin:0;color:var(--muted);font-size:13px;line-height:1.6}.workflow{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:24px}.workflow-step{min-width:0;padding:13px;background:#fff;border:1px solid var(--line);border-top:3px solid var(--green)}.workflow-step b{display:block;margin-bottom:6px;color:var(--green);font-size:11px}.workflow-step strong{display:block;margin-bottom:5px;font-size:13px}.workflow-step span{display:block;color:var(--muted);font-size:11px;line-height:1.45}.ratio-layout{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(420px,.75fr);gap:16px}.ratio-panel{padding:16px;background:#fff;border:1px solid var(--line)}.ratio-panel h3{margin:0 0 12px;font-size:15px}.stack-label{display:flex;justify-content:space-between;gap:12px;margin:10px 0 5px;color:var(--muted);font-size:11px}.stack{display:flex;height:22px;overflow:hidden;background:#e5ebe8;border:1px solid #d5ded9}.stack span{display:block;height:100%;min-width:1px}.segment-interact{background:#445b54}.segment-branch_out{background:#2d7c65}.segment-follow{background:#4f8db7}.segment-reveal{background:#d79a3b}.segment-zoom_in{background:#b96645}.segment-reframe{background:#77599b}.ratio-legend{display:flex;gap:9px;flex-wrap:wrap;margin-top:10px;color:var(--muted);font-size:10px}.ratio-legend span::before{content:"";display:inline-block;width:8px;height:8px;margin-right:4px;background:var(--swatch)}.ratio-table{width:100%;border-collapse:collapse;font-size:11px}.ratio-table th,.ratio-table td{padding:6px 7px;text-align:right;border-bottom:1px solid #e4e9e7}.ratio-table th:first-child,.ratio-table td:first-child{text-align:left}.ratio-table th{color:var(--muted);font-weight:700}.ratio-table .rare{font-weight:800;color:var(--green)}.ratio-note{margin:10px 0 0;color:var(--muted);font-size:10px;line-height:1.5}
.lightbox{width:min(96vw,1600px);height:min(94vh,1000px);max-width:none;max-height:none;padding:0;background:#0c1210;border:1px solid #46534f;border-radius:6px}.lightbox::backdrop{background:rgba(4,8,7,.88)}.lightbox[open]{display:grid;grid-template-rows:54px minmax(0,1fr)}.lightbox-bar{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;color:#fff;background:#18231f}.lightbox-bar button{width:38px;height:36px;color:#fff;background:#2c3b35;border:1px solid #506059;border-radius:4px;font-size:20px;cursor:pointer}.lightbox-view{display:flex;align-items:center;justify-content:center;min-width:0;min-height:0;overflow:auto;padding:15px}.lightbox-view img{display:block;max-width:100%;max-height:100%;object-fit:contain}
@media(max-width:1100px){.workflow{grid-template-columns:repeat(3,1fr)}.ratio-layout{grid-template-columns:1fr}}@media(max-width:900px){.stats{grid-template-columns:repeat(2,1fr)}.branch-grid{grid-template-columns:1fr}.topbar{position:static}.filters{top:0}.shown{width:100%;margin-left:0}}@media(max-width:560px){.nav a:not(.active){display:none}.workflow{grid-template-columns:1fr 1fr}.section-title{align-items:flex-start;flex-direction:column}.stats{grid-template-columns:1fr 1fr}.branch-grid{grid-template-columns:minmax(0,1fr)}.compare{grid-template-columns:1fr}.quality{grid-template-columns:1fr 1fr}.user-header{flex-direction:column}.search{width:100%}}
"""


def js() -> str:
    return """
(() => {
  const buttons = [...document.querySelectorAll('[data-intent-filter]')];
  const search = document.querySelector('#search');
  const branches = [...document.querySelectorAll('.branch')];
  const users = [...document.querySelectorAll('.user-section')];
  const shown = document.querySelector('#shown-count');
  const empty = document.querySelector('#empty');
  let intent = 'all';
  const apply = () => {
    const query = (search.value || '').trim().toLowerCase();
    let count = 0;
    branches.forEach((branch) => {
      const visible = (intent === 'all' || branch.dataset.intent === intent) && (!query || branch.dataset.search.includes(query));
      branch.classList.toggle('hidden', !visible);
      if (visible) count += 1;
    });
    users.forEach((user) => user.classList.toggle('hidden', !user.querySelector('.branch:not(.hidden)')));
    shown.textContent = count + ' / ' + branches.length + ' branches';
    empty.classList.toggle('visible', count === 0);
  };
  buttons.forEach((button) => button.addEventListener('click', () => {
    intent = button.dataset.intentFilter;
    buttons.forEach((item) => item.classList.toggle('active', item === button));
    apply();
  }));
  search.addEventListener('input', apply);
  apply();
  const dialog = document.querySelector('#lightbox');
  const image = document.querySelector('#lightbox-image');
  const title = document.querySelector('#lightbox-title');
  document.addEventListener('click', (event) => {
    const preview = event.target.closest('[data-full]');
    if (preview) {
      image.src = preview.dataset.full;
      image.alt = preview.alt || '';
      title.textContent = preview.alt || 'Image preview';
      dialog.showModal();
      return;
    }
    if (event.target.closest('[data-close]')) dialog.close();
    if (event.target === dialog) dialog.close();
  });
})();
"""


def render_branch(row: dict[str, Any]) -> str:
    slots = row["slots"]
    search_text = " ".join([row["source_dataset"], row["user_id"], row["tree_id"], row["topic_id"], row["target_intent"], row["cohort_intent"], row["natural_language_intent"]]).lower()
    v2 = '<span class="badge v2">人工复核 · v2</span>' if row["image_attempts"] > 1 else ""
    backup = '<span class="badge v2">备用节点</span>' if row["used_backup_node"] else ""
    return f"""
<article class="branch" data-intent="{safe(row['target_intent'])}" data-search="{safe(search_text)}">
<header class="branch-head"><div><div class="badges"><span class="badge new">新增分支</span>{v2}{backup}</div><h3>{safe(row['topic_id'])}</h3><small>{safe(row['tree_id'])}</small></div><code class="intent">{safe(row['target_intent'])}</code></header>
<div class="compare">
<figure class="panel"><img loading="lazy" decoding="async" src="{safe(row['source_thumb'])}" data-full="{safe(row['source_full'])}" alt="{safe(row['tree_id'])} source node {safe(row['source_node_id'])}"><figcaption>源节点 · {safe(row['source_node_id'])}</figcaption></figure>
<figure class="panel new-panel"><img loading="lazy" decoding="async" src="{safe(row['target_thumb'])}" data-full="{safe(row['target_full'])}" alt="{safe(row['tree_id'])} added {safe(row['target_intent'])} branch"><figcaption>新增节点 · {safe(row['target_node_id'])}</figcaption></figure>
</div>
<div class="branch-body"><p class="tree-tag">已增强树 · 主倾向 {safe(row['cohort_intent'])} → 本分支 {safe(row['target_intent'])}</p><p class="intent-text">{safe(row['natural_language_intent'])}</p>
<div class="node-line"><span>{safe(row['source_node_id'])}</span><span>→</span><strong>{safe(row['target_node_id'])}</strong><span>· target: {safe(row['target_label'])}</span></div>
<div class="quality"><div><b>{row['node_affordance']:.2f}</b><span>节点可供性</span></div><div><b>{row['image_intent_alignment']:.2f}</b><span>视觉意图对齐</span></div><div><b>{row['continuity_alignment']:.2f}</b><span>叙事连续性</span></div></div>
<dl class="slots"><dt>action</dt><dd>{safe(slots.get('action'))}</dd><dt>target</dt><dd>{safe(slots.get('target'))}</dd><dt>goal</dt><dd>{safe(slots.get('narrative_goal'))}</dd><dt>mood</dt><dd>{safe(slots.get('mood'))}</dd><dt>continuity</dt><dd>{safe(slots.get('continuity_constraint'))}</dd><dt>scope</dt><dd>{safe(slots.get('scope'))}</dd></dl><div class="branch-actions"><a class="tree-button" href="augmentation-tree.html?id={safe(row['tree_view_id'])}">查看补充后整棵树</a></div></div>
</article>"""


def distribution_overview(quality: dict[str, Any]) -> str:
    added = Counter({key: int(value) for key, value in quality["intent_counts"].items()})
    after = Counter({key: int(value) for key, value in quality["combined_intent_counts"].items()})
    before = Counter({label: after[label] - added[label] for label in INTENT_ORDER})
    before_total = sum(before.values())
    after_total = sum(after.values())

    def percent(count: int, total: int) -> float:
        return 100.0 * count / total if total else 0.0

    def stack(counts: Counter[str], total: int, label: str) -> str:
        segments = "".join(
            f'<span class="segment-{intent}" style="width:{percent(counts[intent], total):.4f}%" title="{intent}: {counts[intent]} ({percent(counts[intent], total):.3f}%)"></span>'
            for intent in INTENT_ORDER
        )
        return f'<div class="stack-label"><strong>{label}</strong><span>{total:,} 条</span></div><div class="stack" aria-label="{label}标签比例">{segments}</div>'

    rows = "".join(
        f'<tr class="{"rare" if intent in {"zoom_in", "reframe"} else ""}"><td><code>{intent}</code></td><td>{before[intent]:,}</td><td>{percent(before[intent], before_total):.3f}%</td><td>+{added[intent]:,}</td><td>{after[intent]:,}</td><td>{percent(after[intent], after_total):.3f}%</td></tr>'
        for intent in INTENT_ORDER
    )
    colors = {"interact": "#445b54", "branch_out": "#2d7c65", "follow": "#4f8db7", "reveal": "#d79a3b", "zoom_in": "#b96645", "reframe": "#77599b"}
    legend = "".join(f'<span style="--swatch:{colors[intent]}">{intent}</span>' for intent in INTENT_ORDER)
    return f'''<section class="method-band"><div class="method-inner">
<div class="section-title"><div><p class="eyebrow">Augmentation pipeline</p><h2>分支补充流程与当前标签比例</h2></div><p>补充仅写入独立 sidecar，不覆盖原树。候选分支同时经过画像倾向、当前节点可供性、剧情连续性和生成图像校准，避免为了数量强行改变故事。</p></div>
<div class="workflow" aria-label="分支补充流程"><article class="workflow-step"><b>01</b><strong>画像群体筛选</strong><span>从长期 8 维画像识别更可能产生少数类意图的用户。</span></article><article class="workflow-step"><b>02</b><strong>源节点候选</strong><span>在不同 topic 和深度中选择可自然扩展的已有节点。</span></article><article class="workflow-step"><b>03</b><strong>节点可供性验证</strong><span>判断该画面是否真实支持 zoom、reframe、follow 等操作。</span></article><article class="workflow-step"><b>04</b><strong>混合意图规划</strong><span>结合画像、topic 和叙事状态分配意图，同一用户不固定单一标签。</span></article><article class="workflow-step"><b>05</b><strong>分支图像生成</strong><span>生成一个新目标节点，保留人物、场景和时间线连续性。</span></article><article class="workflow-step"><b>06</b><strong>视觉校准与接纳</strong><span>复核意图对齐和连续性，达标后作为 sidecar 分支接入。</span></article></div>
<div class="ratio-layout"><article class="ratio-panel"><h3>全数据补充前后</h3>{stack(before, before_total, "补充前")}{stack(after, after_total, "补充后")}<div class="ratio-legend">{legend}</div><p class="ratio-note">补充后共 {after_total:,} 条；本批接纳 {sum(added.values())} / 100 条。稀有标签仍保持低占比，只提升可评测覆盖，而非追求机械六类均衡。</p></article><article class="ratio-panel"><h3>六类精确比例</h3><table class="ratio-table"><thead><tr><th>Intent</th><th>补充前</th><th>比例</th><th>本批新增</th><th>当前</th><th>比例</th></tr></thead><tbody>{rows}</tbody></table></article></div>
</div></section>'''


def tree_css() -> str:
    return '''
:root{--ink:#17211f;--muted:#68736f;--paper:#f4f7f5;--line:#ccd7d2;--green:#176c52;--added:#32a474}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);font-family:Aptos,"Segoe UI","Noto Sans SC",Arial,sans-serif}.topbar{display:flex;align-items:center;gap:14px;min-height:58px;padding:8px clamp(16px,4vw,54px);color:#fff;background:#17211f}.topbar a{color:#fff;text-decoration:none}.topbar .brand{margin-right:auto;font-weight:800}.back{padding:8px 11px;border:1px solid #52615c;border-radius:4px;font-size:12px}.tree-hero{padding:24px clamp(16px,4vw,54px);background:#fff;border-bottom:1px solid var(--line)}.eyebrow{margin:0 0 7px;color:var(--green);font-size:11px;font-weight:800;text-transform:uppercase}.tree-hero h1{margin:0;font-family:Georgia,"Noto Serif SC",serif;font-size:clamp(25px,3vw,38px);letter-spacing:0}.tree-hero>p:not(.eyebrow){margin:7px 0 0;color:var(--muted)}.tree-meta{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.badge{padding:5px 8px;color:#3d5049;background:#eef4f1;border:1px solid #c5d5ce;border-radius:4px;font-size:11px}.badge.added{color:#fff;background:var(--green);border-color:var(--green)}.legend{display:flex;gap:18px;flex-wrap:wrap;padding:12px clamp(16px,4vw,54px);color:var(--muted);background:#edf2ef;border-bottom:1px solid var(--line);font-size:11px}.legend i{display:inline-block;width:22px;height:3px;margin-right:6px;vertical-align:middle;background:#85938e}.legend .new i{height:5px;background:var(--added)}.tree-stage{min-height:560px;overflow:auto;padding:26px}.tree-canvas{position:relative;display:flex;flex-direction:column;gap:72px;width:max-content;min-width:100%;padding:12px 30px 50px}.edge-layer{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none}.edge{fill:none;stroke:#91a09a;stroke-width:2}.edge.added{stroke:var(--added);stroke-width:5}.level{position:relative;z-index:2;display:flex;justify-content:center;gap:15px;min-height:188px}.node{width:210px;overflow:hidden;background:#fff;border:1px solid #bac8c2;border-radius:5px;box-shadow:0 7px 20px rgba(23,33,31,.08)}.node.added{border:3px solid var(--added);box-shadow:0 0 0 4px rgba(50,164,116,.14),0 8px 24px rgba(23,33,31,.12)}.node img{display:block;width:100%;aspect-ratio:3/2;object-fit:cover;background:#18201d;cursor:zoom-in}.node-body{padding:9px}.node-head{display:flex;justify-content:space-between;gap:6px;align-items:center}.node-head strong{font:800 11px ui-monospace,SFMono-Regular,Consolas,monospace}.intent{max-width:105px;overflow:hidden;text-overflow:ellipsis;padding:3px 5px;color:#fff;background:#52645d;border-radius:3px;font:700 9px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.node.added .intent{background:var(--added)}.node p{display:-webkit-box;overflow:hidden;margin:7px 0 0;color:var(--muted);font-size:10px;line-height:1.4;-webkit-box-orient:vertical;-webkit-line-clamp:3}.error{margin:50px auto;padding:22px;max-width:680px;color:#8b351e;background:#fff2ec;border:1px solid #e6b5a7}.lightbox{width:min(96vw,1500px);height:min(94vh,960px);max-width:none;max-height:none;padding:0;background:#0c1210;border:1px solid #46534f;border-radius:6px}.lightbox::backdrop{background:rgba(4,8,7,.88)}.lightbox[open]{display:grid;grid-template-rows:52px minmax(0,1fr)}.lightbox header{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;color:#fff}.lightbox button{width:38px;height:35px;color:#fff;background:#293630;border:1px solid #52615b;border-radius:4px;font-size:20px}.lightbox div{display:flex;align-items:center;justify-content:center;min-height:0;padding:14px}.lightbox img{max-width:100%;max-height:100%;object-fit:contain}@media(max-width:600px){.tree-stage{padding:14px}.tree-canvas{gap:54px;padding-inline:15px}.node{width:170px}.level{gap:10px;min-height:155px}}
'''


def tree_js() -> str:
    return '''
(async () => {
  const root = document.querySelector('#tree-root');
  const id = new URLSearchParams(location.search).get('id');
  try {
    const response = await fetch('augmentation-tree-data.json');
    if (!response.ok) throw new Error('tree data HTTP ' + response.status);
    const payload = await response.json();
    const tree = payload.trees[id];
    if (!tree) throw new Error('未找到指定增强树：' + (id || '(empty id)'));
    document.title = 'WorldWeaver · ' + tree.tree_id;
    document.querySelector('#tree-title').textContent = tree.topic_id;
    document.querySelector('#tree-subtitle').textContent = tree.tree_id;
    const meta = document.querySelector('#tree-meta');
    [['增强用户', 'User ' + tree.user_id], ['数据来源', tree.source_dataset], ['节点', tree.node_count], ['边', tree.edge_count], ['新增意图', tree.target_intent]].forEach(([label, value], index) => {
      const span = document.createElement('span');
      span.className = 'badge' + (index === 4 ? ' added' : '');
      span.textContent = label + ' · ' + value;
      meta.appendChild(span);
    });
    const incoming = new Map(tree.edges.map((edge) => [edge.target, edge]));
    const levels = new Map();
    tree.nodes.forEach((node) => {
      if (!levels.has(node.depth)) levels.set(node.depth, []);
      levels.get(node.depth).push(node);
    });
    const canvas = document.createElement('div');
    canvas.className = 'tree-canvas';
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('edge-layer');
    canvas.appendChild(svg);
    [...levels.keys()].sort((a, b) => a - b).forEach((depth) => {
      const level = document.createElement('section');
      level.className = 'level';
      level.dataset.depth = depth;
      levels.get(depth).sort((a, b) => a.node_id.localeCompare(b.node_id, undefined, {numeric:true})).forEach((node) => {
        const edge = incoming.get(node.node_id);
        const article = document.createElement('article');
        article.className = 'node' + (node.added ? ' added' : '');
        article.dataset.nodeId = node.node_id;
        const img = document.createElement('img');
        img.src = node.image;
        img.alt = node.node_id + (node.added ? ' added node' : '');
        img.loading = 'lazy';
        img.dataset.full = node.image;
        const body = document.createElement('div');
        body.className = 'node-body';
        const head = document.createElement('div');
        head.className = 'node-head';
        const name = document.createElement('strong');
        name.textContent = node.node_id + (node.added ? ' · 新增' : '');
        head.appendChild(name);
        if (edge) {
          const intent = document.createElement('span');
          intent.className = 'intent';
          intent.textContent = edge.intent || 'unknown';
          intent.title = edge.label || edge.intent || '';
          head.appendChild(intent);
        }
        const status = document.createElement('p');
        status.textContent = node.narrative_status || node.location || '';
        body.append(head, status);
        article.append(img, body);
        level.appendChild(article);
      });
      canvas.appendChild(level);
    });
    root.appendChild(canvas);
    const drawEdges = () => {
      svg.replaceChildren();
      const box = canvas.getBoundingClientRect();
      tree.edges.forEach((edge) => {
        const source = canvas.querySelector('[data-node-id="' + CSS.escape(edge.source) + '"]');
        const target = canvas.querySelector('[data-node-id="' + CSS.escape(edge.target) + '"]');
        if (!source || !target) return;
        const a = source.getBoundingClientRect();
        const b = target.getBoundingClientRect();
        const x1 = a.left + a.width / 2 - box.left;
        const y1 = a.bottom - box.top;
        const x2 = b.left + b.width / 2 - box.left;
        const y2 = b.top - box.top;
        const mid = (y1 + y2) / 2;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`);
        path.setAttribute('class', 'edge' + (edge.added ? ' added' : ''));
        svg.appendChild(path);
      });
    };
    requestAnimationFrame(drawEdges);
    window.addEventListener('resize', drawEdges);
    document.fonts?.ready.then(drawEdges);
  } catch (error) {
    root.innerHTML = '<div class="error"></div>';
    root.firstElementChild.textContent = error.message;
  }
  const dialog = document.querySelector('#lightbox');
  const image = document.querySelector('#lightbox-image');
  document.addEventListener('click', (event) => {
    const preview = event.target.closest('[data-full]');
    if (preview) { image.src = preview.dataset.full; image.alt = preview.alt || ''; dialog.showModal(); }
    if (event.target.closest('[data-close]') || event.target === dialog) dialog.close();
  });
})();
'''


def tree_page() -> str:
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="WorldWeaver 补充后的完整树视图。"><title>WorldWeaver · 补充后的完整树</title><link rel="stylesheet" href="augmentation-tree.css"></head><body><header class="topbar"><a class="brand" href="project.html">WorldWeaver</a><a class="back" href="augmentation.html">返回增强分支</a></header><section class="tree-hero"><p class="eyebrow">Original tree + accepted sidecar branch</p><h1 id="tree-title">补充后的完整树</h1><p id="tree-subtitle"></p><div class="tree-meta" id="tree-meta"></div></section><div class="legend"><span><i></i>原始树边 / 节点</span><span class="new"><i></i>本次新增边 / 节点</span><span>点击任意节点图像可放大查看</span></div><main class="tree-stage" id="tree-root"></main><dialog class="lightbox" id="lightbox"><header><strong>节点图像</strong><button type="button" data-close aria-label="关闭">×</button></header><div><img id="lightbox-image" alt=""></div></dialog><script src="augmentation-tree.js"></script></body></html>'''


def build_page(augmentation_dir: Path, web_dir: Path) -> Path:
    web_dir.mkdir(parents=True, exist_ok=True)
    run_config = read_json(augmentation_dir / "run_config.json")
    quality = read_json(augmentation_dir / "quality_summary.json")
    source_root = Path(str(run_config["source_merged_dataset"]))
    records = read_jsonl(augmentation_dir / "usable_branches.jsonl")
    rows = [public_record(record, source_root, web_dir) for record in records]
    with ThreadPoolExecutor(max_workers=8) as executor:
        built_trees = list(executor.map(lambda record: public_tree(record, source_root, web_dir), records))
    trees = {tree["augmentation_id"]: tree for tree in built_trees}
    rows.sort(key=lambda row: (row["source_dataset"], int(row["user_id"]) if row["user_id"].isdigit() else row["user_id"], row["topic_id"]))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_dataset"], row["user_id"])].append(row)
    sections = []
    for (source_dataset, user_id), user_rows in grouped.items():
        intents = Counter(row["target_intent"] for row in user_rows)
        intent_badges = "".join(f'<span class="badge">{safe(label)} {count}</span>' for label, count in sorted(intents.items()))
        sections.append(f'<section class="user-section"><header class="user-header"><div><div class="badges"><span class="badge new">增强用户</span><span class="badge">{safe(source_dataset)}</span></div><h2>User {safe(user_id)}</h2><div class="user-meta">{len(user_rows)} 棵已增强树 · 每棵新增 1 条自然适配分支</div></div><div class="badges">{intent_badges}</div></header><div class="branch-grid">{"".join(render_branch(row) for row in user_rows)}</div></section>')
    counts = Counter(row["target_intent"] for row in rows)
    order = ["zoom_in", "reframe", "follow", "reveal", "branch_out"]
    pills = "".join(f'<div class="count-pill"><code>{label}</code><b>{counts[label]}</b></div>' for label in order)
    buttons = "".join(f'<button class="filter-button" type="button" data-intent-filter="{label}">{label}</button>' for label in order)
    (web_dir / "augmentation-data.json").write_text(json.dumps({"schema_version": "augmentation_web_v2", "summary": quality, "records": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (web_dir / "augmentation-tree-data.json").write_text(json.dumps({"schema_version": "augmentation_tree_web_v1", "trees": trees}, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (web_dir / "augmentation.css").write_text(css(), encoding="utf-8")
    (web_dir / "augmentation.js").write_text(js(), encoding="utf-8")
    (web_dir / "augmentation-tree.css").write_text(tree_css(), encoding="utf-8")
    (web_dir / "augmentation-tree.js").write_text(tree_js(), encoding="utf-8")
    (web_dir / "augmentation-tree.html").write_text(tree_page(), encoding="utf-8")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="WorldWeaver 画像与叙事双约束少数类新增分支数据。"><title>WorldWeaver · 增强分支数据</title><link rel="stylesheet" href="augmentation.css"></head><body>
<header class="topbar"><a class="brand" href="project.html">WorldWeaver</a><nav class="nav"><a href="project.html">项目主页</a><a href="index.html">均衡批次</a><a class="active" href="augmentation.html">增强分支</a></nav></header>
<section class="hero"><p class="eyebrow">Profile-grounded minority augmentation · accepted sidecar</p><h1>画像与叙事双约束的新增分支</h1><p class="lead">从修复后的自然树节点出发，为画像适配的目标用户跨不同 topic 增加五类非 interact 分支。绿色边框表示新增树分支；源节点与目标节点并排展示，旧树 JSON 和旧图像保持不变。</p>
<div class="stats"><div class="stat"><b>{quality['selected_user_count']}</b><span>特别标注的增强用户</span></div><div class="stat"><b>{quality['record_count']}</b><span>已增强树 / 新增分支</span></div><div class="stat"><b>{quality['selected_target_image_count']}</b><span>最终新增图像</span></div><div class="stat"><b>{quality['image_intent_alignment']['mean']:.3f}</b><span>平均视觉意图对齐</span></div><div class="stat"><b>{quality['continuity_alignment']['mean']:.3f}</b><span>平均叙事连续性</span></div></div></section>
{distribution_overview(quality)}
<section class="filters"><input class="search" id="search" type="search" placeholder="搜索 user ID、topic、tree 或描述"><div class="filter-group"><button class="filter-button active" type="button" data-intent-filter="all">全部</button>{buttons}</div><output class="shown" id="shown-count"></output></section>
<main class="main"><div class="distribution">{pills}</div>{"".join(sections)}<div class="empty" id="empty">没有符合当前筛选条件的增强分支。</div></main>
<dialog class="lightbox" id="lightbox"><header class="lightbox-bar"><strong id="lightbox-title">图像预览</strong><button type="button" data-close aria-label="关闭">×</button></header><div class="lightbox-view"><img id="lightbox-image" alt=""></div></dialog><script src="augmentation.js"></script></body></html>"""
    output = web_dir / "augmentation.html"
    output.write_text(page, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the minority-branch augmentation web viewer.")
    parser.add_argument("--augmentation-dir", required=True)
    parser.add_argument("--web-dir", required=True)
    args = parser.parse_args()
    print(build_page(Path(args.augmentation_dir).resolve(), Path(args.web_dir).resolve()))


if __name__ == "__main__":
    main()
