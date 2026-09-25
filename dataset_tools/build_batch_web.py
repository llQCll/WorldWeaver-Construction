from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe(text: Any) -> str:
    return html.escape(str(text if text is not None else ""))


def slug_title(slug: str) -> str:
    return slug.replace("_", " ").title()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def latest_run(user_dir: Path) -> Path | None:
    runs_dir = user_dir / "runs"
    if not runs_dir.exists():
        return None
    runs = sorted(path for path in runs_dir.iterdir() if path.is_dir() and path.name.startswith("run_"))
    return runs[-1] if runs else None


def status_for(user_dir: Path) -> dict[str, Any]:
    path = user_dir / "job_status.json"
    if path.exists():
        return read_json(path)
    return {"status": "pending"}


def collect_topic_cards(run_dir: Path, user_id: str, web_dir: Path) -> list[dict[str, Any]]:
    manifest = read_jsonl(run_dir / "manifest.jsonl") if (run_dir / "manifest.jsonl").exists() else []
    cards: list[dict[str, Any]] = []
    user_dir_name = f"user_{user_id}"
    for row in manifest:
        tree_path = Path(row["tree_path"])
        if not tree_path.exists():
            # Manifests produced on Windows contain absolute drive paths. The
            # run layout is portable, so reconstruct the tree path locally.
            topic_id = str(row.get("topic_id") or "")
            tree_path = run_dir / "users" / user_dir_name / topic_id / "tree.json"
        if not tree_path.exists():
            continue
        tree = read_json(tree_path)
        topic_id = str(tree.get("topic_id") or row.get("topic_id") or tree_path.parent.name)
        topic_title = slug_title(topic_id)
        nodes = tree.get("nodes", [])
        edges = tree.get("edges", [])
        root_image = Path(str(tree.get("root_image") or (tree_path.parent / "images" / "n0.png")))
        if not root_image.exists():
            root_image = tree_path.parent / "images" / "n0.png"
        img_rel = ""
        if root_image.exists():
            dst = web_dir / "assets" / user_id / topic_id / "root.png"
            copy_file(root_image, dst)
            img_rel = dst.relative_to(web_dir).as_posix()
        synopsis = ""
        root_asset = tree_path.parent / "root_asset.json"
        if root_asset.exists():
            asset = read_json(root_asset)
            synopsis = str(asset.get("synopsis") or asset.get("root_synopsis") or "")
        if not synopsis:
            synopsis = str(tree.get("global_synopsis") or tree.get("synopsis") or "")
        run_web_base = Path("..") / user_dir_name / "runs" / run_dir.name / "web"
        topic_web_base = run_web_base / "users" / user_dir_name / topic_id
        cards.append(
            {
                "topic_id": topic_id,
                "topic_title": topic_title,
                "image": img_rel,
                "synopsis": synopsis,
                "nodes": len(nodes),
                "edges": len(edges),
                "story_href": (topic_web_base / "index.html").as_posix(),
                "intent_href": (topic_web_base / "intent_task.html").as_posix(),
                "profile_href": (topic_web_base / "profile_modeling.html").as_posix(),
            }
        )
    return cards


def collect_topic_summaries(root_assets_dir: Path, web_dir: Path) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    if not root_assets_dir.exists():
        return summaries
    for topic_dir in sorted(path for path in root_assets_dir.iterdir() if path.is_dir()):
        asset_path = topic_dir / "root_asset.json"
        if not asset_path.exists():
            continue
        asset = read_json(asset_path)
        topic_id = str(asset.get("topic_id") or topic_dir.name)
        topic_title = str(asset.get("topic") or slug_title(topic_id))
        synopsis = str(asset.get("synopsis") or asset.get("world_bible", {}).get("premise") or "")
        root_image = Path(str(asset.get("root_image") or "root.png"))
        if not root_image.is_absolute():
            root_image = topic_dir / root_image
        image_rel = ""
        if root_image.exists():
            dst = web_dir / "assets" / "topic_library" / topic_id / "root.png"
            copy_file(root_image, dst)
            image_rel = dst.relative_to(web_dir).as_posix()
        summaries.append(
            {
                "topic_id": topic_id,
                "topic_title": topic_title,
                "synopsis": synopsis,
                "image": image_rel,
            }
        )
    return summaries


def css() -> str:
    return """
:root{--ink:#111827;--muted:#667085;--line:#d7dde8;--blue:#155eef;--sky:#eaf3ff;--green:#16805c;--amber:#b7791f;--red:#c2413b}
*{box-sizing:border-box}body{margin:0;font-family:Aptos,Segoe UI,Calibri,Arial,sans-serif;color:var(--ink);background:linear-gradient(180deg,#f7fbff 0,#eef3f8 42%,#f9fafb 100%)}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
.hero{padding:26px 42px;background:#fff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10;box-shadow:0 8px 28px rgba(15,23,42,.05)}.hero-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:24px}.hero-actions{display:grid;gap:8px;min-width:330px}.home-button,.batch-switch a{display:inline-flex;align-items:center;justify-content:center;min-height:40px;padding:8px 14px;border-radius:6px;font-size:13px;font-weight:700;white-space:nowrap}.home-button{color:#fff;background:var(--ink);border:1px solid var(--ink)}.home-button:hover{color:#fff;text-decoration:none;background:#263143}.batch-switch{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.batch-switch a{color:var(--ink);background:#fff;border:1px solid var(--line)}.batch-switch a:hover{text-decoration:none;border-color:var(--blue)}.batch-switch a.active{color:#fff;background:var(--blue);border-color:var(--blue)}.batch-switch a.augmentation{color:#fff;background:var(--green);border-color:var(--green)}h1{margin:0 0 8px;font-size:31px;letter-spacing:0}h2{margin:34px 0 16px;font-size:22px}h3{margin:0 0 8px;font-size:17px}.muted{color:var(--muted);line-height:1.55}.wrap{padding:26px 42px 70px}
.stats{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:12px;margin-top:18px}.stat{background:#f8fbff;border:1px solid var(--line);border-radius:14px;padding:13px 15px}.stat b{font-size:24px}.stat span{display:block;color:var(--muted);font-size:12px;margin-top:4px}
.user-card{background:#fff;border:1px solid var(--line);border-radius:18px;margin:22px 0;padding:18px;box-shadow:0 14px 34px rgba(15,23,42,.06)}.user-head{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;margin-bottom:16px}.badge{display:inline-flex;align-items:center;border-radius:999px;padding:5px 10px;background:#edf7f1;color:var(--green);font-size:12px;font-weight:700}.badge.partial{background:#fff8e6;color:var(--amber)}.badge.failed{background:#fff1f0;color:var(--red)}
.topic-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}.topic{border:1px solid var(--line);border-radius:15px;overflow:hidden;background:#fcfdff}.topic img{width:100%;aspect-ratio:3/2;display:block;object-fit:cover;background:#111827}.topic-body{padding:13px 14px 15px}.topic p{min-height:58px;margin:0 0 12px;color:var(--muted);font-size:13px;line-height:1.45}.links{display:flex;flex-wrap:wrap;gap:8px}.links a{font-size:12px;padding:6px 8px;border-radius:8px;background:var(--sky);border:1px solid #cfe1ff}.meta{font-size:12px;color:var(--muted);margin-bottom:10px}.empty{padding:20px;border:1px dashed var(--line);border-radius:12px;color:var(--muted)}
.library{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 14px 34px rgba(15,23,42,.06);margin-bottom:26px}.library-head{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:16px}.library-head h2{margin:0}.library-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}.topic-intro{display:grid;grid-template-columns:72px 1fr;gap:12px;align-items:start;border:1px solid var(--line);border-radius:13px;padding:10px;background:#fbfdff}.topic-intro img{width:72px;height:72px;border-radius:10px;object-fit:cover;background:#111827}.topic-intro h3{font-size:14px;margin:0 0 5px}.topic-intro p{margin:0;color:var(--muted);font-size:12px;line-height:1.38}
.topic img,.topic-intro img{cursor:zoom-in}.root-lightbox{width:min(96vw,1600px);height:min(94vh,1000px);max-width:none;max-height:none;margin:auto;padding:0;color:var(--ink);background:#0b111b;border:1px solid #354052;border-radius:6px;overflow:hidden}.root-lightbox[open]{display:grid;grid-template-rows:58px minmax(0,1fr)}.root-lightbox::backdrop{background:rgba(3,7,18,.84)}.lightbox-toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:9px 12px;color:#f8fafc;background:#172033;border-bottom:1px solid #354052}.lightbox-toolbar strong{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}.lightbox-controls{display:flex;align-items:center;gap:6px;flex:0 0 auto}.lightbox-controls button{width:38px;height:38px;padding:0;color:#f8fafc;background:#253148;border:1px solid #46536a;border-radius:4px;font-size:21px;line-height:1;cursor:pointer}.lightbox-controls button:hover{background:#35445f}.lightbox-controls output{width:54px;text-align:center;color:#cbd5e1;font-size:12px;font-variant-numeric:tabular-nums}.lightbox-viewport{min-width:0;min-height:0;overflow:auto}.lightbox-stage{display:flex;align-items:center;justify-content:center;min-width:100%;min-height:100%;padding:16px}.lightbox-image{display:block;max-width:none;max-height:none;object-fit:contain;box-shadow:0 8px 32px rgba(0,0,0,.35);cursor:zoom-in}
@media(max-width:900px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.hero{position:static}}@media(max-width:620px){.hero,.wrap{padding-left:18px;padding-right:18px}.hero-heading{align-items:stretch;flex-direction:column}.hero-actions{min-width:0}.stats{grid-template-columns:1fr 1fr}.user-head{flex-direction:column}.topic-grid{grid-template-columns:1fr}}
"""


def lightbox_js() -> str:
    return """
(() => {
  const dialog = document.querySelector("#root-lightbox");
  const image = document.querySelector("#root-lightbox-image");
  const title = document.querySelector("#root-lightbox-title");
  const viewport = document.querySelector("#root-lightbox-viewport");
  const stage = document.querySelector("#root-lightbox-stage");
  const zoomOutput = document.querySelector("#root-lightbox-zoom");
  const previewSelector = ".topic img, .topic-intro img";
  if (!dialog || !image || !title || !viewport || !stage || !zoomOutput) return;

  let zoom = 1;
  let fittedWidth = 0;
  let fittedHeight = 0;

  const renderZoom = () => {
    const width = Math.max(1, Math.round(fittedWidth * zoom));
    const height = Math.max(1, Math.round(fittedHeight * zoom));
    image.style.width = width + "px";
    image.style.height = height + "px";
    stage.style.width = Math.max(viewport.clientWidth, width + 32) + "px";
    stage.style.height = Math.max(viewport.clientHeight, height + 32) + "px";
    zoomOutput.value = Math.round(zoom * 100) + "%";
    image.style.cursor = zoom < 4 ? "zoom-in" : "default";
  };

  const fitImage = () => {
    if (!image.naturalWidth || !image.naturalHeight) return;
    const availableWidth = Math.max(1, viewport.clientWidth - 32);
    const availableHeight = Math.max(1, viewport.clientHeight - 32);
    const ratio = Math.min(
      availableWidth / image.naturalWidth,
      availableHeight / image.naturalHeight,
      1
    );
    fittedWidth = image.naturalWidth * ratio;
    fittedHeight = image.naturalHeight * ratio;
    renderZoom();
  };

  const setZoom = (nextZoom) => {
    zoom = Math.min(4, Math.max(0.5, nextZoom));
    renderZoom();
  };

  const openPreview = (trigger) => {
    zoom = 1;
    title.textContent = trigger.alt || "Topic root image";
    image.alt = trigger.alt || "";
    image.src = trigger.currentSrc || trigger.src;
    dialog.showModal();
    if (image.complete) requestAnimationFrame(fitImage);
  };

  document.addEventListener("click", (event) => {
    const trigger = event.target.closest(previewSelector);
    if (trigger) {
      openPreview(trigger);
      return;
    }
    const action = event.target.closest("[data-lightbox-action]")?.dataset.lightboxAction;
    if (action === "close") dialog.close();
    if (action === "in") setZoom(zoom + 0.25);
    if (action === "out") setZoom(zoom - 0.25);
    if (action === "reset") setZoom(1);
  });

  document.addEventListener("keydown", (event) => {
    const trigger = event.target.closest?.(previewSelector);
    if (trigger && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      openPreview(trigger);
      return;
    }
    if (!dialog.open) return;
    if (event.key === "+" || event.key === "=") setZoom(zoom + 0.25);
    if (event.key === "-") setZoom(zoom - 0.25);
    if (event.key === "0") setZoom(1);
  });

  document.querySelectorAll(previewSelector).forEach((preview) => {
    preview.tabIndex = 0;
    preview.setAttribute("role", "button");
    preview.setAttribute("aria-label", "放大查看 " + (preview.alt || "topic root image"));
  });
  image.addEventListener("load", fitImage);
  image.addEventListener("dblclick", () => setZoom(zoom === 1 ? 2 : 1));
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  window.addEventListener("resize", () => {
    if (dialog.open) fitImage();
  });
})();
"""


def build_index(*, batch_dir: Path, web_dir: Path, root_assets_dir: Path, include_partial: bool) -> Path:
    web_dir.mkdir(parents=True, exist_ok=True)
    topic_summaries = collect_topic_summaries(root_assets_dir, web_dir)
    users: list[dict[str, Any]] = []
    for user_dir in sorted(batch_dir.glob("user_*")):
        status = status_for(user_dir)
        user_status = str(status.get("status") or "pending")
        if user_status != "done" and not include_partial:
            continue
        run_dir = latest_run(user_dir)
        if run_dir is None or not (run_dir / "manifest.jsonl").exists():
            continue
        user_id = user_dir.name.replace("user_", "")
        cards = collect_topic_cards(run_dir, user_id, web_dir)
        if cards:
            run_label = run_dir.relative_to(batch_dir).as_posix()
            users.append(
                {
                    "user_id": user_id,
                    "status": user_status,
                    "run_dir": run_dir,
                    "run_label": run_label,
                    "cards": cards,
                }
            )

    topic_count = sum(len(user["cards"]) for user in users)
    image_count = sum(card["nodes"] for user in users for card in user["cards"])
    tree_count = topic_count
    done_count = sum(1 for user in users if user["status"] == "done")
    partial_count = len(users) - done_count

    user_sections = []
    for user in users:
        badge_class = "badge" if user["status"] == "done" else "badge partial"
        cards_html = []
        for card in user["cards"]:
            img = f'<img loading="lazy" decoding="async" src="{safe(card["image"])}" alt="{safe(card["topic_title"])} root image">' if card["image"] else ""
            cards_html.append(
                f"""
                <article class="topic">
                  {img}
                  <div class="topic-body">
                    <h3>{safe(card["topic_title"])}</h3>
                    <div class="meta">{card["nodes"]} nodes · {card["edges"]} edges</div>
                    <p>{safe(card["synopsis"])}</p>
                    <div class="links">
                      <a href="{safe(card["story_href"])}">Story Tree</a>
                      <a href="{safe(card["intent_href"])}">Intent Task</a>
                      <a href="{safe(card["profile_href"])}">Profile Modeling</a>
                    </div>
                  </div>
                </article>
                """
            )
        user_sections.append(
            f"""
            <section class="user-card">
              <div class="user-head">
                <div>
                  <h2>User {safe(user["user_id"])}</h2>
                  <div class="muted">{safe(user["run_label"])}</div>
                </div>
                <span class="{badge_class}">{safe(user["status"])}</span>
              </div>
              <div class="topic-grid">{''.join(cards_html)}</div>
            </section>
            """
        )

    topic_intro_html = []
    for topic in topic_summaries:
        img = f'<img loading="lazy" decoding="async" src="{safe(topic["image"])}" alt="{safe(topic["topic_title"])} root image">' if topic["image"] else ""
        topic_intro_html.append(
            f"""
            <article class="topic-intro">
              {img}
              <div>
                <h3>{safe(topic["topic_title"])}</h3>
                <p>{safe(topic["synopsis"])}</p>
              </div>
            </article>
            """
        )
    library_html = (
        f"""
        <section class="library">
          <div class="library-head">
            <div>
              <h2>Topic Library</h2>
              <div class="muted">Twelve shared, non-personalized root worlds used before user-specific branches unfold.</div>
            </div>
            <div class="badge">{len(topic_summaries)} topics</div>
          </div>
          <div class="library-grid">{''.join(topic_intro_html)}</div>
        </section>
        """
        if topic_summaries
        else ""
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WorldWeaver · Balanced Intent Dataset</title>
  <style>{css()}</style>
</head>
<body>
  <header class="hero">
    <div class="hero-heading">
      <div>
        <h1>Balanced Intent Dataset</h1>
        <div class="muted">Current 10-user batch with six strictly balanced closed-domain intents. Open a topic to inspect its story tree, intent annotations, and profile trajectory.</div>
      </div>
      <div class="hero-actions">
        <a class="home-button" href="project.html" aria-label="返回 WorldWeaver 主页面">← 返回主页面</a>
        <nav class="batch-switch" aria-label="数据视图切换">
          <a class="active" href="/web_completed/index.html" aria-current="page">新均衡批次</a>
          <a class="augmentation" href="/web_completed/augmentation.html">增强分支</a>
        </nav>
      </div>
    </div>
    <div class="stats">
      <div class="stat"><b>{done_count}</b><span>completed users</span></div>
      <div class="stat"><b>{partial_count}</b><span>partial users shown</span></div>
      <div class="stat"><b>{tree_count}</b><span>topic trees</span></div>
      <div class="stat"><b>{image_count}</b><span>node images expected in shown trees</span></div>
      <div class="stat"><b>{safe(batch_dir.name)}</b><span>batch id</span></div>
    </div>
  </header>
  <main class="wrap">
    {library_html}
    {''.join(user_sections) if user_sections else '<div class="empty">No completed runs found.</div>'}
  </main>
  <dialog class="root-lightbox" id="root-lightbox" aria-label="Topic root image viewer">
    <header class="lightbox-toolbar">
      <strong id="root-lightbox-title">Topic root image</strong>
      <div class="lightbox-controls">
        <button type="button" data-lightbox-action="out" title="缩小" aria-label="缩小">−</button>
        <output id="root-lightbox-zoom" aria-live="polite">100%</output>
        <button type="button" data-lightbox-action="in" title="放大" aria-label="放大">+</button>
        <button type="button" data-lightbox-action="reset" title="复位" aria-label="复位">↺</button>
        <button type="button" data-lightbox-action="close" title="关闭" aria-label="关闭">×</button>
      </div>
    </header>
    <div class="lightbox-viewport" id="root-lightbox-viewport">
      <div class="lightbox-stage" id="root-lightbox-stage">
        <img class="lightbox-image" id="root-lightbox-image" alt="">
      </div>
    </div>
  </dialog>
  <script src="root-lightbox.js"></script>
</body>
</html>
"""
    index_path = web_dir / "index.html"
    (web_dir / "root-lightbox.js").write_text(lightbox_js(), encoding="utf-8")
    index_path.write_text(html_text, encoding="utf-8")
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a batch-level visualization index for generated comic tree runs.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--web-dir", default="")
    parser.add_argument("--root-assets-dir", default="../assets/topic_roots")
    parser.add_argument("--include-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    web_dir = Path(args.web_dir).resolve() if args.web_dir else batch_dir / "web_generated"
    root_assets_dir = Path(args.root_assets_dir)
    if not root_assets_dir.is_absolute():
        root_assets_dir = (Path(__file__).resolve().parent / root_assets_dir).resolve()
    index = build_index(
        batch_dir=batch_dir,
        web_dir=web_dir,
        root_assets_dir=root_assets_dir,
        include_partial=args.include_partial,
    )
    print(index)


if __name__ == "__main__":
    main()
