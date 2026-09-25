from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dataset_pipeline import read_profiles, slugify, write_json


DEFAULT_TOPICS = [
    "Hidden Castle",
    "Neon Hospital",
    "Desert Whale Caravan",
    "Underwater Library",
    "Abandoned Space Elevator",
    "Rainy Cyberpunk Alley",
    "Talking Animal Courtroom",
    "Volcanic Flower Market",
    "Time-loop Train Station",
    "Snow Mountain Radio Tower",
    "Miniature City Inside a Watch",
    "Ocean Festival on Paper Lantern Island",
]


def copy_topic_for_user(*, user_id: str, topic: str, root_assets_dir: Path, output_dir: Path) -> None:
    topic_id = slugify(topic)
    source_dir = root_assets_dir / topic_id
    if not source_dir.exists():
        raise RuntimeError(f"Missing root asset folder for topic {topic!r}: {source_dir}")

    target_dir = output_dir / f"user_{user_id}" / topic_id
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for name in ["root_asset.json", "root_prompt.txt", "synopsis.txt"]:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)

    source_image = source_dir / "root.png"
    if source_image.exists():
        shutil.copy2(source_image, images_dir / "n0.png")
        shutil.copy2(source_image, target_dir / "root.png")
    source_meta = source_dir / "root.json"
    if source_meta.exists():
        shutil.copy2(source_meta, target_dir / "root.image_metadata.json")

    asset = json.loads((source_dir / "root_asset.json").read_text(encoding="utf-8"))
    asset["user_id"] = user_id
    asset["topic_id"] = topic_id
    asset["local_root_image"] = str((images_dir / "n0.png").resolve())
    asset["source_root_image"] = str(source_image.resolve())
    write_json(target_dir / "user_topic_root.json", asset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand shared topic root assets into user/topic folders.")
    parser.add_argument("--profiles", default="user_profiles(1).jsonl")
    parser.add_argument("--profile-limit", type=int, default=1)
    parser.add_argument("--profile-user-id", default="")
    parser.add_argument("--root-assets-dir", default="topic_root_assets")
    parser.add_argument("--output-dir", default="user_topic_assets")
    parser.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    profiles_path = (base_dir / args.profiles).resolve()
    root_assets_dir = Path(args.root_assets_dir)
    if not root_assets_dir.is_absolute():
        root_assets_dir = (base_dir / root_assets_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    profiles = read_profiles(
        profiles_path,
        limit=args.profile_limit if args.profile_limit > 0 else None,
        user_id=args.profile_user_id or None,
    )
    manifest = []
    for profile in profiles:
        user_id = str(profile.get("user_id"))
        for topic in args.topics:
            copy_topic_for_user(user_id=user_id, topic=topic, root_assets_dir=root_assets_dir, output_dir=output_dir)
            manifest.append({"user_id": user_id, "topic_id": slugify(topic), "topic": topic})
    write_json(output_dir / "manifest.json", manifest)
    print(f"Prepared {len(manifest)} user-topic root folders in {output_dir}")


if __name__ == "__main__":
    main()
