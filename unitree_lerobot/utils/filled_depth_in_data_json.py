import json
import re
import shutil
from pathlib import Path

# 루트 경로(episode_* 들이 있는 상위 폴더)
BASE_DIR = Path("/media/eunju/KIAT/dataset/0226/data/filled_bottle")
EPISODE_GLOB = "episode_*"

def fill_depths(data, episode_dir: Path):
    assigned = 0
    depth_dir = episode_dir / "color_depths"

    def walk(node):
        nonlocal assigned

        if isinstance(node, dict):
            # colors가 있고 depths가 있는 구조를 찾음
            if "colors" in node and "depths" in node:
                colors = node.get("colors", {})
                depths = node.get("depths", {})

                if isinstance(colors, dict) and isinstance(depths, dict):
                    for cam_key, color_rel in colors.items():
                        color_path = Path(color_rel)

                        # colors/000000_color_0.jpg
                        # -> color_depths/000000_color_0.png
                        depth_rel = Path("color_depths") / (color_path.stem + ".png")
                        depth_abs = episode_dir / depth_rel

                        if depth_abs.exists():
                            depths[cam_key] = depth_rel.as_posix()
                            assigned += 1
                        else:
                            print(f"[WARN] depth file 없음: {depth_abs}")

                    node["depths"] = depths

            for v in node.values():
                walk(v)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return assigned


def process_episode(episode_dir: Path):
    json_path = episode_dir / "data.json"
    depth_dir = episode_dir / "color_depths"

    if not json_path.exists():
        print(f"[SKIP] data.json 없음: {episode_dir}")
        return 0
    if not depth_dir.exists():
        print(f"[SKIP] color_depths 없음: {episode_dir}")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    backup = json_path.with_suffix(json_path.suffix + ".bak")
    shutil.copy2(json_path, backup)

    assigned = fill_depths(data, episode_dir)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] {episode_dir.name}: filled={assigned}")
    return assigned


def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base dir not found: {BASE_DIR}")

    episodes = sorted([p for p in BASE_DIR.glob(EPISODE_GLOB) if p.is_dir()])
    if not episodes:
        print(f"[INFO] episode 폴더 없음: {BASE_DIR}/{EPISODE_GLOB}")
        return

    total_filled = 0
    for ep in episodes:
        try:
            total_filled += process_episode(ep)
        except Exception as e:
            print(f"[ERR] {ep}: {e}")

    print(f"[DONE] episodes={len(episodes)}, total_filled={total_filled}")


if __name__ == "__main__":
    main()