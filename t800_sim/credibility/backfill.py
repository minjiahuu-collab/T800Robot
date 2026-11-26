"""给现有 session 回填 environment 指纹（meta.json 缺失时）。"""

import json
from pathlib import Path

from ..validate import find_sessions
from .environment import environment_fingerprint


def backfill(root="recordings") -> int:
    sessions = find_sessions(root)
    n = 0
    for s in sessions:
        meta_path = s / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        if "environment" in meta and meta["environment"]:
            continue
        meta["environment"] = environment_fingerprint()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        n += 1
    print(f"回填 environment 指纹: {n}/{len(sessions)}")
    return n


if __name__ == "__main__":
    import sys
    sys.exit(backfill(sys.argv[1] if len(sys.argv) > 1 else "recordings"))
