from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import sys

try:
    # When executed as a module
    from .enforce_rect_white import enforce, _snap_border_whites
except Exception:
    # When executed as a script
    from pathlib import Path as _P
    _here = _P(__file__).resolve().parent
    sys.path.insert(0, str(_here))
    from enforce_rect_white import enforce, _snap_border_whites  # type: ignore


def iter_pngs(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".png":
        yield root
        return
    for p in root.rglob("*.png"):
        if p.is_file():
            yield p


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Batch enforce rectangular white frames on PNGs")
    ap.add_argument("paths", nargs="+", help="Files or directories of PNGs")
    ap.add_argument("--frame", type=int, default=48)
    ap.add_argument("--tol", type=int, default=18)
    ap.add_argument("--inset", type=int, default=0)
    ap.add_argument("--snap", type=int, default=10)
    ap.add_argument("--snap-tol", type=int, default=40)
    args = ap.parse_args(list(argv) if argv is not None else None)

    for s in args.paths:
        for p in iter_pngs(Path(s)):
            canvas, inner = enforce(p, None, frame=args.frame, tol=args.tol, inset=args.inset)
            canvas = _snap_border_whites(canvas, inner, args.snap, args.snap_tol)
            canvas.save(p)


if __name__ == "__main__":
    main()
