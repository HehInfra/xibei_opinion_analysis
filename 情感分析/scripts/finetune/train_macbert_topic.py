from __future__ import annotations

import sys

from train_macbert_single_label import main


if __name__ == "__main__":
    raise SystemExit(main(["--task", "topic", *sys.argv[1:]]))
