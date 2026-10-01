#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MSG="${1:-$(date +%F) 进度}"
git add -A
if git diff --cached --quiet; then echo "没有改动,跳过"; exit 0; fi
git commit -m "$MSG"
git push
