#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f friend_atomiq/src/qcampaign/models/quantum.py ]; then
  echo "Teammate qcampaign source is already bundled at code/friend_atomiq (no network needed)."
else
  echo "ERROR: bundled code/friend_atomiq is missing. Restore it from the submission archive." >&2
  exit 1
fi
