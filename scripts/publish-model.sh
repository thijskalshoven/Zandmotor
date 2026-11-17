#!/bin/sh
# Publishes this checkout's lagoon model (after --outline / --calibrate) as the
# rolling "model" pre-release, which installed copies download.
# Needs the GitHub CLI once: brew install gh && gh auth login
set -eu
cd "$(dirname "$0")/.."
OUT="$(mktemp -d)"
python -m zandmotor --export-model "$OUT/zandmotor-model.zip"
if ! gh release view model >/dev/null 2>&1; then
  # a pre-release, so it never becomes the repo's "latest" release
  gh release create model --prerelease --title "Lagoon model" \
    --notes "The shared lagoon model, updated after each calibration. Installed copies download it automatically."
fi
gh release upload model "$OUT/zandmotor-model.zip" "$OUT/manifest.json" --clobber
echo "Published: $(python -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['created'], '- newest image', m['newest_image'])" "$OUT/manifest.json")"
rm -rf "$OUT"
