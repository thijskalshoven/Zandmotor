#!/bin/sh
# Installs (or updates) the Zandmotor kite nowcast on a Mac:
#   curl -LsSf https://raw.githubusercontent.com/thijskalshoven/Zandmotor/main/install.sh | sh
# It installs uv (which brings its own Python), then the app, then creates the
# "Zandmotor Kite" folder in your home folder with double-click launchers.
# Running it again updates the app and keeps your settings and photos.
set -eu

APP_URL="${ZANDMOTOR_APP_URL:-https://github.com/thijskalshoven/Zandmotor/archive/refs/heads/main.zip}"
export PATH="$HOME/.local/bin:$PATH"

echo "==> Zandmotor kite nowcast installer"
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv (the tool that installs Python and the app)..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi

echo "==> Installing the app (the first time this takes a minute or two)..."
uv tool install --quiet --python 3.12 --reinstall --refresh "zandmotor[tide] @ $APP_URL"

echo "==> Creating your Zandmotor Kite folder..."
"$HOME/.local/bin/zandmotor" --install-launchers

echo
echo "Done. Double-click \"Zandmotor\" on your Desktop to build your first report."
open "${ZANDMOTOR_HOME:-$HOME/Zandmotor Kite}" 2>/dev/null || true
