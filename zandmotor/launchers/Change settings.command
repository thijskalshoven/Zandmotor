#!/bin/zsh
# Asks for your weight and kites again and saves them in settings.toml.
cd "${0:A:h}"
__ZANDMOTOR__ --setup
echo
read -k1 "?Done. Press any key to close this window."
