#!/bin/zsh
# Builds the Zandmotor lagoon report and opens it in the browser.
# The first time, it asks for your weight and kites.
cd "${0:A:h}"
if ! __ZANDMOTOR__; then
  echo
  echo "Something went wrong (see above)."
  read -k1 "?Press any key to close this window."
fi
