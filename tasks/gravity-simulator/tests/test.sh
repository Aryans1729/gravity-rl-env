#!/bin/bash
# Harbor verifier entrypoint. Writes 1 (pass) or 0 (fail) to /logs/verifier/reward.txt.
set -u

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt   # default to fail if anything below crashes

if [ ! -f /app/simulator.py ]; then
    echo "No /app/simulator.py found."
    exit 0
fi

python3 -I /tests/verifier.py /app/simulator.py \
    --reward-file /logs/verifier/reward.txt \
    --details-file /logs/verifier/details.json

exit 0
