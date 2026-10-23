#!/usr/bin/env bash
set -euo pipefail
PRICE="$1"; OUT="$2"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git libglib2.0-0
rm -rf guardian-cv
git clone https://github.com/roshini-t-24/guardian-cv.git
cd guardian-cv
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
nproc; uname -m
python benchmark_cool.py --iterations 100 --price-hourly "$PRICE" --out "../$OUT"