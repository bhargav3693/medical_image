#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt
python download_models.py
python manage.py collectstatic --no-input
python manage.py migrate
