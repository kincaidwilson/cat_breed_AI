#!/bin/bash
gunicorn app:app --timeout 120 --workers 1 --threads 2 --bind 0.0.0.0:$PORT
