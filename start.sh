#!/bin/bash
gunicorn app:app --timeout 120 -b 0.0.0.0:$PORT
