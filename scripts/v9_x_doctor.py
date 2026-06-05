"""Quick X-API setup check (free, no API call). Run: python scripts/v9_x_doctor.py"""
import os
from pathlib import Path

env = Path(".env")
print(f".env exists: {env.is_file()}  ({env.resolve()})")

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("python-dotenv: loaded")
except ImportError:
    print("python-dotenv: NOT installed (pip install python-dotenv)")

tok = (os.environ.get("X_BEARER_TOKEN") or "").strip().strip('"').strip("'").strip()
print(f"X_BEARER_TOKEN set: {bool(tok)}   length: {len(tok)}")
if tok:
    print(f"  starts with: {tok[:6]}...   (a real bearer token is ~100+ chars; OAuth1 keys are ~25)")
    if len(tok) < 40:
        print("  ⚠️ too short to be a Bearer Token — looks like an API key/secret, not the OAuth2 app-only Bearer.")
else:
    print("  -> add to .env:  X_BEARER_TOKEN=AAAA... (the OAuth2 *Bearer Token* from your X app's Keys page)")
