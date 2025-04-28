import os
import random
import time
import requests
from requests.exceptions import ChunkedEncodingError

# ─── Config ────────────────────────────────────────────────────────────────
API_URL            = "https://api.openai.com/v1/images/generations"
API_KEY            = os.getenv("OPENAI_API_KEY")
OUTPUT_DIR         = "dalle"
MODEL              = "dall-e-2"
SIZE               = "1024x1024"
NUM_IMAGES         = 4
CALLS_PER_MINUTE   = 5
SLEEP_BETWEEN      = 60.0 / CALLS_PER_MINUTE 
MAX_SERVER_RETRIES = 5
MAX_DOWNLOAD_RETRIES = 3

# ─── Prompt ingredients ─────────────────────────────────────────────────────
ethnicities  = ["East Asian", "Black", "South Asian", "White", "Latinx"]
genders      = ["man", "woman", "non-binary person"]
ages         = ["young adult", "middle-aged adult", "senior adult"]
lightings    = [
    "soft natural window lighting",
    "warm golden-hour glow",
    "even studio softbox lighting",
    "dramatic Rembrandt-style lighting",
    "neutral overcast daylight"
]
environments = [
    "a misty forest at dawn",
    "a modern office lobby",
    "a brick-walled café interior",
    "an urban street with soft bokeh lights",
    "a sandy ocean beach at sunset"
]

# ─── Setup ─────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def download_image(url: str, path: str):
    """Download with retries on chunked-encoding errors."""
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return
        except ChunkedEncodingError:
            print(f"⚠️ ChunkedEncodingError downloading {path}, retry {attempt}/{MAX_DOWNLOAD_RETRIES}…")
            time.sleep(1)
    raise RuntimeError(f"Failed to download {url} after {MAX_DOWNLOAD_RETRIES} attempts")

# ─── Compute start index ────────────────────────────────────────────────────
existing = [
    int(fn.split(".")[0]) for fn in os.listdir(OUTPUT_DIR)
    if fn.endswith(".png") and fn.split(".")[0].isdigit()
]
counter = max(existing) + 1 if existing else 1
successes = 0

# ─── Sequential generation with no skips ─────────────────────────────────────
while successes < NUM_IMAGES:
    idx = counter
    prompt = (
        f"Hyper-realistic portrait headshot of a {random.choice(ages)} "
        f"{random.choice(ethnicities)} {random.choice(genders)}, "
        f"{random.choice(lightings)}, standing in {random.choice(environments)}, "
        "detailed skin pores, natural facial imperfections, realistic hair strands, "
        "and lifelike eyes"
    )
    payload = {"model": MODEL, "prompt": prompt, "n": 1, "size": SIZE}
    data = None

    for attempt in range(MAX_SERVER_RETRIES):
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except ChunkedEncodingError:
                print(f"⚠️ ChunkedEncodingError parsing JSON at [{idx:04d}], retry {attempt+1}/{MAX_SERVER_RETRIES}…")
                time.sleep(SLEEP_BETWEEN)
                continue
            print(f"→ [{idx:04d}] Generated")
            break
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", SLEEP_BETWEEN))
            print(f"⚠️ 429 at [{idx:04d}], sleeping {wait:.1f}s…")
            time.sleep(wait)
            continue
        if 500 <= resp.status_code < 600:
            backoff = (2 ** attempt) + random.random()
            print(f"⚠️ {resp.status_code} at [{idx:04d}], retry {attempt+1}/{MAX_SERVER_RETRIES} in {backoff:.1f}s…")
            time.sleep(backoff)
            continue
        print(f"❗️ [{idx:04d}] Failed ({resp.status_code}): {resp.text}")
        break

    if not data:
        print(f"❗️ [{idx:04d}] Skipping retry and will retry this index next loop.")
        time.sleep(SLEEP_BETWEEN)
        continue

    out_path = os.path.join(OUTPUT_DIR, f"{idx:04d}.png")
    download_image(data["data"][0]["url"], out_path)
    print(f"   ✅ Saved {out_path}")

    counter += 1
    successes += 1

    time.sleep(SLEEP_BETWEEN)

print(f"Done! Appended {NUM_IMAGES} images into '{OUTPUT_DIR}'.")
