#!/usr/bin/env python3
"""
Step 1: Generate paraphrases of original lyrics via MiniMax API.

For each song in conditions.json, sends the original lyrics (C0) to the
MiniMax chat API and asks for a meaning-preserving rewrite.  Results are
saved incrementally to paraphrases.json so the script can be resumed.

Environment variables:
    MINIMAX_API_KEY   – required; your MiniMax API key
"""
import json, subprocess, re, os, time, sys

API_KEY = os.environ.get("MINIMAX_API_KEY")
if not API_KEY:
    sys.exit("Error: set MINIMAX_API_KEY environment variable")

# ─── Paths (relative to project root) ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONDITIONS_FILE = os.path.join(PROJECT_ROOT, "results", "conditions.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "paraphrases.json")

with open(CONDITIONS_FILE, "r", encoding="utf-8") as f:
    conditions = json.load(f)

keys = sorted(conditions.keys())

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        paraphrases = json.load(f)
else:
    paraphrases = {}

remaining = [k for k in keys if k not in paraphrases or len(paraphrases.get(k, "")) < 50]
print(f"Total: {len(keys)}, Already done: {len([k for k in keys if k in paraphrases and len(paraphrases.get(k,'')) >= 50])}, Remaining: {len(remaining)}", flush=True)


def call_api(lyrics, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = subprocess.run(
                ["curl", "-s", "--max-time", "60",
                 "https://api.minimax.chat/v1/chat/completions",
                 "-H", "Content-Type: application/json",
                 "-H", f"Authorization: Bearer {API_KEY}",
                 "-d", json.dumps({
                     "model": "MiniMax-M2.7",
                     "messages": [{"role": "user", "content": f"改写这段歌词，保持意思不变但换词语换句式，只输出改写结果：\n{lyrics}"}],
                     "max_tokens": 800,
                     "temperature": 0.7,
                 }, ensure_ascii=False)],
                capture_output=True, text=True, timeout=90)
            if r.returncode != 0:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

            resp = json.loads(r.stdout)
            content = resp["choices"][0]["message"]["content"]
            # Clean thinking tags
            content = re.sub(r'<think\s*>[\s\S]*?</think\s*>', '', content).strip()
            # Remove any 【】 markers the model might add
            content = re.sub(r'【[^】]*】\s*', '', content).strip()
            return content if len(content) > 20 else None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            return None
    return None


ok = fail = done = 0
start = time.time()

for i, name in enumerate(remaining):
    lyrics = conditions[name]["C0_orig_lyrics"][:200]  # 200 char limit for speed

    result = call_api(lyrics)
    done += 1

    if result and len(result) > 20:
        paraphrases[name] = result
        ok += 1
    else:
        fail += 1

    # Progress every 5 songs
    if (i + 1) % 5 == 0 or (i + 1) == len(remaining):
        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(remaining)}] ok={ok} fail={fail} rate={rate:.1f}/s ETA={eta/60:.1f}m", flush=True)

    # Save every 10 songs
    if (i + 1) % 10 == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(paraphrases, f, ensure_ascii=False, indent=2)

    # Small delay between requests to avoid rate limiting
    time.sleep(0.3)

# Final save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(paraphrases, f, ensure_ascii=False, indent=2)

elapsed = time.time() - start
print(f"\nDone in {elapsed:.0f}s! ok={ok} fail={fail} total={len(paraphrases)}/{len(keys)}", flush=True)
