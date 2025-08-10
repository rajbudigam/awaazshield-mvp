# AwaazShield — Anti-Spoof + Speaker Verification MVP

**Live demo:** (Hugging Face Spaces URL after deploy)

### What this does
- Detects **voice deepfakes** (calibrated anti-spoof)
- Verifies **speaker similarity** (ECAPA-TDNN)
- Checks a **spoken passphrase** (digits) via Whisper
- Aggregates to a final **risk** + **Safe/Caution/Danger** label

### How to run (local)
```bash
cd app
python -m pip install -r requirements.txt
python app.py
