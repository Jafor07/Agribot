# 🌾 AgriBot — Rice Disease Detector

A YOLO-powered rice disease detector with a built-in remedy advisor, deployable
as a Streamlit web app.

## What changed from the original desktop app

`AgriBot_7.py` (kept in this repo for reference) is a **Tkinter desktop app**
that also drove a 4WD Arduino RC car over a serial/COM port and spoke alerts
through the local machine's speakers. None of that works on a hosted web
server, so `app.py` is a from-scratch **Streamlit** rewrite that keeps the
part that works great on the web — disease detection + remedy advisor — and
drops what can't run remotely:

| Feature                          | Desktop app (`AgriBot_7.py`)      | Web app (`app.py`)                          |
|-----------------------------------|------------------------------------|-----------------------------------------------|
| Disease detection (YOLO)          | ✅ webcam loop + image upload      | ✅ image upload + camera snapshot            |
| Remedy advisor panel              | ✅                                  | ✅                                             |
| Arduino / RC car control          | ✅ via `pyfirmata` + serial port    | ❌ no server access to your local hardware   |
| Voice alerts                      | ✅ `pyttsx3` (server speakers)      | ⚙️ optional `gTTS` clip played in-browser     |

## Files

- `app.py` — Streamlit app (deploy this)
- `AgriBot_7.py` — original Tkinter desktop app (run locally if you need Arduino control)
- `best.pt` — trained YOLO weights
- `requirements.txt` — Python dependencies
- `packages.txt` — system packages Streamlit Cloud installs via apt
- `.streamlit/config.toml` — theme

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (make sure `best.pt` is committed — it's ~15 MB,
   well within GitHub's limits).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click
   **New app**.
3. Pick this repo/branch and set **Main file path** to `app.py`.
4. Click **Deploy**. The first build installs `ultralytics`/`torch`, which
   can take a few minutes.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Detection classes: `false_smut`, `sheath_blight`, `sheath_rot`, `stem_borer`.
- The confidence threshold is adjustable from the sidebar (default `0.45`,
  matching the original app).
- Voice alerts use Google's `gTTS` API, which requires outbound internet
  access from wherever the app is hosted; if unavailable, the app silently
  skips audio instead of erroring.
