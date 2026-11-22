# PS 59 — MediSlot Hospital Appointment Bot

A Streamlit GenAI mini project built with the Google Gemini API and Python CSV-based appointment booking.

## Features
- Natural-language extraction of patient name, department, date and time using Gemini
- Doctor and slot availability from `data/doctor_slots.csv`
- Confirmation buttons and unique booking IDs
- CSV updates with a file lock to prevent double-booking in concurrent sessions
- Sample fictional doctors; no real healthcare or personal records

## Setup (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Edit `.env` locally to paste your Gemini API key. Do not publish it.

```bash
streamlit run app.py
```

Default model is `gemini-2.5-flash`. If your free-tier project has a different available model, change `GEMINI_MODEL` in `.env`.

## Demo prompts
- `Hi, I'm Riya Sharma. Book a Dermatology appointment tomorrow at 10 AM.`
- `My name is Aman Patel. I want General Medicine on the next available day.`
- `I'm Neha. Is there an ENT appointment on Friday?`
- `Hello, I'm Aryan. I need Cardiology at 2 PM.`
- `I'm Sara. Can I see Orthopedics tomorrow?`

Click an available slot button to make the booking. Check `data/appointments.csv` and `data/doctor_slots.csv` after booking.

## Notes
- CSV storage is adequate for a classroom demo; use a real transactional database and authentication for production.
- Dates use the host machine's local calendar. Demo slots are generated once for the next 14 days except Sundays.
- This app does not provide medical advice or diagnosis. Please don't use actual patient information.
