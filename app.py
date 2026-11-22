
"""
PS 59 - Hospital Appointment Bot
Google Gemini + Streamlit + CSV Database
"""

import csv
import fcntl
import json
import os
import random
import re
import time
import uuid

from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
SLOTS_FILE = DATA_DIR / "doctor_slots.csv"
BOOKINGS_FILE = DATA_DIR / "appointments.csv"
LOCK_FILE = DATA_DIR / ".booking.lock"

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

DOCTORS = [
    ("D101", "Dr. Asha Mehta", "Cardiology"),
    ("D102", "Dr. Rohan Shah", "Dermatology"),
    ("D103", "Dr. Neha Kulkarni", "General Medicine"),
    ("D104", "Dr. Arjun Rao", "Orthopedics"),
    ("D105", "Dr. Priya Nair", "ENT"),
]

DEPARTMENTS = sorted(set(d[2] for d in DOCTORS))

SLOT_FIELDS = [
    "slot_id",
    "doctor_id",
    "doctor",
    "department",
    "date",
    "time",
    "status",
]

BOOKING_FIELDS = [
    "booking_id",
    "slot_id",
    "patient_name",
    "doctor",
    "department",
    "date",
    "time",
    "created_at",
]


# ==========================================
# 2. CSV DATABASE FUNCTIONS
# ==========================================

def read_csv(path):
    if not path.exists():
        return []

    with path.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, fields, rows):
    temp_file = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    with temp_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_file, path)


def initialize_database():
    DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    with LOCK_FILE.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        try:
            if not SLOTS_FILE.exists():
                slots = []

                for offset in range(1, 15):
                    day = (
                        date.today()
                        + timedelta(days=offset)
                    )

                    # Sunday closed
                    if day.weekday() == 6:
                        continue

                    for doctor_id, doctor, department in DOCTORS:
                        for hour in (10, 11, 14, 15):

                            slot_time = f"{hour:02d}:00"

                            slots.append({
                                "slot_id": (
                                    f"{doctor_id}-"
                                    f"{day.isoformat()}-"
                                    f"{slot_time}"
                                ),
                                "doctor_id": doctor_id,
                                "doctor": doctor,
                                "department": department,
                                "date": day.isoformat(),
                                "time": slot_time,
                                "status": "Available",
                            })

                write_csv(
                    SLOTS_FILE,
                    SLOT_FIELDS,
                    slots,
                )

            if not BOOKINGS_FILE.exists():
                write_csv(
                    BOOKINGS_FILE,
                    BOOKING_FIELDS,
                    [],
                )

        finally:
            fcntl.flock(
                lock, fcntl.LOCK_UN
            )


# ==========================================
# 3. GEMINI ERROR HANDLING
# ==========================================

def get_error_code(error):
    code = (
        getattr(error, "code", None)
        or getattr(error, "status_code", None)
    )

    try:
        return int(code)
    except (ValueError, TypeError):
        return None


def get_safe_error(error):
    message = str(error)

    if API_KEY:
        message = message.replace(
            API_KEY, "[REDACTED]"
        )

    message = re.sub(
        r"AIza[A-Za-z0-9_-]{30,}",
        "[REDACTED]",
        message,
    )

    return message[:900]


def gemini_generate(prompt):
    """
    Call Gemini with retry logic.

    Retries temporary 503 and 500 errors.
    Retries 429 cautiously using backoff.
    Stops immediately on other errors.
    """

    client = genai.Client(api_key=API_KEY)

    max_attempts = 3

    for attempt in range(max_attempts):

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text

        except Exception as error:
            code = get_error_code(error)

            # Do not retry invalid API requests,
            # authentication errors or missing models.
            if code not in (429, 500, 503):
                raise

            if attempt == max_attempts - 1:
                raise

            # Exponential backoff with jitter
            delay = (
                2 ** attempt
                + random.uniform(0, 0.5)
            )

            time.sleep(delay)

    raise RuntimeError(
        "Gemini request failed after retries."
    )


# ==========================================
# 4. EXTRACT PATIENT DETAILS USING GEMINI
# ==========================================

def extract_patient_details(messages):

    user_messages = [
        msg["content"]
        for msg in messages
        if msg["role"] == "user"
    ]

    prompt = f"""
You are an AI hospital appointment assistant.

Today's date is {date.today().isoformat()}.

Available hospital departments:
{", ".join(DEPARTMENTS)}

Analyze the patient's entire conversation.

Extract the following details:

1. patient_name
2. department
3. preferred_date
4. preferred_time

IMPORTANT RULES:

- Return only valid JSON.
- Never invent patient information.
- Use null for missing values.
- Consider information given in earlier messages.
- If the patient updates a detail, use the latest value.
- Convert date to YYYY-MM-DD.
- Convert time to HH:MM in 24-hour format.
- Interpret tomorrow relative to today's date.
- Department must match an available department.
- Do not diagnose conditions or infer a department
  from symptoms alone.
- Do not book an appointment yourself.

Return JSON in exactly this structure:

{{
    "patient_name": null,
    "department": null,
    "preferred_date": null,
    "preferred_time": null
}}

Patient conversation:

{json.dumps(user_messages, ensure_ascii=False)}
"""

    result = gemini_generate(prompt)

    data = json.loads(result)

    if not isinstance(data, dict):
        raise ValueError(
            "Gemini returned an invalid JSON structure."
        )

    return {
        "patient_name": data.get("patient_name"),
        "department": data.get("department"),
        "preferred_date": data.get("preferred_date"),
        "preferred_time": data.get("preferred_time"),
    }


# ==========================================
# 5. VALIDATE PATIENT INFORMATION
# ==========================================

def validate_information(info):

    name = info.get("patient_name")
    department = info.get("department")
    preferred_date = info.get("preferred_date")
    preferred_time = info.get("preferred_time")

    if not isinstance(name, str):
        name = None
    elif not 2 <= len(name.strip()) <= 80:
        name = None
    else:
        name = name.strip()

    if department not in DEPARTMENTS:
        department = None

    if preferred_date:
        try:
            preferred_date = (
                date.fromisoformat(
                    preferred_date
                ).isoformat()
            )
        except (ValueError, TypeError):
            preferred_date = None

    if preferred_time:
        try:
            preferred_time = (
                datetime.strptime(
                    preferred_time, "%H:%M"
                ).strftime("%H:%M")
            )
        except (ValueError, TypeError):
            preferred_time = None

    return {
        "patient_name": name,
        "department": department,
        "preferred_date": preferred_date,
        "preferred_time": preferred_time,
    }


# ==========================================
# 6. FIND AVAILABLE DOCTOR SLOTS
# ==========================================

def find_available_slots(
    department,
    preferred_date=None,
    preferred_time=None,
):

    slots = read_csv(SLOTS_FILE)

    available = [
        slot
        for slot in slots
        if (
            slot["department"] == department
            and slot["status"] == "Available"
            and slot["date"] > date.today().isoformat()
        )
    ]

    if preferred_date:
        available = [
            slot for slot in available
            if slot["date"] == preferred_date
        ]

    if preferred_time:
        available.sort(
            key=lambda slot: (
                slot["time"] != preferred_time,
                slot["date"],
                slot["time"],
            )
        )
    else:
        available.sort(
            key=lambda slot: (
                slot["date"],
                slot["time"],
            )
        )

    return available[:6]


# ==========================================
# 7. CONFIRM APPOINTMENT
# ==========================================

def book_appointment(slot_id, patient_name):

    if not patient_name or not patient_name.strip():
        return None

    with LOCK_FILE.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        try:
            slots = read_csv(SLOTS_FILE)

            selected = next(
                (
                    slot
                    for slot in slots
                    if slot["slot_id"] == slot_id
                ),
                None,
            )

            if selected is None:
                return None

            if selected["status"] != "Available":
                return None

            if (
                selected["date"]
                <= date.today().isoformat()
            ):
                return None

            bookings = read_csv(BOOKINGS_FILE)

            # Prevent duplicate booking records
            if any(
                b["slot_id"] == slot_id
                for b in bookings
            ):
                return None

            booking = {
                "booking_id": (
                    "APT-"
                    + uuid.uuid4().hex[:8].upper()
                ),
                "slot_id": slot_id,
                "patient_name": patient_name.strip(),
                "doctor": selected["doctor"],
                "department": selected["department"],
                "date": selected["date"],
                "time": selected["time"],
                "created_at": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
            }

            bookings.append(booking)

            write_csv(
                BOOKINGS_FILE,
                BOOKING_FIELDS,
                bookings,
            )

            selected["status"] = "Booked"

            write_csv(
                SLOTS_FILE,
                SLOT_FIELDS,
                slots,
            )

            return booking

        finally:
            fcntl.flock(
                lock, fcntl.LOCK_UN
            )


# ==========================================
# 8. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MediSlot - Hospital Appointment Bot",
    page_icon="🏥",
    layout="centered",
)

initialize_database()

st.title("🏥 MediSlot")
st.subheader(
    "AI Hospital Appointment Booking Assistant"
)

st.caption(
    "PS 59 | Generative AI Mini Project "
    "| Google Gemini + Streamlit"
)

st.info(
    "Educational demonstration only. "
    "All doctors and appointment slots are fictional. "
    "Please use sample patient information."
)


# ==========================================
# 9. CHECK API KEY
# ==========================================

if not API_KEY:

    st.error(
        "Gemini API key not found. "
        "Add GEMINI_API_KEY to your .env file."
    )

    st.stop()


# ==========================================
# 10. SESSION STATE
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to MediSlot.\n\n"
                "I can help you book a hospital appointment.\n\n"
                "Please tell me your name, "
                "preferred medical department, "
                "and appointment date."
            ),
        }
    ]

if "available_slots" not in st.session_state:
    st.session_state.available_slots = []

if "patient_name" not in st.session_state:
    st.session_state.patient_name = None


# ==========================================
# 11. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("🏥 Hospital Information")

    st.subheader("Our Departments")

    for department in DEPARTMENTS:
        st.write("•", department)

    st.divider()

    st.subheader("Appointment Timings")

    st.write("10:00 AM")
    st.write("11:00 AM")
    st.write("02:00 PM")
    st.write("03:00 PM")

    st.caption(
        "The sample hospital is closed on Sundays."
    )

    st.divider()

    st.caption(f"Gemini Model: {MODEL}")

    if st.button(
        "🔄 New Conversation",
        use_container_width=True,
    ):

        for key in (
            "messages",
            "available_slots",
            "patient_name",
        ):
            st.session_state.pop(key, None)

        st.rerun()


# ==========================================
# 12. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 13. DISPLAY AVAILABLE SLOTS
# ==========================================

if st.session_state.available_slots:

    st.subheader("📅 Available Appointments")

    for index, slot in enumerate(
        st.session_state.available_slots
    ):

        label = (
            f"👨‍⚕️ {slot['doctor']} | "
            f"{slot['date']} | "
            f"{slot['time']}"
        )

        if st.button(
            label,
            key=f"slot_{index}_{slot['slot_id']}",
            use_container_width=True,
        ):

            booking = book_appointment(
                slot["slot_id"],
                st.session_state.patient_name,
            )

            st.session_state.available_slots = []

            if booking:

                reply = (
                    "✅ **Appointment Confirmed!**\n\n"
                    f"**Booking ID:** "
                    f"{booking['booking_id']}\n\n"
                    f"**Patient:** "
                    f"{booking['patient_name']}\n\n"
                    f"**Doctor:** "
                    f"{booking['doctor']}\n\n"
                    f"**Department:** "
                    f"{booking['department']}\n\n"
                    f"**Date:** "
                    f"{booking['date']}\n\n"
                    f"**Time:** "
                    f"{booking['time']}\n\n"
                    "Thank you for using MediSlot!"
                )

                st.balloons()

            else:

                reply = (
                    "⚠️ Sorry, the selected slot "
                    "is no longer available.\n\n"
                    "Please request another appointment."
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
            })

            st.rerun()


# ==========================================
# 14. CHAT INPUT
# ==========================================

user_input = st.chat_input(
    "Example: I'm Riya, book Dermatology tomorrow at 10 AM"
)

if user_input:

    st.session_state.available_slots = []

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    try:

        with st.spinner(
            "🤖 Checking appointment availability..."
        ):

            raw_info = extract_patient_details(
                st.session_state.messages
            )

            info = validate_information(raw_info)

            name = info["patient_name"]
            department = info["department"]
            preferred_date = info["preferred_date"]
            preferred_time = info["preferred_time"]

            st.session_state.patient_name = name

            if not name:

                reply = (
                    "May I know your full name "
                    "to proceed with the booking?"
                )

            elif not department:

                reply = (
                    f"Thank you, {name}! 😊\n\n"
                    "Which medical department "
                    "would you like to visit?\n\n"
                    + ", ".join(DEPARTMENTS)
                )

            elif (
                preferred_date
                and preferred_date <= date.today().isoformat()
            ):

                reply = (
                    "⚠️ Please select a future date "
                    "for your appointment."
                )

            else:

                slots = find_available_slots(
                    department,
                    preferred_date,
                    preferred_time,
                )

                if not slots:

                    reply = (
                        "Sorry, there are no available "
                        "appointments matching your request.\n\n"
                        "Please choose another date or time."
                    )

                else:

                    st.session_state.available_slots = slots

                    reply = (
                        f"Great, {name}! 😊\n\n"
                        f"I found available "
                        f"**{department}** appointments.\n\n"
                        "Select a doctor and time slot "
                        "below to confirm your appointment."
                    )

    # ======================================
    # 15. IMPROVED API ERROR HANDLING
    # ======================================

    except Exception as exc:

        status = get_error_code(exc)
        detail = get_safe_error(exc)

        if status == 503:

            reply = (
                "⚠️ **Gemini is temporarily unavailable.**\n\n"
                "The server is experiencing high demand. "
                "Automatic retries were attempted, "
                "but the request could not be completed.\n\n"
                "Please retry your request later.\n\n"
                f"**Details:** `{detail}`"
            )

        elif status == 429:

            reply = (
                "⚠️ **Gemini API quota or rate limit reached.**\n\n"
                "Check your free-tier limits "
                "in Google AI Studio.\n\n"
                f"**Details:** `{detail}`"
            )

        elif status == 404:

            reply = (
                "⚠️ **Gemini model not found.**\n\n"
                f"Selected model: `{MODEL}`\n\n"
                "Check which models are available "
                "for your API key.\n\n"
                f"**Details:** `{detail}`"
            )

        elif status in (401, 403):

            reply = (
                "⚠️ **Gemini authentication or permission error.**\n\n"
                "Check your API key and project permissions.\n\n"
                f"**Details:** `{detail}`"
            )

        else:

            reply = (
                "⚠️ **Gemini request failed.**\n\n"
                f"**Error Type:** "
                f"{type(exc).__name__}\n\n"
                f"**HTTP Status:** "
                f"{status or 'Unknown'}\n\n"
                f"**Details:** `{detail}`"
            )

    # ======================================
    # 16. SAVE ASSISTANT RESPONSE
    # ======================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
    })

    st.rerun()
