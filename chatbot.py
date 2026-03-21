import os
import streamlit as st
import google.generativeai as genai
import pandas as pd

# Load API key securely
# Works with Streamlit Cloud secrets management or local .streamlit/secrets.toml
_GEMINI_API_KEY = "AIzaSyCk19KxqZ_mSUrl5Y1gcYZYpPPOZxKswOY"  # fallback for local dev

try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    elif "GEMINI_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    else:
        genai.configure(api_key=_GEMINI_API_KEY)
except Exception:
    try:
        genai.configure(api_key=_GEMINI_API_KEY)
    except Exception:
        pass

ACADEMIC_CALENDAR_CONTEXT = """
DOCUMENT TITLE: Spring Semester Academic Calendar 2025-2026
INSTITUTION: National Institute of Technology Warangal
COMMENCEMENT OF CLASSWORK: 15 December 2025

IMPORTANT EVENTS:
- 15 Dec 2025: Commencement of Classwork
- 25 Dec 2025: Christmas Day holiday
- 12-14 Jan 2026: Extra Holidays & Makar Sankranti
- 26 Jan 2026: Republic Day holiday
- 31 Jan to 6 Feb 2026: Mid-Semester Examinations (High lab/library load expected)
- 27-28 Feb 2026: Spring Spree (Cultural Festival - high load evening/night, abnormal patterns)
- 4 Mar 2026: Holi holiday
- 13 Mar 2026: Sports Day
- 21 Mar 2026: Id-ul-Fitr holiday
- 31 Mar 2026: Mahavir Jayanti holiday
- 3 Apr 2026: Good Friday holiday
- 8 Apr 2026: Last Working Day
- 10 to 18 Apr 2026: End-Semester Examinations
- 20 Apr 2026: Summer Vacation Begins (Load should drop significantly)

STUDENT WEEKEND RULE:
- Sunday & Saturday: holidays by default unless a specific activity (like exams or Spring Spree) is written.
"""

def get_calendar_event(date_str: str) -> str:
    """Check if a specific date has a calendar event, to enrich hover data."""
    if not isinstance(date_str, str):
        date_str = str(date_str)
        
    events = {
        "2025-12-15": "Commencement of Classwork",
        "2025-12-25": "Christmas Day",
        "2026-01-12": "Extra Holiday",
        "2026-01-13": "Extra Holiday",
        "2026-01-14": "Makar Sankranti",
        "2026-01-26": "Republic Day",
        "2026-01-31": "Mid-Semester Exams Begin",
        "2026-02-01": "Mid-Semester Exams",
        "2026-02-02": "Mid-Semester Exams",
        "2026-02-03": "Mid-Semester Exams",
        "2026-02-04": "Mid-Semester Exams",
        "2026-02-05": "Mid-Semester Exams",
        "2026-02-06": "Mid-Semester Exams End",
        "2026-02-27": "Spring Spree",
        "2026-02-28": "Spring Spree",
        "2026-03-04": "Holi",
        "2026-03-13": "Sports Day",
        "2026-03-21": "Id-ul-Fitr",
        "2026-03-31": "Mahavir Jayanti",
        "2026-04-03": "Good Friday",
        "2026-04-08": "Last Working Day",
        "2026-04-10": "End-Semester Exams Begin",
        "2026-04-11": "End-Semester Exams",
        "2026-04-12": "End-Semester Exams",
        "2026-04-13": "End-Semester Exams",
        "2026-04-14": "End-Semester Exams",
        "2026-04-15": "End-Semester Exams",
        "2026-04-16": "End-Semester Exams",
        "2026-04-17": "End-Semester Exams",
        "2026-04-18": "End-Semester Exams End"
    }
    
    date_only = date_str.split(" ")[0] if " " in date_str else date_str
    event = events.get(date_only, "Normal Academic Day")
    
    # Check weekend rule
    try:
        dt = pd.to_datetime(date_only)
        if dt.weekday() >= 5 and event == "Normal Academic Day":
            event = "Weekend (Low load expected)"
    except Exception:
        pass
        
    return event


def build_system_prompt(dashboard_context: str) -> str:
    return f"""You are the AI Assistant for the EED SmartGrid Analytics dashboard at NIT Warangal.
You summarize energy consumption, explain anomalies, and correlate data with the academic calendar.

ACADEMIC CALENDAR CONTEXT:
{ACADEMIC_CALENDAR_CONTEXT}

DASHBOARD DATA CONTEXT:
{dashboard_context}

YOUR ROLE:
- Be concise and professional.
- Explain potential reasons for anomalies or high loads based on academic events (e.g., exams might mean high nighttime load in hostels or library; festivals might mean irregular patterns).
- Help users interpret terms like Power Factor (a measure of how effectively electricity is used; < 0.95 is bad).
- Format your response nicely using markdown (bullet points, bold text).
- Do not make up data.
"""

def get_ai_response(user_query: str, dashboard_context: str, chat_history: list) -> str:
    """Call Google Gemini to get an answer."""
    try:
        # Check if configured
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception:
            return "⚠️ Gemini API key not configured. Please add `GEMINI_API_KEY` to your Streamlit secrets."
            
        system_instruction = build_system_prompt(dashboard_context)
        
        # Build history
        messages = [{"role": "user", "parts": [system_instruction]}]
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            messages.append({"role": role, "parts": [msg["content"]]})
            
        messages.append({"role": "user", "parts": [user_query]})
        
        response = model.generate_content(messages)
        return response.text
        
    except Exception as e:
        return f"⚠️ Error communicating with AI: {str(e)}\n\n(Make sure your GEMINI_API_KEY is correctly set in Streamlit Cloud Secrets and is valid.)"
