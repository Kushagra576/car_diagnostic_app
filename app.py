import os
import json
import difflib
import html
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Settings
# ----------------------------
DEV_MODE = False
APP_TITLE = "Smart Car Diagnostic"
CSV_FILENAME = "bigger_car_diagnostics.csv"
FEEDBACK_FILENAME = "user_feedback.csv"
MAX_TEXT_LENGTH = 120
MAX_NOTE_LENGTH = 500
MAX_VIDEO_MB = 25


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ----------------------------
# CSS / Mobile UX
# ----------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #111827 0%, #0b0f14 45%, #06080c 100%);
    color: #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1.8rem;
    max-width: 1180px;
}

.app-title {
    margin-top: -6px;
    font-size: 2.45rem;
    font-weight: 850;
    letter-spacing: -0.04em;
    color: #f9fafb;
    margin-bottom: 0.2rem;
}

.app-subtitle {
    color: #9ca3af;
    margin-bottom: 1.1rem;
    font-size: 1rem;
}

.card {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 42, 0.96));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 1.15rem;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    backdrop-filter: blur(14px);
    margin-bottom: 1rem;
}

.section-title {
    font-size: 1.08rem;
    font-weight: 750;
    color: #f9fafb;
    margin-bottom: 0.35rem;
}

.section-subtitle, .muted {
    color: #94a3b8;
    font-size: 0.93rem;
    margin-bottom: 0.8rem;
}

.step-pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(96, 165, 250, 0.12);
    border: 1px solid rgba(96, 165, 250, 0.22);
    color: #bfdbfe;
    font-size: 0.82rem;
    font-weight: 750;
    margin-bottom: 0.75rem;
}

.result-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.25), rgba(15, 23, 42, 0.93));
    border: 1px solid rgba(96, 165, 250, 0.26);
    border-radius: 20px;
    padding: 1.05rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.14), inset 0 1px 0 rgba(255,255,255,0.05);
}

.result-kicker {
    font-size: 0.8rem;
    color: #93c5fd;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.result-title {
    font-size: 1.55rem;
    font-weight: 850;
    color: #f8fafc;
    margin-bottom: 0.2rem;
}

.result-confidence {
    font-size: 1rem;
    color: #cbd5e1;
}

.confidence-wrap { margin-top: 0.8rem; }
.confidence-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    color: #cbd5e1;
    margin-bottom: 0.35rem;
}
.confidence-bar {
    width: 100%;
    height: 10px;
    background: rgba(148, 163, 184, 0.14);
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.12);
}
.confidence-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #38bdf8 0%, #2563eb 55%, #1d4ed8 100%);
}

.prediction-card {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(10, 15, 28, 0.96));
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 16px;
    padding: 0.85rem 0.9rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.26);
}

.prediction-rank {
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #93c5fd;
    margin-bottom: 0.22rem;
}

.prediction-problem {
    font-size: 1rem;
    font-weight: 750;
    color: #f8fafc;
}

.prediction-prob {
    font-size: 0.9rem;
    color: #cbd5e1;
}

.feedback-note {
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.14), rgba(6, 78, 59, 0.2));
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #d1fae5;
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    margin-top: 0.75rem;
}

label, .stSelectbox label {
    color: #e2e8f0 !important;
    font-weight: 650 !important;
    font-size: 0.95rem !important;
}

div[data-testid="stTextInputRootElement"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.95) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    border-radius: 14px !important;
}

div[data-testid="stTextInputRootElement"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

div[data-baseweb="select"] span {
    color: #f8fafc !important;
}

div[role="listbox"] {
    background-color: #0f172a !important;
    color: #f8fafc !important;
}

div.stButton > button {
    width: 100%;
    border-radius: 14px !important;
    font-weight: 750 !important;
    border: 1px solid rgba(96, 165, 250, 0.22) !important;
    color: #f8fafc !important;
    background: linear-gradient(180deg, #1d4ed8 0%, #1e40af 100%) !important;
    box-shadow: 0 8px 20px rgba(29, 78, 216, 0.30);
}

div.stButton > button:hover {
    background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%) !important;
}

/* Desktop-only extra panels */
.mobile-only { display: none; }

/* Phone flow */
@media (max-width: 768px) {
    .block-container {
        padding-top: 0.65rem !important;
        padding-left: 0.72rem !important;
        padding-right: 0.72rem !important;
        padding-bottom: 1.1rem !important;
    }

    .app-title {
        font-size: 1.72rem !important;
        line-height: 1.08 !important;
        margin-bottom: 0.16rem !important;
    }

    .app-subtitle {
        font-size: 0.86rem !important;
        margin-bottom: 0.8rem !important;
    }

    .card {
        border-radius: 18px !important;
        padding: 0.9rem 0.85rem !important;
        margin-bottom: 0.75rem !important;
    }

    .desktop-only { display: none !important; }
    .mobile-only { display: block !important; }

    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    div[data-testid="stTextInputRootElement"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 50px !important;
        font-size: 16px !important;
        border-radius: 12px !important;
    }

    div.stButton > button {
        min-height: 50px !important;
        font-size: 0.97rem !important;
        border-radius: 12px !important;
    }

    .result-title { font-size: 1.25rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Security / validation helpers
# ----------------------------
def clean_text(value: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    """Limit length, remove control characters, and strip formula-like CSV injection prefixes."""
    value = str(value or "").strip()
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    value = value[:max_len]
    if value.startswith(("=", "+", "-", "@")):
        value = "'" + value
    return value


def escape(value: str) -> str:
    return html.escape(str(value or ""))


def safe_video_name(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_VIDEO_MB:
        st.warning(f"Video is too large. Max allowed is {MAX_VIDEO_MB} MB.")
        return ""

    return clean_text(uploaded_file.name, 160)


# ----------------------------
# Data / model helpers
# ----------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, CSV_FILENAME)
    data = pd.read_csv(file_path)

    required_cols = ["Noise", "Vibration", "Smoke", "When", "Location", "CheckEngineLight", "Problem"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    for col in required_cols:
        data[col] = data[col].astype(str).str.lower().str.strip()

    return data, file_path


@st.cache_resource
def train_model_from_data(data: pd.DataFrame):
    X = pd.get_dummies(
        data[["Noise", "Vibration", "Smoke", "When", "Location", "CheckEngineLight"]]
    )
    y = data["Problem"]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)
    return model, X.columns


def match_noise(user_noise, valid_noises):
    if user_noise in valid_noises:
        return user_noise, None

    matches = difflib.get_close_matches(user_noise, valid_noises, n=1, cutoff=0.4)
    if matches:
        return matches[0], f"I haven't learned '{user_noise}', so I used '{matches[0]}' instead."

    return None, f"I haven't learned '{user_noise}' and couldn't find a close match."


def save_feedback_row(feedback_row, filename=FEEDBACK_FILENAME):
    base_path = os.path.dirname(os.path.abspath(__file__))
    feedback_file = os.path.join(base_path, filename)
    df = pd.DataFrame([feedback_row])

    if os.path.exists(feedback_file):
        df.to_csv(feedback_file, mode="a", header=False, index=False)
    else:
        df.to_csv(feedback_file, index=False)


def make_feedback_row(user_inputs, top_problem, top_prob, top_predictions, feedback_type, actual_problem="", note="", video_name=""):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "noise": clean_text(user_inputs["Noise"]),
        "vibration": clean_text(user_inputs["Vibration"]),
        "smoke": clean_text(user_inputs["Smoke"]),
        "when": clean_text(user_inputs["When"]),
        "location": clean_text(user_inputs["Location"]),
        "check_engine_light": clean_text(user_inputs["CheckEngineLight"]),
        "predicted_problem": clean_text(top_problem),
        "prediction_confidence": round(float(top_prob), 2),
        "top_3_predictions": json.dumps(top_predictions),
        "feedback": clean_text(feedback_type),
        "actual_problem": clean_text(actual_problem, 160),
        "user_note": clean_text(note, MAX_NOTE_LENGTH),
        "video_name": clean_text(video_name, 160),
    }


def reset_for_new_input():
    st.session_state.step = "input"
    st.session_state.feedback_status = ""


def render_result_card(diag):
    safe_problem = escape(diag["top_problem"])
    safe_prob = float(diag["top_prob"])

    st.markdown(f"""
    <div class="result-hero">
        <div class="result-kicker">Primary Match</div>
        <div class="result-title">🔧 {safe_problem.title()}</div>
        <div class="result-confidence">{safe_prob:.2f}% confidence</div>
        <div class="confidence-wrap">
            <div class="confidence-label">
                <span>AI Confidence</span>
                <span>{safe_prob:.2f}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {min(max(safe_prob, 0), 100):.2f}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_top_predictions(diag):
    html_cards = ""
    for i, item in enumerate(diag["top_predictions"], start=1):
        html_cards += f"""
        <div class="prediction-card">
            <div class="prediction-rank">Option {i}</div>
            <div class="prediction-problem">{escape(item["problem"]).title()}</div>
            <div class="prediction-prob">{float(item["probability"]):.2f}% match</div>
        </div>
        """
    st.markdown(html_cards, unsafe_allow_html=True)


def run_diagnosis(noise, vibration, smoke, when, location, cel):
    clean_noise = clean_text(noise, 50).lower()

    if not clean_noise:
        st.session_state.feedback_status = ""
        st.error("Please enter a noise before running the diagnosis.")
        return

    matched_noise, noise_message = match_noise(clean_noise, valid_noises)

    if matched_noise is None:
        st.warning(f"{noise_message}\n\nKnown sounds: " + ", ".join(valid_noises))
        return

    new_data = pd.DataFrame(
        [[matched_noise, vibration, smoke, when, location, cel]],
        columns=["Noise", "Vibration", "Smoke", "When", "Location", "CheckEngineLight"]
    )
    new_data = pd.get_dummies(new_data).reindex(columns=model_columns, fill_value=0)

    probs = model.predict_proba(new_data)[0]
    results = list(zip(model.classes_, probs))
    results.sort(key=lambda x: x[1], reverse=True)

    top_problem, top_prob_raw = results[0]
    second_problem, second_prob_raw = results[1] if len(results) > 1 else ("unknown", 0)

    top_prob = float(top_prob_raw) * 100
    second_prob = float(second_prob_raw) * 100

    top_predictions = [
        {"problem": clean_text(problem, 160), "probability": round(float(prob) * 100, 2)}
        for problem, prob in results[:3]
    ]

    user_inputs = {
        "Noise": matched_noise,
        "When": when,
        "Location": location,
        "Vibration": vibration,
        "Smoke": smoke,
        "CheckEngineLight": cel
    }

    if top_prob < 35:
        summary = "Low confidence result. Add more detail or test more symptoms."
    elif abs(top_prob - second_prob) < 5:
        summary = f"Uncertain between {top_problem} ({top_prob:.2f}%) and {second_problem} ({second_prob:.2f}%)."
    else:
        summary = f"Most likely issue: {top_problem} ({top_prob:.2f}%)."

    if noise_message:
        summary = noise_message + " " + summary

    st.session_state.diagnosis_result = {
        "user_inputs": user_inputs,
        "top_problem": clean_text(top_problem, 160),
        "top_prob": round(top_prob, 2),
        "top_predictions": top_predictions,
        "noise_message": clean_text(noise_message or "", 200),
        "summary": clean_text(summary, 320),
    }

    st.session_state.prediction = top_problem
    st.session_state.confidence = top_prob
    st.session_state.step = "result"
    st.session_state.feedback_status = ""
    st.rerun()


# ----------------------------
# Session state
# ----------------------------
defaults = {
    "diagnosis_result": None,
    "chat_history": [],
    "feedback_status": "",
    "step": "input",
    "prediction": None,
    "confidence": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ----------------------------
# Load model
# ----------------------------
try:
    data, csv_path = load_data()
    model, model_columns = train_model_from_data(data)
    valid_noises = sorted(data["Noise"].dropna().astype(str).str.lower().str.strip().unique().tolist())
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()


# ----------------------------
# Header
# ----------------------------
st.markdown(f'<div class="app-title">{escape(APP_TITLE)}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Fast symptom check for common car issues. Results are guidance, not a guaranteed diagnosis.</div>',
    unsafe_allow_html=True
)


# ----------------------------
# Input controls shared by mobile / desktop
# ----------------------------
def symptom_form(prefix=""):
    noise = st.text_input(
        "Noise",
        placeholder="squealing, grinding, knocking...",
        max_chars=50,
        key=f"{prefix}noise"
    )
    vibration = st.selectbox("Vibration", ["no", "yes"], key=f"{prefix}vibration")
    smoke = st.selectbox("Smoke", ["no", "yes"], key=f"{prefix}smoke")
    when = st.selectbox(
        "When?",
        ["startup", "idle", "driving", "accelerating", "braking", "turning", "decelerating"],
        key=f"{prefix}when"
    )
    location = st.selectbox(
        "Location",
        ["front", "rear", "engine bay", "dashboard", "transmission tunnel"],
        key=f"{prefix}location"
    )
    cel = st.selectbox("Check Engine Light", ["no", "yes"], key=f"{prefix}cel")
    return noise, vibration, smoke, when, location, cel


def feedback_controls(prefix=""):
    diag = st.session_state.diagnosis_result
    if not diag:
        return

    st.markdown('<div class="section-title">Was this correct?</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    yes_clicked = col1.button("Yes", use_container_width=True, key=f"{prefix}yes")
    no_clicked = col2.button("No", use_container_width=True, key=f"{prefix}no")
    unsure_clicked = col3.button("Not Sure", use_container_width=True, key=f"{prefix}not_sure")

    with st.expander("Add details / corrected issue", expanded=st.session_state.feedback_status.startswith("Type")):
        actual_problem = st.text_input(
            "Actual issue if wrong",
            placeholder="Example: worn belt",
            max_chars=120,
            key=f"{prefix}actual"
        )
        note = st.text_area(
            "Optional note",
            placeholder="What happened? What fixed it?",
            max_chars=MAX_NOTE_LENGTH,
            height=90,
            key=f"{prefix}note"
        )
        uploaded_video = st.file_uploader(
            "Optional video",
            type=["mp4", "mov", "avi", "mkv"],
            key=f"{prefix}video"
        )
        save_corrected = st.button("Save Corrected Feedback", use_container_width=True, key=f"{prefix}save_corrected")

    video_name = safe_video_name(uploaded_video)

    if yes_clicked:
        row = make_feedback_row(
            diag["user_inputs"], diag["top_problem"], diag["top_prob"], diag["top_predictions"],
            "yes", actual_problem=diag["top_problem"], note=note, video_name=video_name
        )
        save_feedback_row(row)
        st.session_state.feedback_status = "Thanks. Confirmed feedback saved."
        st.rerun()

    if unsure_clicked:
        row = make_feedback_row(
            diag["user_inputs"], diag["top_problem"], diag["top_prob"], diag["top_predictions"],
            "not_sure", actual_problem="", note=note, video_name=video_name
        )
        save_feedback_row(row)
        st.session_state.feedback_status = "Saved as not sure."
        st.rerun()

    if no_clicked:
        st.session_state.feedback_status = "Type the actual issue, then click Save Corrected Feedback."
        st.rerun()

    if save_corrected:
        if not clean_text(actual_problem, 120):
            st.session_state.feedback_status = "Type the actual issue before saving."
        else:
            row = make_feedback_row(
                diag["user_inputs"], diag["top_problem"], diag["top_prob"], diag["top_predictions"],
                "no", actual_problem=actual_problem, note=note, video_name=video_name
            )
            save_feedback_row(row)
            st.session_state.feedback_status = "Corrected feedback saved."
        st.rerun()

    if st.session_state.feedback_status:
        st.markdown(f'<div class="feedback-note">{escape(st.session_state.feedback_status)}</div>', unsafe_allow_html=True)


# ----------------------------
# Single mobile-first flow (no duplicate inputs)
# ----------------------------
# This same flow works on desktop and phone. On desktop it stays centered and clean;
# on phone it behaves like a simple 3-step app: symptoms -> result -> feedback.

if st.session_state.step == "input":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-pill">Step 1 of 3 · Symptoms</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What is your car doing?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Fill this out once. After the result, you can easily come back and edit it.</div>', unsafe_allow_html=True)

    noise, vibration, smoke, when, location, cel = symptom_form("main_")

    if st.button("Analyze Symptoms", use_container_width=True, key="main_analyze"):
        run_diagnosis(noise, vibration, smoke, when, location, cel)

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == "result":
    diag = st.session_state.diagnosis_result

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-pill">Step 2 of 3 · Result</div>', unsafe_allow_html=True)

    if diag:
        render_result_card(diag)
        st.info(diag["summary"])
        render_top_predictions(diag)

        col1, col2 = st.columns(2)
        if col1.button("Edit Symptoms", use_container_width=True, key="main_edit"):
            reset_for_new_input()
            st.rerun()
        if col2.button("Give Feedback", use_container_width=True, key="main_to_feedback"):
            st.session_state.step = "feedback"
            st.rerun()
    else:
        st.info("Run a diagnosis first.")
        if st.button("Go to Symptoms", use_container_width=True, key="main_no_diag"):
            reset_for_new_input()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == "feedback":
    diag = st.session_state.diagnosis_result

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-pill">Step 3 of 3 · Feedback</div>', unsafe_allow_html=True)

    if diag:
        render_result_card(diag)
        feedback_controls("main_feedback_")

        col1, col2 = st.columns(2)
        if col1.button("Back to Result", use_container_width=True, key="main_back_result"):
            st.session_state.step = "result"
            st.rerun()
        if col2.button("New Diagnosis", use_container_width=True, key="main_new"):
            st.session_state.diagnosis_result = None
            st.session_state.prediction = None
            st.session_state.confidence = None
            reset_for_new_input()
            st.rerun()
    else:
        st.info("Run a diagnosis first.")
        if st.button("Go to Symptoms", use_container_width=True, key="main_feedback_no_diag"):
            reset_for_new_input()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Optional follow-up stays below the main flow so it does not crowd the phone screen.
with st.expander("Ask a follow-up question", expanded=False):
    followup_question = st.text_input(
        "Follow-up",
        placeholder="Example: Can I still drive like this?",
        max_chars=160,
        key="main_followup"
    )
    if st.button("Send", use_container_width=True, key="main_send_followup"):
        if not st.session_state.diagnosis_result:
            st.session_state.chat_history.append(("Mechanic Assistant", "Please run a diagnosis first."))
        elif followup_question.strip():
            st.session_state.chat_history.append(("You", clean_text(followup_question, 160)))
            st.session_state.chat_history.append((
                "Mechanic Assistant",
                "Follow-up chat is disabled in this version because the local LLM is not installed."
            ))

    if st.session_state.chat_history:
        for speaker, message in st.session_state.chat_history:
            st.markdown(f"**{escape(speaker)}:**  \n{escape(message)}")

# ----------------------------
# Sidebar diagnostics
# ----------------------------
with st.sidebar:
    st.markdown("### App Info")
    st.write(f"CSV file: `{CSV_FILENAME}`")
    st.write(f"Feedback file: `{os.path.join(os.path.dirname(os.path.abspath(__file__)), FEEDBACK_FILENAME)}`")
    st.write(f"Loaded from: `{csv_path}`")
    st.write(f"Rows: `{len(data)}`")
    st.write(f"Problems: `{len(sorted(data['Problem'].unique()))}`")
    st.write(f"Dev mode: `{DEV_MODE}`")
    st.markdown("### Security basics added")
    st.write("- Input length limits")
    st.write("- CSV injection guard")
    st.write("- HTML escaping for displayed user/model text")
    st.write(f"- Video upload size warning: {MAX_VIDEO_MB} MB")
