import os
import json
import difflib
import time
from textwrap import dedent
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


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

/* -----------------------------
   Base app
----------------------------- */
.stApp {
    background: radial-gradient(circle at top, #111827 0%, #0b0f14 45%, #06080c 100%);
    color: #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
}

/* Main container */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1.8rem;
    max-width: 1280px;
}

/* Remove weird spacing/ghost layout look */
div[data-testid="stHorizontalBlock"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* -----------------------------
   Typography
----------------------------- */
.app-title {
    margin-top: -6px;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f9fafb;
    margin-bottom: 0.2rem;
    animation: fadeSlideDown 0.7s ease-out;
}

.app-subtitle {
    color: #9ca3af;
    margin-bottom: 1.2rem;
    font-size: 1rem;
    animation: fadeSlideDown 0.9s ease-out;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f9fafb;
    margin-bottom: 0.35rem;
    letter-spacing: -0.01em;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 0.94rem;
    margin-bottom: 0.85rem;
}

.muted {
    color: #94a3b8;
    font-size: 0.92rem;
}

/* -----------------------------
   Cards
----------------------------- */
.card {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 42, 0.96));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 1.15rem 1.15rem 1.2rem 1.15rem;
    box-shadow:
        0 12px 40px rgba(0, 0, 0, 0.35),
        inset 0 1px 0 rgba(255,255,255,0.04);
    backdrop-filter: blur(14px);
    transition: all 0.25s ease;
    animation: fadeUp 0.7s ease-out;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow:
        0 18px 46px rgba(0, 0, 0, 0.42),
        0 0 0 1px rgba(96, 165, 250, 0.08);
}

/* -----------------------------
   Inputs (FIXED VISIBILITY)
----------------------------- */

/* Labels (Noise, Vibration, etc.) */
label, .stSelectbox label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Text input */
div[data-testid="stTextInputRootElement"] input {
    background: rgba(15, 23, 42, 0.95) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    border-radius: 14px !important;
    padding: 10px !important;
}

/* Placeholder text */
div[data-testid="stTextInputRootElement"] input::placeholder {
    color: #64748b !important;
}

/* Dropdown (selectbox main box) */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.95) !important;
    color: #f8fafc !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
}

/* Dropdown text (selected value) */
div[data-baseweb="select"] span {
    color: #f8fafc !important;
    font-weight: 500 !important;
}

/* Dropdown menu options */
div[role="listbox"] {
    background-color: #0f172a !important;
    color: #f8fafc !important;
}

/* Hover option */
div[role="option"]:hover {
    background-color: #1e293b !important;
}

/* Focus glow */
div[data-testid="stTextInputRootElement"] input:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border: 1px solid #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.25) !important;
}
            
/* -----------------------------
   Buttons
----------------------------- */
div.stButton > button {
    width: 100%;
    border-radius: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    border: 1px solid rgba(96, 165, 250, 0.22) !important;
    color: #f8fafc !important;
    background: linear-gradient(180deg, #1d4ed8 0%, #1e40af 100%) !important;
    box-shadow:
        0 8px 20px rgba(29, 78, 216, 0.30),
        0 0 0 rgba(59,130,246,0);
    transition: all 0.22s ease !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    border-color: rgba(147, 197, 253, 0.55) !important;
    background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%) !important;
    box-shadow:
        0 12px 28px rgba(37, 99, 235, 0.42),
        0 0 18px rgba(59, 130, 246, 0.30);
}

div.stButton > button:active {
    transform: translateY(0px) scale(0.995);
}

/* -----------------------------
   Result area
----------------------------- */
            
/* Confidence bar */
.confidence-wrap {
    margin-top: 0.8rem;
}

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
    box-shadow: 0 0 18px rgba(59, 130, 246, 0.35);
    animation: fillBar 1s ease-out forwards;
    transform-origin: left;
}

/* Top prediction cards */
.prediction-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(10, 15, 28, 0.96));
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 16px;
    padding: 0.85rem 0.9rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.26);
    animation: fadeUp 0.45s ease-out;
}

.prediction-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 20% 20%, rgba(59,130,246,0.16), transparent 60%);
    pointer-events: none;
}

.prediction-rank {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #93c5fd;
    margin-bottom: 0.25rem;
}

.prediction-problem {
    font-size: 1rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.2rem;
}

.prediction-prob {
    font-size: 0.92rem;
    color: #cbd5e1;
}

@keyframes fillBar {
    from {
        width: 0%;
    }
}
            
.result-hero {
    position: relative; 
    overflow: hidden; 

    background: linear-gradient(135deg, rgba(37, 99, 235, 0.22), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    margin-bottom: 0.9rem;
    box-shadow:
        0 10px 30px rgba(37, 99, 235, 0.14),
        inset 0 1px 0 rgba(255,255,255,0.05);
    animation: resultReveal 0.55s ease-out;
}

/* Glow effect */
.result-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 18px;
    background: radial-gradient(circle at 20% 20%, rgba(59,130,246,0.25), transparent 60%);
    opacity: 0.4;
    pointer-events: none;
}

.feedback-note {
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.14), rgba(6, 78, 59, 0.2));
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #d1fae5;
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    margin-top: 0.75rem;
    animation: fadeUp 0.5s ease-out;
}

/* -----------------------------
   Info / alert boxes
----------------------------- */
div[data-testid="stInfo"] {
    border-radius: 16px !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    background: rgba(15, 23, 42, 0.78) !important;
}

/* -----------------------------
   File uploader
----------------------------- */
div[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.55);
    border-radius: 16px;
    padding: 0.35rem;
    border: 1px dashed rgba(148, 163, 184, 0.25);
}

/* -----------------------------
   Subtle divider polish
----------------------------- */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(148,163,184,0.18), transparent);
    margin: 1rem 0;
}

/* -----------------------------
   Animations
----------------------------- */
@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(14px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeSlideDown {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes resultReveal {
    from {
        opacity: 0;
        transform: scale(0.97) translateY(10px);
    }
    to {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

/* -----------------------------
   Mobile responsive
----------------------------- */
@media (max-width: 768px) {
    .block-container {
        padding-top: 0.7rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-bottom: 1.2rem !important;
    }

    .app-title {
        font-size: 1.85rem !important;
        line-height: 1.1 !important;
        margin-top: 0 !important;
    }

    .app-subtitle {
        font-size: 0.92rem !important;
        margin-bottom: 0.9rem !important;
    }

    .card {
        border-radius: 18px !important;
        padding: 0.9rem 0.85rem 1rem 0.85rem !important;
    }

    .section-title {
        font-size: 1rem !important;
    }

    .section-subtitle,
    .muted {
        font-size: 0.88rem !important;
    }

    div[data-testid="stTextInputRootElement"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 52px !important;
        font-size: 16px !important;
        border-radius: 12px !important;
    }

    div.stButton > button {
        min-height: 52px !important;
        font-size: 0.98rem !important;
        border-radius: 12px !important;
    }

    .result-hero {
        padding: 1rem 0.95rem !important;
        border-radius: 16px !important;
    }

    .result-hero div:nth-child(2) {
        font-size: 1.25rem !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ----------------------------
# Data / model helpers
# ----------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, CSV_FILENAME)
    data = pd.read_csv(file_path)

    for col in ["Noise", "Vibration", "Smoke", "When", "Location", "CheckEngineLight", "Problem"]:
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
        "noise": user_inputs["Noise"],
        "vibration": user_inputs["Vibration"],
        "smoke": user_inputs["Smoke"],
        "when": user_inputs["When"],
        "location": user_inputs["Location"],
        "check_engine_light": user_inputs["CheckEngineLight"],
        "predicted_problem": top_problem,
        "prediction_confidence": round(float(top_prob), 2),
        "top_3_predictions": json.dumps(top_predictions),
        "feedback": feedback_type,
        "actual_problem": actual_problem.strip(),
        "user_note": note.strip(),
        "video_name": video_name,
    }


# ----------------------------
# Session state
# ----------------------------
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback_status" not in st.session_state:
    st.session_state.feedback_status = ""


# ----------------------------
# Load model
# ----------------------------
try:
    data, csv_path = load_data()
    model, model_columns = train_model_from_data(data)
    valid_noises = sorted(
        data["Noise"].dropna().astype(str).str.lower().str.strip().unique().tolist()
    )
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()


# ----------------------------
# Header
# ----------------------------
st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enter your symptoms, review the most likely issues, and optionally collect user feedback to improve the model.</div>',
    unsafe_allow_html=True
)

if DEV_MODE:
    st.info("Dev mode is ON. Feedback controls are hidden so you can test the app without clicking Yes/No every time.")

left_col, right_col = st.columns([1.02, 1], gap="large")

# ----------------------------
# Left panel
# ----------------------------
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Symptoms</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Fill out the details below to analyze the issue.</div>', unsafe_allow_html=True)

    noise = st.text_input("Noise", placeholder="Type a sound like squealing, grinding, chirping...")
    vibration = st.selectbox("Vibration", ["yes", "no"], index=1)
    smoke = st.selectbox("Smoke", ["yes", "no"], index=1)
    when = st.selectbox(
        "When does it happen?",
        ["startup", "idle", "driving", "accelerating", "braking", "turning", "decelerating"]
    )
    location = st.selectbox(
        "Where is it coming from?",
        ["front", "rear", "engine bay", "dashboard", "transmission tunnel"]
    )
    cel = st.selectbox("Check Engine Light", ["yes", "no"], index=1)

    analyze = st.button("Analyze Symptoms", use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top: 1rem;">Summary</div>', unsafe_allow_html=True)
    summary_placeholder = st.empty()

    st.markdown('<div class="section-title" style="margin-top: 1rem;">Follow-up</div>', unsafe_allow_html=True)
    followup_question = st.text_input("Ask more questions", placeholder="Example: Can I still drive like this?")
    send_followup = st.button("Send", use_container_width=True)
    chat_placeholder = st.empty()

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Right panel
# ----------------------------
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Diagnostic Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Predictions appear here after you analyze symptoms.</div>', unsafe_allow_html=True)

    result_placeholder = st.empty()
    predictions_placeholder = st.empty()

    st.markdown(
        '<div class="muted" style="margin-top: 0.85rem;">These probabilities reflect patterns in the dataset, not guaranteed real-world accuracy.</div>',
        unsafe_allow_html=True
    )

    if not DEV_MODE:
        st.markdown('<div class="section-title" style="margin-top: 1rem;">Feedback</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Was this diagnosis correct?</div>', unsafe_allow_html=True)

        fb1, fb2, fb3 = st.columns(3)
        yes_clicked = fb1.button("Yes", use_container_width=True)
        no_clicked = fb2.button("No", use_container_width=True)
        not_sure_clicked = fb3.button("Not Sure", use_container_width=True)

        actual_problem = st.text_input("If no, what was the actual issue?", placeholder="Example: worn belt, bad starter, brake pad wear")
        user_note = st.text_area("Optional note", placeholder="Add extra detail about what really happened...", height=90)
        uploaded_video = st.file_uploader("Optional video upload", type=["mp4", "mov", "avi", "mkv"])
        save_corrected = st.button("Save Corrected Feedback", use_container_width=True)

        if st.session_state.feedback_status:
            st.markdown(f'<div class="feedback-note">{st.session_state.feedback_status}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Diagnosis
# ----------------------------
if analyze:
    if noise.strip() == "":
        st.session_state.diagnosis_result = None
        result_placeholder.error("Please enter a noise before running the diagnosis.")
        predictions_placeholder.empty()
        summary_placeholder.info("Enter symptoms to get a prediction summary.")
    else:
        user_noise = noise.lower().strip()
        matched_noise, noise_message = match_noise(user_noise, valid_noises)

        if matched_noise is None:
            st.session_state.diagnosis_result = None
            result_placeholder.warning(
                f"{noise_message}\n\nKnown sounds:\n" + ", ".join(valid_noises)
            )
            predictions_placeholder.empty()
            summary_placeholder.info("I could not confidently map that sound to something I know yet.")
        else:
            new_data = pd.DataFrame(
                [[matched_noise, vibration, smoke, when, location, cel]],
                columns=["Noise", "Vibration", "Smoke", "When", "Location", "CheckEngineLight"]
            )
            new_data = pd.get_dummies(new_data).reindex(columns=model_columns, fill_value=0)

            probs = model.predict_proba(new_data)[0]
            results = list(zip(model.classes_, probs))
            results.sort(key=lambda x: x[1], reverse=True)

            top_problem, top_prob = results[0]
            second_problem, second_prob = results[1]

            top_prob = top_prob * 100
            second_prob = second_prob * 100

            top_predictions = []
            prediction_lines = []
            for problem, prob in results[:3]:
                top_predictions.append({
                    "problem": problem,
                    "probability": round(float(prob) * 100, 2)
                })
                prediction_lines.append(f"- **{problem}** — {prob * 100:.2f}%")

            user_inputs = {
                "Noise": matched_noise,
                "When": when,
                "Location": location,
                "Vibration": vibration,
                "Smoke": smoke,
                "CheckEngineLight": cel
            }

            st.session_state.diagnosis_result = {
                "user_inputs": user_inputs,
                "top_problem": top_problem,
                "top_prob": round(float(top_prob), 2),
                "top_predictions": top_predictions,
                "noise_message": noise_message or "",
            }

            summary_lines = []
            if noise_message:
                summary_lines.append(noise_message)

            if top_prob < 35:
                summary_lines.append("Low confidence result. Try adding more detail or testing more symptoms.")
            elif abs(top_prob - second_prob) < 5:
                summary_lines.append(
                    f"The result is uncertain between {top_problem} ({top_prob:.2f}%) and {second_problem} ({second_prob:.2f}%)."
                )
            else:
                summary_lines.append(f"Most likely issue: {top_problem} ({top_prob:.2f}%).")

            result_placeholder.empty()

            # Step 1: Thinking message
            result_placeholder.markdown(
                "<div class='muted'>Analyzing symptoms...</div>",
                unsafe_allow_html=True
            )

            time.sleep(0.8)

            # Step 2: Typing effect
            typing_placeholder = result_placeholder.empty()

            response_text = f"I analyzed your symptoms.\n\nMost likely issue: {top_problem}.\nConfidence: {top_prob:.2f}%"

            typed_text = ""
            for char in response_text:
                typed_text += char
                typing_placeholder.markdown(
                    f"<div class='muted' style='white-space: pre-line;'>{typed_text}<span style='opacity:0.6'>|</span></div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.015)

            time.sleep(0.3)

            # Step 3: Final result card
            typing_placeholder.markdown(
                dedent(f"""
                <div class="result-hero">
                    <div style="font-size: 0.85rem; color: #93c5fd; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem;">
                        Primary Match
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.2rem;">
                        {top_problem}
                    </div>
                    <div style="font-size: 1rem; color: #cbd5e1;">
                        {top_prob:.2f}% confidence
                    </div>
                    <div class="confidence-wrap">
                        <div class="confidence-label">
                            <span>AI Confidence</span>
                            <span>{top_prob:.2f}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {top_prob:.2f}%"></div>
                        </div>
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )
            predictions_placeholder.empty()

            current_html = ""

            for i, item in enumerate(top_predictions, start=1):
                card_html = f"""
                <div class="prediction-card">
                    <div class="prediction-rank">Prediction {i}</div>
                    <div class="prediction-problem">{item["problem"]}</div>
                    <div class="prediction-prob">{item["probability"]:.2f}% match</div>
                </div>
                """

                current_html += card_html
                predictions_placeholder.markdown(current_html, unsafe_allow_html=True)

                time.sleep(0.15)

            summary_placeholder.info(" ".join(summary_lines))
            st.session_state.chat_history = []
            st.session_state.feedback_status = ""

# ----------------------------
# Render existing diagnosis
# ----------------------------
if st.session_state.diagnosis_result:
    diag = st.session_state.diagnosis_result
    if not analyze:
        result_placeholder.markdown(
            f'''
            <div class="result-hero">
                <div style="font-size: 0.85rem; color: #93c5fd; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem;">
                    Primary Match
                </div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.2rem;">
                    {diag["top_problem"]}
                </div>
                <div style="font-size: 1rem; color: #cbd5e1;">
                    {diag["top_prob"]:.2f}% confidence
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        top3_html = ""
        for i, item in enumerate(diag["top_predictions"], start=1):
            top3_html += f"""
            <div class="prediction-card" style="animation-delay: {i * 0.08}s;">
                <div class="prediction-rank">Option {i}</div>
                <div class="prediction-problem">{item["problem"]}</div>
                <div class="prediction-prob">{item["probability"]:.2f}% match</div>
            </div>
            """

        predictions_placeholder.markdown(top3_html, unsafe_allow_html=True)

        summary_text = []
        if diag["noise_message"]:
            summary_text.append(diag["noise_message"])
        summary_text.append(
            f"Based on the current symptom pattern, the strongest match is {diag['top_problem']}."
        )
        summary_text.append(
            f"Treat this as guidance rather than a guaranteed diagnosis."
        )
        summary_placeholder.info(" ".join(summary_text))

# ----------------------------
# Follow-up
# ----------------------------
if send_followup:
    if not st.session_state.diagnosis_result:
        st.session_state.chat_history.append(("Mechanic Assistant", "Please run a diagnosis first."))
    elif followup_question.strip():
        st.session_state.chat_history.append(("You", followup_question.strip()))
        st.session_state.chat_history.append((
            "Mechanic Assistant",
            "Follow-up chat is disabled in this version because the local LLM is not installed."
        ))

if st.session_state.chat_history:
    chat_lines = []
    for speaker, message in st.session_state.chat_history:
        chat_lines.append(f"**{speaker}:**  \n{message}")
    chat_placeholder.markdown("\n\n".join(chat_lines))
else:
    chat_placeholder.markdown("")

# ----------------------------
# Feedback
# ----------------------------
if not DEV_MODE and st.session_state.diagnosis_result:
    diag = st.session_state.diagnosis_result
    video_name = uploaded_video.name if uploaded_video is not None else ""

    if yes_clicked:
        row = make_feedback_row(
            diag["user_inputs"],
            diag["top_problem"],
            diag["top_prob"],
            diag["top_predictions"],
            "yes",
            actual_problem=diag["top_problem"],
            note=user_note,
            video_name=video_name,
        )
        save_feedback_row(row)
        st.session_state.feedback_status = "Thanks. Confirmed feedback saved."

    if not_sure_clicked:
        row = make_feedback_row(
            diag["user_inputs"],
            diag["top_problem"],
            diag["top_prob"],
            diag["top_predictions"],
            "not_sure",
            actual_problem="",
            note=user_note,
            video_name=video_name,
        )
        save_feedback_row(row)
        st.session_state.feedback_status = "Saved as not sure."

    if no_clicked:
        st.session_state.feedback_status = "Type the actual issue, then click Save Corrected Feedback."

    if save_corrected:
        if actual_problem.strip() == "":
            st.session_state.feedback_status = "Type the actual issue before saving."
        else:
            row = make_feedback_row(
                diag["user_inputs"],
                diag["top_problem"],
                diag["top_prob"],
                diag["top_predictions"],
                "no",
                actual_problem=actual_problem,
                note=user_note,
                video_name=video_name,
            )
            save_feedback_row(row)
            st.session_state.feedback_status = "Corrected feedback saved."

# ----------------------------
# Sidebar diagnostics
# ----------------------------
with st.sidebar:
    st.markdown("### App Info")
    st.write(f"CSV file: `{CSV_FILENAME}`")
    st.write(f"Loaded from: `{csv_path}`")
    st.write(f"Rows: `{len(data)}`")
    st.write(f"Problems: `{len(sorted(data['Problem'].unique()))}`")
    st.write(f"Dev mode: `{DEV_MODE}`")
