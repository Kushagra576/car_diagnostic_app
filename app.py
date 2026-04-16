import difflib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Smart Car Diagnostic",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f6efe7;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.main-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #7a3e12;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    font-size: 1rem;
    color: #8a6a55;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #7a3e12;
    margin-bottom: 0.25rem;
}

.section-subtitle {
    font-size: 0.95rem;
    color: #8a6a55;
    margin-bottom: 1rem;
}

.result-good {
    background-color: #f6eadf;
    border-left: 6px solid #c96b2c;
    padding: 14px 16px;
    border-radius: 12px;
    color: #4e3426;
    font-weight: 600;
    margin-bottom: 1rem;
}

.result-warn {
    background-color: #fff4e8;
    border-left: 6px solid #d97706;
    padding: 14px 16px;
    border-radius: 12px;
    color: #4e3426;
    font-weight: 600;
    margin-bottom: 1rem;
}

.small-note {
    font-size: 0.9rem;
    color: #8a6a55;
    margin-top: 1rem;
}

.prediction-row {
    background-color: #fffaf5;
    border: 1px solid #e6d3c3;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 10px;
    color: #4e3426;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data = pd.read_csv("bigger_car_diagnostics.csv")

    for col in ["Noise", "Vibration", "Smoke", "When", "Location", "CheckEngineLight", "Problem"]:
        data[col] = data[col].astype(str).str.lower().str.strip()

    return data


@st.cache_resource
def train_model(data):
    X = pd.get_dummies(
        data[["Noise", "Vibration", "Smoke", "When", "Location", "CheckEngineLight"]]
    )
    y = data["Problem"]

    model = RandomForestClassifier(
        n_estimators=200,
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


try:
    data = load_data()
    model, model_columns = train_model(data)

    valid_noises = sorted(data["Noise"].unique())
    valid_vibration = ["yes", "no"]
    valid_smoke = ["yes", "no"]
    valid_when = [
        "startup", "idle", "driving", "accelerating",
        "braking", "turning", "decelerating"
    ]
    valid_location = [
        "front", "rear", "engine bay", "dashboard", "transmission tunnel"
    ]
    valid_cel = ["yes", "no"]

    st.markdown('<div class="main-title">Smart Car Diagnostic</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Enter your symptoms and get intelligent predictions for possible car issues.</div>',
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">Enter Symptoms</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Fill out the details below to analyze the issue.</div>',
                unsafe_allow_html=True
            )

            noise = st.text_input("Noise", placeholder="Type a sound like squealing, grinding, chirping...")
            vibration = st.selectbox("Vibration", valid_vibration)
            smoke = st.selectbox("Smoke", valid_smoke)
            when = st.selectbox("When does it happen?", valid_when)
            location = st.selectbox("Where is it coming from?", valid_location)
            cel = st.selectbox("Check Engine Light", valid_cel)

            diagnose = st.button("Analyze Symptoms", use_container_width=True)

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">Diagnostic Results</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Your results will appear here after running the diagnosis.</div>',
                unsafe_allow_html=True
            )

            if diagnose:
                noise = noise.lower().strip()
                vibration = vibration.lower().strip()
                smoke = smoke.lower().strip()
                when = when.lower().strip()
                location = location.lower().strip()
                cel = cel.lower().strip()

                matched_noise, noise_message = match_noise(noise, valid_noises)

                if noise == "":
                    st.markdown(
                        '<div class="result-warn">Please enter a noise before running the diagnosis.</div>',
                        unsafe_allow_html=True
                    )

                elif matched_noise is None:
                    st.markdown(
                        f'<div class="result-warn">{noise_message}</div>',
                        unsafe_allow_html=True
                    )
                    st.write("Known sounds:")
                    st.write(", ".join(valid_noises))

                elif vibration not in valid_vibration:
                    st.markdown(
                        '<div class="result-warn">Vibration must be yes or no.</div>',
                        unsafe_allow_html=True
                    )

                elif smoke not in valid_smoke:
                    st.markdown(
                        '<div class="result-warn">Smoke must be yes or no.</div>',
                        unsafe_allow_html=True
                    )

                elif when not in valid_when:
                    st.markdown(
                        '<div class="result-warn">Invalid value for when it happens.</div>',
                        unsafe_allow_html=True
                    )

                elif location not in valid_location:
                    st.markdown(
                        '<div class="result-warn">Invalid location.</div>',
                        unsafe_allow_html=True
                    )

                elif cel not in valid_cel:
                    st.markdown(
                        '<div class="result-warn">Check engine light must be yes or no.</div>',
                        unsafe_allow_html=True
                    )

                else:
                    if noise_message:
                        st.info(noise_message)

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

                    if top_prob < 0.35:
                        st.markdown(
                            '<div class="result-warn">Low confidence result. Try adding more detail or testing more symptoms.</div>',
                            unsafe_allow_html=True
                        )
                    elif abs(top_prob - second_prob) < 0.05:
                        st.markdown(
                            f'<div class="result-warn">The result is uncertain between <b>{top_problem}</b> ({top_prob * 100:.2f}%) and <b>{second_problem}</b> ({second_prob * 100:.2f}%).</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="result-good">Most likely issue: <b>{top_problem}</b> ({top_prob * 100:.2f}%)</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("### Top Predictions")
                    for problem, prob in results[:3]:
                        st.markdown(
                            f'<div class="prediction-row"><b>{problem}</b> — {prob * 100:.2f}%</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        '<div class="small-note">These probabilities reflect patterns in your dataset, not guaranteed real-world accuracy.</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.write("No diagnosis yet. Enter symptoms on the left and click **Analyze Symptoms**.")

except Exception as e:
    st.error(f"Error: {e}")
