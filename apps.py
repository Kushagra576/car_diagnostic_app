import difflib
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.title("🚗 Car Diagnostic Tool")


@st.cache_data
def load_data():
    data = pd.read_csv("bigger_car_diagnostics.csv")

    data["Noise"] = data["Noise"].str.lower().str.strip()
    data["Vibration"] = data["Vibration"].str.lower().str.strip()
    data["Smoke"] = data["Smoke"].str.lower().str.strip()
    data["When"] = data["When"].str.lower().str.strip()
    data["Location"] = data["Location"].str.lower().str.strip()
    data["CheckEngineLight"] = data["CheckEngineLight"].str.lower().str.strip()
    data["Problem"] = data["Problem"].str.lower().str.strip()

    return data


@st.cache_resource
def train_model(data):
    X = pd.get_dummies(
        data[["Noise", "Vibration", "Smoke", "When", "Location", "CheckEngineLight"]]
    )
    y = data["Problem"]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns


def match_noise(user_noise, valid_noises):
    if user_noise in valid_noises:
        return user_noise, None

    matches = difflib.get_close_matches(user_noise, valid_noises, n=1, cutoff=0.6)
    if matches:
        return matches[0], f"Using closest match: {matches[0]}"
    return None, None


try:
    data = load_data()
    model, model_columns = train_model(data)

    valid_noises = sorted(data["Noise"].unique())
    valid_vibration = ["yes", "no"]
    valid_smoke = ["yes", "no"]
    valid_when = ["startup", "idle", "driving", "accelerating", "braking", "turning", "decelerating"]
    valid_location = ["front", "rear", "engine bay", "dashboard", "transmission tunnel"]
    valid_cel = ["yes", "no"]

    noise = st.text_input("Noise")
    vibration = st.text_input("Vibration? (yes/no)")
    smoke = st.text_input("Smoke? (yes/no)")
    when = st.text_input("When does it happen? (startup, idle, driving, accelerating, braking, turning, decelerating)")
    location = st.text_input("Where is it coming from? (front, rear, engine bay, dashboard, transmission tunnel)")
    cel = st.text_input("Check engine light? (yes/no)")

    if st.button("Diagnose"):
        noise = noise.lower().strip()
        vibration = vibration.lower().strip()
        smoke = smoke.lower().strip()
        when = when.lower().strip()
        location = location.lower().strip()
        cel = cel.lower().strip()

        matched_noise, noise_message = match_noise(noise, valid_noises)

        if matched_noise is None:
            st.warning(f"Unknown noise. Try one of these: {', '.join(valid_noises)}")
        elif vibration not in valid_vibration:
            st.warning("Vibration must be yes or no.")
        elif smoke not in valid_smoke:
            st.warning("Smoke must be yes or no.")
        elif when not in valid_when:
            st.warning("Invalid 'when' value.")
        elif location not in valid_location:
            st.warning("Invalid location.")
        elif cel not in valid_cel:
            st.warning("Check engine light must be yes or no.")
        else:
            if noise_message:
                st.info(noise_message)

            new_data = pd.DataFrame(
                [[matched_noise, vibration, smoke, when, location, cel]],
                columns=["Noise", "Vibration", "Smoke", "When", "Location", "CheckEngineLight"]
            )

            new_data = pd.get_dummies(new_data).reindex(columns=model_columns, fill_value=0)

            probs = model.predict_proba(new_data)
            confidence_score = max(probs[0]) * 100
            prediction = model.predict(new_data)[0]

            st.success(f"Predicted Issue: {prediction}")
            st.write(f"Model confidence score: {confidence_score:.2f}%")
            st.caption("This score reflects how sure the model is on this dataset, not real-world accuracy.")

except Exception as e:
    st.error(f"Error: {e}")