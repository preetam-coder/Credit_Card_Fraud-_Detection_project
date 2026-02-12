
import os

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


def find_model_files(directory: str):
    """Return a list of possible model files in the given directory."""
    candidates = []
    for name in os.listdir(directory):
        lower = name.lower()
        if lower.endswith((".pkl", ".joblib")):
            candidates.append(name)
    return sorted(candidates)


@st.cache_resource
def load_model(path: str):
    """Load a trained model (or pipeline) from disk."""
    full_path = os.path.abspath(path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")
    return joblib.load(full_path)


def main():
    st.title("Credit Card Fraud Detection")
    st.markdown(
        "Fill in the transaction details below and this app will use your "
        "**trained model** to predict whether the transaction is "
        "**fraudulent** or **legitimate**."
    )

    cwd = os.getcwd()
    model_files = find_model_files(cwd)

    if not model_files:
        st.error(
            "No model files found in this folder.\n\n"
            "Please place your trained model (for example `fraud_model.pkl` or "
            "`pipeline.joblib`) in the same directory as `app.py` and refresh "
            "the page."
        )
        return

    st.sidebar.header("Model selection")
    selected_model = st.sidebar.selectbox(
        "Choose a model file",
        model_files,
        help="These are all `.pkl` / `.joblib` files found next to `app.py`.",
    )

    try:
        model = load_model(selected_model)
        st.sidebar.success(f"Loaded model: {selected_model}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model `{selected_model}`:\n\n{e}")
        return

    st.sidebar.header("Transaction input")

    # Let the user choose between single-transaction form and batch CSV upload
    mode = st.sidebar.radio(
        "Prediction mode",
        ["Single transaction (manual)", "Batch from CSV upload"],
    )

    # Features from your notebook: all columns except 'Class'
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    if mode == "Single transaction (manual)":
        st.subheader("Enter transaction details")
        st.write(
            "These fields correspond to the columns used when training the model.\n\n"
            "- **Time**: seconds elapsed between this transaction and the first transaction in the dataset.\n"
            "- **Amount**: transaction amount.\n"
            "- **V1–V28**: anonymized numeric features created using PCA on the original card and transaction data. "
            "They do **not** directly correspond to things like merchant, location, or card type; they are internal "
            "components the model uses for its decision."
        )

        # Use a form so all inputs are submitted together
        with st.form("transaction_form"):
            cols = st.columns(3)
            values = {}

            for idx, feat in enumerate(feature_names):
                col = cols[idx % 3]
                label = feat

                if feat.lower() == "time":
                    values[feat] = col.number_input(
                        "Time (seconds since first transaction)",
                        value=0.0,
                        step=1.0,
                    )
                elif feat.lower() == "amount":
                    values[feat] = col.number_input(
                        "Amount",
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                    )
                else:
                    # V1–V28 anonymized PCA features
                    values[feat] = col.number_input(
                        label,
                        value=0.0,
                        step=0.1,
                        help=(
                            f"{label} is an anonymized numeric feature derived from the original "
                            "transaction details using PCA. It does not have a direct, human-readable meaning "
                            "like 'merchant' or 'location', but the model uses it as an internal signal."
                        ),
                    )

            submitted = st.form_submit_button("Predict fraud")

        if submitted:
            try:
                # Build a single-row DataFrame with the same columns as training
                input_df = pd.DataFrame([values], columns=feature_names)

                st.subheader("Input summary")
                st.dataframe(input_df)

                # Because you trained a Pipeline (StandardScaler + LogisticRegression),
                # we can pass the raw values directly.
                pred = model.predict(input_df)[0]

                prob = None
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_df)[0][1]

                st.subheader("Prediction")
                if int(pred) == 1:
                    if prob is not None:
                        st.error(
                            f"🚨 The transaction is LIKELY FRAUDULENT "
                            f"(fraud probability: {prob:.2%})."
                        )
                    else:
                        st.error("🚨 The transaction is LIKELY FRAUDULENT.")
                else:
                    if prob is not None:
                        st.success(
                            f"✅ The transaction is NOT fraudulent "
                            f"(fraud probability: {prob:.2%})."
                        )
                    else:
                        st.success("✅ The transaction is NOT fraudulent.")

            except Exception as e:
                st.error(f"Error during prediction:\n\n{e}")

    else:
        # Batch prediction mode: user uploads a CSV of transactions
        st.subheader("Batch prediction from CSV")
        st.write(
            "Upload a CSV file containing one or more transactions.\n\n"
            "- The file should include at least the columns: **Time, V1–V28, Amount**.\n"
            "- If a **Class** column is present, it will be kept so you can compare "
            "true labels with the model predictions."
        )

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help=(
                "The CSV should have the same columns as the data used for training "
                "(Time, V1–V28, Amount, and optionally Class)."
            ),
        )

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read the uploaded CSV file:\n\n{e}")
                return

            if data.empty:
                st.warning(
                    "The uploaded CSV is empty. Please upload a file with at least one row."
                )
                return

            st.subheader("Uploaded data (first 10 rows)")
            st.dataframe(data.head(10), use_container_width=True)

            missing = [c for c in feature_names if c not in data.columns]
            if missing:
                st.error(
                    "The following required feature columns are missing from the CSV:\n\n"
                    + ", ".join(missing)
                )
                return

            if st.button("Predict fraud for dataset"):
                try:
                    X_batch = data[feature_names]
                    preds = model.predict(X_batch)

                    probas = None
                    if hasattr(model, "predict_proba"):
                        try:
                            proba_all = model.predict_proba(X_batch)
                            probas = proba_all[:, 1]  # probability of fraud class
                        except Exception:
                            probas = None

                    result_df = data.copy()
                    result_df["is_fraud_prediction"] = preds
                    if probas is not None:
                        result_df["fraud_probability"] = probas

                    st.subheader("Prediction results (first 50 rows)")
                    st.dataframe(result_df.head(50), use_container_width=True)

                    n_rows = len(result_df)
                    fraud_count = int((result_df["is_fraud_prediction"] == 1).sum())
                    non_fraud_count = n_rows - fraud_count

                    st.markdown(
                        f"- **Total transactions scored**: {n_rows}\n"
                        f"- **Predicted fraudulent**: {fraud_count}\n"
                        f"- **Predicted legitimate**: {non_fraud_count}"
                    )

                except Exception as e:
                    st.error(f"Error while running batch prediction:\n\n{e}")


if __name__ == "__main__":
    main()
