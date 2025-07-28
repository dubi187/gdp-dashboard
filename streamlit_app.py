# dictionary_classification_streamlit_app.py
# -----------------------------------------------------------------------------
# Streamlit app that classifies marketing statements in an uploaded CSV file
# based on configurable keyword dictionaries. Users can edit the dictionaries
# directly in the UI (JSON format) and download the result as a new CSV.
# -----------------------------------------------------------------------------

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------ DEFAULT CONFIGURATION -------------------------------

DEFAULT_DICTIONARIES = {
    "urgency_marketing": {
        "limited",
        "limited time",
        "limited run",
        "limited edition",
        "order now",
        "last chance",
        "hurry",
        "while supplies last",
        "before they're gone",
        "selling out",
        "selling fast",
        "act now",
        "don't wait",
        "today only",
        "expires soon",
        "final hours",
        "almost gone",
    },
    "exclusive_marketing": {
        "exclusive",
        "exclusively",
        "exclusive offer",
        "exclusive deal",
        "members only",
        "vip",
        "special access",
        "invitation only",
        "premium",
        "privileged",
        "limited access",
        "select customers",
        "insider",
        "private sale",
        "early access",
    },
}

# ----------------------------- HELPER FUNCTIONS ------------------------------


def classify_statement(text: str, dictionaries: dict[str, set[str]]) -> str:
    """Return a semicolon‑separated list of dictionary labels found in *text*.
    If no keywords match, returns "none"."""
    text_lower = str(text).lower()
    matches = [
        label
        for label, terms in dictionaries.items()
        if any(term in text_lower for term in terms)
    ]
    return ";".join(matches) if matches else "none"


@st.cache_data(show_spinner=False)
def parse_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    """Read uploaded CSV into a DataFrame, return None on failure."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:  # pylint: disable=broad-except
        st.error(f"❌ Failed to read CSV: {e}")
        return None


# ------------------------------ STREAMLIT UI ---------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Dictionary‑Based Statement Classifier",
        page_icon="📑",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("📑 Dictionary‑Based Statement Classifier")
    st.markdown(
        """
        Upload a CSV file that contains a **`Statement`** column. The app will
        classify each statement based on the keyword dictionaries you provide
        (or edit below) and let you download the augmented file.
        """
    )

    # --------------------------- Upload Section -----------------------------

    uploaded_file = st.file_uploader(
        "1️⃣ Upload your CSV file", type=["csv"], accept_multiple_files=False
    )

    # ---------------------- Dictionary Editor Section -----------------------

    with st.expander("2️⃣ Edit keyword dictionaries (JSON)", expanded=False):
        st.markdown(
            "Each **key** becomes a category label. The **value** for each key is a list of keyword strings.\n"
            "If a keyword appears anywhere in a statement (case‑insensitive), the corresponding label is assigned."
        )

        default_json = json.dumps({k: sorted(v) for k, v in DEFAULT_DICTIONARIES.items()}, indent=2)
        dict_text = st.text_area("Dictionaries JSON", value=default_json, height=300, key="dict_area")

        # Attempt to parse JSON input
        try:
            user_dict_raw = json.loads(dict_text)
            # Convert lists back to sets for faster lookup
            user_dict: dict[str, set[str]] = {
                label: set(map(str.lower, terms)) for label, terms in user_dict_raw.items()
            }
            dict_is_valid = True
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON: {e.msg}")
            dict_is_valid = False
            user_dict = {}

    # ------------------------- Classification Button ------------------------

    run_button_disabled = not (uploaded_file and dict_is_valid)
    if st.button("3️⃣ Run Classification", disabled=run_button_disabled, use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a CSV file first.")
            st.stop()

        df = parse_uploaded_csv(uploaded_file)
        if df is None:
            st.stop()

        if "Statement" not in df.columns:
            st.error("❌ The uploaded CSV must contain a 'Statement' column.")
            st.stop()

        # Perform classification
        with st.spinner("Classifying statements..."):
            df["Category"] = df["Statement"].apply(lambda s: classify_statement(s, user_dict))

        st.success("🎉 Classification complete!")

        # Show preview
        st.subheader("Preview of classified data")
        st.dataframe(df.head(50), use_container_width=True)

        # Prepare download
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue().encode("utf‑8")
        st.download_button(
            "⬇️ Download full classified CSV",
            data=csv_content,
            file_name="classified_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Optionally cache full result so user can explore further without rerun
        st.session_state["last_result_df"] = df

    # ---------------------- Optional Result Exploration ---------------------

    if "last_result_df" in st.session_state:
        with st.expander("🔍 Explore previous results", expanded=False):
            result_df: pd.DataFrame = st.session_state["last_result_df"]
            st.dataframe(result_df, use_container_width=True)

            # Simple category count visualization
            category_counts = result_df["Category"].value_counts().sort_values(ascending=False)
            st.bar_chart(category_counts)


if __name__ == "__main__":
    main()

