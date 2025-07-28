import streamlit as st
import pandas as pd
from io import BytesIO

"""
Streamlit app: Keyword Dictionary Classifier

Updates (2025-07-28)
-------------------
* **Label column** now stores a *Python-style list* representation of all matched keywords, e.g. `['exclusive', 'limited']`, aligning with the latest standalone script.
* Default value for `Label` is the empty list string `[]` instead of an empty string.
* Minor doc tweaks.
"""

# -------------------------------
# Default keyword dictionaries – extend or modify these as needed
# -------------------------------
DEFAULT_DICTIONARIES: dict[str, set[str]] = {
    "urgency_marketing": {
        "limited", "limited time", "limited run", "limited edition", "order now",
        "last chance", "hurry", "while supplies last", "before they're gone",
        "selling out", "selling fast", "act now", "don't wait", "today only",
        "expires soon", "final hours", "almost gone",
    },
    "exclusive_marketing": {
        "exclusive", "exclusively", "exclusive offer", "exclusive deal", "members only",
        "vip", "special access", "invitation only", "premium", "privileged",
        "limited access", "select customers", "insider", "private sale", "early access",
    },
}

# -------------------------------
# Helper functions
# -------------------------------

def find_keywords(text: str, terms: set[str]) -> list[str]:
    """Return a sorted list of *terms* present in *text* (case-insensitive)."""
    text_lower = str(text).lower()
    return sorted({term for term in terms if term in text_lower})


def classify_dataframe(df: pd.DataFrame, text_col: str, dictionaries: dict[str, set[str]]) -> pd.DataFrame:
    """Return *df* with added classification columns based on *dictionaries*.

    * **Label**: list-style string of all matches, e.g. `['exclusive', 'limited']`
    * One column per dictionary category: semicolon-separated matches for that category
    """
    df_result = df.copy()
    df_result["Label"] = "[]"  # empty list representation by default

    # Ensure all category columns exist before iteration
    for cat in dictionaries:
        if cat not in df_result.columns:
            df_result[cat] = ""

    for idx, text in df_result[text_col].items():
        all_matches: set[str] = set()
        for cat, terms in dictionaries.items():
            matches = find_keywords(text, terms)
            df_result.at[idx, cat] = ";".join(matches)
            all_matches.update(matches)
        df_result.at[idx, "Label"] = str(sorted(all_matches))  # list-style string

    return df_result


def parse_keywords(text: str) -> set[str]:
    """Convert comma- or newline-separated keywords into a set of stripped, lower-cased terms."""
    if not text:
        return set()
    parts = [p.strip().lower() for p in text.replace("\n", ",").split(",")]
    return {p for p in parts if p}


# -------------------------------
# Streamlit app
# -------------------------------

def main() -> None:
    st.set_page_config(page_title="Dictionary Classification", page_icon="🗂️", layout="wide")
    st.title("🗂️ Keyword Dictionary Classifier")

    st.markdown(
        """Upload a CSV file, choose the column to scan, adjust your keyword dictionaries, and download a
        classified copy with match labels.  
        **Label** column now contains a Python-style list of all keywords matched in the row.
        """
    )

    # 1️⃣ Upload section
    uploaded_file = st.file_uploader("**Upload your CSV**", type=["csv"], help="CSV only – no Excel files.")
    if uploaded_file is None:
        st.info("Awaiting CSV upload …")
        st.stop()

    # Load CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to read CSV – {exc}")
        st.stop()

    # 2️⃣ Pick text column
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_cols:
        st.error("No string columns detected in your CSV.")
        st.stop()

    text_col = st.selectbox("Select the text column to classify", text_cols, index=0)

    # 3️⃣ Dictionary editor
    st.subheader("Keyword dictionaries")

    if "dictionaries" not in st.session_state:
        st.session_state.dictionaries = {k: v.copy() for k, v in DEFAULT_DICTIONARIES.items()}

    dictionaries = st.session_state.dictionaries
    updated_dicts: dict[str, set[str]] = {}

    for cat, terms in dictionaries.items():
        with st.expander(f"Category: {cat}"):
            term_text = st.text_area(
                "Comma- or newline-separated keywords", value="\n".join(sorted(terms)), key=f"ta_{cat}"
            )
            updated_dicts[cat] = parse_keywords(term_text)

    # ➕ Add new category
    with st.expander("➕ Add new category"):
        new_cat_name = st.text_input("Category name", key="new_cat_name")
        new_cat_terms = st.text_area("Keywords", key="new_cat_terms")
        if new_cat_name:
            updated_dicts[new_cat_name] = parse_keywords(new_cat_terms)

    # 🚀 Run classification
    if st.button("Run classification"):
        st.session_state.dictionaries = updated_dicts  # persist changes
        result_df = classify_dataframe(df, text_col, st.session_state.dictionaries)
        st.success("Classification complete!")

        # Preview first 20 rows
        st.subheader("Preview (first 20 rows)")
        st.dataframe(result_df.head(20), use_container_width=True)

        # Download full CSV
        csv_buf = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download classified CSV",
            data=csv_buf,
            file_name="classified_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()


