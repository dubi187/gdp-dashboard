import streamlit as st
import pandas as pd
import re
import unicodedata

"""
Streamlit app: Keyword Dictionary Classifier

Updates (2025-07-28, rev 2)
---------------------------
* **Whole-word/phrase matching** with robust Unicode-normalized regex (no accidental substring hits).
* Added **personalized_service_product** category (keywords: *custom*, *monogram*).
* Re-used Python-style list for **Label** column (`['exclusive', 'limited']`).
* Helper `normalize()` now removes accents, unifies punctuation & hyphens before matching.
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
    "personalized_service_product": {
        "custom", "monogram",
    },
}

# -------------------------------
# Helper functions
# -------------------------------

def normalize(text: str) -> str:
    """Lower-case, strip accents, unify dash/quote punctuation so regex matches are robust."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[–—-]", " ", text)          # dashes → space
    text = re.sub(r"[’‘`]", "'", text)           # curly quotes → straight
    text = re.sub(r"[!?.,:;()\[\]]", " ", text) # punct → space
    return text.lower()


def _phrase_to_regex(phrase: str) -> str:
    """Convert a keyword phrase into a word-boundary regex pattern supporting flexible whitespace."""
    escaped = re.escape(phrase.lower())
    spaced = re.sub(r"\\\s+", r"\\s+", escaped)  # keep escaped \s+ intact if term already uses it
    spaced = spaced.replace(" ", r"\\s+")           # replace literal spaces with \s+
    return rf"\\b{spaced}\\b"


def find_keywords(text: str, terms: set[str]) -> list[str]:
    """Return sorted list of *terms* present in *text* as whole words/phrases (case-insensitive)."""
    text_norm = normalize(text)
    matches: list[str] = []
    for term in terms:
        pattern = _phrase_to_regex(term)
        if re.search(pattern, text_norm):
            matches.append(term)
    return sorted(matches)


def classify_dataframe(df: pd.DataFrame, text_col: str, dictionaries: dict[str, set[str]]) -> pd.DataFrame:
    """Return *df* with added classification columns based on *dictionaries*.

    * **Label**: list-style string of all matches, e.g. `['exclusive', 'limited']`
    * One column per dictionary category: semicolon-separated matches for that category
    """
    df_result = df.copy()
    df_result["Label"] = "[]"  # default list representation

    # Ensure all category columns exist
    for cat in dictionaries:
        if cat not in df_result.columns:
            df_result[cat] = ""

    for idx, text in df_result[text_col].items():
        all_matches: set[str] = set()
        for cat, terms in dictionaries.items():
            kw_matches = find_keywords(str(text), terms)
            df_result.at[idx, cat] = ";".join(kw_matches)
            all_matches.update(kw_matches)
        df_result.at[idx, "Label"] = str(sorted(all_matches))
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
        """Upload a CSV, choose the text column, tweak your dictionaries, and download a
        classified version. Matching is **whole-word**, Unicode-normalized (e.g. accented
        characters & dashes handled).  
        **Label** column lists all matched keywords per row, keeping Python list syntax.
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


