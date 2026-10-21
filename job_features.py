"""Feature engineering for the Fake Job Postings project (shared by the notebook and the Streamlit app)."""
import re
import html
import numpy as np
import pandas as pd

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
BODY_COLS = ["company_profile", "description", "requirements", "benefits"]
CAT_COLS = ["country", "employment_type", "required_experience", "required_education", "industry", "function"]
NUM_COLS = ["telecommuting", "has_company_logo", "has_questions",
            "has_salary", "has_company_profile", "has_requirements", "has_benefits",
            "has_email", "has_phone", "has_url",
            "log_n_exclaim", "caps_ratio",
            "log_len_description", "log_len_requirements", "log_len_company_profile", "log_len_benefits"]

URL_RE = re.compile(r"#URL_[0-9a-fA-F]+#")
EMAIL_RE = re.compile(r"#EMAIL_[0-9a-fA-F]+#")
PHONE_RE = re.compile(r"#PHONE_[0-9a-fA-F]+#")


def clean_text(s):
    """Lower-case, de-HTML and de-mask a text field; returns '' for missing values."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = URL_RE.sub(" urltoken ", s)
    s = EMAIL_RE.sub(" emailtoken ", s)
    s = PHONE_RE.sub(" phonetoken ", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)          # "ManagerLOCATION" -> "Manager LOCATION"
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def _caps_ratio(text):
    letters = sum(ch.isalpha() for ch in text)
    return sum(ch.isupper() for ch in text) / letters if letters else 0.0


def build_features(df):
    """Turn raw posting columns into the engineered feature table used by the models."""
    out = pd.DataFrame(index=df.index)
    raw_all = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)

    out["title_clean"] = df["title"].map(clean_text)
    out["body_clean"] = df[BODY_COLS].fillna("").agg(" ".join, axis=1).map(clean_text)

    out["country"] = df["location"].fillna("Missing").str.split(",").str[0].str.strip().replace("", "Missing")
    for c in ["employment_type", "required_experience", "required_education", "industry", "function"]:
        out[c] = df[c].fillna("Missing")

    for c in ["telecommuting", "has_company_logo", "has_questions"]:
        out[c] = df[c].astype(int)
    out["has_salary"] = df["salary_range"].notna().astype(int)
    out["has_company_profile"] = df["company_profile"].notna().astype(int)
    out["has_requirements"] = df["requirements"].notna().astype(int)
    out["has_benefits"] = df["benefits"].notna().astype(int)

    out["has_email"] = raw_all.str.contains(EMAIL_RE).astype(int)
    out["has_phone"] = raw_all.str.contains(PHONE_RE).astype(int)
    out["has_url"] = raw_all.str.contains(URL_RE).astype(int)
    out["log_n_exclaim"] = np.log1p(raw_all.str.count("!"))
    out["caps_ratio"] = df["description"].fillna("").map(_caps_ratio)
    for c in ["description", "requirements", "company_profile", "benefits"]:
        out[f"log_len_{c}"] = np.log1p(df[c].fillna("").str.len())
    return out
