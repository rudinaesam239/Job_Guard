# JobGuard – Fake Job Ad Detector (Flask + scikit-learn)
Run locally: `pip install -r requirements.txt && python app.py` → http://127.0.0.1:5000
Docker: `docker build -t jobguard . && docker run -p 80:8000 -v jg_data:/app/data jobguard` (the volume keeps user feedback)
Cloud (AWS EC2 / Azure VM / GCP VM): install Docker, copy this folder (git clone or scp), run the two Docker commands above, open port 80 in the security group / NSG / firewall. Health check: `/health`.
Keep scikit-learn at 1.8.0 (the version used to train the model).

## What changed in v3.1
- **EN / AR toggle** (top corner): switches the whole UI between Arabic (RTL) and English (LTR) instantly, no reload. The choice is remembered in the browser.
- Red flags, scam types and "what to do now" advice are returned in both languages by the API (`red_flags_en`, `name_en` / `about_en` / `advice_en`, `flags_en`).
- Downloaded CSV: the `red_flags` column follows the selected language (`lang=en` form field; default Arabic).

## What changed in v3
- **New Arabic RTL neon UI**: radar header, animated gauge, glass cards.
- **What-if simulator** (`/api/whatif`): flip logo / company profile / screening questions and watch the score move live; shows the model's weak point.
- **"Is this verdict wrong?"** (`/api/feedback`): saves the ad + correct label to `data/feedback.jsonl`. `python export_feedback.py` turns it into a training CSV.
- **Scam type** (fee, Gulf travel, money mule, reshipping, data theft, lure) with "what to do now" advice.
- **Egyptian rules**: Vodafone Cash / InstaPay / Fawry, Gulf travel + visa fees, upfront commission, paying before hiring.

## What changed in v2
- **Check a CSV** now shows results on screen (summary + table with verdict, fake %, red flags) and has a Download button.
- **Hybrid verdict**: model score + rule-based red flags. Four outcomes: SUSPICIOUS, CAUTION, LIKELY REAL, UNSUPPORTED LANGUAGE.
- **Non-English ads** (e.g. Arabic) are no longer called "LIKELY REAL" by mistake: the model reads English only, so rules decide.
- **New red-flag rules**: bank details/ID requests, money forwarding, package reshipping, no-interview hiring, Arabic scam phrases. "Direct deposit" no longer counts as a fee.
- Example buttons (scam / real) on the Check an ad tab.
- Downloaded CSV includes a `red_flags` column and is protected against Excel formula injection.
- `debug` mode is off unless `DEBUG=1`.

## Known limitation
The ML model leans heavily on logo / company profile / screening-question fields (adding them dropped a scam ad from 41% to under 1%), which is why the rule flags can override it.
