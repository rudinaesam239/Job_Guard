import io, os, re, time, json, threading, html as _html
from collections import defaultdict, deque
import joblib, numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from job_features import build_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
BUNDLE = joblib.load("fake_job_model.joblib")
MODEL, THR = BUNDLE["model"], BUNDLE["threshold"]
PREP, CLF = MODEL.named_steps["prep"], MODEL.named_steps["clf"]
FEATS = PREP.get_feature_names_out()
MAX_LEN, HITS = 20000, defaultdict(deque)
DATA_DIR = os.environ.get("DATA_DIR", "data"); os.makedirs(DATA_DIR, exist_ok=True)
FB_FILE, LOCK = os.path.join(DATA_DIR, "feedback.jsonl"), threading.Lock()
PROFILE = "Acme Technologies is a 15-year-old software company with 400 employees, building payment infrastructure for banks."
FIELDS = ["title", "location", "company_profile", "description", "requirements", "benefits",
          "employment_type", "required_experience", "required_education", "industry", "function", "salary_range"]

# real emails/phones/links -> the same masked tokens the model saw in training
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)\S+")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
def mask(s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = _html.unescape(s)
    return PHONE.sub(" #PHONE_0# ", URL.sub(" #URL_0# ", EMAIL.sub(" #EMAIL_0# ", s)))

FLAGS = [  # (label, regex, scam type, strong?) -> strong = enough alone to call the ad suspicious
    ("بيطلب فلوس أو رسوم", r"registration fee|training fee|processing fee|(?<!direct )deposit|pay (?:a|the) fee|send (?:us )?money|gift card|رسوم|ادفع|دفع مبلغ|تحويل مبلغ", "fee", False),
    ("دفع عن طريق فودافون كاش / انستاباي / فوري", r"vodafone ?cash|فودافون ?كاش|insta ?pay|[اإ]نستا ?باي|فوري|orange ?cash|[اأ]ورنج ?كاش|etisalat ?cash|اتصالات ?كاش|محفظة (?:[اإ]لكترونية|الكترونية)", "fee", True),
    ("عمولة أو مبلغ تأمين مقدّم", r"عمولة مقدم|دفع عمولة|مبلغ (?:تأمين|ضمان)|تأمين (?:العمل|مقدم)|رسوم عقد", "fee", True),
    ("تحويل فلوس قبل التعيين", r"قبل (?:ال)?(?:تعيين|استلام العمل|التوقيع)|(?:حوّل|حول)\s+(?:لنا|مبلغ|فلوس)|ابعت(?:ي)? (?:لنا )?(?:فلوس|مبلغ)", "fee", True),
    ("سفر للخليج / رسوم تأشيرة", r"تأ?شير[ةه]|فيز[اه]|(?:سفر|عقد عمل|شغل)[^.\n]{0,25}(?:خليج|الخليج|السعودية|الإمارات|الامارات|الكويت|قطر|دبي)|work visa|visa (?:fee|processing)", "gulf", False),
    ("شيك أو تحويل أموال (Money Mule)", r"wire transfer|western union|send you a (?:check|cheque)|cash (?:a|the) (?:check|cheque)|forward (?:the |your )?(?:payments?|funds?)|transfer (?:the )?(?:remaining|funds?)|keep \d+% commission|استلم (?:التحويلات|الأموال)|حساب(?:ك)? البنكي لاستقبال", "mule", False),
    ("بيطلب بيانات شخصية أو بنكية", r"bank (?:details|account)|copy of your id|passport|social security|national id|الرقم القومي|صورة (?:البطاقة|الباسبور|جواز)|بيانات (?:البنك|البطاقة)|كلمة (?:السر|المرور)|\bOTP\b|كود التفعيل", "data", False),
    ("استلام وإعادة شحن طرود", r"re-?ship|receive packages|forward packages|استلام (?:طرود|شحنات)|إعادة (?:شحن|إرسال) (?:الطرود|الشحنات)", "reship", False),
    ("إيميل شخصي (Gmail/Yahoo/Hotmail)", r"@(?:gmail|yahoo|hotmail|outlook|aol)\.", "lure", False),
    ("نقل المحادثة لواتساب/تليجرام", r"whatsapp|telegram|signal app|واتس ?اب|تيليجرام|تليجرام", "lure", False),
    ("وعد بدخل خيالي", r"earn \$?\d{3,}.{0,20}(?:per|a) (?:day|week|hour)|\$\d{4,}\s*(?:/|per)\s*(?:week|day)|no experience (?:needed|required).{0,40}\$|بدون خبرة|اربح|دخل غير محدود|ربح مضمون", "lure", False),
    ("تعيين فوري بدون مقابلة", r"no interview|without (?:an? )?interview|immediate start|hired immediately|بدون مقابلة|تعيين فوري", "lure", False),
    ("ضغط واستعجال", r"apply (?:now|immediately|today)|limited (?:spots|positions)|urgent(?:ly)? hiring|act fast|hurry|الحق نفسك|عدد محدود|فرصة (?:أخيرة|اخيرة)", "lure", False),
]
TYPES = {  # id -> (name, what it is, what to do now)
    "fee": ("رسوم مقدمة", "الشركة بتاخد منك فلوس قبل ما تشتغل (رسوم تسجيل، تدريب، تأمين، عمولة). الشركات الحقيقية مبتطلبش فلوس من المتقدم.",
            ["متدفعش ولا جنيه ومتحولش أي مبلغ، حتى لو صغير.", "لو دفعت: كلّم فودافون كاش/البنك فورًا واطلب إيقاف التحويل.", "احتفظ بسكرين شوت للمحادثة وبلّغ المباحث الإلكترونية (الإنترنت) أو النيابة."]),
    "gulf": ("نصب سفر للخليج", "وعد بعقد شغل في الخليج مقابل رسوم تأشيرة أو سفر. التأشيرة الحقيقية بتجيلك من الشركة مش بتدفعلها.",
             ["اتأكد إن الوكالة مرخّصة من وزارة القوى العاملة (الإدارة العامة للتشغيل).", "اطلب عقد موثّق من الشركة نفسها وتأكد منه من السفارة.", "متدفعش رسوم تأشيرة لأي سمسار قبل ما تتأكد من الشركة."]),
    "mule": ("شيك مزيف / تحويل أموال (Money Mule)", "بيبعتولك فلوس أو شيك وبيطلبوا منك تحوّل الباقي. الشيك بيطلع مزيف وتتحمل الخسارة، وممكن تتورط قانونيًا.",
             ["متستلمش ولا تحوّل أي فلوس لحساب حد متعرفوش.", "غسل الأموال جريمة: وقّف التعامل فورًا ومتفتحش حسابك لحد.", "بلّغ البنك والجهات المختصة واحتفظ بالمراسلات."]),
    "reship": ("إعادة شحن طرود", "بيطلبوا منك تستلم طرود في بيتك وتبعتها لعنوان تاني. غالبًا الطرود مشتراة بكروت مسروقة.",
               ["متستلمش ولا تبعت طرود لحساب حد.", "لو استلمت طرد: متبعتوش، واتواصل مع شركة الشحن.", "بلّغ عن الإعلان للمنصة اللي شفته عليها."]),
    "data": ("سرقة بيانات", "الهدف إنهم ياخدوا بطاقتك أو باسبورك أو بيانات البنك أو كود OTP لاستخدامها في سرقة هويتك أو فلوسك.",
             ["متبعتش صورة البطاقة أو الباسبور قبل ما تتأكد إن الشركة حقيقية.", "عمرك ما تدّي كود OTP أو كلمة السر لأي حد.", "لو بعت بيانات: غيّر الباسورد وكلّم البنك وراقب حساباتك."]),
    "lure": ("عرض براق وضغط نفسي", "وعود بدخل كبير من غير خبرة أو مقابلة، وتحويل المحادثة لواتساب، واستعجال. دي أساليب لإنك متفكرش.",
             ["اتأكد من الشركة: موقع رسمي، صفحة لينكدإن، عنوان حقيقي.", "اطلب مقابلة رسمية، والشركات الحقيقية مش بتستعجل.", "ابحث عن اسم الشركة + كلمة «نصب» قبل ما تتواصل."]),
}

EN_FLAGS = [  # English labels, same order as FLAGS
    "Asks for money or fees",
    "Payment via Vodafone Cash / InstaPay / Fawry",
    "Upfront commission or security deposit",
    "Sending money before being hired",
    "Gulf travel / visa fees",
    "Check or money transfer (money mule)",
    "Asks for personal or banking details",
    "Receiving and reshipping packages",
    "Personal email (Gmail/Yahoo/Hotmail)",
    "Moves the chat to WhatsApp/Telegram",
    "Promises unrealistic income",
    "Instant hiring without an interview",
    "Pressure and urgency",
]
assert len(EN_FLAGS) == len(FLAGS)
EN_FLAG = {f[0]: e for f, e in zip(FLAGS, EN_FLAGS)}
TYPES_EN = {  # id -> (name, what it is, what to do now)
    "fee": ("Upfront fees", "The company takes money from you before you start working (registration, training, deposit, commission). Real employers never charge applicants.",
            ["Don't pay a single pound or transfer any amount, even a small one.", "If you already paid: contact Vodafone Cash or your bank immediately and ask them to stop the transfer.", "Keep screenshots of the conversation and report it to the Egyptian cybercrime police (Internet crimes unit) or the public prosecutor."]),
    "gulf": ("Gulf travel scam", "A promise of a job contract in the Gulf in exchange for visa or travel fees. A real visa is arranged by the employer, not paid for by you.",
             ["Check that the agency is licensed by the Ministry of Manpower (General Administration of Employment).", "Ask for an attested contract from the company itself and verify it with the embassy.", "Don't pay visa fees to any broker before you have verified the company."]),
    "mule": ("Fake check / money transfer (money mule)", "They send you money or a check and ask you to transfer the rest. The check turns out to be fake, you bear the loss, and you may get into legal trouble.",
             ["Don't receive or transfer money for anyone you don't know.", "Money laundering is a crime: stop all contact immediately and don't let anyone use your account.", "Report it to your bank and the authorities, and keep all messages."]),
    "reship": ("Package reshipping", "They ask you to receive parcels at your home and forward them to another address. The parcels are usually bought with stolen cards.",
               ["Don't receive or forward packages for anyone.", "If you already received a package: don't forward it, and contact the shipping company.", "Report the ad to the platform where you saw it."]),
    "data": ("Data theft", "The goal is to get your ID card, passport, bank details or OTP code to steal your identity or your money.",
             ["Don't send a photo of your ID or passport until you have verified the company is real.", "Never give anyone an OTP code or password.", "If you already sent your details: change your passwords, call your bank, and monitor your accounts."]),
    "lure": ("Flashy offer and psychological pressure", "Promises of big income with no experience or interview, moving the chat to WhatsApp, and rushing you. These tactics exist so that you don't stop and think.",
             ["Verify the company: official website, LinkedIn page, a real address.", "Ask for a formal interview. Real companies don't rush you.", "Search the company name plus the word \u201cscam\u201d before contacting them."]),
}
assert set(TYPES_EN) == set(TYPES)

def en_flags(fl):
    return [EN_FLAG[n] for n in fl]

def hits_for(raw):
    return [(n, t, st) for n, rx, t, st in FLAGS if re.search(rx, raw, re.I)]

def types_for(raw):
    cnt = {}
    for _, t, _ in hits_for(raw): cnt[t] = cnt.get(t, 0) + 1
    return [dict(id=t, name=TYPES[t][0], about=TYPES[t][1], advice=TYPES[t][2],
                 name_en=TYPES_EN[t][0], about_en=TYPES_EN[t][1], advice_en=TYPES_EN[t][2], hits=c)
            for t, c in sorted(cnt.items(), key=lambda x: -x[1])]

def latin_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 1.0

def judge(s, prob, raw):
    """Combine model score + rule flags. The model only reads English, so other languages fall back to rules."""
    h = hits_for(raw); fl = [n for n, _, _ in h]; strong = any(st for _, _, st in h)
    if latin_ratio(raw) < 0.5:
        return ("SUSPICIOUS" if len(fl) >= 2 or strong else "UNSUPPORTED LANGUAGE"), fl
    if s >= THR or len(fl) >= 2 or strong: return "SUSPICIOUS", fl
    if fl or prob >= 0.30: return "CAUTION", fl
    return "LIKELY REAL", fl

def score_rows(df):
    for c in FIELDS:
        if c not in df: df[c] = None
        df[c] = df[c].map(lambda v: None if pd.isna(v) or v == "" else str(v)[:MAX_LEN])
    for c in ["telecommuting", "has_company_logo", "has_questions"]:
        if c not in df: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    raws = df[["title", "company_profile", "description", "requirements", "benefits"]].fillna("").agg(" ".join, axis=1).tolist()
    for c in ["title", "company_profile", "description", "requirements", "benefits"]:
        df[c + "_raw"] = df[c]; df[c] = df[c].map(mask)
    X = build_features(df)
    s = MODEL.decision_function(X)
    prob = 1 / (1 + np.exp(-s))
    return X, s, prob, raws

def explain(X, k=6):
    v = PREP.transform(X.iloc[[0]]).multiply(CLF.coef_[0]).tocsr()
    keep = [i for i, n in zip(v.indices, v.data) if FEATS[i].startswith(("title__", "body__"))]
    idx = np.array(keep, dtype=int); vals = np.array([v[0, i] for i in keep])
    pretty = lambda n: re.sub(r"^(title|body|cat|num)__", "", n).replace("_", " ")
    order = np.argsort(vals)
    up = [(pretty(FEATS[idx[i]]), float(vals[i])) for i in order[::-1] if vals[i] > 0][:k]
    down = [(pretty(FEATS[idx[i]]), float(vals[i])) for i in order if vals[i] < 0][:k]
    return [{"name": n, "weight": round(w, 2)} for n, w in up], [{"name": n, "weight": round(w, 2)} for n, w in down]

def limited():
    ip, now = request.headers.get("X-Forwarded-For", request.remote_addr), time.time()
    q = HITS[ip]
    while q and now - q[0] > 60: q.popleft()
    q.append(now)
    return len(q) > 30

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return {"status": "ok", "model": BUNDLE["name"], "sklearn": BUNDLE["sklearn_version"]}

@app.post("/api/predict")
def predict():
    if limited(): return jsonify(error="Too many requests. Wait a minute and try again."), 429
    d = request.get_json(silent=True) or {}
    if len(str(d.get("description", "")).strip()) < 30:
        return jsonify(error="Paste the job description (at least a few sentences)."), 400
    raw = " ".join(str(d.get(k, "")) for k in ["title", "company_profile", "description", "requirements", "benefits"])
    X, s, prob, _ = score_rows(pd.DataFrame([{k: d.get(k) for k in FIELDS + ["telecommuting", "has_company_logo", "has_questions"]}]))
    up, down = explain(X)
    verdict, fl = judge(s[0], prob[0], raw)
    return jsonify(verdict=verdict, score=round(float(s[0]), 2),
                   probability=round(float(prob[0]), 3), threshold=round(THR, 2),
                   red_flags=fl, red_flags_en=en_flags(fl), types=types_for(raw), raises=up, lowers=down)

@app.post("/api/whatif")
def whatif():
    """What-if simulator: same text, flip logo / company profile / screening questions and see the score move."""
    if limited(): return jsonify(error="Too many requests."), 429
    d = request.get_json(silent=True) or {}
    desc = str(d.get("description", "")).strip()
    if len(desc) < 30: return jsonify(error="Description is too short."), 400
    keys = ("has_company_logo", "has_profile", "has_questions")
    cur = {k: int(bool(d.get(k))) for k in keys}
    states = [cur] + [{**cur, k: 1 - cur[k]} for k in keys] + [dict.fromkeys(keys, 1), dict.fromkeys(keys, 0)]
    rows = [{"title": d.get("title"), "description": desc, "company_profile": PROFILE if c["has_profile"] else None,
             "has_company_logo": c["has_company_logo"], "has_questions": c["has_questions"]} for c in states]
    _, s, prob, _ = score_rows(pd.DataFrame(rows))
    raw = f"{d.get('title', '')} {desc}"
    verdict, fl = judge(s[0], prob[0], raw)
    p = lambda i: round(float(prob[i]), 3)
    return jsonify(verdict=verdict, probability=p(0), red_flags=fl, red_flags_en=en_flags(fl), types=types_for(raw),
                   impact={k: p(i + 1) for i, k in enumerate(keys)}, all_on=p(4), all_off=p(5))

@app.post("/api/feedback")
def feedback():
    """'Is this verdict wrong?' -> store the ad with the user's correct label (training-ready JSONL)."""
    if limited(): return jsonify(error="Too many requests."), 429
    d = request.get_json(silent=True) or {}
    if d.get("label") not in ("fake", "real"): return jsonify(error="label must be fake or real"), 400
    if len(str(d.get("description", "")).strip()) < 30: return jsonify(error="Missing ad text."), 400
    rec = {"ts": int(time.time()), "label": d["label"], "model_verdict": str(d.get("verdict", ""))[:30],
           "probability": d.get("probability") if isinstance(d.get("probability"), (int, float)) else None}
    rec.update({k: str(d.get(k) or "")[:MAX_LEN] for k in FIELDS})
    rec.update({k: int(bool(d.get(k))) for k in ("telecommuting", "has_company_logo", "has_questions")})
    with LOCK:
        with open(FB_FILE, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total = sum(1 for _ in open(FB_FILE, encoding="utf-8"))
    return jsonify(ok=True, total=total)

@app.post("/api/batch")
def batch():
    if limited(): return jsonify(error="Too many requests."), 429
    f = request.files.get("file")
    if not f: return jsonify(error="Upload a CSV file."), 400
    try: df = pd.read_csv(f).head(2000)
    except Exception: return jsonify(error="Could not read the CSV."), 400
    if "description" not in df: return jsonify(error="CSV needs at least a 'description' column."), 400
    _, s, prob, raws = score_rows(df.copy())
    res = [judge(x, p, r) for x, p, r in zip(s, prob, raws)]
    verdicts = [v for v, _ in res]
    if request.form.get("download"):
        out = df.copy()
        out["score"], out["fake_probability"] = s.round(3), prob.round(3)
        en = request.form.get("lang") == "en"  # language of the red_flags column (default Arabic)
        out["verdict"], out["red_flags"] = verdicts, ["; ".join(en_flags(f) if en else f) for _, f in res]
        safe = lambda v: "'" + v if isinstance(v, str) and v[:1] in "=+-@" else v  # stop Excel formula injection
        out = out.apply(lambda col: col.map(safe) if col.dtype == object else col)
        buf = io.BytesIO(out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")); buf.seek(0)
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="jobguard_results.csv")
    titles = df["title"].fillna("").astype(str) if "title" in df else df["description"].astype(str).str[:40]
    rows = [dict(n=i + 1, title=titles.iloc[i][:60], verdict=v, probability=round(float(prob[i]), 3), flags=f, flags_en=en_flags(f))
            for i, (v, f) in enumerate(res)]
    cnt = {k: verdicts.count(k) for k in set(verdicts)}
    return jsonify(total=len(rows), counts=cnt, rows=rows)

if __name__ == "__main__":
    app.run(debug=os.environ.get("DEBUG") == "1")
