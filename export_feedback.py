"""Turn data/feedback.jsonl into a training-ready CSV (EMSCAD-style columns + `fraudulent`).
Usage: python export_feedback.py [data/feedback.jsonl] [feedback_train.csv]
Then append it to the original EMSCAD data and re-run the training notebook."""
import sys, pandas as pd
src, dst = (sys.argv + [None, None])[1:3]
df = pd.read_json(src or "data/feedback.jsonl", lines=True)
df["fraudulent"] = (df["label"] == "fake").astype(int)
df.drop(columns=["ts", "label", "model_verdict", "probability"]).to_csv(dst or "feedback_train.csv", index=False)
print(f"{len(df)} corrected ads -> {dst or 'feedback_train.csv'}")
