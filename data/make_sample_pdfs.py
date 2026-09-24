import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = "/home/claude/AI-Resume-Screening/data/sample_resumes"

for txt_path in sorted(glob.glob(os.path.join(SRC_DIR, "*.txt"))):
    with open(txt_path, "r") as f:
        text = f.read()

    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.08, 0.95, text, va="top", ha="left", fontsize=10, family="monospace", wrap=True)
    plt.axis("off")

    pdf_path = txt_path.replace(".txt", ".pdf")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print("Wrote", pdf_path)
