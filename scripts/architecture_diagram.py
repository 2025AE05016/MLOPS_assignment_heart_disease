"""Draw the system architecture diagram used in the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LANES = [("Data & training", "#E8F0FB", 7.6), ("CI/CD (GitHub Actions)", "#FDF1E6", 5.1),
         ("Serving & deployment", "#EAF6EC", 2.6), ("Monitoring", "#F5ECF7", 0.1)]
BOXES = {
    "uci": (0.3, 8.0, "UCI archive\n(download + SHA-256)"),
    "csv": (2.9, 8.0, "Cleaned CSV\n303 × 14"),
    "eda": (5.5, 8.0, "EDA figures\n(notebook 01)"),
    "cv": (8.1, 8.0, "Pipeline + GridSearch\n5-fold CV"),
    "mlflow": (10.7, 8.0, "MLflow\nruns · registry"),
    "lint": (0.3, 5.5, "ruff lint"),
    "test": (2.9, 5.5, "train +\npytest"),
    "build": (5.5, 5.5, "docker build\n+ smoke test"),
    "kind": (8.1, 5.5, "kind cluster\ndeploy + verify"),
    "art": (10.7, 5.5, "Uploaded\nartifacts"),
    "model": (0.3, 3.0, "model.joblib (from\ntraining: preproc+clf)"),
    "api": (2.9, 3.0, "FastAPI\n/predict /metrics"),
    "docker": (5.5, 3.0, "Docker image\nnon-root"),
    "k8s": (8.1, 3.0, "K8s Deployment\n2 replicas + Svc"),
    "ing": (10.7, 3.0, "Ingress / LB\nclient JSON"),
    "logs": (2.9, 0.5, "JSON request\nlogs"),
    "prom": (5.5, 0.5, "Prometheus\nscrape 15s"),
    "graf": (8.1, 0.5, "Grafana\ndashboard"),
}
EDGES = [("uci", "csv"), ("csv", "eda"), ("eda", "cv"), ("cv", "mlflow"), ("lint", "test"),
         ("test", "build"), ("build", "kind"), ("kind", "art"), ("model", "api"),
         ("api", "docker"), ("docker", "k8s"), ("k8s", "ing"), ("api", "logs"),
         ("api", "prom"), ("prom", "graf")]
W, H = 2.2, 1.1

fig, ax = plt.subplots(figsize=(15, 10))
for title, color, y in LANES:
    ax.add_patch(FancyBboxPatch((0.05, y), 13.1, 2.3, boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor="#bbbbbb"))
    ax.text(0.15, y + 2.05, title, fontsize=11, weight="bold", color="#333333")
for x, y, label in BOXES.values():
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.05",
                                facecolor="white", edgecolor="#4C72B0", linewidth=1.5))
    ax.text(x + W / 2, y + H / 2, label, ha="center", va="center", fontsize=9.5)
for start, end in EDGES:
    (x1, y1, _), (x2, y2, _) = BOXES[start], BOXES[end]
    if y1 == y2:
        a, b = (x1 + W, y1 + H / 2), (x2, y2 + H / 2)
    elif x1 == x2:
        a, b = (x1 + W / 2, y1), (x2 + W / 2, y2 + H)
    else:
        a, b = (x1 + W / 2, y1), (x2 + W / 2, y2 + H)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                 color="#555555", connectionstyle="arc3,rad=0"))
ax.set(xlim=(0, 13.3), ylim=(0, 10.1))
ax.axis("off")
ax.set_title("HeartOps architecture", fontsize=15, weight="bold")
plt.tight_layout()
plt.savefig("reports/figures/architecture.png", dpi=120)
