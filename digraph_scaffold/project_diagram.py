#1. Fraud Detection System Overview

from graphviz import Digraph

dot = Digraph(comment="Fraud Detection System Overview", format="png")
dot.attr(rankdir="LR", size="8")

dot.node("D", "Transaction Data")
dot.node("P", "Preprocessing")
dot.node("F", "Feature Engineering")
dot.node("M", "ML Model")
dot.node("R", "Risk Scoring")
dot.node("A", "Alerts/Action")

dot.edges(["DP", "PF", "FM", "MR", "RA"])

dot.render("fraud_detection_overview", view=True)

# 2. ML Approaches (Concept Diagram)

dot = Digraph(comment="ML Approaches", format="png")
dot.attr(rankdir="TB")

dot.node("ML", "Fraud Detection ML Approaches")
dot.node("S", "Supervised\n(Logistic Regression,\nRandom Forest, XGBoost)")
dot.node("U", "Unsupervised\n(Isolation Forest,\nClustering)")
dot.node("DL", "Deep Learning\n(Neural Networks,\nAutoencoders)")
dot.node("H", "Hybrid\n(Supervised + Unsupervised)")

dot.edges([("MLS", "MLU"), ("MLU", "MLDL"), ("MLDL", "MLH")])

dot.render("ml_approaches", view=True)

# 3. Workflow Pipeline
dot = Digraph(comment="Workflow Pipeline", format="png")
dot.attr(rankdir="LR", size="10")

steps = [
    ("C", "1. Collect & Preprocess Data"),
    ("F", "2. Feature Engineering"),
    ("T", "3. Train & Evaluate Models"),
    ("D", "4. Deploy into Production"),
    ("S", "5. Real-time Scoring"),
    ("M", "6. Monitoring & Retraining"),
]

for code, label in steps:
    dot.node(code, label, shape="box", style="rounded,filled", color="lightblue")

dot.edges([s[0] + steps[i+1][0] for i, s in enumerate(steps[:-1])])

dot.render("workflow_pipeline", view=True)

# 4. Feedback Loop Diagram
dot = Digraph(comment="Feedback Loop", format="png")
dot.attr(rankdir="LR")

dot.node("T", "Transactions")
dot.node("S", "Scored by Model")
dot.node("F", "Flagged for Review")
dot.node("R", "Human Review")
dot.node("B", "Feedback Data")
dot.node("M", "Retrain Model")

dot.edge("T", "S")
dot.edge("S", "F")
dot.edge("F", "R")
dot.edge("R", "B")
dot.edge("B", "M")
dot.edge("M", "S", label="Improved Model")

dot.render("feedback_loop", view=True)
