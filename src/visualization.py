import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import seaborn as sns


def plot_feature_importance(importances, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="#5b6fd6")
    ax.invert_yaxis()
    ax.set_title("Random Forest Feature Importance — Credit Risk Attributes")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def plot_circuit_diagram(kernel_circuit, x1, x2, save_path=None):
    fig, ax = qml.draw_mpl(kernel_circuit, style="pennylane")(x1, x2)
    fig.suptitle("Quantum Kernel Circuit: U(x1) followed by U(x2)-adjoint", y=1.05)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_measurement_probs(probs_same_class, probs_diff_class, n_qubits, top_k=12, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, probs, title, color in zip(
        axes, [probs_same_class, probs_diff_class],
        ["Two applicants: same class (both 'Good')", "Two applicants: different class ('Good' vs 'Bad')"],
        ["#3fa34d", "#d1495b"],
    ):
        order = np.argsort(probs)[::-1][:top_k]
        labels = [format(i, f'0{n_qubits}b') for i in order]
        ax.bar(labels, probs[order], color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Basis state |q5 q4 q3 q2 q1 q0>")
        ax.tick_params(axis='x', rotation=90)
    axes[0].set_ylabel("Measurement probability")
    plt.suptitle("Measurement Probability Distribution (top 12 basis states)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, axes


def plot_kernel_matrix(K_train, y_train, save_path=None):
    order = np.argsort(y_train)
    K_ordered = K_train[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(K_ordered, cmap="viridis", ax=ax, cbar_kws={"label": "Quantum kernel value"})
    n_bad = np.sum(y_train[order] == 0)
    ax.axhline(n_bad, color="white", lw=1.5)
    ax.axvline(n_bad, color="white", lw=1.5)
    ax.set_title("Quantum Kernel Gram Matrix (train set, grouped by class)\n"
                 "top-left/bottom-right blocks = within-class similarity")
    ax.set_xlabel("Applicant index (sorted: Bad credit | Good credit)")
    ax.set_ylabel("Applicant index (sorted: Bad credit | Good credit)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def plot_accuracy_comparison(results, save_path=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ["#5b6fd6", "#8c8c8c", "#c2c2c2"]
    bars = ax.bar(results["Model"], results["Test Accuracy"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        "Quantum Kernel SVM vs. Classical SVM Baselines\n(Credit Risk Classification, 60 held-out applicants)"
    )
    for b, v in zip(bars, results["Test Accuracy"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2%}", ha="center", fontsize=9)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def plot_confusion_matrix(cm, save_path=None):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Quantum Kernel SVM — Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax
