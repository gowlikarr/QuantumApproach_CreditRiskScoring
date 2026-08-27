import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC


def train_quantum_svm(K_train, y_train):
    svm = SVC(kernel="precomputed")
    svm.fit(K_train, y_train)
    return svm


def train_classical_svms(X_train, y_train):
    svm_rbf = SVC(kernel="rbf").fit(X_train, y_train)
    svm_linear = SVC(kernel="linear").fit(X_train, y_train)
    return svm_rbf, svm_linear


def evaluate_models(svm_quantum, K_test, svm_rbf, svm_linear, X_test, y_test):
    pred_quantum = svm_quantum.predict(K_test)
    acc_quantum = accuracy_score(y_test, pred_quantum)
    acc_rbf = accuracy_score(y_test, svm_rbf.predict(X_test))
    acc_linear = accuracy_score(y_test, svm_linear.predict(X_test))

    results = pd.DataFrame({
        "Model": [
            "Quantum Kernel SVM (6 qubits)",
            "Classical SVM (RBF kernel)",
            "Classical SVM (Linear kernel)",
        ],
        "Test Accuracy": [acc_quantum, acc_rbf, acc_linear],
    })
    report = classification_report(y_test, pred_quantum, target_names=["Bad credit", "Good credit"])
    cm = confusion_matrix(y_test, pred_quantum)
    return pred_quantum, results, report, cm
