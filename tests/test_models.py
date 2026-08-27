import numpy as np

from src.models import evaluate_models, train_classical_svms, train_quantum_svm


def test_train_and_evaluate_models():
    rng = np.random.RandomState(0)
    X_train = rng.rand(40, 3)
    y_train = np.array([0, 1] * 20)
    X_test = rng.rand(10, 3)
    y_test = np.array([0, 1] * 5)

    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T

    svm_quantum = train_quantum_svm(K_train, y_train)
    svm_rbf, svm_linear = train_classical_svms(X_train, y_train)

    pred_quantum, results, report, cm = evaluate_models(
        svm_quantum, K_test, svm_rbf, svm_linear, X_test, y_test
    )

    assert len(pred_quantum) == len(y_test)
    assert list(results["Model"]) == [
        "Quantum Kernel SVM (6 qubits)",
        "Classical SVM (RBF kernel)",
        "Classical SVM (Linear kernel)",
    ]
    assert cm.shape == (2, 2)
    assert isinstance(report, str)
