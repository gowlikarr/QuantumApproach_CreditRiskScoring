import numpy as np

from src.quantum_kernel import build_kernel_matrix, make_device, make_kernel_circuit, quantum_kernel_value


def test_self_kernel_is_one():
    n_qubits = 2
    dev = make_device(n_qubits)
    kernel_circuit = make_kernel_circuit(dev, n_qubits)
    x = np.array([0.3, 0.7])
    value = quantum_kernel_value(kernel_circuit, x, x)
    assert np.isclose(value, 1.0, atol=1e-6)


def test_build_kernel_matrix_is_symmetric_with_unit_diagonal():
    n_qubits = 2
    dev = make_device(n_qubits)
    kernel_circuit = make_kernel_circuit(dev, n_qubits)
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.1]])

    K = build_kernel_matrix(X, X, kernel_circuit)
    assert K.shape == (3, 3)
    assert np.allclose(np.diag(K), 1.0, atol=1e-6)
    assert np.allclose(K, K.T, atol=1e-6)
