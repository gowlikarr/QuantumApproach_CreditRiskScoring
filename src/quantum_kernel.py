import numpy as np
import pennylane as qml


def make_device(n_qubits):
    return qml.device("default.qubit", wires=n_qubits)


def feature_map(x, wires):
    n_qubits = len(wires)
    qml.AngleEmbedding(x, wires=wires, rotation="Y")
    for i in range(n_qubits):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n_qubits]])
    qml.AngleEmbedding(x, wires=wires, rotation="Z")


def make_kernel_circuit(dev, n_qubits):
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        feature_map(x1, wires=range(n_qubits))
        qml.adjoint(feature_map)(x2, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    return kernel_circuit


def quantum_kernel_value(kernel_circuit, x1, x2):
    return kernel_circuit(x1, x2)[0]


def build_kernel_matrix(A, B, kernel_circuit):
    K = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            K[i, j] = quantum_kernel_value(kernel_circuit, a, b)
    return K
