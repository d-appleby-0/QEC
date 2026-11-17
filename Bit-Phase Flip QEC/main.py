from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator, transpile
import random
import numpy as np

def encode_circuit(init_state=None):
    
    qr = QuantumRegister(3, 'q')            # Quantum register w/ 3 qubits
    cr = ClassicalRegister(3, 'out')
    qc = QuantumCircuit(qr)

    if init_state != None:
        theta, phi = init_state
        qc.ry(theta, qr[0])         

    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[2])             # Entangle qubit 0 with qubits 1 and 2

    return qc, cr, qr

def bit_flip_error_injection(qr, qc, p):
    # Injects a bit-flip error (X-gate) with probability p on each qubit in the quantum register qr
    for i in range(len(qr)):
        if random.random() < p:
            qc.x(qr[i])             # Apply bit-flip to qubit i
            print(f"Bit-flip error injected on qubit {i}")


def fixed_bit_flip_error_injection(qr, qc, qubit_index):
    # Injects a bit-flip error (X-gate) on a specific qubit in the quantum register qr
    qc.x(qr[qubit_index])           # Apply bit-flip to the specified qubit
    print(f"Bit-flip error injected on qubit {qubit_index}")


def run_qec_sim(p, init_state=None, shots=2000, use_qec=True):
    qr = QuantumRegister(3, 'q')
    anc = QuantumRegister(2, 'anc')
    syn = ClassicalRegister(2, 'syn')
    out = ClassicalRegister(3, 'out')
    qc = QuantumCircuit(qr, anc, syn, out)

    """
    Encodes a single qubit state into a 3-qubit bit-flip code.
    init_state: optional state of the first qubit according to Bloch sphere representation as a tuple (theta, phi)
        None defaults to |0>
    """
    if init_state:
        theta, phi = init_state
        qc.ry(theta, qr[0]); qc.rz(phi, qr[0])  # Rotate qubit 0 by theta around the Y-axis, rotate qubit 0 by phi around the Z-axis
    qc.cx(qr[0], qr[1]); qc.cx(qr[0], qr[2])    # Entangle qubit 0 with qubit 1, 2

    bit_flip_error_injection(qr, qc, p)

    if use_qec:         
        qc.cx(qr[0], anc[0]); qc.cx(qr[1], anc[0])
        qc.cx(qr[1], anc[1]); qc.cx(qr[2], anc[1])
        qc.measure(anc, syn)                        # measure parity difference between adjacent qubits as syndrome
        with qc.if_test((syn, 0b01)): qc.x(qr[0])   # correct errors: s0=1, s1=0
        with qc.if_test((syn, 0b11)): qc.x(qr[1])
        with qc.if_test((syn, 0b10)): qc.x(qr[2])

    qc.measure(qr, out)
    counts = AerSimulator().run(transpile(qc, AerSimulator()), shots=shots).result().get_counts()

    correct = 0
    for outcome, count in counts.items():
        """
        outcome format: 'out syn' (registers printed in order of circuit addition,
        last-added register appears leftmost in the string)
        """
        data_bits = outcome.split(' ')[0]                       # the 'out' register (added last, printed first)
        if init_state is not None:
            # Determine the expected logical state based on the initial state
            theta, phi = init_state
            expected_logical = 0 if theta < np.pi/2 else 1
            logical = 0 if data_bits.count('1') <= 1 else 1
            if logical == expected_logical:
                correct += count
        else:
            logical = 0 if data_bits.count('1') <= 1 else 1
            if logical == 0:                                    # Default is |0> (logical 0)
                correct += count

    return correct / shots


if __name__ == "__main__":
    init_state = (np.pi/4, 0)
    error_rates = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    print(f"{'p_error':>10} {'Fidelity (no QEC)':>20} {'Fidelity (QEC)':>18}")
    print("-" * 52)  
    for p in error_rates:
        fid_raw = run_qec_sim(p, init_state=init_state, shots=2000, use_qec=False)
        fid_qec = run_qec_sim(p, init_state=init_state, shots=2000, use_qec=True)
        print(f"{p:>10.2f} {fid_raw:>20.4f} {fid_qec:>18.4f}")


