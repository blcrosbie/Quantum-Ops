"""
Quantum-Ops: IonQ QAOA Logistics Starter Script
Purpose: Runs the Max-Cut QAOA demo from use-cases/01_Logistics_Optimization_MaxCut.ipynb.
"""

import os
import itertools

import networkx as nx
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider


def classical_max_cut_brute_force(graph):
    nodes = list(graph.nodes)
    n = len(nodes)
    max_cut = 0
    best_partition = None

    for partition in itertools.product([0, 1], repeat=n):
        current_cut = 0
        for u, v in graph.edges:
            if partition[u] != partition[v]:
                current_cut += 1
        if current_cut > max_cut:
            max_cut = current_cut
            best_partition = partition

    return max_cut, best_partition


def cut_size(graph, bitstring):
    return sum(1 for u, v in graph.edges if bitstring[u] != bitstring[v])


def create_qaoa_circ(graph, theta):
    """
    theta[0] = gamma (cost)
    theta[1] = beta (mixer)
    """
    nqubits = len(graph.nodes)
    qc = QuantumCircuit(nqubits)

    qc.h(range(nqubits))

    for i, j in graph.edges:
        qc.rzz(2 * theta[0], i, j)

    qc.barrier()

    for i in range(nqubits):
        qc.rx(2 * theta[1], i)

    qc.measure_all()
    return qc


def evaluate_graph(graph, gamma, beta, backend, shots=1024):
    classical_best, _ = classical_max_cut_brute_force(graph)
    circuit = create_qaoa_circ(graph, [gamma, beta])

    job = backend.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    best_bitstring = max(counts, key=counts.get)
    node_bitstring = best_bitstring[::-1]
    qaoa_cut = cut_size(graph, node_bitstring)
    ratio = qaoa_cut / classical_best if classical_best else 0

    return {
        "classical_best": classical_best,
        "qaoa_cut": qaoa_cut,
        "ratio": ratio,
        "counts": counts,
        "best_bitstring": best_bitstring,
        "node_bitstring": node_bitstring,
        "circuit": circuit,
    }


def run_qaoa_demo():
    load_dotenv()
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        print("Error: IONQ_API_KEY not found in environment variables.")
        return

    provider = IonQProvider(api_key)
    backend = provider.get_backend("ionq_simulator")

    gamma, beta = 0.35, 0.25
    benchmarks = [12, 16, 20]

    for n_nodes in benchmarks:
        d_degree = 5
        graph = nx.random_regular_graph(d=d_degree, n=n_nodes, seed=40 + n_nodes)
        print(f"\nNodes: {len(graph.nodes)} | Edges: {len(graph.edges)}")

        result = evaluate_graph(graph, gamma, beta, backend, shots=1024)
        circuit = result["circuit"]

        print("--- Circuit Diagnostics ---")
        print(f"Depth: {circuit.depth()}")
        print(f"Gate Counts: {circuit.count_ops()}")
        print(f"RZZ Gates (one per edge): {circuit.count_ops().get('rzz', 0)}")

        print(
            f"Classical Best Cut (brute force): {result['classical_best']} | "
            f"QAOA Cut: {result['qaoa_cut']} | "
            f"Approx Ratio: {result['ratio']:.3f}"
        )

        print(f"Best bitstring: {result['best_bitstring']} (node order: {result['node_bitstring']})")
        print("Execution Results (top 5):")
        for bitstring, shots in sorted(result["counts"].items(), key=lambda item: item[1], reverse=True)[:5]:
            node_bits = bitstring[::-1]
            print(f"  {bitstring} | shots={shots} | cut={cut_size(graph, node_bits)}")


if __name__ == "__main__":
    run_qaoa_demo()
