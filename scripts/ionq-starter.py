"""
Quantum-Ops: IonQ QAOA Logistics Starter Script
Purpose: Runs the Max-Cut QAOA demo from use-cases/01_Logistics_Optimization_MaxCut.ipynb.
"""

import os
import itertools
import math

import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider

import warnings

warnings.filterwarnings('ignore')


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


def score_counts(graph, counts):
    best_cut = -1
    best_bitstring = None
    total_shots = sum(counts.values()) or 1
    expected_cut = 0.0
    most_frequent_bitstring = max(counts, key=counts.get)

    for bitstring, shots in counts.items():
        node_bits = bitstring[::-1]
        cut = cut_size(graph, node_bits)
        expected_cut += cut * shots
        if cut > best_cut:
            best_cut = cut
            best_bitstring = bitstring

    expected_cut /= total_shots
    return {
        "best_cut": best_cut,
        "best_bitstring": best_bitstring,
        "expected_cut": expected_cut,
        "most_frequent_bitstring": most_frequent_bitstring,
    }


def evaluate_graph(graph, gamma, beta, backend, shots=1024):
    classical_best, _ = classical_max_cut_brute_force(graph)
    circuit = create_qaoa_circ(graph, [gamma, beta])

    job = backend.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    scores = score_counts(graph, counts)
    best_cut = scores["best_cut"]
    expected_cut = scores["expected_cut"]
    best_bitstring = scores["best_bitstring"]
    node_bitstring = best_bitstring[::-1]

    ratio_best = best_cut / classical_best if classical_best else 0
    ratio_expected = expected_cut / classical_best if classical_best else 0

    return {
        "classical_best": classical_best,
        "best_cut": best_cut,
        "expected_cut": expected_cut,
        "ratio_best": ratio_best,
        "ratio_expected": ratio_expected,
        "counts": counts,
        "best_bitstring": best_bitstring,
        "node_bitstring": node_bitstring,
        "circuit": circuit,
    }


def linspace(start, stop, num):
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def param_sweep(graph, classical_best, backend, shots=256, grid=7):
    gamma_values = linspace(0.1, math.pi, grid)
    beta_values = linspace(0.05, math.pi / 2, grid)

    best = None
    for gamma in gamma_values:
        for beta in beta_values:
            circuit = create_qaoa_circ(graph, [gamma, beta])
            job = backend.run(circuit, shots=shots)
            counts = job.result().get_counts()
            scores = score_counts(graph, counts)
            ratio = scores["best_cut"] / classical_best if classical_best else 0

            if best is None or ratio > best["ratio"]:
                best = {
                    "gamma": gamma,
                    "beta": beta,
                    "ratio": ratio,
                    "best_cut": scores["best_cut"],
                }

    return best


def run_qaoa_demo():
    load_dotenv()
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        print("Error: IONQ_API_KEY not found in environment variables.")
        return

    provider = IonQProvider(api_key)
    backend = provider.get_backend("ionq_simulator")

    n_nodes = 12
    d_degree = 5
    graph = nx.random_regular_graph(d=d_degree, n=n_nodes, seed=42)
    classical_best, _ = classical_max_cut_brute_force(graph)

    print(f"\nNodes: {len(graph.nodes)} | Edges: {len(graph.edges)}")
    print(f"Classical Best Cut (brute force): {classical_best}")

    print("\nRunning parameter sweep (p=1)...")
    sweep = param_sweep(graph, classical_best, backend, shots=256, grid=7)
    print(
        f"Best sweep params: gamma={sweep['gamma']:.3f}, beta={sweep['beta']:.3f} | "
        f"best_cut={sweep['best_cut']} | ratio={sweep['ratio']:.3f}"
    )

    result = evaluate_graph(graph, sweep["gamma"], sweep["beta"], backend, shots=1024)
    circuit = result["circuit"]

    print("\n--- Circuit Diagnostics ---")
    print(f"Depth: {circuit.depth()}")
    print(f"Gate Counts: {circuit.count_ops()}")
    print(f"RZZ Gates (one per edge): {circuit.count_ops().get('rzz', 0)}")

    print(
        f"QAOA Best Cut: {result['best_cut']} | "
        f"Expected Cut: {result['expected_cut']:.2f} | "
        f"Best Ratio: {result['ratio_best']:.3f} | "
        f"Expected Ratio: {result['ratio_expected']:.3f}"
    )

    print(f"Best bitstring: {result['best_bitstring']} (node order: {result['node_bitstring']})")
    print("Execution Results (top 5):")
    for bitstring, shots in sorted(result["counts"].items(), key=lambda item: item[1], reverse=True)[:5]:
        node_bits = bitstring[::-1]
        print(f"  {bitstring} | shots={shots} | cut={cut_size(graph, node_bits)}")

    # Visualize the best bitstring assignment on the graph
    best_str = result["best_bitstring"]
    colors = ['red' if best_str[::-1][i] == '1' else 'blue' for i in range(len(graph.nodes))]

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 7))

    cut_edges = [(u, v) for u, v in graph.edges if colors[u] != colors[v]]
    uncut_edges = [(u, v) for u, v in graph.edges if colors[u] == colors[v]]

    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=600)
    nx.draw_networkx_labels(graph, pos, font_color='white')
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color='green', width=2, label='Cut (The Bridge)')
    nx.draw_networkx_edges(graph, pos, edgelist=uncut_edges, edge_color='gray', style='dashed', alpha=0.5, label='Internal')

    plt.title(f"Max-Cut Visualization: {len(cut_edges)} Redundant Bridges")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_qaoa_demo()
