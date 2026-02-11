"""
Quantum-Ops: IonQ QAOA Logistics Starter Script
Purpose: Runs the Max-Cut QAOA demo from use-cases/01_Logistics_Optimization_MaxCut.ipynb.
"""

import os
import itertools
import math
import time
import csv
import platform
import sys
import qiskit
import qiskit_ionq
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit("tqdm not installed. Run: python -m pip install tqdm") from exc

import warnings

warnings.filterwarnings('ignore')


def classical_max_cut_brute_force(graph, time_limit_s=None, check_every=10000):
    nodes = list(graph.nodes)
    n = len(nodes)
    max_cut = 0
    best_partition = None
    start = time.perf_counter()

    for idx, partition in enumerate(itertools.product([0, 1], repeat=n)):
        current_cut = 0
        for u, v in graph.edges:
            if partition[u] != partition[v]:
                current_cut += 1
        if current_cut > max_cut:
            max_cut = current_cut
            best_partition = partition
        if time_limit_s and idx % check_every == 0:
            if time.perf_counter() - start > time_limit_s:
                return max_cut, best_partition, False

    return max_cut, best_partition, True


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


def run_qaoa(graph, gamma, beta, backend, shots=1024):
    circuit = create_qaoa_circ(graph, [gamma, beta])
    job = backend.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    scores = score_counts(graph, counts)
    best_cut = scores["best_cut"]
    expected_cut = scores["expected_cut"]
    best_bitstring = scores["best_bitstring"]
    node_bitstring = best_bitstring[::-1]

    return {
        "best_cut": best_cut,
        "expected_cut": expected_cut,
        "counts": counts,
        "best_bitstring": best_bitstring,
        "node_bitstring": node_bitstring,
        "circuit": circuit,
    }


def evaluate_graph(graph, gamma, beta, backend, shots=1024, classical_time_limit_s=None):
    start_classical = time.perf_counter()
    classical_best, _, classical_exact = classical_max_cut_brute_force(
        graph, time_limit_s=classical_time_limit_s
    )
    classical_time = time.perf_counter() - start_classical

    start_qaoa = time.perf_counter()
    qaoa = run_qaoa(graph, gamma, beta, backend, shots=shots)
    qaoa_time = time.perf_counter() - start_qaoa

    ratio_best = qaoa["best_cut"] / classical_best if classical_exact and classical_best else None
    ratio_expected = (
        qaoa["expected_cut"] / classical_best if classical_exact and classical_best else None
    )

    return {
        "classical_best": classical_best,
        "classical_exact": classical_exact,
        "classical_time": classical_time,
        "qaoa_time": qaoa_time,
        "best_cut": qaoa["best_cut"],
        "expected_cut": qaoa["expected_cut"],
        "ratio_best": ratio_best,
        "ratio_expected": ratio_expected,
        "counts": qaoa["counts"],
        "best_bitstring": qaoa["best_bitstring"],
        "node_bitstring": qaoa["node_bitstring"],
        "circuit": qaoa["circuit"],
    }


def linspace(start, stop, num):
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def param_sweep(graph, classical_best, backend, shots=256, grid=7, gamma_min=0.1, gamma_max=math.pi, beta_min=0.05, beta_max=math.pi / 2):
    gamma_values = linspace(gamma_min, gamma_max, grid)
    beta_values = linspace(beta_min, beta_max, grid)

    best = None
    total = len(gamma_values) * len(beta_values)
    with tqdm(total=total, desc="Param sweep", unit="eval") as pbar:
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
                pbar.update(1)

    return best


def get_next_epoch(csv_path):
    if not csv_path.exists():
        return 1
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        epochs = [int(row["epoch"]) for row in reader if row.get("epoch")]
    return max(epochs, default=0) + 1


def append_benchmark(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())

    if not csv_path.exists():
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        existing_fieldnames = reader.fieldnames or []
        rows = list(reader)

    if existing_fieldnames != fieldnames:
        merged_fieldnames = list(dict.fromkeys(existing_fieldnames + fieldnames))
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=merged_fieldnames)
            writer.writeheader()
            for old_row in rows:
                writer.writerow(old_row)
            writer.writerow(row)
        return

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writerow(row)


def run_qaoa_demo():
    load_dotenv()
    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        print("Error: IONQ_API_KEY not found in environment variables.")
        return

    provider = IonQProvider(api_key)
    backend = provider.get_backend("ionq_simulator")

    base_nodes = 12
    d_degree = 5
    graph_type = "random_regular"
    p_layers = 1
    base_seed = 42
    gamma_min = 0.1
    gamma_max = math.pi
    beta_min = 0.05
    beta_max = math.pi / 2
    classical_time_limit_s = 120.0

    graph = nx.random_regular_graph(d=d_degree, n=base_nodes, seed=base_seed)
    classical_best, _, classical_exact = classical_max_cut_brute_force(
        graph, time_limit_s=classical_time_limit_s
    )

    print(f"\nNodes: {len(graph.nodes)} | Edges: {len(graph.edges)}")
    print(
        f"Classical Best Cut (brute force, exact={classical_exact}, "
        f"time limit {classical_time_limit_s:.0f}s): {classical_best}"
    )

    print("\nRunning parameter sweep (p=1)...")
    sweep_grid = 7
    sweep_shots = 256
    eval_shots = 1024

    sweep_start = time.perf_counter()
    sweep = param_sweep(
        graph,
        classical_best,
        backend,
        shots=sweep_shots,
        grid=sweep_grid,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        beta_min=beta_min,
        beta_max=beta_max,
    )
    sweep_time = time.perf_counter() - sweep_start
    print(
        f"Best sweep params: gamma={sweep['gamma']:.3f}, beta={sweep['beta']:.3f} | "
        f"best_cut={sweep['best_cut']} | ratio={sweep['ratio']:.3f}"
    )
    print(f"Param sweep time: {sweep_time:.2f}s")

    result = evaluate_graph(
        graph,
        sweep["gamma"],
        sweep["beta"],
        backend,
        shots=eval_shots,
        classical_time_limit_s=classical_time_limit_s,
    )
    circuit = result["circuit"]

    print("\n--- Circuit Diagnostics ---")
    print(f"Depth: {circuit.depth()}")
    print(f"Gate Counts: {circuit.count_ops()}")
    print(f"RZZ Gates (one per edge): {circuit.count_ops().get('rzz', 0)}")

    print(
        f"QAOA Best Cut: {result['best_cut']} | "
        f"Expected Cut: {result['expected_cut']:.2f} | "
        f"Best Ratio: {result['ratio_best'] if result['ratio_best'] is not None else 'n/a'} | "
        f"Expected Ratio: {result['ratio_expected'] if result['ratio_expected'] is not None else 'n/a'}"
    )
    print(f"Classical time: {result['classical_time']:.2f}s | QAOA time: {result['qaoa_time']:.2f}s")

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

    # Benchmark logging
    csv_path = Path("benchmarks") / "qaoa_maxcut_runs.csv"
    epoch = get_next_epoch(csv_path)

    machine_id = platform.node() or os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME") or "unknown"
    backend_name = backend.name() if hasattr(backend, "name") else str(backend)
    python_version = sys.version.split()[0]
    qiskit_version = getattr(qiskit, "__version__", "unknown")
    qiskit_ionq_version = getattr(qiskit_ionq, "__version__", "unknown")

    def log_run(n_nodes, graph, result, sweep, sweep_time, eval_shots, graph_seed):
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        row = {
            "timestamp_utc": timestamp_utc,
            "machine_id": machine_id,
            "backend_name": backend_name,
            "python_version": python_version,
            "qiskit_version": qiskit_version,
            "qiskit_ionq_version": qiskit_ionq_version,
            "epoch": epoch,
            "nodes": n_nodes,
            "edges": graph.number_of_edges(),
            "degree": d_degree,
            "graph_type": graph_type,
            "graph_seed": graph_seed,
            "p_layers": p_layers,
            "gamma": f"{sweep['gamma']:.6f}",
            "beta": f"{sweep['beta']:.6f}",
            "gamma_min": f"{gamma_min:.6f}",
            "gamma_max": f"{gamma_max:.6f}",
            "beta_min": f"{beta_min:.6f}",
            "beta_max": f"{beta_max:.6f}",
            "sweep_grid": sweep_grid,
            "sweep_shots": sweep_shots,
            "sweep_time_s": f"{sweep_time:.4f}",
            "eval_shots": eval_shots,
            "classical_best_cut": result["classical_best"],
            "qaoa_best_cut": result["best_cut"],
            "qaoa_expected_cut": f"{result['expected_cut']:.4f}",
            "best_ratio": f"{result['ratio_best']:.6f}" if result["ratio_best"] is not None else "",
            "expected_ratio": f"{result['ratio_expected']:.6f}" if result["ratio_expected"] is not None else "",
            "classical_exact": result["classical_exact"],
            "classical_time_limit_s": f"{classical_time_limit_s:.2f}",
            "classical_time_s": f"{result['classical_time']:.4f}",
            "qaoa_time_s": f"{result['qaoa_time']:.4f}",
            "best_bitstring_le": str(result["best_bitstring"]),
        }
        append_benchmark(csv_path, row)

    log_run(base_nodes, graph, result, sweep, sweep_time, eval_shots, base_seed)

    # Re-run with tuned parameters at 20 and 28 nodes
    for n_nodes in (20, 28):
        graph_seed = 40 + n_nodes
        graph = nx.random_regular_graph(d=d_degree, n=n_nodes, seed=graph_seed)
        print(f"\nNodes: {len(graph.nodes)} | Edges: {len(graph.edges)}")
        print("Re-using tuned parameters from 12-node sweep...")
        result = evaluate_graph(
            graph,
            sweep["gamma"],
            sweep["beta"],
            backend,
            shots=eval_shots,
            classical_time_limit_s=classical_time_limit_s,
        )
        print(
            f"Classical Best Cut (brute force, exact={result['classical_exact']}): {result['classical_best']} | "
            f"QAOA Best Cut: {result['best_cut']} | "
            f"Expected Cut: {result['expected_cut']:.2f} | "
            f"Best Ratio: {result['ratio_best'] if result['ratio_best'] is not None else 'n/a'} | "
            f"Expected Ratio: {result['ratio_expected'] if result['ratio_expected'] is not None else 'n/a'}"
        )
        print(f"Classical time: {result['classical_time']:.2f}s | QAOA time: {result['qaoa_time']:.2f}s")
        log_run(n_nodes, graph, result, sweep, sweep_time, eval_shots, graph_seed)


if __name__ == "__main__":
    run_qaoa_demo()
