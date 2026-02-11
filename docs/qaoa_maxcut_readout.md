# QAOA Max-Cut Readout (12-Node Demo)

This page explains the QAOA parameters, tuning methods, and how to interpret the binary string in the Max-Cut solution. It also records a recent 12-node demo run.

## Demo Readout (12 Nodes)
```
Nodes: 12 | Edges: 30
Classical Best Cut (brute force): 22

Running parameter sweep (p=1)...
Best sweep params: gamma=0.100, beta=0.050 | best_cut=22 | ratio=1.000

--- Circuit Diagnostics ---
Depth: 18
Gate Counts: OrderedDict([('rzz', 30), ('h', 12), ('rx', 12), ('measure', 12), ('barrier', 2)])
RZZ Gates (one per edge): 30
QAOA Best Cut: 21 | Expected Cut: 14.48 | Best Ratio: 0.955 | Expected Ratio: 0.658
Best bitstring: 001011100111 (node order: 111001110100)
Execution Results (top 5):
  010001001010 | shots=3 | cut=14
  100010011110 | shots=3 | cut=14
  100100001010 | shots=3 | cut=16
  000000001110 | shots=2 | cut=11
  000000010010 | shots=2 | cut=8
```

## What The Parameters Mean
- `gamma` (cost angle): Scales the problem (cost) Hamiltonian. For Max-Cut, this is the set of `RZZ(2*gamma)` gates on every edge. Increasing `gamma` increases the phase separation between "good" and "bad" cuts.
- `beta` (mixer angle): Scales the mixer Hamiltonian, implemented as `RX(2*beta)` on every qubit. This controls how much the circuit explores different bitstrings.
- `p` (QAOA depth): Number of alternating cost/mixer layers. This demo uses `p=1` (one cost layer + one mixer layer). Higher `p` can improve results but increases circuit depth.
- `shots`: Number of samples from the circuit. More shots gives a more reliable estimate of the distribution (and improves the chance of seeing a high-quality cut).

## What The Binary String Represents
Each output bit corresponds to one node in the logistics graph.
- Bit `1` -> node assigned to Group A
- Bit `0` -> node assigned to Group B

For Max-Cut, an edge contributes to the cut when its endpoints land in different groups. The objective is to maximize the number of such edges.

Important: Qiskit returns bitstrings in little-endian order (qubit 0 is the rightmost bit).  
That is why the log shows:

```
Best bitstring: 001011100111 (node order: 111001110100)
```

We reverse the string (node order) to align index `i` with node `i`.

## How We Tune Parameters (p=1)
We use a grid sweep over `(gamma, beta)` for the 12-node graph:
- `gamma` in a small range from low to high (example: 0.1 to pi)
- `beta` in a smaller range (example: 0.05 to pi/2)
- Evaluate each pair by running the circuit and computing a cut score

This is fast and robust enough to show improvement over fixed parameters.

## How We Score Results
- Best Cut: The maximum cut found among all sampled bitstrings.
- Expected Cut: The shot-weighted average cut across the full sample distribution.
- Approximation Ratio: `best_cut / classical_best` or `expected_cut / classical_best`

In practice, we report both:
- Best Ratio (peak performance seen)
- Expected Ratio (more conservative, distribution-aware)
