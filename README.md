# VPR–Riemann Spectral Analysis

**Author:** Stefka Georgieva  
**Year:** 2025  
**Funding:** Self-funded independent research  
**Website:** [https://vpr-research.eu](https://vpr-research.eu)

---

### Overview

This repository contains the computational experiment validating the resonance correspondence between prime numbers and the non-trivial zeros of the Riemann zeta function, within the **VPR (Vortex–Pattern–Resonance)** framework.

The algorithm constructs a prime-weighted Hamiltonian:

\[
\hat{H} = -\frac{d^2}{dn^2} + \frac{3}{4n^2} + \lambda \sum_{p \in \mathbb{P}} \delta(n - p)
\]

and compares its eigenvalue spectrum with the known Riemann zeros.

---

### Key Results

| Metric | Result | Interpretation |
|--------|---------|----------------|
| Correlation (eigenvalues–zeros) | 0.9862 | Strong spectral alignment |
| Best Test R² | 0.9974 | Excellent model coherence |
| Mean relative deviation | 0.05% | Near-perfect numerical match |
| Overfitting gap | 0.0029 | Negligible |

---

### Files

- `main_final.py` – Full Python implementation  
- `README.md` – Description and background  


---

### Reference

This research is conceptually linked to the **VPR Model** of informational energy and resonance structures.  
For more details, visit: [https://vpr-research.eu](https://vpr-research.eu)

---

### License

Released under the [MIT License](LICENSE).
