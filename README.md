# VPR–Riemann: Spectral Analysis and Numerical Validation

**Author:** Stefka Georgieva  
**Project DOI:** [10.5281/zenodo.17552652](https://doi.org/10.5281/zenodo.17552652)  
**Website:** [https://vpr-research.eu](https://vpr-research.eu)  
**License:** MIT  
**Funding:** Self-funded independent research  

---

## Abstract

This repository presents the computational and theoretical foundations of the **VPR (Vortex–Pattern–Resonance)** framework applied to the **Riemann Hypothesis**.  
The work introduces a novel differential operator:

\[
\hat{H} = -\frac{d^2}{dn^2} + \frac{3}{4n^2} + \sum_{p \in \mathbb{P}} \lambda_p \delta(n - p)
\]

whose eigenvalues numerically reproduce the first non-trivial zeros of the Riemann zeta function.  
The model interprets primes as nodal points in a resonant field, revealing a hidden spectral structure underlying number theory.

---

## Numerical Validation

The code computes the eigenvalue spectrum of the prime-weighted Hamiltonian and compares it to the first 95 non-trivial Riemann zeros.  
Two analytical modes are implemented:

- **Nonlinear (Adaptive) Analysis:** Gradient boosting regression achieving \( R^2 = 0.9974 \), mean relative error 0.05%.  
- **Conservative (Linear) Analysis:** Ridge regression baseline used for contrast and validation of model nonlinearity.

These results confirm the spectral resonance predicted by the VPR model — a bridge between mathematical chaos and quantum structure.

---


### Files

- `vpr_riemann_analysis.py` – Full Python implementation  
- `README.md` – Description and background  


---

## Run Instructions

To reproduce the experiment:

```bash
pip install numpy scipy sympy scikit-learn matplotlib
python vpr_riemann_analysis.py

---

### Reference

This research is conceptually linked to the **VPR Model** of informational energy and resonance structures.  
For more details, visit: [https://vpr-research.eu](https://vpr-research.eu)

---
## 📚 Citation

If you use or reference this work, please cite:

> **Georgieva, S.** (2025). *Resonance Model and the Riemann Hypothesis: Spectral Analysis and Numerical Validation.*  
> Zenodo. [https://doi.org/10.5281/zenodo.17552652](https://doi.org/10.5281/zenodo.17552652)

---

## 📬 Contact

- 🌐 **Website:** [https://vpr-research.eu](https://vpr-research.eu)  
- 💼 **LinkedIn:** [linkedin.com/in/stefkageorgieva](https://linkedin.com/in/stefkageorgieva)  


---



### License

Released under the [MIT License](LICENSE).
