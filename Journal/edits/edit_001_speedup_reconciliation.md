# Edit 001 — Speedup Denominator Reconciliation (1 h vs 1–4 h)

**Date**: 2026-08-27
**Requested by**: Professor / Reviewer — inline placeholder in Sec I
**Paper section**: Sec I — Introduction (speedup sentence) + Sec VIII-E — Inference Speed
**Edit type**: clarification + replacement
**Status**: ready-for-review

---

## Context

The professor's placeholder (paper_full_text.txt, lines 75–82) reads:

> [Input/confirmation required from the students: the speedup figure reported later in the paper
> (Section VIII, ∼180,000×) is computed against a ∼3,600 s (≈1 hour) reference run, i.e., the
> low end of the 1–4 hour range just quoted. Please confirm which Fluent runtime was actually
> measured/assumed for the speedup calculation and reconcile the two figures; if 1 hour was a
> deliberately conservative (lower-bound) choice, state that explicitly where the speedup is
> first reported.]

**The tension**: Sec I states that ANSYS Fluent requires "1–4 hours per scenario"; but
the speedup figure everywhere in the paper (Sec I abstract sentence, Sec VIII-E, Sec X
conclusion) divides the measured ~20 ms inference latency by **≈3,600 s (1 hour)** to
obtain ~180,000×.  The 1-hour denominator is the *low end* of the cited 1–4-hour range
and was never explicitly flagged as a conservative (lower-bound) assumption.

---

## Evidence from Codebase / Literature

### Paper text occurrences of the speedup (paper_full_text.txt)

| Line | Location | Verbatim text |
|------|----------|---------------|
| 40   | Sec I abstract sentence | "under 20 ms—a ∼180,000× speedup over an approximately 1hour Fluent reference run." |
| 1265 | Sec VIII-E | "Compared to ANSYS Fluent (≈3,600 s), this is a ∼180,000× speedup in surrogate inference…" |
| 1539 | Sec X conclusion | "∼180,000× speedup over the Fluent reference run—" |

### Code search results

A full grep across `src/`, `fluent/`, `configs/`, and `scripts/` for the strings
`3600`, `speedup`, `fluent.*hour`, `runtime` returned **no hits**. This means:

- The ≈3,600 s denominator was **not derived from a measured Fluent timing** stored in
  the repository; it is an **assumed/literature-sourced reference value**.
- The 20 ms GPU latency is the only measured figure (NVIDIA RTX 4060, Sec VIII-E).

### Conclusion

The 1-hour (3,600 s) denominator is a **conservatively chosen lower bound** from the
stated 1–4-hour Fluent runtime range. It was not directly measured in this study.
Using the lower bound makes the speedup claim the **most conservative defensible figure**
(a 1-hour Fluent run gives ~180,000×; a 4-hour run would give ~720,000×). This is
scientifically sound but must be stated explicitly in both Sec I and Sec VIII-E per the
professor's request.

---

## Proposed Text

### A. Replacement for Sec I — Introduction (abstract speedup sentence)

> [DROP-IN REPLACEMENT BEGINS — replaces the sentence ending "…approximately 1hour
> Fluent reference run." in the abstract/introduction paragraph]

End-to-end inference completes in under 20 ms—a $\sim$180,000$\times$ speedup relative to a
conservative 1-hour ($\approx$3,600 s) lower-bound Fluent reference, corresponding to the
low end of the 1–4 hour per-scenario runtime typical of full-fidelity ANSYS Fluent cold-leg
CFD runs [VERIFY: cite Fluent/AP1000 runtime source, e.g., Westinghouse DCD or published
CFD benchmark]; using a 4-hour upper bound would yield a $\sim$720,000$\times$ figure.
Because the Fluent denominator is a literature-sourced assumption rather than a directly
measured value in this study, the 1-hour lower bound is adopted throughout as the most
conservative and reproducible baseline.

> [DROP-IN REPLACEMENT ENDS]

---

### B. Replacement for Sec VIII-E — Inference Speed (full subsection)

> [DROP-IN REPLACEMENT BEGINS — replaces Sec VIII-E in paper_full_text.txt, lines 1261–1267]

\subsubsection*{E. Inference Speed}

End-to-end pipeline latency remains below 20~ms under the present GPU implementation
(NVIDIA RTX~4060): neural operator ($\sim$15~ms) + feature extraction ($<$1~ms) +
classification ($<$1~ms). As a reference denominator, we adopt $\approx$3,600~s
(1~hour) per scenario, which represents the \emph{conservative lower bound} of the
1–4~hour runtime range reported for full-fidelity ANSYS~Fluent cold-leg LOCA simulations
[VERIFY: insert Fluent runtime citation here]. This choice yields a surrogate speedup of
$\sim$180,000$\times$; adopting the 4-hour upper bound would increase the figure to
$\sim$720,000$\times$. The 1-hour denominator is used consistently throughout this paper
because it produces the most conservative, reproducibly defensible comparison. It must be
noted that the Fluent runtime is a literature-sourced assumption, not a value measured
directly in this study; a controlled wall-clock comparison on matched hardware remains
necessary for a definitive speedup claim.

> [DROP-IN REPLACEMENT ENDS]

---

### C. Addition for Sec X — Conclusion (after the ~180,000× mention)

> [ADDITION BEGINS — append parenthetical after "∼180,000× speedup over the Fluent reference run" at line 1539]

(adopting a conservative 1-hour lower-bound denominator; the full 1–4-hour Fluent range
would yield 180,000–720,000$\times$)

> [ADDITION ENDS]

---

## Notes / Caveats

1. **Team must decide**: Was the 3,600 s value taken from a specific published source
   (e.g., Westinghouse AP1000 DCD, a prior Fluent benchmark paper), or is it a general
   community estimate? If a citable source exists, insert the IEEE citation at all three
   `[VERIFY: ...]` markers above.

2. **If no citable source exists**, the paper should state clearly that the runtime is
   a "commonly cited community benchmark" and add a footnote or inline qualifier. Do
   NOT leave the denominator uncited in the final submitted version.

3. **Consistency**: Once the team confirms the denominator source, update all three
   occurrences (Sec I, Sec VIII-E, Sec X) identically so reviewers can follow the claim.

4. **Upper-bound speedup option**: The edit exposes the 720,000× upper-bound figure
   for completeness. The team may choose to quote a range (180,000–720,000×) or retain
   only the lower-bound figure; either approach is acceptable provided it is clearly
   labelled.

5. **No code change required**: The 3,600 s value is not hard-coded anywhere in the
   repository; it exists only in the paper text.
