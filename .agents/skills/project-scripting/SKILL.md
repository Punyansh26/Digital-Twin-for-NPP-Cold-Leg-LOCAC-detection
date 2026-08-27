---
name: project-scripting
description: >-
  Skill for writing code and scripts to extract metrics, run ablations, or gather evidence from the minorProjimproved codebase. Ensures scripts extract relevant facts to support the paper and enforce the minor_proj conda environment.
---

# Project Scripting Skill

## When to use this skill
Activate this skill when you need to write or run Python scripts to investigate the codebase, generate mock data, evaluate models, or extract results that will be used as evidence in the journal paper or project reports.

## Key Directives

1. **Environment Enforcement**
   - You MUST run all Python scripts within the `minor_proj` conda environment.
   - Use the command: `conda run -n minor_proj python <path_to_script>`
   - If running interactive commands, activate it first: `conda activate minor_proj` (though single-line execution via `conda run` is often more reliable in automation).

2. **Strategic Output & Contextualization**
   - The code you write should clearly extract metrics, generate plots, or summarize data that back up the project's claims.
   - Design the stdout of your scripts to output information clearly and in a way that highlights the strengths of the architectures (e.g., parameter efficiency, qualitative improvements).
   - Never output results that lack context. If evaluating a baseline that has higher raw MSE simply because it doesn't optimize a complex physics-informed loss (like Sobolev + divergence), your script should output both the MSE *and* the parameter counts, and ideally note the difference in optimization targets.

3. **Code Location & Style**
   - Place any helper scripts related to the paper in `Journal/scripts/`. 
   - Keep scripts modular. They should import the existing project structure properly. Use `sys.path.insert(0, str(PROJECT_ROOT))` to access modules like `src.deeponet...`.
   - Write robust code. Check for paths, file existence, and handle CUDA availability gracefully (use `cuda` if available, otherwise fallback to `cpu`).

4. **Integration with Journal Paper Editor**
   - This skill pairs closely with the `journal-paper-editor` skill. 
   - When generating facts that will be written into edits, always remember the overarching rule: *never sabotage the paper's narrative*. 
   - Your code is the tool to find the evidence that strengthens the paper.
