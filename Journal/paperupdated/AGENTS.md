# AGENTS.md — Tracked Revision Rules for `paperupdated`

> **Scope:** These rules apply to the editable manuscript and its supporting
> files in `Journal/paperupdated/`. They extend the repository-root and
> `Journal/AGENTS.md` instructions. The `.tex` file in this directory is the
> working tracked-revision copy; the PDF and LaTeX build by-products are not
> editing targets unless explicitly requested.

## Purpose

This copy is intended to show completed manuscript revisions unambiguously to
the authors, supervisor, and reviewers while retaining the paper's established
IEEE voice, scientific scope, and confident proof-of-concept framing.

## Required visual tracking

For every direct manuscript revision in the working `.tex` file:

- **Added text** must be blue: `\textcolor{blue}{...}`.
- **Removed or superseded manuscript text must be deleted from the working
  source.** Do not retain it as red text or strikethrough, and do not introduce
  any new red markup.
- For a replacement, remove the old wording and show only the new wording in
  blue. Preserve punctuation, citations, cross-references, mathematical
  notation, labels, and surrounding LaTeX structure unless the requested
  change requires otherwise.
- **Unresolved author work** must be written in green as
  `\textcolor{green!50!black}{\textbf{TODO:\{...\}}}`. A TODO must state the
  concrete evidence, experiment, citation, or verified author information that
  is still required.
- Do not colour unchanged context merely to make a revision more conspicuous.
  Colour only the smallest complete unit needed to represent the edit.

`xcolor` is already loaded. The working manuscript must not depend on `ulem`
or a deletion/strikethrough macro after red markup is removed. Compile after
every substantive tracked revision.

## Completing requests and placeholders

- `[Input/confirmation required ...]` and `[Input for the students ...]` blocks
  are author-action requests, not manuscript prose. Once their request has
  been fully addressed with evidenced text, **remove the request block
  entirely**.
- Do not remove a request merely because related wording was drafted. Remove
  it only when the required information, experiment, citation, figure, or
  decision has been supplied and the resulting claim is supported.
- If the request cannot yet be completed, replace it with a concise green
  `TODO:{...}` that states exactly what must be supplied. Do not invent a
  result, citation, implementation detail, or author biography.

## Writing and scientific safeguards

- Match the existing language, section style, terminology, tense, citation
  style, and level of technical detail. The manuscript should read as one
  coherent, professional IEEE journal paper rather than as stitched-in notes.
- Maintain confident, evidence-based framing of the contribution as a
  proof-of-concept surrogate methodology. State the scope precisely without
  weakening supported contributions or implying safety certification,
  operational deployment, or full-fidelity validation that has not occurred.
- Keep synthetic mock-data, Fluent-CFD, and NPPAD-derived evidence distinct.
  Do not present a source-code module, option, or planned experiment as an
  evaluated result. Use a clearly scoped future-work statement where evidence
  is absent.
- Preserve limitations that are material to scientific accuracy, but phrase
  them proportionately and constructively. Do not introduce adversarial,
  apologetic, or self-undermining language.
- Before making a technical claim, inspect the relevant source, configuration,
  dataset/result artifact, and invocation path. Preserve exact reported
  values, protocol details, citations, and qualifiers unless verified evidence
  supports a change.

## Safe editing workflow

1. Locate the precise target in the working `.tex` source and inspect nearby
   prose, labels, citations, and any related red request.
2. Confirm the requested facts from reproducible project evidence. If the
   evidence is incomplete, leave the request in place and report what is
   needed rather than guessing.
3. Apply the smallest coherent tracked revision using blue additions and green
   TODOs only. Delete superseded wording; never add red or strikethrough text.
   Remove the associated request only when it is genuinely complete.
4. Compile the manuscript with its normal LaTeX workflow and fix only errors
   introduced by the edit. Do not overwrite the existing PDF or build outputs
   unless explicitly authorised.
5. Report the edited location, the resolved request, the evidence used, and
   any remaining verification required.

## File hygiene

- Edit the `.tex` source with focused patches. Treat the existing PDF, `.aux`,
  `.fls`, `.fdb_latexmk`, `.out`, and `.synctex.gz` files as generated
  artifacts; do not hand-edit them.
- Do not overwrite or delete existing user changes or generated artifacts.
  Give any newly requested output a distinct name unless the user expressly
  asks to rebuild the working PDF.
- Do not modify the canonical paper outside `Journal/paperupdated/` unless the
  user explicitly expands the scope.

*Last updated: 2026-08-31 by Codex.*
