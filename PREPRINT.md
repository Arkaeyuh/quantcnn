# Portfolio preprint

This branch adds a publication-neutral, 11pt, single-column, author–year version of the QuantCNN manuscript.

- **Source:** [`paper/preprint.tex`](paper/preprint.tex)
- **Original ICML source:** [`paper/main.tex`](paper/main.tex), unchanged.
- **Shared content:** `paper/sections/`, `paper/figures/`, `paper/math_commands.tex`, and `paper/references.bib` are reused without modification.
- **Build:** From the `paper/` directory, run `latexmk -pdf -interaction=nonstopmode -halt-on-error preprint.tex`. The output is `paper/preprint.pdf`.
- **Automated build:** The GitHub Actions workflow `Build portfolio preprint` compiles the PDF on pushes to this branch and makes it downloadable as a workflow artifact. The workflow does not alter the default branch.

**Before publishing:** Verify the author name in `paper/preprint.tex` and inspect the resulting PDF for table overflow, figure placement, reference formatting and completeness. This file is an independent research preprint, not a claim of conference acceptance or peer review.

The existing `paper/main.pdf` continues to be the ICML-layout PDF; it has not been replaced.
