## Current focus
- [in progress] VisCon workshop paper revision

## Completed (this revision)
- [x] Added compact Related Work section (COD, SAM adaptation, PEFT)
- [x] Added bib entries for SegMaR, LoRA, prompt tuning
- [x] Expanded LLPM mention in Discussion with architecture, training, and result details
- [x] Updated `ablation_prompt.sh` to use full 2,026-image test set (`--max-samples 0`)
- [x] Updated `eval_camo.yaml` and `eval_nc4k.yaml` to evaluate all four prompt types
- [x] Updated `eval_crossdataset.sh` time limit for expanded evaluation
- [x] Updated Table 2 caption to reflect full test set
- [x] Reduced number repetition across intro/discussion (replaced with table references)
- [x] Fixed ambiguous "both encoders" phrasing in Method section
- [x] Humanized prose: removed AI writing patterns (copula avoidance, trailing -ing clauses, promotional modifiers, vague hedging), tightened sentence structure across all sections
- [x] Rewrote qualitative figure script: contour-based overlays (not opaque fills), GT contour on prediction columns, IoU badges, reduced default to 4 rows

## Next (requires HPC)
- [ ] Run `sbatch scripts/ablation_prompt.sh` -- rerun prompt ablation on full COD10K test set
- [ ] Run `sbatch scripts/eval_crossdataset.sh` -- cross-dataset eval with all four prompt types
- [ ] Update Table 2 numbers with full-set ablation results
- [ ] Update Table 3 with cross-dataset Range row once results are in
- [ ] Remove the TODO comment in main.tex after Table 3 is finalized
- [ ] Final page-budget and submission readiness review

