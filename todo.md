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
- [x] Added --best selection mode (default): scores both models, picks examples with largest IoU delta where ours succeeds, deduplicates by instance
- [x] Deep humanizer pass across all 8 sections: removed promotional language ("rich", "Crucially", "highly effective", "complementary contribution"), hedging ("appears to", "largely yes", "suggests that"), copula avoidance ("serves as"), trailing -ing clauses, duplicate phrasing between abstract/intro, verbose filler, and sycophantic constructions. Tightened sentence structure, varied rhythm, and differentiated abstract vs. intro openings.
- [x] Trimmed LLPM paragraph from 7 sentences to 3 (freed ~4 lines for 4-page budget)
- [x] Added Conclusion section (2 sentences)
- [x] Strengthened contribution 3 with connection to medical imaging and underwater domains
- [x] Added S_alpha column to Table 1 (Base and Ours, from full paper data)

## HPC done; synced from origin
- [x] Prompt ablation `5854374` — Table 2 + `ablation_comparison_table.txt`
- [x] Cross-dataset CAMO `5854375_0` — `camo_results.csv`; Table 3 CAMO + Range
- [x] Cross-dataset NC4K `5854375_1` — `nc4k_results.csv`; Table 3 NC4K Range (5.8× / 1.1×) + prose in `paper/VisCon/main.tex`
- [x] Qualitative `5854478` — `paper/VisCon/figures/qualitative_comparison.png` (commit on HPC: `17e818d`)
- [x] Full HPC result bundle in git (`58ab349`): ablation CSVs including `point_only` / `box_only`, per-image exports, cross-dataset tables; docs synced in `logs/HPC_RUN_LOG.md` + `EXPERIMENTS.md`

## Final (no HPC)
- [ ] Page-budget / VisCon submission checklist

