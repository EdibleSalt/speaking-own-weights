# Reproducing the paper, section by section

All commands run from the repository root. The "recompute from raw" column names the tool that rebuilds a number from the shipped data **where such a tool exists** — for the C0/A/B/C5/C7 families the shipped per-cell `summary.json` / logs *are* the primary record (`analyze_all.py` reconciles them but does not independently recompute them), and regenerating those numbers means rerunning the experiment (last column, GPU).

A note on the schedulers (`run_pending.py`, `run_tonight.py`, `run_extra.py`): they are kept verbatim because they are the authoritative record of which cells were launched with which hyperparameters. They grew phase by phase over the project and are not a polished pipeline — read them as documentation first, launcher second.

| Paper section | Claim / artifact | Data cells (`results/data/`) | Recompute from raw | Retrain / regenerate |
| --- | --- | --- | --- | --- |
| §3 zero-shot (≈0.01) | A-family spontaneous confabulation | `A3_v3_clean_multiseed`, `A3-1_persona_ablation`, `B4_a_sweep` | per-cell `summary.json` / logs (predate `analyze_all.py`; BLOCK 1 reconciles, does not recompute) | `src/poc.py`, `src/ablate_system_prompt.py` |
| §3 probe ceiling (0.44–0.68), Fig. 2 | C0 layer-scan ridge probe | `C0_probe_ceiling` | per-cell `summary.json`; `src/diagnose_p4.py` needs the omitted hidden-state caches — rerun the probe to regenerate | `src/probe_ceiling.py` |
| §3 bridging (0.50/0.48/0.66), Fig. 3 | C1 input-embed / C2 unembedding / C3 mid-layer | `C1_lora_finetune`, `C2_hard_target`, `C3_deep_hidden` | BLOCKs 1–3 | `run_pending.py`, `run_tonight.py`, `run_extra.py` (exact hyperparameters inside) |
| §4 layer 1 (synonym cut, 0.16–0.18 vs 0.29), Fig. 4 | G1 clean-subset residual + static baseline | `G1_non_activation_query`, `G1_static_baseline` | BLOCK 5 | `tools/g1_build_queries.py` (dictionaries already shipped in `materials/G1_queries/`), then `src/c1_lora.py --query_dict_path ...`; `tools/g1_static_synonym_baseline.py` |
| §4 layer 2 (partial correlation ≈0.60), Fig. 5 | G2 physical targets + frequency baseline | `G2_physical_target`, `G2_freq_baseline` | BLOCK 4 | `run_pending.py` (G2 cells); `tools/g2_freq_baseline.py` |
| §4 layer 3 (2×2, +0.05 / +0.02), Fig. 6, Table 1, Eq. (1) | C4 cross-model targets | `C4_cross_model_target` | `tools/analyze_cross_model_specificity.py`, `tools/matched_eval.py`, `tools/bootstrap_partial_selfnet.py` (CI of the +0.05) | `run_pending.py` (C4 cells) |
| §5 causal verdict (k=0.46/0.30 vs exactly 0), Fig. 7, Table 2 | C7 single-row perturbation | `C7_causal_perturb` | numbers are in `*_perturb.json` (`summary`); preregistration in `docs/` | `tools/causal_perturb.py` or `tools/run_tonight_causal.py` (pure inference; needs models + a trained C1 adapter) |
| §6 base vs instruct (0.39 vs 0.50) | E-family | `C1_lora_finetune/olmo2-1b-base_*` | BLOCKs 1–2 | `run_pending.py` |
| §6 dual-space adapter, Fig. 8 | C6 mixed target | `C6_mixed_target` | BLOCK 7 | `src/c6_mixed_target.py` |
| §6 capability cost, Fig. 9 | C5 five-task eval | `C5_capability_eval` | per-cell `summary.json` | `src/c5_capability_eval.py` (wraps `lm_eval`; needs HF datasets access) |
| §6 Ogden-850, Fig. 10 | C1 `--train_vocab basic` + geometry | `C1_lora_finetune/*basic*` | BLOCK 1; `tools/analyze_ogden_geometry.py` | `run_pending.py` |
| All figures | Figs. 1–10 | hard-coded authoritative values | `tools/make_paper_figs.m`, `tools/make_fig_layer12.m`, `tools/make_fig8_causal.m` (base MATLAB, no toolboxes) | — |

Figure files vs. paper (PDF) numbers — file names were kept stable across draft revisions, numbering is assigned by LaTeX: `fig0_method_en`=Fig. 1, `fig1_gap`=Fig. 2, `fig2_bridge`=Fig. 3, `fig_g1_residual`=Fig. 4, `fig_g2_semantic`=Fig. 5, `fig4_crossmodel`=Fig. 6, `fig8_causal`=Fig. 7, `fig5_shared`=Fig. 8, `fig6_capability`=Fig. 9, `fig7_ogden`=Fig. 10.

Notes:
- NLTK corpora required once: `python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('brown')"`.
- Several `tools/` scripts resolve `results/data` relative to the current working directory — always run from the repository root.
- `tools/fetch_curl.py` downloads model weights from Hugging Face (or a mirror); gated models need `HF_TOKEN` set as an environment variable.
