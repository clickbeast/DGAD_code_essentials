# DGAD — Code Essentials

Core components I wrote from scratch for my MSc thesis, *Optimizing Diffusion Models for Generative Antibody Design* (Ghent University, 2025). The full project extends **DiffAb**, a joint structure/sequence diffusion model for antibody CDR design. This repo isolates the files where my main contributions live, so the changes are easy to read without diffing against the upstream codebase.

This is a curated subset for review, not a standalone runnable repo: it depends on the DiffAb framework and project configs.

## Contributions

1. **SIMS guidance for antibody diffusion.** Adapted SIMS, a guidance technique from image diffusion, to DiffAb. At inference an auxiliary model's score is subtracted from the base model's to steer sampling and counter data scarcity. ~8% reduction in CDR structural error (RMSD).

2. **Sampling-bias correction.** Diagnosed a frequency bias in the SAbDab training distribution and built a Word2Vec-style, frequency-dampened cluster sampler to correct it. A further ~7.7% RMSD reduction.

3. **Plausibility evaluation (wider thesis, not in this subset).** Beyond RMSD, integrated ESM-2 pseudo-perplexity / pseudo-log-likelihood as a proxy for biological plausibility, to separate genuinely novel structures from hallucinations.

## File map

**Model & diffusion**
- `diffab.py` — DiffAb model plus `DiffusionAntibodyDesignSims`, the SIMS variant wrapping a base and an auxiliary model.
- `diffab_sims.py` — auxiliary-model sampling variant (the path that produces the aux score used for guidance).
- `dpm_full.py` — full diffusion process (`FullDPM`, `FullDPMSims`); combines base and aux scores under the guidance weighting.
- `transition.py` — position, rotation (SO(3)) and amino-acid categorical noise/denoise schedules.

**Sampling-bias fix**
- `sampling.py` — `BalancedClusterSampler` (frequency-dampened cluster sampling) and `Tracker`.

**Data**
- `sabdab_dgad.py` — SAbDab dataset loading for the DGAD setup.
- `dgad_aux.py` — synthetic auxiliary dataset (LMDB) loading used for guidance.
- `mask.py` — CDR / antibody masking transforms, including `MaskSingleCDRDgad`.

**Training**
- `train_base.py` — base-model training: balanced sampler, W&B logging, validation and checkpointing.
- `train_aux.py` — auxiliary-model training / auxiliary-dataset generation.

**Inference**
- `diffab_inference_sims_testset.py` — SIMS inference entry point: loads base + aux models and runs guided sampling over a test set.
- `design_for_testset.py`, `design_for_testset_base.py` — de novo and optimization sampling for the base model.

---
Full engineering write-up and other projects: [simonvermeir.com](https://simonvermeir.com)
