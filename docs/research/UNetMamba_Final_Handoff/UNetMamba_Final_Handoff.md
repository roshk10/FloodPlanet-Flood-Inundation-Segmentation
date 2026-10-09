# UNetMamba Final Handoff — FloodPlanet Research Project

**Prepared:** 2026-10-09
**Purpose:** Carry the completed UNetMamba baseline into a new conversation for evidence-led selection of one research extension.
**Status:** Baseline training, test evaluation, and qualitative figure generation are complete. Do not restart training or select the extension from this handoff alone.

> **Evidence policy:** Numbers below are transcribed from the training/evaluation logs supplied in the project conversation. Values not present in those logs are marked unknown. The literature review spreadsheets and the 135-page `Research_for_flood_inundation.docx` are included alongside this handoff. Current public GitHub source was also inspected on 2026-10-09 to identify configuration/source mismatches; the live `main` source is not automatically proof of the exact tree used to produce `best.pt`.

## Executive summary

- Dataset: FloodPlanet only; 366 PlanetScope chips across 19 flood events; four PS bands (RGB + NIR); label encoding after conversion: `-1=ignore`, `0=background`, `1=flood`.
- Common event split: 14 train events / 2 validation events / 3 test events. Test events are Bangladesh, Cambodia, and Nepal.
- The successful model had **14 epochs in the final run configuration**; it resumed from a pre-existing checkpoint and the output logged epochs 8–14. The selected checkpoint was **epoch 10** by validation mIoU.
- Best validation mIoU: **0.646375**. This is a validation result, not the test result.
- On the held-out test events, event-mean two-class mIoU was **0.738343 ± 0.091969**. Event-mean flood IoU was **0.745913 ± 0.139541**. Global confusion-count metrics accumulated over valid patch pixels gave mIoU **0.742300** and flood F1/Dice **0.860608**.
- Qualitative figures were generated for 9 test patches: one best-IoU case, one false-positive case, and one false-negative case per event. Their files exist in the downloaded results ZIP, but they were not visually inspected in the text logs available for this handoff. Therefore no unverified visual claim about specific boundaries, small regions, or spectral confusion is made here.
- **Important measurement caveat:** evaluation summed confusion counts across overlapping patches rather than reconstructing each original chip before scoring. Because the 512-pixel patches use stride 256 and overlap, the reported global aggregate counts patch-pixels, so original source pixels can contribute more than once. Do not describe that aggregate as a unique-pixel full-chip metric without qualification.
- **Important reproducibility caveat:** the repository config currently visible on GitHub `main` has `STRIDE=512` and `MAX_EPOCHS=20`, while the successful training script/checkpoint used stride 256 and max epochs 14. The live `setup.py` also still shows the incompatible C++20/compiler flags; the T4-compatible local patch was not pushed to GitHub. Exact commit hash for the final checkpoint was not recorded.
- The final downloaded large ZIP is **not self-contained source code**: the ZIP-creation cell included checkpoint, logs, qualitative artifacts, training script and configs, but did **not** add `unetmamba/src/` or the full selective-scan source tree. Use the existing GitHub repository for the base source and restore the compatibility patch from the notes below. The small results ZIP is only results/visualization material.

---

# A. Final architecture and code version

## A1. Model actually evaluated

- Model class: `UNetMamba`.
- Parameter count reported by the actual training log: **14,754,182 (14.754M)**.
- Instantiation used by the successful training/test code:
  ```python
  UNetMamba(
      pretrained=False,
      decode_channels=64,
      embed_dim=64,
      in_chans=4,
      num_classes=2,
  )
  ```
- Checkpoint selected: `/kaggle/working/FloodPlanet-Flood-Inundation-Segmentation/unetmamba/results/checkpoints/best.pt`.
- Checkpoint metadata reported by the test-evaluation cell: epoch **10**, best validation mIoU **0.646375**.
- Training-mode tested tensor contract: input `(B, 4, 512, 512)`; main logits `(B, 2, 512, 512)` plus auxiliary logits `(B, 2, 512, 512)`. In `eval()` mode, the model returns the main logits `(B, 2, 512, 512)` only.
- The model was initialized with `pretrained=False`. No pretrained backbone weights were used.

## A2. Architecture description grounded in the model source

`unetmamba/src/models/UNetMamba.py` defines the actual model wrapper and imports `ResT` and VMamba components.

- **Encoder/backbone:** the `rest_lite(...)` constructor creates a ResT-Lite encoder. With `embed_dim=64`, the decoder expects hierarchical channel widths `[64, 128, 256, 512]`. The configured ResT-Lite definition uses embedding widths `[64, 128, 256, 512]`, heads `[1, 2, 4, 8]`, depths `[2, 2, 2, 2]`, and spatial-reduction ratios `[8, 4, 2, 1]`.
- **Mamba / selective scan:** the decoder contains `VSSLayer` / `VSSBlock` modules imported from `src.models.vmamba`. The VMamba source provides the visual selective-state-space/scan implementation. The compiled selective-scan CUDA core is a dependency used by this implementation; VMamba is not being trained as a separate model.
- **Decoder:** `MambaSegDecoder`, using `PatchExpand`, skip-feature concatenation/projection, VSS layers and a final `FinalPatchExpand_X4` stage.
- **Main head:** `self.seg`, a 1×1 convolution producing `num_classes=2` logits, upsampled to the input spatial resolution through the final patch-expansion path.
- **Auxiliary supervision:** training mode returns a second auxiliary output formed by summing local supervision outputs (`LocalSupervision`) from intermediate decoder stages. During evaluation mode the model returns only the main segmentation logits.
- **Original architecture source files:**
  - `unetmamba/src/models/UNetMamba.py`
  - `unetmamba/src/models/ResT.py`
  - `unetmamba/src/models/vmamba.py`
  - `unetmamba/src/models/csm_triton.py`
  - `unetmamba/src/mamba_ssm/ops/selective_scan_interface.py` and related `mamba_ssm/ops/` implementation
  - `unetmamba/kernels/selective_scan/` including `csrc/selective_scan/`
  - `unetmamba/src/losses/`
  - `unetmamba/src/dataset.py`
  - `unetmamba/configs/`
  - `unetmamba/scripts/train_unetmamba.py` (created in Kaggle during this run; GitHub push has not been confirmed)

## A3. Modifications made during the actual working run

1. **Dataset edge-padding fix** in `unetmamba/src/dataset.py`: assignment now preserves the channels dimension when padding channel-first images:
   ```python
   padded[:, :height, :width] = image
   ```
   The previous form omitted the channel slice and failed for rectangular edge crops. The current public `main` copy of `dataset.py` inspected on 2026-10-09 contains this channel-preserving form.

2. **Selective-scan build compatibility patch** in the Kaggle working tree: Linux C++ compiler flag changed from `-std=c++20` to `-std=c++17`; NVCC `-std=c++20` changed to `-std=c++17`; the NVCC argument pair `-Xcompiler`, `/Zc:preprocessor` was removed. The compiled module imported successfully on the T4 and training then proceeded. This was a build-compatibility change, not an architecture change. The live GitHub `main` `setup.py` inspected on 2026-10-09 still displayed C++20 and the `/Zc:preprocessor` NVCC flags, so the working compatibility patch is **not confirmed pushed**.

3. **FP32 loss arithmetic while retaining AMP forward:** the initial smoke test produced NaN loss with reduced-precision logits. The training script casts the main and auxiliary outputs to FP32 before passing them to `UnetMambaLoss`. The real smoke test then produced a finite loss and completed backward/optimizer step. No `UNetMamba.py` architecture alteration was required for this numerical fix.

4. **New training script:** `unetmamba/scripts/train_unetmamba.py` was created during the Kaggle work. It records checkpoints/history and performs train/validation. It is not known to have been pushed to GitHub.

## A4. Commit / branch / repository state

- Official repository: `https://github.com/roshk10/FloodPlanet-Flood-Inundation-Segmentation.git`
- Exact Git commit hash associated with the final `best.pt`: **not recorded**. Earlier conversational snapshots mentioned hashes `50196d0` and `936b56d` at different clone/runtime points; neither can safely be asserted as the exact code revision that produced the final checkpoint.
- The local Kaggle working tree included the script and `setup.py` patch. Kaggle push was known to be unreliable in this workflow; a final local-PC commit/push for UNetMamba was still pending.
- Current public `main` config file inspected 2026-10-09 declares `STRIDE=512` and `MAX_EPOCHS=20`, and current public `setup.py` still uses `-std=c++20` plus `-Xcompiler /Zc:preprocessor`. These do **not** match the final successful run configuration/build. The checkpoint's embedded config and `train_unetmamba.py` are better evidence for actual runtime settings than the stale general config file.

## A5. Environment observed

- Python: `3.13.15`
- PyTorch: `2.11.0+cu128`
- CUDA runtime reported by PyTorch: `12.8`
- GPU: NVIDIA Tesla T4, compute capability `(7, 5)`, reported memory `15,360 MiB` / `14.562 GiB`
- Driver seen in the environment: `580.178.04`
- `fvcore` had to be installed in the active runtime.
- Imports also relied on `timm`, `einops`, `rasterio`, NumPy, OpenCV, PyTorch and the compiled selective-scan extension. Exact versions for every dependency were not recorded in the training output.
- Compiler build targeted `sm_75`. Binary extension name observed: `selective_scan_cuda_core.cpython-313-x86_64-linux-gnu.so`.
- Deprecation `FutureWarning`s from `timm` and `torch.cuda.amp.custom_fwd/custom_bwd` were observed but were not fatal.
- The compiled `.so` is environment-specific and is not a portable source artifact. The handoff does not include a binary `.so`.

**Architecture citation:** Enze Zhu, Zhan Chen, Dingkai Wang, Hanru Shi, Xiaoxuan Liu, Lei Wang, “UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images,” *IEEE Geoscience and Remote Sensing Letters*, 22 (2025), article 6001205, DOI: https://doi.org/10.1109/LGRS.2024.3505193. This is the source architecture paper, not a FloodPlanet result from this experiment.

---

# B. Actual final training configuration

| Setting | Actual successful run | Evidence / qualification |
|---|---|---|
| Dataset | FloodPlanet only | `build_records(..., expected_records=366)` |
| Model | `UNetMamba` / ResT-Lite encoder + Mamba/VSS decoder | 14,754,182 parameters |
| Input channels | 4 (PlanetScope RGB + NIR) | Dataset loader requires 4 PS bands |
| Classes | 2 | Background and flood |
| Patch size | 512 × 512 | Training/evaluation script and logs |
| Patch stride | **256** | Actual script/dataset call; 50% stride relative to patch side |
| Batch size | 4 | Actual script / logs |
| DataLoader workers | 2 | Actual script |
| Training batches per epoch | 832 | `3328 / 4` with `drop_last=True` |
| Validation batches per epoch | 456 | `1824 / 4` |
| Maximum epochs in successful run | **14** | Output showed `Epoch 08/14` through `Epoch 14/14`, `Completed epochs: 14` |
| Resume | Resumed existing checkpoint from the epoch-7 boundary (log: “Resuming from epoch 8”) | This was not a clean fresh 14-epoch run in the displayed process; it continued a saved state/history |
| Optimizer | AdamW | Confirmed by log/script |
| Decoder LR | `6e-4` | Param group `model.decoder.parameters()` |
| Encoder/backbone LR | `6e-5` | Param group `model.encoder.parameters()` |
| Weight decay | `2.5e-4` | Confirmed by log/script |
| Scheduler | `CosineAnnealingWarmRestarts` | `T_0=15`, `T_mult=2` |
| Scheduler stepping | Every training batch, using fractional epoch progress `epoch + (batch_idx + 1) / len(loader)` | Confirmed in saved training script construction from the conversation |
| AMP | Enabled, FP16 autocast on CUDA | Loss computed after FP32 casts of output logits |
| GradScaler | Enabled using `torch.amp.GradScaler("cuda", enabled=True)` | Training script |
| Seed | 2026 | Python/NumPy/PyTorch/CUDA seeding |
| Pretrained initialization | `False` | No pretrained encoder weights loaded |
| Input scaling | Convert to `float32`, divide source pixels by `65536.0`, clip to `[0,1]` | `dataset.py` currently visible in GitHub source; the model sample check recorded values in approximately `1e-4` to `4e-3` range |
| Label conversion | Source `1 → background 0`; source `2 → flood 1`; other pixels become `-1` | Dataset implementation |
| Training augmentation | Random horizontal flip (p=0.5), vertical flip (p=0.5), and orthogonal 90/180/270-degree rotation when selected (p=0.5) | Dataset implementation; synchronized image/label transform |
| Validation/test augmentation | Off | Dataset created with `augment=False` |
| Loss | `UnetMambaLoss` | Main output: soft cross entropy + Dice, equal weights; auxiliary output: soft cross entropy |
| Soft CE smoothing | 0.05 | `src/losses/useful_loss.py` |
| Dice smoothing | 0.05 | `src/losses/useful_loss.py` |
| Auxiliary loss weight | 0.4 | `main_loss + 0.4 * aux_loss` |
| Ignore index | -1 | Confirmed config/code/logs |
| Gradient clipping | Not used in the saved training script | Do not report clipping as applied |
| Early stopping | Not used | Fixed epoch cap; all configured epochs proceeded |
| Model-selection metric | Validation macro mIoU over background and flood classes | Best checkpoint was saved when validation mIoU improved |
| Threshold | Argmax over 2-class logits | No separate threshold tuning reported |
| TTA | False / not used | Config states `TTA=False`; evaluation code used direct forward+argmax |
| Checkpoints | `best.pt`, `last.pt`, per-epoch checkpoints during training; intermediate `epoch_*.pt` files were later deleted after the backup ZIPs were created | Current Kaggle directory was cleaned; only `best.pt` and `last.pt` were kept, and both were packed into the large ZIP |

## Planned versus actual

- Initial plan: up to 20 epochs; later the runtime target was discussed as 17 epochs. The successful saved script was ultimately changed to **14 epochs**. Do not report the planned 17 or 20 as actual.
- The standalone `unetmamba/configs/unetmamba_config.py` currently visible on GitHub sets `STRIDE=512`, `MAX_EPOCHS=20`; that file was not synchronized with the actual training script. For the final checkpoint, actual stride was 256 and successful max epochs was 14.
- A first attempt using the reference `selective_scan_ref` path ran out of memory. That attempt is diagnostic only and must not be combined with the final results. Once the compiled kernel import was required and `fvcore` restored, the successful resumed run completed on the T4.
- The earlier benchmark is not a training result: 1 warm-up + 5 measured real batches at batch size 4; its estimated training-only runtime was ~2.954 hours for 30 epochs and peak benchmark VRAM ~4.331 GB. A separate smoke test recorded peak VRAM ~7.91 GB for one forward/backward/optimizer step. Neither value is the measured peak GPU memory of the successful full 14-epoch training run.

---

# C. Dataset, split and patch verification

## C1. Dataset

- Dataset root used by Kaggle: `/kaggle/input/datasets/leviosan/floodplanet/FloodPlanet`
- `build_records` found **366** paired PlanetScope (`PS/*.tif`) image chips and `labels/*.tif` label chips across **19** event folders.
- The model in this implementation reads **PlanetScope four-band input (RGB+NIR)**. The broader FloodPlanet resource contains PlanetScope, Sentinel-1, and Sentinel-2 data, but this UNetMamba loader reads the PlanetScope `PS` folder only. Do not claim this trained model fused S1 and S2.
- Source imagery described in the project notes: approximately 1024×1024 at ~3 m spatial resolution.

## C2. Event split actually used

**Train — 14 events (208 records):** Colombia, Ghana, Nigeria, Paraguay, Somalia, Spain, US-Alabama, US-Arkansas, US-Carolina, US-Kansas, US-Nebraska, US-Oklahoma, US-Texas, Uzbekistan.

**Validation — 2 events (114 records):** Bolivia, US-Dakota.

**Test — 3 events (44 records):** Bangladesh, Cambodia, Nepal.

| Partition | Events | Records/chips | Patches | Batches (batch size 4) |
|---|---|---:|---:|---:|
| Train | 14 | 208 | 3,328 | 832 |
| Validation | 2 | 114 | 1,824 | 456 |
| Test | 3 | 44 | 704 | 176 total (Bangladesh 60, Cambodia 80, Nepal 36) |
| Total | 19 | 366 | 5,856 | — |

Seed: **2026**. The script explicitly checked event lists for equality and checked train/validation/test sets for intersection. The three final test events were excluded from training and validation and were evaluated only after model selection.

## C3. Patch construction

- Patch: 512×512; stride: 256 pixels (nominal half-patch shift / 50% overlap between adjacent full patches).
- The dataset implementation generates full crops plus right-edge, bottom-edge, and bottom-right partial crops. Partial image patches are zero-padded to 512×512; their corresponding labels are padded with `ignore_index=-1`, so padded target pixels do not contribute to loss or metrics.
- Patch counts above are actual logged values, not estimates.
- Because the evaluation script scores patches independently and sums their confusion counts, overlap means a source pixel may be evaluated multiple times. No chip-level overlap stitching or averaging was performed before metric calculation.

---

# D. Training history and convergence

## D1. Selection result

- Total successful configured epochs: **14**.
- Best validation mIoU: **0.646375 at epoch 10**.
- Best checkpoint: `unetmamba/results/checkpoints/best.pt`.
- Last checkpoint: `unetmamba/results/checkpoints/last.pt`.
- Best epoch validation loss: **0.695003**.
- Best epoch validation flood F1/Dice: **0.721875**.
- Best epoch validation precision: **0.780820**.
- Best epoch validation recall: **0.671205**.
- Best epoch validation accuracy: **0.798941**.
- Best epoch validation specificity: **0.880177**.
- Best epoch validation **flood IoU was not separately printed** in the displayed epoch logs. Do not invent or report a back-calculated value as an official logged metric.

## D2. Actual visible history segment

The full `training_history.csv` was saved, but in the conversation the displayed detailed rows were epochs 8–14. They are listed exactly here:

| Epoch | Train loss | Val loss | Val mIoU | Val F1 | Precision | Recall | Accuracy | Specificity | Decoder LR printed | Epoch time (min) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.701654 | 0.716646 | 0.642456 | 0.728458 | 0.740515 | 0.716788 | 0.792263 | 0.840263 | 0.00026864 | 8.57 |
| 9 | 0.674457 | 0.706278 | 0.646363 | 0.730110 | 0.750644 | 0.710670 | 0.795752 | 0.849862 | 0.00020729 | 7.64 |
| 10 | 0.662746 | 0.695003 | **0.646375** | 0.721875 | 0.780820 | 0.671205 | 0.798941 | 0.880177 | 0.00015000 | 7.63 |
| 11 | 0.649031 | 0.710758 | 0.640230 | 0.710011 | 0.798878 | 0.638936 | 0.797108 | 0.897700 | 0.00009926 | 7.64 |
| 12 | 0.643530 | 0.715155 | 0.637738 | 0.704975 | 0.807612 | 0.625484 | 0.796487 | 0.905240 | 0.00005729 | 7.63 |
| 13 | 0.630371 | 0.712535 | 0.641627 | 0.709734 | 0.808100 | 0.632716 | 0.798813 | 0.904445 | 0.00002594 | 7.64 |
| 14 | 0.622735 | 0.713200 | 0.641535 | 0.708523 | 0.813313 | 0.627654 | 0.799247 | 0.908375 | 0.00000656 | 7.66 |

Epochs 8–14 consumed **54.41 minutes** according to the displayed epoch times; mean across those seven displayed epochs: **7.77 min/epoch**. This is **not the exact total wall-clock time of the whole run**, because the process resumed an existing checkpoint and the earlier epochs happened before the displayed continuation. Exact full-run duration and actual peak VRAM were not recorded in the visible output. Training loss continued to decrease through epoch 14, while the best validation mIoU occurred at epoch 10 and did not improve afterward. No early stopping was used; the configured 14-epoch run completed.

## D3. History/curve artifacts

- Training history CSV: `unetmamba/results/metrics/training_history.csv` inside the Kaggle working tree and included in the large and small downloaded packages.
- A training-curve PNG was **not generated** in the visible workflow. Curves can later be plotted from the CSV without retraining.
- Per-epoch checkpoint files `epoch_*.pt` were removed during storage cleanup after the two ZIPs had been created. `best.pt` and `last.pt` were retained.

---

# E. Final test metrics per event

These are from the final test-evaluation cell loading `best.pt` from epoch 10. No test-event training or model-selection was performed.

| Event | Records | Patches | Background IoU | Flood IoU | Two-class mIoU | Flood F1/Dice | Precision | Recall | Accuracy | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bangladesh | 15 | 240 | 0.819796 | 0.598739 | 0.709267 | 0.749014 | 0.849631 | 0.669705 | 0.857983 | 0.945135 |
| Cambodia | 20 | 320 | 0.566153 | 0.762699 | 0.664426 | 0.865376 | 0.978335 | 0.775802 | 0.818808 | 0.948280 |
| Nepal | 9 | 144 | 0.806368 | 0.876302 | 0.841335 | 0.934074 | 0.939238 | 0.928966 | 0.918358 | 0.900860 |

### Per-event confusion counts

| Event | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Bangladesh | 31,120,657 | 1,806,540 | 5,034,279 | 10,207,484 |
| Cambodia | 15,185,910 | 828,255 | 10,808,836 | 37,402,279 |
| Nepal | 9,826,212 | 1,081,383 | 1,278,176 | 16,715,605 |

The counts are accumulated from predictions on **valid patch pixels** for each event. Since overlapping patches are not reconstructed, these are sums over patch pixels and can count the same original chip pixel more than once.

---

# F. Aggregate test metrics and aggregation methodology

## F1. Event mean ± sample standard deviation

Mean and sample standard deviation (`ddof=1`) over the three event-level metric values:

| Metric | Event mean | Event SD |
|---|---:|---:|
| Background IoU | 0.730772 | 0.142723 |
| Flood IoU | 0.745913 | 0.139541 |
| Two-class mIoU | **0.738343** | **0.091969** |
| Flood F1/Dice | 0.849488 | 0.093547 |
| Precision | 0.922401 | 0.065983 |
| Recall | 0.791491 | 0.130340 |
| Accuracy | 0.865050 | 0.050150 |
| Specificity | 0.931425 | 0.026517 |

## F2. Pooled confusion-count metrics

The code also summed TN/FP/FN/TP across event-level patch evaluations and calculated metrics once from the summed counts:

| Metric | Pooled-from-patch-confusions value |
|---|---:|
| Background IoU | 0.729279 |
| Flood IoU | 0.755322 |
| Two-class mIoU | **0.742300** |
| Flood F1/Dice | 0.860608 |
| Precision | 0.945384 |
| Recall | 0.789785 |
| Accuracy | 0.852526 |
| Specificity | 0.937907 |

Summed confusion counts: **TN=56,132,779; FP=3,716,178; FN=17,121,291; TP=64,325,368**.

## F3. Exact metric definitions used by the evaluator

For each event, after flattening patch predictions and labels, target pixels equal to `ignore_index=-1` are removed. The binary confusion counts use class 1 as flood and class 0 as background:

- `IoU_flood = TP / (TP + FP + FN)`.
- `IoU_background = TN / (TN + FP + FN)`.
- `mIoU = (IoU_flood + IoU_background) / 2`; this is the macro mean over both classes.
- `F1/Dice = 2TP / (2TP + FP + FN)`; this is **flood-class F1**, not a macro average over two classes.
- `Precision = TP / (TP + FP)`.
- `Recall = TP / (TP + FN)`.
- `Accuracy = (TP + TN) / (TP + TN + FP + FN)`.
- `Specificity = TN / (TN + FP)`.
- TP/TN/FP/FN are integer confusion counts over valid patch pixels.

**Aggregation limitation:** predictions from overlapping patches were not blended or reconstructed into a single mask per original chip. Event metrics were calculated using confusion sums over each event's patches, then the event-level metrics were averaged for the `event mean ± SD`. The pooled table combines confusion counts before calculating metrics, but its denominator is still **valid patch pixels, with overlap repetitions**, not unique original-image pixels. This distinction must be kept in figure captions and comparisons.

## F4. Result files

- `unetmamba/results/metrics/unetmamba_test_metrics.csv`
- `unetmamba/results/metrics/unetmamba_test_summary.csv`
- `unetmamba/results/metrics/unetmamba_test_config.json`

These were saved in the Kaggle working tree and included in the two downloaded ZIPs. The small GitHub-results ZIP contains them under `results/unetmamba/` names.

---

# G. Qualitative results and supported limitations

## G1. Actual figures generated

Output folder in Kaggle:
`/kaggle/working/FloodPlanet-Flood-Inundation-Segmentation/unetmamba/results/unetmamba/qualitative/`

Each figure has five panels: `Input RGB | Ground Truth | UNetMamba Prediction | Predicted Flood Overlay | Prediction Errors`. Its error panel marks false positives red and false negatives blue.

Files generated:

- `Bangladesh_patch_0187_Correct_Best_IoU.png`
- `Bangladesh_patch_0002_False_Positive.png`
- `Bangladesh_patch_0176_False_Negative.png`
- `Cambodia_patch_0254_Correct_Best_IoU.png`
- `Cambodia_patch_0062_False_Positive.png`
- `Cambodia_patch_0141_False_Negative.png`
- `Nepal_patch_0010_Correct_Best_IoU.png`
- `Nepal_patch_0003_False_Positive.png`
- `Nepal_patch_0029_False_Negative.png`

Index: `unetmamba/results/unetmamba/qualitative_index.csv`
Contact sheet: `unetmamba/results/unetmamba/UNetMamba_qualitative_contact_sheet.png`

The visual selection routine evaluated all 704 test patches and selected a max-flood-IoU candidate plus high false-positive and false-negative cases per event. Selected patch numbers are local to each event dataset's deterministic patch order.

## G2. What is supported by metrics

- **Nepal is the strongest of the three test events** in the reported figures: mIoU 0.841335, flood F1 0.934074, recall 0.928966.
- **Bangladesh shows lower flood recall** (0.669705) and a relatively low flood IoU (0.598739). Its patch-aggregated FN count is 5,034,279. This supports a quantitative under-detection concern, but does not alone prove which visual mechanism caused it.
- **Cambodia shows a precision/recall trade-off**: precision 0.978335 but recall 0.775802. Its background IoU is 0.566153 and FN count is 10,808,836, so its binary confusion profile indicates a substantial number of flood target pixels were predicted as background relative to its true positives. This does not by itself identify the scene type or boundary morphology responsible.
- Event-wise variation is material (event-mean mIoU SD 0.091969; flood-IoU SD 0.139541). This supports discussing event generalization rather than citing only the pooled number.

## G3. What has **not** been verified visually here

The filenames indicate how the selection code categorized examples, but the actual PNG pixels were not available in the text logs for independent visual inspection in this handoff step. Do **not** claim specific visible small-object failures, fragmentation, boundary errors, shadows, wet soil, vegetation confusion, or spectral causes until the PNGs are actually opened and reviewed. The current quantitative evidence supports event-dependent performance and the recall/precision trade-off above; it does not prove the cause.

---

# H. Previously reviewed literature

Two source workbooks and the long research notes are included in `research_sources/`. The list below preserves what was documented there. Paper-reported metrics belong to those papers, **not to our FloodPlanet experiments**. Years/titles from the workbook are reproduced; any metadata conflict or missing author field is flagged instead of silently resolved.

## H1. Recent 2024–2026 review matrix (10 entries)

1. **“A rapid high-resolution multi-sensory urban flood mapping framework via DEM upscaling.”** Listed as 2024 in the review workbook; *Remote Sensing of Environment*, 301, 113956. DOI: https://doi.org/10.1016/j.rse.2023.113956. The DOI suffix indicates 2023, so verify the publisher's publication/online year before final citation. Technique discussed: high-resolution multi-sensor urban mapping with DEM upscaling; includes flood segmentation plus GIS-oriented mapping/depth outputs. Dataset/case: 2013 Calgary flood, very-high-resolution RGB aerial imagery and DEM. Relevance: the paper emphasizes that terrain and auxiliary information can affect flood mapping; it is not a FloodPlanet experiment.

2. **Binayak Ghosh, Shagun Garg, Mahdi Motagh, Sandro Martinis. “Automatic Flood Detection from Sentinel-1 Data Using a Nested UNet Model and a NASA Benchmark Dataset.”** *PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science*, 92, 1–18 (2024). DOI: https://doi.org/10.1007/s41064-024-00275-1. Nested U-Net/UNet++-style encoder-decoder with EfficientNet-B7; NASA/IMPACT flood dataset and external geographic evaluation. Relevance: event/geographic holdouts and U-Net-family comparisons. Not FloodPlanet.

3. **Ali Jamali et al. “Residual wave vision U-Net for flood mapping using dual polarization Sentinel-1 SAR imagery.”** *International Journal of Applied Earth Observation and Geoinformation*, 127, 103662 (2024). DOI: https://doi.org/10.1016/j.jag.2024.103662. WVResU-Net combines ResU-Net with wave-vision/Vision-MLP contextual components and dual-polarization Sentinel-1 features. The literature note reports 96.20% OA, 92.97% precision, 69.67% recall and 82.03% F1 on its experiment; do not compare those numbers directly with FloodPlanet because sensor/data/split differ. Relevance: global/positional feature mixing, local detail, and complexity trade-off. Not FloodPlanet.

4. **Tamer Saleh, Xingxing Weng, Shimaa Holail, Chen Hao, Gui-Song Xia. “DAM-Net: Flood detection from SAR imagery using differential attention metric-based vision transformers.”** *ISPRS Journal of Photogrammetry and Remote Sensing*, 212, 440–453 (2024). DOI: https://doi.org/10.1016/j.isprsjprs.2024.05.018. Uses pre-/post-flood SAR pairs, a weight-sharing Siamese ViTAEv2 backbone, temporal change attention and fusion; S1GFloods with 5,360 image pairs and 46 events. Relevant to temporal change and SAR ambiguity; it requires paired SAR inputs and is not the same single-PS-image setup used here. Not FloodPlanet.

5. **Mulham Fawakherji et al. “DeepFlood for Inundated Vegetation High-Resolution Dataset for Accurate Flood Mapping and Segmentation.”** *Scientific Data*, 12, 271 (2025). DOI: https://doi.org/10.1038/s41597-025-04554-3. Dataset paper for high-resolution aerial/UAV/SAR imagery with detailed inundated-vegetation labels. Relevant as evidence of difficult vegetation cases and label needs; dataset must not be substituted into our experiments. Not FloodPlanet.

6. **Zhijie Zhang et al. “Assessing Inundation Semantic Segmentation Models Trained on High- versus Low-Resolution Labels using FloodPlanet, a Manually Labeled Multi-Sourced High-Resolution Flood Dataset.”** *Journal of Remote Sensing*, 5, 0575 (2025). DOI: https://doi.org/10.34133/remotesensing.0575. This is the selected base paper/dataset foundation: FloodPlanet's 366 manually labelled 1024×1024 chips at ~3 m, 19 global flood events (2017–2020), and comparisons involving PlanetScope/Sentinel-1/Sentinel-2/label resolution with event-level evaluation. The review workbook reports mean PlanetScope U-Net IoU 0.691 (SD 0.227) in the paper. That is the paper's result, not our matched U-Net result and not our UNetMamba result.

7. **Enrique Portalés-Julià, Gonzalo Mateo-García, Luis Gómez-Chova. “Understanding flood detection models across Sentinel-1 and Sentinel-2 modalities and benchmark datasets.”** *Remote Sensing of Environment*, 328, 114882 (2025). DOI: https://doi.org/10.1016/j.rse.2025.114882. Multidataset cross-modality evaluation and dual-stream multimodal U-Net using public datasets including Kuro Siwo, WorldFloods, UNOSAT, S1S2Water and Sen1Floods11. Relevant to generalization and modality-aware evaluation. Not a FloodPlanet experiment according to the review notes.

8. **Binbin Wang, Zijie Chen, Hailin Zou, Anran Yuan, Yuanyuan Pan, Hongfei Guo, Jianqing Li. “FM-Mamba: end-to-end non-causal Mamba-based network for efficient flood mapping.”** *Scientific Reports*, 16, 24857 (2026). DOI: https://doi.org/10.1038/s41598-026-56046-y. The workbook describes a bidirectional/non-causal Mamba encoder and context-aware decoder with channel/global-context enhancement and high-resolution lateral connections; datasets include Sen1Floods11 and S1GFloods, not FloodPlanet. The note reports 3.93M parameters and cautions about degradation in dense built-up, mountainous and vegetated scenes. Relevance: Mamba is already an active flood-mapping direction; the extension must solve a specific limitation instead of claiming novelty for using Mamba.

9. **Zhengguang Zhao, Ruixin Zhang, Haoran Guo, Jun Zhang, Yaohui Liu, Xiaoxian Chen, Chunlei Wang. “FloodSeg: A Shift and Sequence-Shuffle Based Mamba-CNN for Flood Segmentation Using Remote Sensing Images.”** *ISPRS International Journal of Geo-Information*, 15(7), 279 (2026). DOI: https://doi.org/10.3390/ijgi15070279. A Mamba-CNN encoder-decoder with spatial Shift and sequence-shuffle modules; benchmarks on the Kaggle Flood Detection Dataset and FloodNet, not FloodPlanet. The workbook reports mIoU 81.85% and 91.21% on those two respective benchmarks. It explicitly discusses imprecise boundaries, fragmented masks and confusion with shadows/dark surfaces/wet soil, and compares with UNetMamba/VMamba/RS3Mamba among other methods. This is important novelty-risk evidence: a plain Shift/sequence-shuffle copy or a general “Mamba for flood segmentation” claim would overlap prior work.

10. **Li Li, Yu Zhao, Shengwu Qin, Dianqi Pan, Jiquan Zhang, Anglin Li, Qiandong Hu. “PhysWRNet: A physics-guided deep learning framework for flood inundation mapping with SAR and hydrodynamic simulations.”** *Journal of Hydrology*, 665, 134662; DOI: https://doi.org/10.1016/j.jhydrol.2025.134662. The review workbook classifies it in the recent 2026 literature set. It combines a physics-guided WVResU-Net, Sentinel-1 VV/VH scattering features and an HEC-RAS flood-probability prior, evaluated for a 2024 Dongting Lake event. The review notes report 68.3% boundary-F1 improvement, 48.6% MAE reduction and 90.2% overall accuracy in that experiment. It relies on hydrodynamic inputs not confirmed to be available within our current FloodPlanet-only inputs, so direct reproduction is not presently scoped.

## H2. Earlier papers in the 2020–2026 review matrix (overlapping recent papers are cross-referenced above)

11. **Varun Tiwari et al. “Flood inundation mapping—Kerala 2018; Harnessing the power of SAR, automatic threshold detection method and Google Earth Engine.”** *PLOS ONE*, 15(8), e0237324 (2020). DOI: https://doi.org/10.1371/journal.pone.0237324. Sentinel-1 VV, Lee filtering and Otsu thresholding/time comparisons; the paper reports OA values 94.3% and 94.1% for two flood dates. Classical low-compute baseline but region-focused and without learned semantic context. Not FloodPlanet.

12. **Bai et al. “Enhancement of Detecting Permanent Water and Temporary Water in Flood Disasters by Fusing Sentinel-1 and Sentinel-2 Imagery Using Deep Learning Algorithms: Demonstration of the Sen1Floods11 Benchmark Dataset.”** *Remote Sensing*, 13, 2220 (2021). DOI: https://doi.org/10.3390/rs13112220. BASNet with ResNet-34, S1/S2 fusion, focal loss, multi-scale losses and refinement; Sen1Floods11. The review records mIoU 52.99%, IoU 52.30%, OA 92.81% for the reported experiment. Not FloodPlanet.

13. **Goutam Konapala, Sujay V. Kumar, Shahryar Khalique Ahmad. “Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning.”** *ISPRS Journal of Photogrammetry and Remote Sensing*, 180, 163–173 (2021). DOI: https://doi.org/10.1016/j.isprsjprs.2021.08.016. U-Net across 32 S1/S2 feature combinations and elevation/indices; Sen1Floods11. Review notes report median F1 ~0.62 for S1 alone, ~0.73 for S1+elevation, and ~0.90 for one HSV-based S2 setup. Supports the view that input modality matters, but not a FloodPlanet result.

14. **Chuan Xu et al. “SAR image water extraction using the attention U-net and multi-scale level set method: flood monitoring in South China in 2020 as a test case.”** *Geo-spatial Information Science*, 25(2), 155–168 (2022). DOI: https://doi.org/10.1080/10095020.2021.1978275. Attention U-Net generates a coarse water mask, then multi-scale level-set optimization/refinement and DEM-based shadow removal. Relevance: boundaries and post-processing; added engineering complexity. Not FloodPlanet.

15. **Zhouyayan Li, Ibrahim Demir. “U-net-based semantic classification for flood extent extraction using SAR imagery and GEE platform: A case study for 2019 central US flooding.”** *Science of the Total Environment*, 869, 161757 (2023). DOI: https://doi.org/10.1016/j.scitotenv.2023.161757. Modified U-Net with input/terrain experiments (VV/VH, slope, DEM, HAND); case study is one central-US flood. The review flags narrow river channels as a limitation. Not FloodPlanet.

16. **Marc Wieland, Sandro Martinis, Ralph Kiefl, Veronika Gstaiger. “Semantic segmentation of water bodies in very high-resolution satellite and aerial images.”** *Remote Sensing of Environment*, 287, 113452 (2023). DOI: https://doi.org/10.1016/j.rse.2023.113452. U-Net and DeepLabV3+ variants across 1,120 very-high-resolution satellite/aerial images and sensor sources; U-Net with MobileNetV3 performed strongly, with NIR/slope/augmentation benefits. It includes normal and flood water, not a flood-only FloodPlanet study.

## H3. Additional reviewed/discussed work: DeepSARFlood

**Nirdesh Kumar Sharma and Manabendra Saharia. “DeepSARFlood: Rapid and automated SAR-based flood inundation mapping using vision transformer-based deep ensembles with uncertainty estimates.”** *Science of Remote Sensing*, 11, 100203 (2025). DOI: https://doi.org/10.1016/j.srs.2025.100203. Available project PDF in the research materials. Uses Sen1Floods11 and additional global flood-event data, with CNN and ViT models, weak labels generated from concurrent optical imagery, multitask learning, model soups, a gain-based selection algorithm and deep-ensemble uncertainty. The paper reports IoU around 0.72 on Sen1Floods11 and operational processing under 40 seconds for a 1°×1° area. It does not use our FloodPlanet experiment split. It is a source of candidate methods, not our base paper.

## H4. Additional FloodPlanet/GFM work discussed in the longer notes

- **“Assessing Geo-Foundational Models for Flood Inundation Mapping: Benchmarking models for Sentinel-1, Sentinel-2, and PlanetScope.”** Discussed in the notes with arXiv link `https://arxiv.org/abs/2511.01990`. The notes state the work evaluated several existing foundation/segmentation families on FloodPlanet, including U-Net, Attention U-Net, DeepLabv3+, Prithvi, Clay, DOFA, UViT and TransNorm. Exact author/year/version metadata should be verified against its arXiv record before a formal bibliography is finalized. Its key relevance is a **do-not-duplicate warning**: simply applying a GFM/Transformer or one of these named models to FloodPlanet is not by itself an unexplored contribution.
- **Saurabh Kaushik, Lalit Maurya, Beth Tellman. “Prithvi-Complimentary Adaptive Fusion Encoder (CAFE): unlocking full-potential for flood inundation mapping.”** CVF/WACV Workshops, CV4EO, 2026, pp. 1435–1444; arXiv: https://arxiv.org/abs/2601.02315. The notes classify it as a workshop/conference paper, not the journal paper required by the instructor for the qualifying base paper. It combines a pretrained Prithvi/GFM branch with a CNN/local-detail branch and attention-based adaptive fusion and includes FloodPlanet work. The notes state the FloodPlanet experiment uses Sentinel-2 imagery paired with PlanetScope-derived labels; do not assume it is a full PlanetScope+S1+S2 three-way input fusion experiment.

---

# I. Previously discussed extension candidates (none selected)

The literature notes explicitly say these are candidates, not final decisions. Complexity and cost descriptions below are **planning estimates**, not measured outcomes.

## I1. Candidates arising from DeepSARFlood

### 1) FloodPlanet-adapted multi-task learning (MTL)

- **Concept:** retain flood segmentation and add an auxiliary water-related target, with MNDWI from aligned Sentinel-2 information listed as a possible target if availability and alignment are confirmed.
- **Problem hypothesis:** an auxiliary water-related prediction may improve shared representations around ambiguous flood/water pixels.
- **Literature evidence:** DeepSARFlood reports an MTL ablation improvement from IoU 0.682 to 0.712 on its own benchmark; this is not a FloodPlanet result.
- **Risks:** MNDWI and manual flood labels are not identical concepts; auxiliary supervision can conflict with the target. MTL itself is not novel. S2 input/band availability and spatial/temporal alignment must be checked using FloodPlanet only.
- **Integration/weights:** add a second prediction head and loss. Encoder and existing segmentation decoder can likely initialize from `best.pt`; the new auxiliary head needs initialization; fine-tuning is required.
- **Parameters/contract:** adds trainable auxiliary-head parameters during training; can keep inference output as the existing two-class flood mask if the auxiliary head is only a training regularizer. Requires a justified auxiliary target and evaluation ablation.
- **Complexity/compute:** medium. Likely one baseline-scale fine-tune plus the cost of auxiliary forward/loss; the exact overhead has not been benchmarked. Avoid claiming novelty based only on applying MTL to FloodPlanet.

### 2) Gain-based ensemble/model selection

- **Concept:** train or fine-tune multiple candidate models, quantify complementary per-image/event contribution, then select or combine ensemble members using a gain criterion.
- **Problem hypothesis:** models may fail differently across FloodPlanet regions/events or sensing conditions.
- **Literature evidence:** distinctive gain-based selection is part of DeepSARFlood; no direct FloodPlanet-specific application was found in the limited review, but absence from that search is not proof of novelty.
- **Risks:** ensemble-selection can overfit the small validation-event set; multiple models raise training/inference cost; this is an ensemble/selection strategy rather than a single architectural module.
- **Weights/contract:** can reuse `best.pt` as one member, but other candidates require more trained weights. The ensemble can output the same 2-class logits/mask, but inference memory and latency rise.
- **Complexity/compute:** high relative to one single-model run; roughly multiple baseline runs. Not the obvious fit if the goal is a single architectural extension under a tight budget.

### 3) Uncertainty-guided difficult-region / boundary analysis and refinement

- **Concept:** obtain an uncertainty/confidence map; inspect uncertain regions, especially boundary regions, and optionally introduce a targeted refinement mechanism.
- **Problem hypothesis:** identify uncertain flood edges and difficult pixels rather than optimize only global mIoU.
- **Literature evidence:** DeepSARFlood discusses uncertainty along boundaries; the prior notes say no direct DeepSARFlood uncertainty-guided boundary-refinement implementation on FloodPlanet was found, but a wider boundary/refinement search is required.
- **Risks:** boundary refinement is heavily researched. A generic edge module is not sufficient novelty. Actual UNetMamba images have not been visually inspected in this handoff; the need must be grounded in those images and a boundary-specific metric before selecting this direction.
- **Weights/contract:** uncertainty from logits can be estimated without changing the model, but analysis alone is not an architectural extension. A trainable refinement head would add parameters and would require fine-tuning; it could be placed after the decoder while keeping the final 2-class mask contract.
- **Complexity/compute:** low for a post-hoc uncertainty analysis; medium for a small trainable refinement head; potentially higher if uncertainty requires an ensemble. Cost is not measured.

### 4) Model-soup optimization

- **Concept:** train several versions of the same architecture under controlled recipes/loss variants and average weights where valid.
- **Evidence:** DeepSARFlood reports model-soup improvement from IoU 0.712 to 0.722 after MTL on its own benchmark.
- **Risks:** model soups are not new as a technique; usefulness on FloodPlanet is unknown and the label distribution differs from the original benchmark.
- **Weights/contract:** no new inference branch or output head in a model soup; can initialize variants from `best.pt`, but multiple fine-tunes/checkpoints are needed. Output contract is unchanged.
- **Complexity/compute:** medium-to-high because several training variants are needed; inference cost is close to one model after a successful soup. This is an optimization/training recipe rather than the clearest architectural extension.

## I2. Candidates arising from the FloodPlanet / GFM review

### 5) Multimodal late fusion (PlanetScope + Sentinel-1 + Sentinel-2)

- **Concept:** separate sensor branches and late feature fusion before segmentation.
- **Problem hypothesis:** complementary sensors may resolve spectral ambiguity or conditions where the optical input is weak.
- **Prior work caveat:** FloodPlanet is already used in multimodal/foundation-model work; the GFM/CAFE notes discuss adaptive fusion and CAFE has already combined a GFM branch and CNN branch on FloodPlanet. Simply proposing “multisensor fusion” is not automatically novel. CAFE’s own S1 addition/cloud-aware fusion was described as a future direction, which also makes the concept higher-risk for novelty.
- **Current project fit:** the current UNetMamba model actually reads PlanetScope PS four-band data only. FloodPlanet contains S1/S2 observations, but current alignment/loading of the other sensors is not implemented in this model.
- **Weights/contract:** the existing PS encoder might be retained as one branch; new S1/S2 branches and fusion layers would add trainable parameters and require aligned data prep, fine-tuning and controlled ablation. Could preserve output mask shape.
- **Complexity/compute:** high; more data paths/encoders, alignment work and memory. Cost unmeasured; not a good choice without strong evidence and a carefully defined input protocol.

### 6) Different/task-specific decoder

- **Concept:** retain the core encoder but change the decoder/feature fusion mechanism for flood-specific boundaries/details.
- **Literature basis:** the GFM review notes mention decoder design as a possible remaining area, but that is not proof of novelty. FloodSeg 2026 already adds shift and sequence-shuffle modules targeting boundary/context behavior.
- **Weights/contract:** encoder can reuse `best.pt`; modified decoder layers need new initialization and fine-tuning. Output can remain `(B,2,H,W)`.
- **Complexity/compute:** medium-to-high depending on the module. Must define a precise module and compare against literature before committing. No named final decoder candidate was selected.

### 7) Parameter-efficient tuning / GFM + CNN hybrid fusion

- **Status:** explicitly ruled out as a standalone novelty claim in the project notes because Prithvi-CAFE already uses adapters and CNN/attention fusion and is evaluated on FloodPlanet. Do not select “add adapters to Prithvi” or a generic GFM+CNN hybrid as our extension without a sharply differentiated hypothesis.

### 8) Add SAR to the CAFE-style optical/GFM branch; cloud-aware adaptive fusion

- **Concept:** use S1 when S2 is cloudy or low quality, possibly dynamically weighting modalities.
- **Status:** recorded as a candidate prompted by CAFE's cloud/cloud-shadow failures and future-scope comments, but not as a final direction. CAFE authors already suggest SAR addition, and adaptive fusion is an active research area. Needs broader verification. High complexity. Does not reuse UNetMamba in a direct way unless integrated as a branch within the UNetMamba model.

### 9) Event/geographic generalization as a research question

- **Concept:** test whether an improvement persists when a whole event/region is held out, rather than relying on random/image-level splits.
- **Status:** endorsed as a scientifically useful evaluation concern by the literature notes, but **not by itself an architectural extension**. Our current UNetMamba split already reserves Bangladesh/Cambodia/Nepal for testing; the matched U-Net and extension must use the same fixed split.

### 10) Physics-guided or hydrodynamic prior

- **Concept:** use physically informed inundation prior/probabilities with SAR and a learned segmentation model, as in PhysWRNet.
- **Status:** literature inspiration only. It depends on HEC-RAS/hydrodynamic inputs not confirmed in our permitted FloodPlanet input package; it is not a ready-to-implement candidate under the current dataset-only constraint. High integration cost.

### 11) Small-object handling, fragmentation, or spectral ambiguity

- **Status:** literature-supported problem categories (especially in FloodSeg and the GFM/CAFE notes), not yet proven as the cause of the current UNetMamba errors. The visual images must be inspected and an appropriate analysis/metric added before choosing one of these as a hypothesis. Do not claim these as observed limitations merely because the literature discusses them.

**No extension has been chosen.** The next conversation should compare these candidates against the actual qualitative PNGs, available FloodPlanet bands/labels, recent literature and time budget, then choose exactly one defensible direction.

---

# J. Research-protocol inconsistencies and controls before extension

1. **Metric definition consistency is mandatory.** UNetMamba reports both flood IoU and macro two-class mIoU. Its headline event mean mIoU (0.738343) is not the same metric as flood-IoU-only summaries that may have been used in the matched U-Net CSV. For direct table comparisons, compare the same definition: e.g., U-Net flood IoU versus UNetMamba flood IoU, or recompute both macro two-class mIoU using the same evaluator.

2. **Overlap-aware evaluation needs a shared rule.** The current UNetMamba evaluator sums patch confusions for patches with stride 256, so overlapping valid pixels can be counted more than once. Before final U-Net/UNetMamba/extension claims, verify the matched U-Net evaluation uses the same patch generation, ignore-mask handling and aggregation. Prefer one shared evaluation implementation. If changing to chip-level reconstructed masks, rerun evaluation for all models from frozen checkpoints; that requires no retraining but will change metric values.

3. **U-Net LOEO numbers are a separate experiment.** Do not compare old LOEO U-Net numbers to fixed-split UNetMamba. The user’s project notes report a *matched fixed-split U-Net* result of IoU `0.7326 ± 0.1222`, F1/Dice `0.8417 ± 0.0852`, precision `0.8539 ± 0.1036`, recall `0.8345 ± 0.1047`, accuracy `0.8491 ± 0.0047`, specificity `0.8338 ± 0.1103`; event flood-IoUs listed were Bangladesh `0.5915`, Cambodia `0.8036`, Nepal `0.8028`. These values are in earlier project notes; their exact CSV was not part of this tool handoff. Verify their metric names/definitions before placing them next to UNetMamba. They should not be directly compared to the old LOEO experiment.

4. **Do not overstate the advantage.** On the metrics supplied, UNetMamba’s event-mean flood IoU is 0.745913 and event-mean macro mIoU is 0.738343. The matched U-Net’s headline `IoU=0.7326 ± 0.1222` appears to be flood IoU, not macro two-class mIoU, so those two numbers are only comparable if both definitions and aggregation methods are verified. Precision/recall trade-offs also differ. Use a matched evaluator before drawing superiority conclusions.

5. **The base config is stale relative to the run.** `unetmamba/configs/unetmamba_config.py` on current GitHub `main` states stride 512 and 20 max epochs. The final successful training used stride 256 and max epochs 14. Update the config/README during later repository maintenance only after documenting the frozen experiment; do not silently claim the existing config exactly reproduces the checkpoint.

6. **Selective-scan build patch is not reflected in public main.** Current public `main` setup.py still contains C++20 and `/Zc:preprocessor` flags. Reapply the known working local patch before attempting a T4 build and commit it from the local PC after reviewing the diff. Do not include the compiled `.so` in Git.

7. **No exact checkpoint commit hash.** Record a commit hash only when the final local worktree is staged/committed and training source/checkpoint provenance has been reconciled. Do not retroactively assign a clone hash to `best.pt` without evidence.

8. **Actual model input is PlanetScope only.** The project title mentions multi-sensor imagery and the dataset includes S1/S2, but the current `FloodPlanetDataset.load_image` reads the 4-band PS image. An extension using other sensors would be a genuine input-pipeline/model change and must be documented as such.

9. **Test set integrity:** test events were not used during training, validation, or best-checkpoint selection. After baseline freeze, use them only for the final standardized comparison and report all metric revisions transparently. Do not use test metrics to pick hyperparameters.

10. **No independent evaluation script was saved.** Test evaluation and qualitative visualization were run from Kaggle notebook cells (“22 - UNetMamba Test Evaluation” and “24 - UNetMamba Qualitative Visualization”). The current large ZIP includes the training script but not a standalone evaluation script. Preserve the notebook or reconstruct a common evaluation script before comparing the extension.

---

# K. Required files and artifact locations

## K1. Artifacts saved in the Kaggle project tree

Project root:
`/kaggle/working/FloodPlanet-Flood-Inundation-Segmentation`

- Best checkpoint: `unetmamba/results/checkpoints/best.pt`
- Last checkpoint: `unetmamba/results/checkpoints/last.pt`
- Training history: `unetmamba/results/metrics/training_history.csv`
- Event test metrics: `unetmamba/results/metrics/unetmamba_test_metrics.csv`
- Summary metrics: `unetmamba/results/metrics/unetmamba_test_summary.csv`
- Test configuration: `unetmamba/results/metrics/unetmamba_test_config.json`
- Qualitative figure directory: `unetmamba/results/unetmamba/qualitative/`
- Qualitative index: `unetmamba/results/unetmamba/qualitative_index.csv`
- Contact sheet: `unetmamba/results/unetmamba/UNetMamba_qualitative_contact_sheet.png`
- Training script: `unetmamba/scripts/train_unetmamba.py`
- General config: `unetmamba/configs/unetmamba_config.py` (stale stride/epoch values; see Section J)
- Event split: `unetmamba/configs/event_split.json`
- Dataset loader: `unetmamba/src/dataset.py`
- Model code: `unetmamba/src/models/UNetMamba.py`, `ResT.py`, `vmamba.py`, `csm_triton.py`
- Loss source: `unetmamba/src/losses/`
- Selective scan source/build: `unetmamba/kernels/selective_scan/`

## K2. Two downloaded ZIPs (created on Kaggle)

1. **`UNetMamba_FloodPlanet_Final_Package.zip` — 320.94 MB**
   Contains `best.pt`, `last.pt`, training history, test metrics/summary/config, qualitative folder/index/contact sheet, `train_unetmamba.py`, and `configs/`. **It does not contain `unetmamba/src/` or the complete `kernels/selective_scan/` source tree**, because the cell that created it did not add those directories. The earlier backup ZIP that had copied `src/` and `setup.py` was removed in storage cleanup. Thus, this archive is checkpoint/results/training-entry/config backup, not a fully self-contained source recovery package.

2. **`UNetMamba_FloodPlanet_GitHub_Results.zip` — 10.58 MB**
   Contains lightweight training/test results, qualitative PNGs, qualitative index and contact sheet. Contains no `.pt` checkpoint or source code. Good for presentation/results staging, not code recovery.

The user reported downloading both. Do not assume files on the local PC are accessible from a new ChatGPT conversation; attach the ZIPs only if their bytes are needed. No FloodPlanet dataset should be included in any handoff.

## K3. Source/research notes bundled with this handoff ZIP

- `research_sources/flood_literature_review_2020_2026.xlsx`
- `research_sources/flood_literature_review_2024_2026.xlsx`
- `research_sources/Research_for_flood_inundation.docx`

These are the pre-existing literature notes; retain them as the authoritative record of the review, but validate bibliographic details before final paper submission.

## K4. What the next conversation should request/attach if it needs direct artifact analysis

- `UNetMamba_FloodPlanet_Final_Package.zip` (contains the actual best/last checkpoint, history, results, training script and configs, but not all model source)
- `UNetMamba_FloodPlanet_GitHub_Results.zip` (contains qualitative PNGs/CSVs/contact sheet)
- Ideally the repository checkout or at least the current GitHub source file tree plus the compatibility patch diff.
- The source review workbooks/docx are already included in this handoff ZIP.
- Do not attach the full FloodPlanet dataset.

---

# L. Recommended next steps for extension selection (do not execute yet)

1. **Freeze this baseline.** Keep `best.pt`, `last.pt`, `training_history.csv`, test metrics, test config and all nine figures. Do not train again just to reach the originally planned epoch cap; the actual run completed 14 and selected epoch 10. Store the 320.94 MB package locally in at least two locations.

2. **Inspect the actual nine qualitative PNGs.** Confirm visible FP/FN/boundary/small-target characteristics. Tie each proposed limitation to a specific image. Current logged filenames categorize cases, but detailed visible descriptions still need to be written after image inspection.

3. **Standardize evaluation before drawing model-comparison conclusions.** Verify the matched U-Net uses the same 512/256 patching, ignore-index treatment, metrics and event aggregation. Determine whether to preserve patch-confusion aggregation for all models or implement chip-level overlap blending and rerun evaluation for all frozen checkpoints. Do not retrain only because of an evaluation-method correction.

4. **Read the relevant recent papers against the evidence:** FloodSeg (2026), FM-Mamba (2026), PhysWRNet (2026), DeepSARFlood (2025), and the FloodPlanet/GFM/CAFE work. The practical novelty question is not “Can we add Mamba?” Mamba, UNetMamba, sequence/shift blocks, transformer/GFM branches, generic multimodal fusion and model soups already exist in the literature. The question should be narrowly tied to a verified, demonstrated limitation and a well-supported mechanism.

5. **Filter candidates against data and compute reality.** The current network sees PS 4-band inputs only. Additional sensors require data alignment and architectural changes. A small refinement head can reuse the current checkpoint more directly than replacing the backbone or adding multiple sensor branches. This is only a feasibility distinction, not a recommendation to pick that candidate.

6. **Use validation events only for design selection.** Keep Bangladesh/Cambodia/Nepal completely untouched until final evaluation of the candidate is frozen. Use the same fixed split and seed across the matched U-Net, UNetMamba and extension where possible.

7. **Select exactly one extension only after the above analysis.** Predefine the main hypothesis, module integration point, ablation, metrics, compute budget and decision rule. Keep extension additions minimal and measurable. Do not combine multiple techniques at once.

8. **Before final GitHub push from the local PC:** inspect `git diff`; include the actual training script and a synchronized config/README, repair the `setup.py` T4 compatibility flags, keep dataset/checkpoints/`.so` out of Git, add lightweight metrics/qualitative assets under `results/unetmamba/`, and commit with a recorded hash. The final local code commit and the checkpoint provenance should be stated separately if the model was trained before that commit.

---

## Appendix: original checkpoint output metrics (for quick copying)

**Selected checkpoint:** epoch 10, validation mIoU 0.646375.
**Test event mean:** mIoU 0.738343 ± 0.091969; flood IoU 0.745913 ± 0.139541; flood F1/Dice 0.849488 ± 0.093547; precision 0.922401 ± 0.065983; recall 0.791491 ± 0.130340; accuracy 0.865050 ± 0.050150; specificity 0.931425 ± 0.026517.
**Pooled-from-patch-confusions:** mIoU 0.742300; flood IoU 0.755322; flood F1/Dice 0.860608; precision 0.945384; recall 0.789785; accuracy 0.852526; specificity 0.937907.
**Pooled confusion counts:** TN 56,132,779; FP 3,716,178; FN 17,121,291; TP 64,325,368.
