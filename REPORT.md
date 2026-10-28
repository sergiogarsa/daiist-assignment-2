# Assignment 2 Report: when should a winter road segment be treated?

- **Name**: Sergio García
- **Student ID**: 19912
- **Email**: sgarcia.ieu2022@student.ie.edu
- **Group**:  BBADBA 5B

Every number in this report comes from files written by `uv run python main.py train`
(`artifacts/results.json`, `artifacts/dataset_audit.json`, `artifacts/predictions_*.csv`) and can be
re-checked in the dashboard (`uv run python main.py app`).

---

## 0. The decision in one paragraph (summary for the comprehension check)

A winter-maintenance duty officer looks at road camera images and decides whether to send a gritting /
ploughing truck to a road segment. I fine-tuned **yolo11n-cls** on the public **UDOT Winter Road
Conditions v1** dataset (Clear / Wet / Slush / Snow) and turned its output into one score per image,
**s = P(Snow) + P(Slush)**. On the untouched test split the model reaches **91.3 % 4-class accuracy** and
**0.991 case-level AUC**. The ImageNet zero-shot baseline reaches 0.590 AUC, and the majority class gets
55.4 % accuracy. With a wasted sortie costing **EUR 400**, an untreated snowy hour costing **EUR 1,000** and
an assumed **12 %** production prevalence, the cheapest policy chosen on validation is:

- **PASS** (no action) if s < **0.180**
- **HUMAN REVIEW** if 0.180 ≤ s < **0.998**
- **FAIL** (dispatch) if s ≥ **0.998**

On test, this costs **EUR 5,170 per 1,000 cases** (EUR 7,445/day). The alternatives cost:

- **Model alone** at t* = **0.960**: EUR 21,262 per 1,000.
- **A person reviewing every case**: EUR 24,044 per 1,000, needing 16 h/day of reviewing.
- **Doing nothing**: EUR 120,000 per 1,000.

The review band sends **18.4 %** of cases (≈265/day, **2.9 h/day, 0.49 FTE**) to the duty officer. That
fits inside the spare time of the existing 24/7 duty rota, so no hire is needed. The biggest caveats:

- The dataset is in-vehicle (dashcam) footage, not fixed roadside cameras.
- 11 % of test images are near-duplicates of training frames.
- C_FN is a scenario estimate, not a measured fact.

---

## 1. Business problem, and the cost of being wrong

### 1.1 Provisional framing (written before browsing any dataset)

- **Decision made today by looking at an image.** In winter, road authorities' duty officers watch road
  camera images and decide, segment by segment, whether the surface needs a treatment run now. The
  NYSDOT describes exactly this practice: it evaluates road conditions "by driving on roads and
  observing live cameras" for operational winter decisions (Sutter et al., arXiv:2510.06440).
- **Who decides, who acts.** The duty officer in the maintenance control room decides; a gritting/plough
  crew acts on a dispatch.
- **One operational case** = one camera snapshot of one road segment at one decision cycle (hourly).
- **Positive / negative.** *Positive* = the road surface is snow- or slush-covered (needs treatment now).
  *Negative* = dry/clear or merely wet.
- **Pass / fail.** *FAIL* = the segment fails the safety check, so a spot treatment is dispatched.
  *PASS* = no action this cycle. *HUMAN REVIEW* = the duty officer looks before deciding.
- **Why the two errors cost different amounts.** A false positive wastes a sortie: a bounded,
  well-known operating cost. A false negative leaves a hazardous surface untreated for a cycle. Its cost
  is a small probability of a crash times a large crash cost, plus delay. It is larger per case and much
  more uncertain.
- **Volume.** About 60 hot-spot camera views checked hourly, 24 h/day in winter, gives **1,440 cases/day**.
  Snow/slush is a minority of observations even in winter.

### 1.2 Refinement after the dataset audit (stated openly)

The audit (§3) showed that the images are **forward-facing in-vehicle cameras**: dashcam screenshots and
fisheye video frames from vehicles driving Utah roads. They are not fixed pole cameras. The decision and
cost logic do not change. I re-state the case as **one camera snapshot (vehicle-mounted, e.g. a
patrol/fleet vehicle, or fixed) of one segment at one hourly decision cycle**. The mismatch with fixed
roadside cameras is a production-risk limitation (§10), not something I can fix with this dataset.

**Treating Slush as positive and Wet as negative is a business choice.** Slush needs ploughing and
salting now, and it refreezes. A wet road needs no treatment *from the image alone*: wet roads near
0 °C are a black-ice risk, but that is decided from pavement-temperature sensors, which an image cannot
see. The hardest boundary is therefore exactly where the money is: **Wet vs Slush**.

---

## 2. Dataset identity

| item | value |
|---|---|
| Platform | Roboflow Universe (public), used as provided: no fork, no relabelling, official splits |
| Workspace | `udot-classification` |
| Project | `udot-winter-road-conditions` |
| Version | **1** (generated 2026-01-30) |
| Task | Image **classification**, 4 classes: Clear, Slush, Snow, Wet |
| URL | https://universe.roboflow.com/udot-classification/udot-winter-road-conditions/dataset/1 |
| License | CC BY 4.0 (as shown on the Universe page and in the export README) |
| Download format | `folder` (train/valid/test, one sub-folder per class) |
| Download verified | Yes: `train.py` downloads it through the Roboflow SDK with my API key; counts match the Universe page (4,363 / 1,261 / 621) |
| Preprocessing / augmentation in the version | none |

**Why it serves the decision.** The label is the road-surface state, which is exactly what drives the
treat / don't-treat decision. Roadside snow banks do not make an image "Snow": many Clear and Wet
images have snowy verges. It has more than three classes, and the positive group (Snow + Slush) maps
cleanly onto "needs treatment".

---

## 3. Dataset audit (before training) and what it implies

Computed by `audit()` in `train.py` → `artifacts/dataset_audit.json`, `artifacts/class_counts.png`.

### 3.1 Split and class counts

| class | train | valid | test | group |
|---|---:|---:|---:|---|
| Clear | 410 | 117 | 52 | negative |
| Slush | 762 | 203 | 107 | positive |
| Snow | 2,289 | 678 | 344 | positive |
| Wet | 902 | 263 | 118 | negative |
| **total** | **4,363** | **1,261** | **621** | |
| Snow+Slush share | 69.9 % | 69.9 % | 72.6 % | |

**Bounding-box size: not applicable.** This is a classification dataset with one label per image and no
boxes, so object-size statistics do not exist. The equivalent "what does the model see" question is
answered by image size and source below.

### 3.2 Image size and source

The filenames reveal three image sources, with very different formats:

| source (from filename) | size | Clear | Slush | Snow | Wet |
|---|---|---:|---:|---:|---:|
| `screenshot_<date>_<time>` (letter-boxed dashcam screenshots) | 3840×2160 | 272 | 836 | **3,217** | 907 |
| `frame<N>` (fisheye dashcam video frames) | 640×640 | **276** | 154 | 94 | 233 |
| `Test_Image_<N>` | 640×640 | 31 | 82 | **0** | 143 |

### 3.3 Duplicates and near-duplicates across splits

- **Exact duplicates** (byte-identical): 42 groups, 20 of them spanning two splits. None have conflicting
  labels.
- **Re-used original filenames:** 259 names appear more than once, 208 of them with *different* labels.
  These are different videos that share frame numbers (e.g. `frame11400` exists as Clear, Wet and Snow),
  so **filenames are not reliable identifiers**.
- **Near-duplicates** were measured with a 256-bit difference hash, checking each valid/test image's
  closest training image. I calibrated the cut-off by eye: up to ~24 differing bits the pair is visibly
  the same scene seconds apart in the same drive.

| max differing bits | valid images with a train near-duplicate | test images with a train near-duplicate | label agreement (test) |
|---:|---:|---:|---:|
| 0 | 1.0 % | 1.1 % | 100 % |
| 10 | 1.3 % | 1.3 % | 100 % |
| 16 | 2.9 % | 3.4 % | 95 % |
| **24 (used)** | **12.8 %** | **11.4 % (71 / 621)** | 90 % |
| 30 | 26.3 % | 24.0 % | 91 % |

### 3.4 What these findings imply

1. **Moderate class imbalance, and Clear is the thin class.** Clear has 52 test images. Its recall
   (0.77) has a 95 % Wilson interval of **0.64–0.86**, which is too wide to rank it precisely. Snow
   (344 test images) dominates every average.
2. **The split prevalence (~70 % positive) is curated, not a production rate.** I never use raw split
   counts for cost. Costs use class-conditional rates (TPR, FPR) re-weighted to an *assumed* production
   prevalence of 12 % (§7). Ignoring this would overstate the value of dispatching by roughly 6×.
3. **The split is random at image level, and drives leak across it.** 11.4 % of test images have a
   same-scene training frame. On those the model scores 97.2 % accuracy, against 90.5 % on the other 550.
   **The honest "unseen scene" accuracy is ≈ 90.5 %, not 91.3 %.** Accuracy on genuinely new cameras or
   routes is likely lower: NYSDOT reports 81.5 % on unseen cameras for a similar task.
4. **Source is confounded with class.** Snow is 97 % screenshots, Clear is 47 % fisheye frames, and
   Test_Image has no Snow at all. A model can partly learn "letter-boxed 4K screenshot ⇒ snow". It shows
   in the results: test accuracy is 94.6 % on screenshots (n = 537), but only **72.1 % on fisheye frames**
   (n = 68) and **62.5 % on Test_Image** (n = 16).
5. **Production mismatch.** In-vehicle dashcams, one US state, a lot of night footage, and screenshots
   whose filename dates (May–June 2025) are capture dates of recorded winter footage, not weather dates.
   Fixed pole cameras in another region would look different.

I did **not** repair any of this. No de-duplication, relabelling or re-splitting: the assignment requires
the official splits. Instead I quantify the effect (leakage-stratified and source-stratified accuracy)
and treat the headline as an upper bound.

---

## 4. Evaluation protocol (fixed before training)

**Two layers, kept separate:**

1. **Model performance (multiclass).** 4-class accuracy, per-class precision/recall, and the 4×4 confusion
   matrix on test.
2. **Operational decision (binary, case level).**
   - **Ground truth:** positive if the label is Snow or Slush, negative if Clear or Wet.
   - **Score:** s = P(Snow) + P(Slush) from the softmax. All four probabilities are kept for every image,
     so nothing is discarded upstream and every threshold in [0, 1] can be evaluated later.
   - **Model-only rule:** FAIL if s ≥ t, else PASS. **Ties at the boundary go to FAIL** (≥).
   - **Review rule:** PASS if s < t_low; REVIEW if t_low ≤ s < t_high; FAIL if s ≥ t_high.
     t_low ≤ t_high is enforced in code. t_low = t_high means no review band.
   - **Missed snow stays a false negative:** a Snow/Slush image with a low score is PASSed and costed as
     an FN. There is no "no prediction" escape, because a classifier always produces a score.
   - The **binary case-level confusion matrix** (TP / FN / FP / TN, plus REVIEW counts) is what gets
     costed. Multiclass mistakes *within* a group (Snow↔Slush, Clear↔Wet) cost nothing operationally.
3. **Data roles:**
   - **train** fits the weights.
   - **valid** selects the epoch, the single threshold t*, and the pair (t_low, t_high).
   - **test** is used once to report the frozen choices.

   Thresholds were never tuned on test. The dashboard's test view is labelled exploratory.
4. **Threshold-selection tie rule:** among equal-cost thresholds, pick the lower one (more treatment,
   safer). Among equal-cost pairs, pick the smaller review share, then the lower t_low.

---

## 5. Fine-tuning and baselines

### 5.1 Training configuration

| setting | value |
|---|---|
| Model / init | `yolo11n-cls.pt` (Ultralytics, ImageNet-pretrained), head reshaped to 4 classes |
| Input | 224×224; Ultralytics `classify_transforms` (resize short side, centre crop) for eval |
| Augmentation | Ultralytics `classify_augmentations`: random resized crop scale 0.35–1, h-flip 0.5, HSV jitter (h 0.015, s 0.4, v 0.3); no RandAugment, no erasing. Mild, because road texture is the signal |
| Optimiser | AdamW, lr 1e-3, weight decay 0.01, ½-epoch linear warm-up then cosine decay to 2 % |
| Epochs / batch / seed | 12 / 32 / 0 |
| Model selection | epoch with the highest **valid top-1** (ties → lower valid loss) → **epoch 11, valid top-1 0.891** |
| Working copy | Images downscaled once to longest side 640 px (originals are up to 3840×2160) so CPU training is practical. Same files, labels and splits; the network sees 224 px either way |
| Environment | CPU sandbox, 2 vCPU (AMD Ryzen 7 5800H), no GPU; Python 3.12.15, torch 2.11.0+cpu, ultralytics 8.4.173 |
| Training time | ≈ 52 min of training compute (3,133 s cumulative) |

**Why a small PyTorch loop instead of `model.train()`.** The compute environment kills any process
after about 3 minutes, and an Ultralytics epoch took longer than that. Ultralytics can only resume at
epoch boundaries. `train.py` therefore fine-tunes the same Ultralytics model with Ultralytics' own
transforms in a short loop. The loop checkpoints every ~45 s and resumes automatically, then saves
`weights/best.pt` in the standard Ultralytics checkpoint format: `YOLO("weights/best.pt")` loads it and
`check.py` confirms it. On a normal machine `main.py train` runs start to finish with no intervention.

| epoch | 1 | 3 | 5 | 7 | 9 | 10 | **11** | 12 |
|---|---|---|---|---|---|---|---|---|
| valid top-1 | 0.770 | 0.832 | 0.842 | 0.883 | 0.889 | 0.890 | **0.891** | 0.890 |
| valid loss | 0.595 | 0.434 | 0.385 | 0.332 | 0.296 | 0.287 | **0.280** | 0.282 |

The curve has flattened by epochs 9–12, so more epochs would add little (`artifacts/training_curves.png`).

### 5.2 Results on the same test split (621 images)

| model / baseline | 4-class accuracy | case AUC | TPR | FPR | EUR / 1,000 cases* |
|---|---:|---:|---:|---:|---:|
| **Fine-tuned yolo11n-cls** (t* = 0.960 from valid) | **0.913** | **0.991** | 0.909 | 0.029 | **21,262** |
| Zero-shot ImageNet yolo11n-cls (snow-class mapping) | undefined | 0.590 | 0 | 0 | 120,000 |
| Zero-shot COCO yolo11n detector (skis/snowboard ⇒ FAIL) | undefined | 0.514 | 0 | 0 | 120,000 |
| Trivial: majority class "Snow" (⇒ always FAIL) | 0.554 | 0.5 | 1 | 1 | 352,000 |
| Trivial: predict nothing (⇒ always PASS = do nothing) | n/a | 0.5 | 0 | 0 | 120,000 |

\*Model-only policies at baseline costs and 12 % prevalence (§7).

**How the zero-shot baselines were handled honestly.**

- **ImageNet classifier (the corresponding `-cls` pretrained model).** ImageNet has no
  Clear/Wet/Slush/Snow classes, so 4-class accuracy is undefined. I mapped, *a priori* and without tuning,
  six ImageNet classes that imply a snowy scene: *snowplow, snowmobile, ski, dogsled, bobsled, alp*. The
  score is the sum of their probabilities. Its top-1 labels on test are alp, snowplow, snowmobile and
  traffic_light, so it "sees winter" but not the road surface (AUC 0.59). At the cost-optimal threshold
  chosen on valid, it never dispatches, so it collapses to "do nothing".
- **COCO detector (asked for by the brief).** COCO's 80 classes include no road-surface condition. Its
  detections are cars, traffic lights, trucks and people, in similar proportions across all four classes.
  The only snow-related COCO classes are *skis* and *snowboard*. Under the stated mapping "a skis/snowboard
  box with conf ≥ 0.25 ⇒ FAIL", it never fires on test (AUC 0.51), which is again identical to doing
  nothing. **Calling the COCO model a road-condition baseline would be an undefined comparison.** I report
  its operational behaviour and state that it is not a meaningful competitor.

**The fine-tuned model beats both baselines by a wide margin** on every layer: accuracy, AUC and cost.

---

## 6. Per-class evaluation and failure modes

### 6.1 Per-class results (test; positive group = Snow + Slush)

| class | precision | recall (95 % CI) | F1 | support |
|---|---:|---:|---:|---:|
| Clear | 0.800 | 0.769 (0.64–0.86) | 0.784 | 52 |
| Slush | 0.814 | 0.897 (0.83–0.94) | 0.853 | 107 |
| Snow | 0.982 | 0.971 (0.95–0.98) | 0.977 | 344 |
| Wet | 0.858 | 0.822 (0.74–0.88) | 0.840 | 118 |
| **overall** | accuracy **0.913** | macro-F1 0.864 | | 621 |

Valid accuracy is 0.891; per-class valid recalls are Clear 0.795, Slush 0.842, Snow 0.945 and Wet 0.833.

**Test confusion matrix (rows = true, columns = predicted)** (`artifacts/confusion_multiclass_test.png`)

| true \ pred | Clear | Slush | Snow | Wet |
|---|---:|---:|---:|---:|
| Clear | 40 | 2 | 2 | 8 |
| Slush | 0 | 96 | 4 | 7 |
| Snow | 4 | 5 | 334 | 1 |
| Wet | 6 | 15 | 0 | 97 |

### 6.2 Case-level confusion matrix (test, positive = Snow/Slush)

| frozen policy | TP (FAIL, pos) | FN (PASS, pos) | REVIEW pos | FP (FAIL, neg) | TN (PASS, neg) | REVIEW neg |
|---|---:|---:|---:|---:|---:|---:|
| Model only, t* = 0.960 | 410 | 41 | — | 5 | 165 | — |
| Model + review [0.180, 0.998) | 340 | 2 | 109 | 0 | 140 | 30 |

### 6.3 What carries the headline and where it fails

- **Snow carries the headline.** It is 55 % of test and has recall 0.97. Remove it and the remaining
  three classes average ~0.83 recall.
- **The costly confusion is Wet ↔ Slush.**
  - 15 of 118 Wet images are called Slush. That is a potential false dispatch.
  - 7 of 107 Slush images are called Wet. That is a potential missed treatment.

  This is the business boundary in §1.2, and the model is weakest exactly there.
- **Source matters more than class** (§3.4). Fisheye frames and Test_Image are much worse (72 % and 63 %).
  Wet fisheye frames reach only 47 % (9/19), so the strong screenshot results do not transfer to the
  minority camera types.
- **Concrete errors at t* = 0.960** (`artifacts/error_examples.png`, thumbnails in `artifacts/errors/`):
  - **FN, true Slush → predicted Wet (s = 0.14–0.31):** night-time fisheye frames
    (`test/Slush/frame8600…`, `frame29400…`, `frame5000…`). Headlight glare on slush looks like a wet
    sheen.
  - **FN, true Snow → predicted Clear (s = 0.12):** a dark, letter-boxed night screenshot
    (`test/Snow/screenshot_2025-05-29_08-53-26…`).
  - **FP, true Clear → predicted Slush (s = 0.995):** `test/Clear/screenshot_2025-06-04_09-59-23…`. A
    bare carriageway between thick snow banks: the model reacts to roadside snow.
  - **FP, true Wet → predicted Slush (s = 0.97–0.98):** wet streets in snowy towns
    (`test/Wet/screenshot_2025-05-29_11-28-03…`, `test/Wet/frame16800…`).
  - **Where FNs come from at t* (41 total):** 19 Snow (17 screenshots, 2 fisheye frames) and 22 Slush
    (10 screenshots, 9 fisheye frames, 3 Test_Image). Fisheye frames are 11 % of test but 27 % of misses.
- **Small samples.** Clear has 52 test images and Test_Image has 16. Statements about those subgroups
  are indicative only.

---

## 7. Cost model

### 7.1 Unit and accounting

One unit = **one camera check of one road segment at one hourly decision cycle**. Costs are in euros
per case, as **excess cost over a perfect decision** (regret):

- **C_TP = C_TN = 0.** Treating a truly snowy segment is paid under *every* policy. Even "do nothing"
  treats it later when a patrol finds it, so it does not discriminate between policies. Only the *excess*
  is counted, which avoids double counting.
- **C_FN is priced for one cycle (1 hour).** If the road is still snowy at the next hourly check, that is
  a new case with its own chance of being caught. Counting several hours per FN would double count.

### 7.2 Figures, with their basis

| item | value | basis | type |
|---|---:|---|---|
| **C_FP**: wasted spot-treatment sortie | **EUR 400** | Cheshire East charges **£8.47 per km per gritting run** (operational) [1]. Norfolk reports **≈ £49,000 per full run** over 49 routes on 2,200 miles, i.e. ≈ £1,000 per ≈ 72 km route [2]. A ~10 km spot treatment plus ~20 km dead-heading ⇒ ≈ £250–420 ⇒ EUR 300–500 | sourced inputs, scenario combination |
| **C_FN**: one untreated snowy/slushy cycle | **EUR 1,000** | 10 km × 600 veh/h × 1 h = 6,000 veh-km; **+3 crashes per million veh-km** when untreated (between Finland's ≈ +20 % injury crashes under reduced salting and the ≈ 8× before/after-salting ratio in Kuemmel & Hanbali 1992, an industry-sponsored study) [3]; average crash **EUR 40,000** (mix of property-damage-only crashes and injuries valued with the EU external-cost handbook: MAIS1 ≈ EUR 12.9k, MAIS2 ≈ EUR 189k, fatality ≈ EUR 4.3M) [4] ⇒ EUR 720; + EUR 150 delay (FHWA: arterial speeds fall 30–40 % on snowy/slushy pavement) [5]; + EUR 130 harder de-icing once snow is compacted | **scenario assumption** (most uncertain number) |
| C_TP, C_TN | 0 | regret accounting (§7.1) | definition |
| Production prevalence π | **12 %** | share of winter-season hourly checks showing snow/slush; the split's ~70 % is curated | **scenario assumption** |
| Cases per day | **1,440** | 60 hot-spot views × 24 hourly checks | scenario assumption |
| Review time per case | **40 s** | open the image, glance at a neighbouring camera and the pavement sensor | scenario assumption |
| Reviewer cost | **EUR 40 / h** ⇒ **EUR 0.44 per case** | Eurostat 2024 average hourly labour cost **EUR 37.3** (euro area), **EUR 33.5** (EU) [6], rounded up for 24/7 shift premia | sourced + adjustment |
| Productive review hours per reviewer-day | 6 of 8 | breaks and other duties | scenario assumption |
| Spare review capacity, no new hire | **6 h/day** | 3 duty-officer shifts × ~2 h | scenario assumption |
| Reviewer error | 5 % miss, 5 % false alarm | the hard cases are the ones sent to review; tested at 0 %, 10 % and 20 % | **scenario assumption** |

The asymmetry is C_FN / C_FP = **2.5**. It is deliberately modest, because pricing a miss for one hour
rather than a whole storm keeps it honest. It is still enough to move the threshold, and §8 shows the
answer at ratios up to ~80×.

### 7.3 Expected cost formula (used identically by `train.py`, the dashboard, and every policy)

Expected cost per case for a model-only policy:

```
E = π [TPR · C_TP + FNR · C_FN] + (1 − π) [FPR · C_FP + TNR · C_TN]
```

With review, cases with t_low ≤ s < t_high cost review labour c_rev plus the reviewer's own errors:

```
E = π [P(FAIL|pos) C_TP + P(PASS|pos) C_FN + P(REV|pos)(c_rev + m C_FN + (1−m) C_TP)]
  + (1 − π)[P(FAIL|neg) C_FP + P(PASS|neg) C_TN + P(REV|neg)(c_rev + f C_FP + (1−f) C_TN)]
```

The rates are class-conditional, measured on the split, and re-weighted by the assumed π. Raw split
counts are never used as production estimates.

---

## 8. Operating point (single threshold)

**Procedure.** I swept every distinct threshold on the **valid** scores (midpoints between unique scores
plus a 0.01 grid), computed expected cost at each, and took the minimum. Then I froze it and evaluated
once on **test** (`artifacts/cost_vs_threshold.png`, `artifacts/cost_sweep_valid.csv`).

- **Selected t* = 0.960** (valid cost EUR 18,171 per 1,000).
- **Test at t* = 0.960:** TPR 0.909 (95 % CI 0.879–0.932), FPR 0.029 (0.013–0.067), **EUR 21,262 per
  1,000 cases** (stratified-bootstrap 95 % CI EUR 12,656–31,881), i.e. EUR 30,617 per day.

**Why so high, when FN is the more expensive error?** Because of **prior shift**. The model learned on
data that is ~70 % positive, but production is assumed 12 % positive. If s were a calibrated probability
at the 70 % training prior, the Bayes rule "FAIL when the production posterior exceeds
C_FP / (C_FP + C_FN) = 0.286" translates to s ≥ **0.872** after correcting the odds by
(0.12/0.88) / (0.70/0.30). The empirical t* = 0.96 lands in the same region. I did not use the
calibrated formula, because softmax scores are not calibrated probabilities. I used the empirical sweep,
and the formula only explains the direction.

**Sensitivity** (thresholds re-selected on valid for each value, costs on test rates;
`artifacts/sensitivity.png`, `artifacts/policy_map.png`):

| C_FN (EUR) | t* | model only /1,000 | review all /1,000 | model + review /1,000 | do nothing /1,000 |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.979 | 7,702 | 18,644 | 1,873 | 12,000 |
| 485 | 0.960 | 15,644 | 20,954 | 4,037 | 58,200 |
| 1,392 | 0.888 | 24,124 | 26,396 | 10,816 | 167,040 |
| 3,991 | 0.540 | 45,872 | 41,990 | 13,319 | 478,920 |
| 11,443 | 0.359 | 69,911 | 86,702 | 20,515 | 1,373,160 |
| 32,809 | 0.123 | 70,400 | 214,898 | 31,296 | 3,937,080 |

**Break-even points** (frozen test rates at t* unless stated):

- The **model beats doing nothing as long as C_FN > EUR 95**, or prevalence > 1.3 %. Both are far below
  any plausible value.
- **Treating everything only beats the frozen model if C_FN > EUR 31,318 or prevalence > 81 %.** With
  thresholds re-selected, the model never loses to treat-all in the range tested.
- **A person reviewing every case beats the model alone once C_FN exceeds roughly EUR 2,500–4,000**: at
  EUR 3,991, review-all costs 41,990 against 45,872 for model-only. Above that, misses are expensive
  enough that 5 % human error beats ~9 % model misses.
- **Model + review is the cheapest policy in every scenario tested**: the 2-D grid C_FN EUR 200–30,000 ×
  π 2–60 %, plus the 1-D sweeps C_FN EUR 100–50,000 and π 1–80 %. If no reviewer were available,
  model-only would be cheapest across the whole 2-D grid.
- t* falls steadily as C_FN rises (0.98 → 0.12) or as prevalence rises (0.98 at π = 1 % → 0.12 at
  π = 80 %), as it should.

**The assumption that would change the answer most** is C_FN, together with the prevalence. They move t*
from 0.98 to 0.12 across the plausible range. They do not, however, change *which policy* to run.

---

## 9. What a person still has to check (pass / review / fail)

**Selection.** I searched every (t_low, t_high) pair with t_low ≤ t_high on **valid**, minimising expected
cost including review labour and reviewer errors. The search was subject to review hours/day ≤ the
6 h/day spare capacity. The unconstrained optimum already fits, so both coincide
(`artifacts/review_pair_heatmap.png`).

**Chosen: t_low = 0.180, t_high = 0.998** (valid: EUR 6,701 per 1,000, review share 18.8 %).

**Frozen on test:**

| | value |
|---|---|
| Production-weighted share sent to review | **18.4 %** |
| Cases reviewed per day | **265** |
| Review hours per day (× 40 s) | **2.95 h** |
| Reviewer FTE (6 productive h/day) | **0.49 FTE** → 1 whole person-equivalent, but absorbed by existing duty officers |
| Fits the 6 h/day spare capacity? | **Yes** (≈ 1 h per 8-h shift; ≈ 11 reviews/hour, one every ~5 min) |
| Remaining **automated** errors / day | 0.0 false dispatches, **0.8** missed snowy segments |
| Errors made by **reviewers** / day (assumed 5 %/5 %) | 11.2 false dispatches, 2.1 misses |
| Expected cost | **EUR 5,170 per 1,000 cases = EUR 7,445 / day** (bootstrap 95 % CI EUR 3,936–6,524 per 1,000) |

The reviewed cases are not counted as correct model predictions. The dashboard shows the automated
decisions, the deferred counts, and a separately labelled "expected outcome after review" that uses the
*assumed* reviewer error rates.

### 9.1 Priced against the alternatives (same costs, same π, test rates)

| policy (concrete action) | EUR / 1,000 | EUR / day | review h/day | staffing |
|---|---:|---:|---:|---|
| **Do nothing**: no camera-based dispatch; snowy segments are treated only when a patrol or public report finds them (every positive becomes a one-cycle FN) | 120,000 | 172,800 | 0 | none |
| **Treat every segment** (always dispatch; reference) | 352,000 | 506,880 | 0 | trucks, not people |
| **Human reviews every case** | 24,044 | 34,624 | **16.0** | **2.67 FTE: exceeds** spare capacity; needs ~3 extra reviewers per day (more for 24/7 coverage) |
| **Model only**, t* = 0.960 | 21,262 | 30,617 | 0 | none |
| **Model + review**, [0.180, 0.998) | **5,170** | **7,445** | 2.95 | 0.49 FTE, **fits** existing staff |

The review policy saves about **EUR 23,000 per day** against the model alone and EUR 27,000 per day
against reviewing everything. It does this with about one-fifth of the reviewing effort.

### 9.2 Is human review assumed perfect? No; alternatives tested

Thresholds were re-selected on valid for each scenario; costs are per 1,000 test cases.

| reviewer error (miss / false alarm) | chosen band | review share | model + review | frozen base band | review all |
|---|---|---:|---:|---:|---:|
| perfect 0 % / 0 % (optimistic) | [0.070, 1.000) | 29.0 % | 129 | 614 | 444 |
| **base 5 % / 5 %** | [0.180, 0.998) | 18.4 % | **5,170** | 5,170 | 24,044 |
| pessimistic 10 % / 10 % | [0.360, 0.979) | 11.1 % | 13,601 | 9,726 | 47,644 |
| hard-case 20 % / 20 % | [0.540, 0.979) | 7.9 % | 17,165 | 18,838 | 94,844 |

As reviewers become less reliable, the optimal band narrows: send fewer, harder cases. Even at 20 %
reviewer error, the frozen base band (EUR 18,838) still beats the model alone (EUR 21,262).

### 9.3 Is the staffing realistic?

Yes, under these assumptions. About 265 reviews a day spread over three shifts is roughly one image
every 5 minutes, which a duty officer already watching cameras can absorb. The risk is peaks: during a
storm, review volume concentrates into a few hours. Even at a storm-hour prevalence of ~40 %, the band
sends ~20 % of that hour's ~60 cases (~12 images, ~8 minutes of work per hour), so it still fits one
officer. Reviewing *every* image would need about 3 extra reviewers per day.

---

## 10. Limitations and next steps

| limitation | why it matters | what I would do |
|---|---|---|
| Random image-level split with same-drive near-duplicates (11 % of test) | headline accuracy is optimistic (97 % vs 90.5 % on non-duplicates) | re-evaluate on a split grouped by drive/camera, or on a fresh month of footage, before deployment |
| In-vehicle dashcam images, one US state; source confounded with class | fisheye frames 72 %, Test_Image 63 % accuracy; fixed pole cameras untested | collect a labelled sample from the actual production cameras; report per-camera accuracy; consider source-balanced training |
| C_FN and π are scenario assumptions | they set t* (0.12–0.98 across the plausible range) | calibrate π from a season of RWIS/camera logs; estimate C_FN from local crash records by surface state |
| Softmax scores are not calibrated | can't read s as a probability | fit temperature scaling on valid if probabilities are needed (not required for empirical thresholding) |
| Reviewer error rates assumed, not measured | dominate the review-policy cost (13.3 of 14.0 residual errors/day) | measure duty-officer agreement on a blinded sample of in-band images |
| Small subgroups (Clear 52, Test_Image 16 test images) | wide CIs | collect more Clear and non-screenshot data |
| Image alone can't see black ice on wet roads | Wet is treated as negative by design | combine with pavement-temperature sensors (RWIS) in the decision rule |
| One training run, one seed, CPU budget | no variance estimate across seeds | repeat with 3–5 seeds on a GPU and report the spread |

---

## 11. Key decisions and trade-offs (what I would defend orally)

1. **Positive = Snow + Slush; Wet is negative.** The action (plough/salt now) is only justified for
   snow or slush on the surface. Wet-and-freezing is a sensor problem, not an image problem.
2. **Score = P(Snow) + P(Slush).** It is one number per case and matches the positive group exactly.
   Within-group confusions (Snow↔Slush) cost nothing operationally.
3. **Regret accounting (C_TP = C_TN = 0) and C_FN priced per hourly cycle.** This avoids double counting
   treatment cost and multi-hour exposure.
4. **Prevalence re-weighting.** The split is 70 % positive; production is assumed 12 %. Using raw split
   counts would hugely overstate the value of dispatching. Prior shift is why t* is 0.96, not 0.29.
5. **Everything is selected on valid and reported on test.** The epoch, t* and (t_low, t_high) were all
   frozen before looking at test costs.
6. **Review band within existing capacity.** It is the cheapest policy under every assumption tested,
   and its staffing need (0.49 FTE) fits the existing rota. Reviewing everything does not.
7. **Honest baselines.** The COCO and ImageNet zero-shot models have no road-surface vocabulary. Under
   stated a-priori mappings they never fire, so they are equivalent to doing nothing. I report that
   rather than inventing a semantic equivalence.
8. **Training loop.** A small resumable PyTorch loop around the Ultralytics model was needed for the
   compute environment. It produces standard Ultralytics weights.

---

## 12. Reproduction

```bash
uv sync
cp .env.example .env              # paste ROBOFLOW_API_KEY=<your key>; never commit .env
uv run python api_key.py          # verifies the key
uv run python main.py train       # download, audit, fine-tune, evaluate, cost sweep -> weights/ + artifacts/
uv run python main.py app         # dashboard on http://127.0.0.1:7860 (never trains)
uv run python check.py            # course self-check
```

- `main.py train` is idempotent. Training state lives in `runs/` (git-ignored) and resumes if
  interrupted. Delete `runs/` to retrain from scratch. On a fresh clone it always trains.
- Expected runtime: about 1 hour on 2 CPU cores (most of it training); faster on a laptop with more cores
  or a GPU.
- Outputs:
  - `weights/best.pt`: committed model
  - `artifacts/results.json`: all metrics, selected thresholds, policies, sensitivity, uncertainty
  - `artifacts/dataset_audit.json`
  - `artifacts/predictions_valid.csv` / `artifacts/predictions_test.csv`: per-image probabilities and
    scores; the dashboard recomputes every policy from these
  - plots in `artifacts/*.png`
- Shared definitions live in `roadcost/config.py` (dataset identity, training settings, economic
  assumptions) and `roadcost/policy.py` (decision rules and cost formula). Both stages import them, so
  every policy is priced identically.

---

## 13. Sources

1. Congleton Town Council, *Winter Gritting 2023/24*, committee report (June 2022), quoting Cheshire East
   Highways' top-up gritting prices (£725/km/year fixed; £8.47/km per run):
   https://www.congleton-tc.gov.uk/wp-content/uploads/2022/06/21-Winter-Gritting-Cheshire-East-Offer.pdf
2. Norfolk County Council, *Twenty-seven brand new gritters to join Norfolk's fleet* (Nov 2023, updated
   May 2024): 49 routes, 2,200 miles per full run, ≈ £49,000 per full gritting run:
   https://norfolk.gov.uk/article/44452/Twenty-seven-brand-new-gritters-to-join-Norfolks-fleet
3. Kuemmel, D. & Hanbali, R. (1992), *Accident Analysis of Ice Control Operations*, Marquette University
   (Salt Institute-sponsored): https://trid.trb.org/View/382235. Finnish reduced-salting experiment:
   https://trid.trb.org/View/467970 and https://trid.trb.org/View/408730
4. CE Delft et al. for the European Commission, *Handbook on the External Costs of Transport, Version
   2019*: https://cedelft.eu/publications/handbook-on-the-external-costs-of-transport-version-2019.
   Injury values as reproduced in the EUROCONTROL ANS CBA standard inputs:
   https://ansperformance.eu/economics/cba/standard-inputs/chapters/value_of_a_statistical_injury.html
5. FHWA Road Weather Management, *Snow & Ice*:
   https://ops.fhwa.dot.gov/weather/weather_events/snow_ice.htm
6. Eurostat, *EU hourly labour costs ranged from €11 to €55 in 2024* (28 March 2025):
   https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20250328-1
7. Sutter et al., *Machine Learning Detection of Road Surface Conditions: A Generalizable Model using
   Traffic Cameras and Weather Data*, arXiv:2510.06440: https://arxiv.org/abs/2510.06440
8. Dataset: UDOT CLASSIFICATION, *UDOT WINTER ROAD CONDITIONS* v1, Roboflow Universe, CC BY 4.0:
   https://universe.roboflow.com/udot-classification/udot-winter-road-conditions/dataset/1

---

## 14. Generative AI use disclosure

I used AI occasionally to help me understand the assignment, explore possible dataset ideas, and organize parts of the code and report. I reviewed the suggestions and made or confirmed the final decisions myself, including the problem framing, dataset choice, cost assumptions, and thresholds. I also checked the results against the actual dataset and model outputs. AI helped me with the process, but I am responsible for the choices and conclusions in this submission.
I reviewed the proposed problem framing, cost figures, prevalence, review-capacity assumptions, and threshold policy against the assignment requirements. I confirmed the choices I considered reasonable and take responsibility for understanding and defending them. Where an assumption needs to change to better reflect the dataset or a realistic operation, I will update it and explain why.
