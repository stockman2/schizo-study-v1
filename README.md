# Systematic biases in spatial working memory: a reanalysis of open data with pre-specified tests

This repository is a worked example of a careful reanalysis of open behavioural data.
It does not claim new findings. The main results reproduce known ones
(category adjustment model, dynamic field theory, attractor models of working memory);
see the section "Relation to prior work" in `report/wm_report_en.pdf`.

## What was done

- Data: Stein, Barbosa et al. (2020), single-item spatial working memory, delays 0, 1, 3 s;
  52 participants (controls, schizophrenia, anti-NMDAR encephalitis), plus follow-up sessions of 22 participants.
- For scope tests: 30 open experiments on orientation and motion direction
  from the collection of Ozkirli, Chetverikov and Pascucci (2025).
- Procedure, the same for every test:
  1. the decision rule is written in the script header before the computation;
  2. the script is first run on synthetic data with a known answer (`--selftest`);
  3. exploration uses the baseline sessions; confirmation uses the held-out post sessions, once.
- Negative and undecided results are reported together with the positive ones.

## Results in one paragraph

About half of the error variance is a location-dependent bias (attraction toward the diagonals).
Its strength is proportional to the random variance (the weighting rule of the category adjustment model);
a constant-rate drift is rejected. A second group of biases (attraction toward the 0-180 axis, attraction toward
the previous target) grows at an approximately constant rate. A model with drift during the delay predicts the size
of the location dependence of the random variance; a model with a single correction at the response overpredicts it
by a factor of about two; neither model passes all pre-specified tests. Details, numbers and limitations:
`report/wm_report_en.pdf`.

## Repository layout

| Path | Content |
|---|---|
| `scripts/` | analysis scripts (Python) |
| `report/` | report in English (PDF, LaTeX source, figures and figure scripts) |
| `docs_ru/` | two earlier working summaries in Russian |
| `results/` | put here the `report.md` files produced by the scripts |

## Data

The data are not included. The scripts download them automatically, or you can place them yourself:

- `data/behavior.pkl`, `data/behavior_retest.pkl` from https://github.com/comptelab/serialNMDA
- `data_open/*.csv` from https://github.com/aozkirli/Large-scale-mega-analysis-on-serial-dependence

The `.pkl` files are Python pickles; load them only from the original repository.
If you use the data, cite the original studies (see "Sources" in the report).

## How to run

```
pip install -r requirements.txt
cd scripts
python wm_stein.py --selftest      # method check on synthetic data
python wm_stein.py                 # analysis; writes wm_stein_out/report.md
```

Each script has `--selftest`. Run time is seconds, except where noted.

| Script | Question | Section of the report |
|---|---|---|
| `wm_stein.py` | decomposition of the error variance; what grows with the delay | Result 1 |
| `wm_stein2.py` | harmonics of the bias map; weighting rule against constant-rate drift | Result 2 |
| `wm_stein3.py` | replication on the post sessions; recovery from encephalitis | Results 2, 7; Section 7 |
| `wm_stein4.py` | exploration of the first and second harmonics | Results 3, 4 |
| `wm_stein5.py` | vector fields of the bias on the circle | figures only |
| `wm_stein6.py` | confirmation of the harmonic results on the post sessions | Results 3, 4, 6; Section 7 |
| `wm_stein7.py` | drift model against correction-at-response model, sawtooth shape (2-3 min) | Result 9 |
| `wm_stein8.py` | the same with a smooth shape fitted to the bias map (4 min) | Result 9 |
| `wm_stein9.py` | growth law of the serial bias; late drift by group | Results 5, 8 |
| `wm_open1.py` | inventory of the 19 open data sets | Section 6 |
| `wm_open2b.py` | sign test: is the bias directed toward low-variance locations | Result 10 |
| `wm_open2.py` | first version of the sign test; superseded by `wm_open2b.py` (kept for the record) | --- |

Later scripts import earlier ones (`wm_stein2.py`, `wm_stein3.py`, `wm_stein4.py`, `wm_stein5.py`, `wm_stein7.py`),
so keep all scripts in one folder.

Note: comments, console messages and the generated `report.md` files are in Russian.
The decision rules are in the header of each script.

## Corrections made during the work

The record of the work includes errors that were found and corrected, for example:
a calibration error in the model test (found by the synthetic-data check), a bias of the first sign test toward
one answer (found after the first run; corrected in `wm_open2b.py`), and a late literature check that showed
that the main results were already known.

## Use of AI

The analysis scripts and the texts were prepared with the assistance of a large language model (Claude, Anthropic).
The author ran all analyses, checked the results, and is responsible for the content.

## License

Code: MIT (see `LICENSE`). Report and figures: CC BY 4.0.
