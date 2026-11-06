# One Neuron, One Bias Vector, No Training

Code and data for **"One Neuron, One Bias Vector, No Training: A Frontier LLM Edits 0.0002% of a Language Model's Weights and Recovers Hidden Accuracy on Medical Exam Questions"** by Mohammad Alnajdawi (paper coming soon).

Claude Opus 5.5 (Anthropic), working as a coding agent, was asked to improve Qwen2.5-0.5B-Instruct on the MedMCQA medical exam benchmark by editing its weights directly, with no training. It found that the model answers "D" 75% of the time, traced most of this habit to a single neuron (block 21, neuron 2921), and removed it by changing 1,015 of the model's 494 million weights (0.0002%).

## Results on 9,183 held-out questions

| | Original | Edit 1 | Edit 2 |
|---|---|---|---|
| Weights changed | 0 | 1,015 | 1,905 |
| Real-exam test (NEET PG, 4,183 questions) | 28.1% | 34.4% | 34.7% |
| Mock-exam test (5,000 questions) | 29.6% | 36.4% | 36.4% |
| Both tests (9,183 questions) | 28.9% | **35.5%** | 35.6% |
| Share of answers that are "D" | 75.4% | 27.3% | 26.5% |
| Same option in all 4 option orders | 10.1% | 20.4% | 20.8% |

- Edit 1 vs. the original: 1,447 answers fixed, 844 broken (McNemar p = 1.1e-36).
- Edit 2 vs. Edit 1: +0.16 points (p = 0.45). Edit 2 looked better on the questions used to choose it, but the gain did not hold up on held-out data (a winner's curse).
- The edit removes a letter habit; it does not add medical knowledge. Accuracy averaged over all four option orders stays at 36.5%.

## Use the edit

Python 3.13 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
python medmcqa/apply_edit.py edits/edit1.json models/medmcqa-edit-qwen25
python medmcqa/evaluate.py models/medmcqa-edit-qwen25
```

The first script downloads the original model (pinned revision, about 1 GB) and writes the edited model. The edit itself is the small file `edits/edit1.json` (the 1,015 changed weights); `edits/edit2.json` is Edit 2. Expected output of `evaluate.py` on Apple silicon: 28.07% for the original model, 34.40% for Edit 1, 34.69% for Edit 2.

## Repository layout

```
edits/              the two edits, as lists of changed weights (JSON)
medmcqa/            all code
  lab.py            data splits, prompt and scoring
  apply_edit.py     apply an edit file to the original model
  evaluate.py       score any model on the real-exam test
  export_edit.py    write an edit file from an edited model
  try_edits.py      the edit tools (surgery, counterweight) and a tester for candidate edits
  recipe.py         Edit 1
  results/qwen25/   saved per-question predictions behind every number in the paper
```

## Reproduce the paper

Run from the repository root. Each step overwrites the saved results in `medmcqa/results/qwen25/` with your run.

| Step | Command | What it does | Time* |
|---|---|---|---|
| 1 | `python medmcqa/baseline.py` | Scores the original model | about 4 min |
| 2 | `python medmcqa/analyze.py` | Logit lens and attribution on 1,500 practice questions (finds neuron 2921) | about 2 min |
| 3 | `python medmcqa/edit.py` | Builds Edit 1, scores it on the real-exam test, saves it | about 10 min |
| 4 | `python medmcqa/test5000.py` | Mock-exam test in 4 option orders, before and after | about 20 min |
| 5 | `python medmcqa/round2.py > medmcqa/results/qwen25/round2.log` | Chooses Edit 2 on the tryout set | about 15 min |
| 6 | `python medmcqa/round2_heldout.py` | Builds Edit 2 and scores it on all test questions | about 15 min |
| 7 | `python medmcqa/lens3.py` | Block-by-block readout of all three models | about 3 min |
| 8 | `python medmcqa/lens_rot.py` and `python medmcqa/probes.py` | Later checks: order-averaged lens and linear probes | about 3 min each |
| 9 | `python medmcqa/ablation.py original` (then `surgery`, `counterweight`, `both`) | What each part of Edit 1 does | about 5 min each |

\*On an Apple M-series laptop. Results are deterministic there (batch size 1, bfloat16); on CUDA or CPU a few near-tie answers can flip.

## How this work was done

The diagnosis, the code and both edits were produced by Claude Opus 5.5 (Anthropic) at maximum reasoning effort, working as a coding agent (GitHub Copilot agent mode in VS Code) on a laptop. The author chose the model and the benchmark, set the goal and a budget of at most 4% of the weights, and asked the follow-up questions. The work was inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang and Khattab, MIT CSAIL, 2026).

## Citation

```bibtex
@misc{alnajdawi2026oneneuron,
  title        = {One Neuron, One Bias Vector, No Training: A Frontier {LLM} Edits 0.0002\% of a Language Model's Weights and Recovers Hidden Accuracy on Medical Exam Questions},
  author       = {Alnajdawi, Mohammad},
  year         = {2026},
  howpublished = {\url{https://github.com/Mohamednajdawi/one-neuron-edit}}
}
```

## License and use

- Code: MIT (see `LICENSE`).
- The edit files modify Qwen2.5-0.5B-Instruct, released by the Qwen team under Apache-2.0.
- MedMCQA (Pal et al., 2022) is listed under Apache-2.0 on Hugging Face.

Not for medical use: the edited model answers about one question in three correctly.
