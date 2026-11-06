"""Edit 1, as chosen by the agent (selected on practice questions only; the test sets are scored once).

(1) Neuron surgery: block 21 neuron 2921 is active on almost every question and adds about +2 logits to "D";
    remove only the letter-voting part of its output vector.
(2) Counterweight: value biases pass through attention unchanged (attention weights sum to 1), so block 23's
    128 v_proj biases add a constant vector at every position. Solve for the change that cancels the remaining
    content-free letter preference, measured on unlabeled practice questions in all 4 option orders (3 rounds).
No answer labels are used.
"""
from try_edits import calibrate_uniform, unletter_neurons

DEV_ACCURACY = {"none": 0.290, "surgery": 0.311, "counterweight": 0.342, "both": 0.342}  # tryout set, at design time
FINAL_RMS = 1.54  # mean RMS of the final residual at the answer position (analyze.py)


def apply(model, tok, practice):
    unletter_neurons(model, 21, [2921])
    calibrate_uniform(model, tok, 23, practice[:500], rms=FINAL_RMS)
    return {"recipe": "remove letter-logit component of block 21 neuron 2921's output; "
                      "re-solve block 23 v_proj bias to cancel the content-free letter preference (3 rounds, no labels)",
            "dev_accuracy": DEV_ACCURACY}
