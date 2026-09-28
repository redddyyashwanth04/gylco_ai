"""
Track B's LSTM/GRU sequence model -- the piece neither base paper builds.

This is the project's most specific, defensible contribution: Huang et al.
(2025) had access to MIMIC-IV's repeated-visit structure and used only the
first 24 hours; we deliberately use the FULL trajectory
(data/processed/mimic_trajectories.csv) as input, testing whether the
longitudinal signal they left unused actually improves prediction over
their static baseline (mimic_static_baseline.py).

ARCHITECTURE (a standard sequence model, appropriately scoped -- NOT
attempting to reproduce Xiao et al. 2025's full Hypergraph Neural ODE,
which needs 8xA100 GPUs and a custom ODE solver; see
Base_Papers_Gap_Analysis.md for why that was ruled out)

    Input:  padded sequences of (time_gap, hba1c, glucose) per patient,
            from preprocess_mimic.py
    Model:  Embedding/projection layer -> LSTM or GRU (2 layers, ~32-64
            hidden units given the small data size -- do not over-parameterize
            a model for a ~13-35 patient dataset) -> attention or final-hidden-
            state pooling -> linear output layer -> sigmoid per complication
    Output: per-complication probability, same target set as Track A where
            applicable (nephropathy, cardiovascular -- neuropathy and
            retinopathy are too sparse in this cohort per the earlier
            prevalence check, exclude them here too)

TRAINING NOTES
    - Use masked loss (ignore padded positions) -- see preprocess_mimic.py's
      per-patient sequence length tracking.
    - Train via lopo_cv.py, not a single train/val split, given the small N.
    - If using pretrain_finetune.py, initialize from the pretrained UCI
      weights before this LSTM's final training loop, and use a lower
      learning rate.
"""

import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    """
    A deliberately small LSTM, sized for a ~13-35 patient dataset -- do not
    scale this up without first checking it doesn't just memorize.
    """
    def __init__(self, n_input_features=2, hidden_size=32, n_layers=2, n_targets=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, n_targets)

    def forward(self, x, lengths):
        """
        x: (batch, max_seq_len, n_input_features), padded
        lengths: (batch,) real sequence length per patient, for masking
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]  # final layer's hidden state
        logits = self.output_layer(last_hidden)
        return torch.sigmoid(logits)


def train_fn(X_train, y_train):
    # TODO: wraps TrajectoryLSTM in a scikit-learn-style .fit() interface so
    # it can be passed directly into lopo_cv.run_lopo_cv() alongside the
    # static baseline models -- implement once preprocess_mimic.py produces
    # real padded sequence tensors to train on.
    raise NotImplementedError("Training loop not yet implemented")


def predict_fn(model, X_test):
    # TODO: matching .predict_proba()-style wrapper for lopo_cv.py
    raise NotImplementedError("Prediction wrapper not yet implemented")


if __name__ == "__main__":
    # smoke test: confirm the model builds and runs a forward pass on
    # random data (NOT real patient data -- just an architecture check)
    model = TrajectoryLSTM()
    batch_size, max_len, n_features = 4, 10, 2
    dummy_x = torch.randn(batch_size, max_len, n_features)
    dummy_lengths = torch.tensor([10, 7, 3, 5])
    output = model(dummy_x, dummy_lengths)
    print(f"Model output shape: {output.shape} (expected: [{batch_size}, 2])")
    print("Architecture smoke test passed.")
