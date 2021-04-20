#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import pandas as pd
import matplotlib.pyplot as plt

# POS =
# NEG =

p = "/home/riedlinger/MetaDetect-TestEvaluation/grads_lr=1e-3_bs=64/pareto_evaluation"

bl_errors = pd.read_csv(f"{p}/baseline_errors.csv")
score_errors = pd.read_csv(f"{p}/score_errors.csv")
norms_errors = pd.read_csv(f"{p}/norms_errors.csv")
grads_errors = pd.read_csv(f"{p}/grads_errors.csv")
nc_errors = pd.read_csv(f"{p}/nc_errors.csv")
gc_errors = pd.read_csv(f"{p}/gc_errors.csv")

auc_scores = {}

labels = ["Threshold Baseline", "GB Score", "GB $|\!|\\nabla|\!|$", "GB $\\mu(\\nabla)$", "GB $s + |\!|\\nabla|\!|$", "GB $s + \\mu(\\nabla)$"]

for i, df in enumerate([bl_errors, score_errors, norms_errors, grads_errors, nc_errors, gc_errors]):
    df["tpr"] = 1 - df["fn"]/POS
    df["fpr"] = df["fp"]/NEG
