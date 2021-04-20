#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse

import configs.uq_config as uq_conf
from src.api.loading_data import load_or_prepare_metrics
from src.uncertainty_quantification.parameter_search import optimize_params

parser = argparse.ArgumentParser(description="Perform parameter search for metrics constellation as given in src.uncertainty_quantification.uq_config.")
parser.add_argument("--df-path", dest="df_path")

df_path = uq_conf.df_path
gt_path = uq_conf.gt_path

if __name__ == "__main__":
    d = {"model": "logistic",
         "score_thresholds": uq_conf.PARAMETER_SEARCH_SCORE_THRESHOLDS,
         "scoring_method": "roc_auc",
         "augmentation_method": False,
         "pca_transform": 15,
         "metrics_const": uq_conf.metrics_constellation,
         "path": uq_conf.df_path,
         "n_jobs": 16}

    metrics_constellation = ["output"]
    # Get variables and targets to perform parameter search for.
    variables_df, target_df = load_or_prepare_metrics(df_path, uq_conf.aggregation_model, metrics_constellation)
    optimize_params(variables_df, target_df, augmentation_method="smote", metrics_const=metrics_constellation, n_jobs=16)
