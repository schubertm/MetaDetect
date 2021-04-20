#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os

from api.loading_data import load_or_prepare_metrics
from configs.uq_config import df_path, metrics_constellation
import configs.uq_config as uq_conf
from src.uncertainty_quantification.train_and_evaluate import train_and_evaluate
from uncertainty_quantification.parameter_search import optimize_params


def uncertainty_evaluation(var_df, tar_df,
                           model=uq_conf.aggregation_model,
                           score_thresholds=uq_conf.PARAMETER_SEARCH_SCORE_THRESHOLDS,
                           scoring_method=None,
                           augmentation_method=None,
                           pca_transform=False,
                           metrics_const=uq_conf.metrics_constellation,
                           path=uq_conf.df_path,
                           n_jobs=16):


    param_dir = f"{df_path}/{model}/{'+'.join(metrics_constellation)}"
    os.makedirs(param_dir, exist_ok=True)

    best_params = optimize_params(var_df, tar_df, model, score_thresholds, scoring_method, augmentation_method, pca_transform, n_jobs, metrics_const, path)
    train_and_evaluate(var_df, tar_df, best_params, path)

d = {"model"                : uq_conf.aggregation_model,
     "score_thresholds"     : [0.0],
     "scoring_method"       : "r2",
     "augmentation_method"  : False,
     "pca_transform"        : False,
     "metrics_const"        : uq_conf.metrics_constellation,
     "path"                 : uq_conf.df_path,
     "n_jobs"               : 16}

if __name__ == "__main__":
    p, m, c = d["path"], d["model"], d["metrics_const"]
    # variables_df, target_df = load_or_prepare_metrics(p, m, c)
    # uncertainty_evaluation(variables_df, target_df, **d)
    #
    # c += ["gradient_metrics"]
    # variables_df, target_df = load_or_prepare_metrics(p, m, c)
    # uncertainty_evaluation(variables_df, target_df, **d)

    # c += ["output_metrics"]
    variables_df, target_df = load_or_prepare_metrics(p, m, c)
    uncertainty_evaluation(variables_df, target_df, **d)
