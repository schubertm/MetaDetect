#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import configs.data_config as data_cfg

df_path = data_cfg.default_df_path

gt_path = data_cfg.default_gt_path

num_classes = data_cfg.num_classes

# Set of uncertainty metrics to use. Should contain "score" xor "output"
"""
Options: 
    - "score" / "output"
    - "meta_detect"
    - "gradient_metrics"
"""
# metrics_constellation = ["score"]
# metrics_constellation = ["output"]
# metrics_constellation = ["output", "meta_detect", "gradient_metrics"]
# metrics_constellation = ["score", "meta_detect"]
# metrics_constellation = ["score", "gradient_metrics"]
# metrics_constellation = ["output", "gradient_metrics"]
metrics_constellation = ["output", "meta_detect"]

"""
Options:
    - "logistic_regression"
    - "gb_classifier"
    - "linear_regression"
    - "gb_regression"
"""
# aggregation_model = "gb_classifier"
aggregation_model = "gb_regression"

# CLASSIFICATION_MODELS = ["logistic_regression", "gb_classifier"]
# REGRESSION_MODELS = ["linear_regression", "gb_regression"]

OUTPUT_METRICS = data_cfg.OUTPUT_METRICS
STD_OUTPUT_METRICS = data_cfg.STD_OUTPUT_METRICS
META_DETECT_METRICS = data_cfg.META_DETECT_METRICS

PERFORM_PARAMETER_SEARCH = False

# PARAMETER_SEARCH_MODELS = {"gb_classifier" : gb_methods.gb_classifier_parameter_selection,
#                            "logistic" : log_reg_methods.logistic_regression_parameter_selection,
#                            "gb_regression" : gb_methods.gb_regression_parameter_selection}
# PARAMETER_SEARCH_OPTIONS = {"gb_classifier" : gb_methods.GB_CLASSIFIER_PARAMETERS,
#                             "logistic" : log_reg_methods.LOGISTIC_REGRESSION_PARAMETERS,
#                             "gb_regression" : gb_methods.GB_REGRESSION_PARAMETERS}

# PARAMETER_SEARCH_SCORE_THRESHOLDS = [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]
PARAMETER_SEARCH_SCORE_THRESHOLDS = [1e-2] #, 1e-3, 1e-2, 0.1, 0.3, 0.5]
# PARAMETER_SEARCH_SCORE_THRESHOLDS = [1e-1] #, 1e-3, 1e-2, 0.1, 0.3, 0.5]
