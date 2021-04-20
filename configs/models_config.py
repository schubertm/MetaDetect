#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import src.models.gradient_boosting as gb
import src.models.logistic_regression as logreg
import src.models.linear_regression as linreg
import sklearn.linear_model as sklin

CLASSIFICATION_MODELS = ["logistic", "gb_classifier"]
REGRESSION_MODELS = ["linear_regression", "gb_regression"]

GB_CLASSIFIER_PARAMETERS = {"n_estimators" : list(range(45, 55)),
                        "max_depth" : list(range(10, 21)),
                        "learning_rate" : [0.3],
                        "reg_alpha" : [0.5, 1.0, 1.5],
                        "reg_lambda" : [0.0]}

GB_REGRESSION_PARAMETERS = {"n_estimators" : list(range(10, 20)),
                            "max_depth" : list(range(2, 4)),
                            "learning_rate" : [0.3],
                            "reg_alpha" : [0.5, 1.0, 2.0],
                            "reg_lambda" : [0.0, 0.5]}

LOGISTIC_REGRESSION_PARAMETERS = {"penalty" : ["l2"],
                                  "C" : [0.5, 0.3, 0.1, 0.05, 0.01],
                                  "solver" : ["saga"],
                                  "max_iter" : [5000]}

LINEAR_REGRESSION_PARAMETERS = {"fit_intercept" : [True]}

PARAMETER_SEARCH_MODELS = {"gb_classifier" : gb.gb_classifier_parameter_selection,
                           "logistic" : logreg.logistic_regression_parameter_selection,
                           "gb_regression" : gb.gb_regression_parameter_selection,
                           "linear_regression" : sklin.LinearRegression}
PARAMETER_SEARCH_OPTIONS = {"gb_classifier" : GB_CLASSIFIER_PARAMETERS,
                            "logistic" : LOGISTIC_REGRESSION_PARAMETERS,
                            "gb_regression" : GB_REGRESSION_PARAMETERS,
                            "linear_regression" : LINEAR_REGRESSION_PARAMETERS}

get_model_from_parameter_dict = {"gb_classifier" : gb.gb_classifier_from_parameter_dict,
                                 "logistic" : logreg.logistic_regression_from_parameter_dict,
                                 "gb_regression" : gb.gb_regression_from_parameter_dict,
                                 "linear_regression" : linreg.linear_regression_from_parameter_dict}

# DEFAULT_PARAMETER_SEARCH_SCORE_THRESHOLDS = [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]
DEFAULT_PARAMETER_SEARCH_SCORE_THRESHOLDS = [0.5]
