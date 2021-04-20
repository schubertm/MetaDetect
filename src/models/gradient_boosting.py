#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import xgboost

def initialize_new_gb_classifier(n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda, colsample_bytree=0.5):
    model = xgboost.XGBClassifier(verbosity=1, max_depth=max_depth, colsample_bytree=colsample_bytree, n_estimators=n_estimators, learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    return model

def load_gb_classifier(file_path):
    model = xgboost.XGBClassifier()
    model.load_model(file_path)
    return model

def gb_classifier_from_parameter_dict(params):
    return xgboost.XGBClassifier(tree_method="gpu_hist", gpu_id=0, **params)

def gb_classifier_parameter_selection(use_gpu=True):
    options = {True : {"tree_method" : "gpu_hist", "gpu_id" : 0}, False: {}}
    return xgboost.XGBClassifier(**options[use_gpu])

#################################################

def initialize_new_gb_regression(n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda, colsample_bytree=0.5):
    model = xgboost.XGBRegressor(verbosity=1, max_depth=max_depth, colsample_bytree=colsample_bytree, n_estimators=n_estimators, learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    return model

def load_gb_regression(file_path):
    model = xgboost.XGBRegressor()
    model.load_model(file_path)
    return model

def gb_regression_from_parameter_dict(params):
    return xgboost.XGBRegressor(tree_method="gpu_hist", gpu_id=0, **params)

def gb_regression_parameter_selection(use_gpu=True):
    options = {True : {"tree_method" : "gpu_hist", "gpu_id" : 0}, False: {}}
    return xgboost.XGBRegressor(**options[use_gpu])
