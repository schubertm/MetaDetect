#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import smogn
import pandas as pd

"""
While we implement the SMOGN updampling method form https://github.com/nickkunz/smogn
note that applying it in practice (especially in this project) takes an exceeding amount of time
and seems to not improve meta regression results by that much.
That is why in src.uncertainty_quantification.parameter_search.py we call None as
default_augmentation for regressors.
"""
def get_augmented_vars_and_targets(var_df, target_df, augmentation_method):
    if augmentation_method == "smote":
        oversample = SMOTE()
        return oversample.fit_resample(var_df, target_df)

    elif augmentation_method == "smote+rus":
        over = SMOTE(sampling_strategy=0.5)
        under = RandomUnderSampler(sampling_strategy=1.0)
        steps = [("o", over), ("u", under)]
        pipeline = Pipeline(steps=steps)
        return pipeline.fit_resample(var_df, target_df)

    elif augmentation_method == "smogn":
        smogn_df = smogn.smoter(data=pd.concat([var_df, target_df], axis=1), y="true_iou")
        iou_df = smogn_df["true_iou"].copy(deep=True)
        variables_df = smogn_df.drop("true_iou", axis=1)

        return variables_df, iou_df

