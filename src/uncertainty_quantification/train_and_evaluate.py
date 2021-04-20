#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import os
import pandas as pd
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from api.pre_process import center_np_columns
import configs.models_config as model_cfg
from src.meta_detect_metrics.md_metrics import add_meta_detect_metrics
from src.uncertainty_quantification.performance_evaluation import evaluate_classifier_performance, evaluate_regression_performance
from src.bbox_tools.nms_algorithms import perform_nms_on_dataframe
from data_tools.sampling_methods import get_augmented_vars_and_targets
import configs.data_config as data_cfg


def threshold_variables_and_targets(var_df, target_df, column="s", threshold=0.3):
    id_str = "dataset_box_id"
    variables_thresh = var_df[var_df[column] >= threshold]
    targets_thresh = target_df[target_df[id_str].isin(variables_thresh[id_str])]

    targets_thresh = targets_thresh.drop(id_str, axis=1)

    return variables_thresh, targets_thresh

def train_and_evaluate(var_df, target_df, parameter_dict, path):
    model_name = parameter_dict["model"]
    metrics_const = parameter_dict["log"]["metrics"]
    augmentation = parameter_dict["log"]["augmentation"]
    pca_transform = parameter_dict["log"]["pca_transform"]

    print(f"Evaluation of {model_name}...")
    for score_thr in parameter_dict["parameters"].keys():
        print("Score threshold = {}".format(score_thr))
        res_path = f"{path}/{model_name}/{'+'.join(metrics_const)}/{score_thr}"
        os.makedirs(res_path, exist_ok=True)
        params = parameter_dict["parameters"][score_thr]
        print(f"Parameters: {params}")

        variables_thresh, targets_thresh = threshold_variables_and_targets(var_df, target_df, threshold=float(score_thr))
        # if "score" not in metrics_const:
        #     variables_thresh = variables_thresh.drop("s", axis=1)

        if "meta_detect" in metrics_const:
            variables_thresh_md = add_meta_detect_metrics(variables_thresh, metrics_const)
            variables_thresh_md.columns = data_cfg.META_DETECT_METRICS
            variables_thresh = pd.concat([variables_thresh, variables_thresh_md], axis=1)
        gradient_identifiers = [s for s in variables_thresh.columns if "grad" in s]
        num_gradient_metrics = len(gradient_identifiers)

        if data_cfg.PERFORM_NMS:
            variables_thresh = perform_nms_on_dataframe(pd.concat([variables_thresh, targets_thresh], axis=1))
            targets_thresh = pd.DataFrame(variables_thresh["true_iou"], columns=["true_iou"])
            variables_thresh = variables_thresh.drop(["dataset_box_id", "true_iou"], axis=1)

        if "score" in metrics_const and "output" not in metrics_const:
            variables_thresh = variables_thresh[['file_path', 's']]

        variables_thresh = variables_thresh.drop("file_path", axis=1)

        variables_thresh, targets_thresh = np.array(variables_thresh), np.ravel(targets_thresh)

        evaluations = {"accuracy" : [], "auroc" : []} if model_name in model_cfg.CLASSIFICATION_MODELS else {"r2" : []}

        for i in tqdm.tqdm(range(15)):
            x_train, x_test, y_train, y_test = train_test_split(variables_thresh, targets_thresh, test_size=0.3)
            if augmentation:
                x_train, y_train = get_augmented_vars_and_targets(x_train, y_train, augmentation)

            #todo:outsource transformation
            if pca_transform:
                print(f"Performing PCA transformation ({pca_transform} comps) of gradient metrics...")
                pca = PCA(n_components=pca_transform)
                grads_train = x_train[:, -num_gradient_metrics:]
                grads_test = x_test[:, -num_gradient_metrics:]
                grads_train, grads_test = center_np_columns(grads_train), center_np_columns(grads_test)

                grads_train = pca.fit_transform(grads_train)
                grads_test = pca.transform(grads_test)

                x_train = np.hstack((x_train[:, :-num_gradient_metrics], grads_train))
                x_test = np.hstack((x_test[:, :-num_gradient_metrics], grads_test))
                # print(f"xtrain: {x_train.shape}, xtest: {x_test.shape}")

            model = model_cfg.get_model_from_parameter_dict[model_name](params)

            model.fit(x_train, y_train)
            if "gb" in model_name:
                model.save_model(f"{res_path}/model_{i}.model")

            if model_name in model_cfg.CLASSIFICATION_MODELS:
                y_pred = model.predict_proba(x_test)[:, 1]
                acc, auroc = evaluate_classifier_performance(y_test, y_pred)
                evaluations["accuracy"].append(acc)
                evaluations["auroc"].append(auroc)
            else:
                y_pred = model.predict(x_test)
                r2 = evaluate_regression_performance(y_test, y_pred)
                evaluations["r2"].append(r2)

            comp_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
            comp_df.to_csv(f"{res_path}/eval_data_{i}.csv")

        eval_results = pd.DataFrame(evaluations)
        eval_results.loc["mean"] = eval_results.mean()
        eval_results.loc["std"] = eval_results.std()
        eval_results.to_csv(f"{res_path}/evaluation_results.csv")

        print(f"Evaluation results for eps_s = {score_thr}: ")
        print(eval_results)
