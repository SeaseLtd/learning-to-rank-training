import pandas as pd
import shap
import xgboost
import matplotlib.pyplot as plt
import logging

def train_model(output_dir, training_set, test_set, name, eval_metric, images_path, feature_to_analyze):
    # train LTR model with XGBoost
    training_set_store = pd.HDFStore(training_set, 'r')
    training_set_data_frame = training_set_store['training_set']
    training_set_store.close()

    test_set_store = pd.HDFStore(test_set, 'r')
    test_set_data_frame = test_set_store['test_set']
    test_set_store.close()

    training_data_set = training_set_data_frame[
        training_set_data_frame.columns.difference(
            ['Ranking', 'ID', 'query_ID'])]
    training_query_id_column = training_set_data_frame['query_ID']
    training_query_groups = training_query_id_column.value_counts(sort=False)
    training_label_column = training_set_data_frame['Ranking']

    # query_first_row = training_set_data_frame.iloc[0]['query_ID']
    # dataset_one_query = training_set_data_frame[training_set_data_frame['query_ID'] == query_first_row].drop(
    #     columns=['ID', 'query_ID'])

    test_data_set = test_set_data_frame[
        test_set_data_frame.columns.difference(
            ['Ranking', 'ID', 'query_ID'])]
    test_query_id_column = test_set_data_frame['query_ID']
    test_query_groups = test_query_id_column.value_counts(sort=False)
    test_label_column = test_set_data_frame['Ranking']

    training_xgb_matrix = xgboost.DMatrix(training_data_set, label=training_label_column)
    training_xgb_matrix.set_group(training_query_groups)
    logging.debug("Query Id Groups in Training Set: " + str(training_query_groups))
    test_xgb_matrix = xgboost.DMatrix(test_data_set, label=test_label_column)
    test_xgb_matrix.set_group(test_query_groups)
    logging.debug("Query Id Groups in Test Set: " + str(test_query_groups))
    params = {'objective': 'rank:ndcg', 'eval_metric': eval_metric, 'verbosity': 2, 'early_stopping_rounds': 10}
    watch_list = [(test_xgb_matrix, 'eval'), (training_xgb_matrix, 'train')]
    logging.info('- - - - Training the model')

    xgb_model = xgboost.train(params, training_xgb_matrix, num_boost_round=999, evals=watch_list)
    preds = xgb_model.predict(test_xgb_matrix)
    print(preds)

    logging.info('- - - - Saving  XGBoost model')
    xgboost_model_json = output_dir + "/xgboost-" + name + ".json"
    xgb_model.dump_model(xgboost_model_json, fmap='', with_stats=True, dump_format='json')

    feature_importance(xgb_model, training_data_set, images_path, feature_to_analyze)

def feature_importance(xgb_model, training_data_set, images_path, feature_to_analyze):
    # explain the model prediction using SHAP library
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(training_data_set)

    # The SHAP feature importance plots
    # SUMMARY PLOT
    shap.summary_plot(shap_values, training_data_set, show=False)
    plt.savefig(images_path + '/summary_plot.png', bbox_inches='tight')
    plt.close()

    # SUMMARY PLOT with bars
    shap.summary_plot(shap_values, training_data_set, plot_type="bar", show=False)
    plt.savefig(images_path + '/summary_plot_bars.png', bbox_inches='tight')
    plt.close()

    # Decision plot (total)
    shap.decision_plot(explainer.expected_value, shap_values, training_data_set,
            feature_names=training_data_set.columns.tolist(), show=False, ignore_warnings=True)
    plt.savefig(images_path + '/decision_plot.png', bbox_inches='tight')
    plt.close()

    # Decision plot (one observation)
    shap.decision_plot(explainer.expected_value, shap_values[0], training_data_set.iloc[0],
                       feature_names=training_data_set.columns.tolist(), show=False, ignore_warnings=True)
    plt.savefig(images_path + '/decision_plot_0.png', bbox_inches='tight')
    plt.close()

    # Decision plot (only 500000 observations)
    shap.decision_plot(explainer.expected_value, shap_values[0:500000, :], training_data_set.iloc[0:500000, :],
                           feature_names=training_data_set.columns.tolist(), show=False, ignore_warnings=True)
    plt.savefig(images_path + '/decision_plot_500000.png', bbox_inches='tight')
    plt.close()


    # Create a dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(feature_to_analyze, shap_values, training_data_set, show=False)
    plt.savefig(images_path + '/dependence_plot_' + feature_to_analyze + '.png', bbox_inches='tight')
    plt.close()

    if 'Artists' in training_data_set.columns:
        shap.dependence_plot(feature_to_analyze, shap_values, training_data_set,
                             interaction_index='Artists', show=False)
        plt.savefig(images_path + '/dependence_plot_' + feature_to_analyze + '_with_Artists.png', bbox_inches='tight')
        plt.close()

    # Force Plot(one observation)
    shap.force_plot(explainer.expected_value, shap_values[0], training_data_set.iloc[0], show=False,
                        matplotlib=True)
    plt.savefig(images_path + '/force_plot_0.png', bbox_inches='tight')
    plt.close()

    # Visualize the training set prediction
    html_img = shap.force_plot(explainer.expected_value, shap_values[0:100000, :],
                                   training_data_set.iloc[0:100000, :],
                                   show=False)
    shap.save_html(images_path + '/prediction_explanation_1000.html', html_img)

    # Visualize the training set prediction
    # html_img = shap.force_plot(explainer.expected_value, shap_values, training_data_set, show=False)
    # shap.save_html(images_path + '/full_prediction_explanation.html', html_img)













