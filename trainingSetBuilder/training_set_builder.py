import pandas as pd
import os
import time
import logging


def write_training_test_set(data_frame, output_dir, test_set_size=10000, query_id_sample_threshold=25):
    #logging.info('- Splitting training/test set\n')
    #initial_time = time.time()
    #start_time = time.time()
    #if not os.path.exists(output_dir):
        #os.makedirs(output_dir)
    # Add count of num rows per query id
    #logging.info('Num rows count')
    query_ids_sample_counts = pd.DataFrame(data_frame['query_ID'].value_counts())
    query_ids_sample_counts.index.rename('query_ID', inplace=True)
    query_ids_sample_counts.rename(columns={'count': "queryIdCount"}, inplace=True)
    interactions_data_frame = pd.merge(data_frame, query_ids_sample_counts,
                                       on='query_ID', how='outer', validate="many_to_one")
    #count_finish_time = time.time()
    #logging.info('- - - - Num rows count computed in {:f} seconds\n'.format(count_finish_time - start_time))

    # Remove query id with only one relevance label
    #logging.info('One label removal')
    #start_time = time.time()
    interactions_data_frame = clean_data_frame_from_single_label(interactions_data_frame)
    #one_label_finish_time = time.time()
    #logging.info('- - - - One label removed in {:f} seconds\n'.format(one_label_finish_time - start_time))


    percentage = test_set_size / len(interactions_data_frame)
    percentage = min(percentage, 0.2)
    test_set_size = min(test_set_size, int(len(interactions_data_frame) * percentage))

    # Split in training and test set
    #logging.info('Split')
    #start_time = time.time()
    interactions_train, interactions_test = training_test_set_split(
       interactions_data_frame, query_id_sample_threshold, percentage, test_set_size)
    #split_finish_time = time.time()
    #logging.info('- - - - Split computed in {:f} seconds\n'.format(split_finish_time - start_time))

    interactions_train = interactions_train.sort_values('query_ID')
    interactions_test = interactions_test.sort_values('query_ID')
    #logging.info('- - - - Writing Training Set of : {:d} feature vectors - - - -'.format(len(interactions_train)))

    #start_time = time.time()
    #print_query_id_stats(interactions_train, query_id_sample_threshold)
    write_set(interactions_train, output_dir, 'training_set')
    #train_finish_time = time.time()
    #logging.info('- - - - Training set written in {:f} seconds\n'.format(train_finish_time - start_time))
    #logging.info('- - - - Writing Test Set of : {:d} feature vectors - - - -'.format(len(interactions_test)))
    #start_time = time.time()
    #print_query_id_stats(interactions_test, query_id_sample_threshold)
    write_set(interactions_test, output_dir, 'test_set')
    #test_finish_time = time.time()
    #logging.info('- - - - Test set written in {:f} seconds\n'.format(test_finish_time - start_time))
    #logging.info(
       # '- - - - Training/Test Set split Successfully produced in {:f} seconds\n\n'.format(time.time() - initial_time))


def write_set(data_frame, output_dir, set_name):
    # Binary save
    store = pd.HDFStore(output_dir + '/' + set_name + '.h5')
    store[set_name] = data_frame
    store.close()
    # interactions_data_frame.to_hdf(output_dir + '/' + set_name + '.h5', key='interactions_data_frame', format='t')
    # set_name = interactions_data_frame


def clean_data_frame_from_single_label(data_frame):
    # Add count of num different relevance label per query id
    relevance_counts = data_frame.groupby("query_ID")["Ranking"].value_counts()
    different_relevance_counts_per_query_id = pd.DataFrame(relevance_counts.index.get_level_values(0).value_counts())
    different_relevance_counts_per_query_id.index.rename('query_ID', inplace=True)
    different_relevance_counts_per_query_id.rename(columns={"count": "relevance_count"}, inplace=True)
    interactions_data_frame = pd.merge(data_frame, different_relevance_counts_per_query_id,
                                       on='query_ID', how='outer', validate="many_to_one")

    # Separate query id with just one relevance label (we don't want it in the test set)
    #logging.debug("- - - - Number of rows with one type of relevance label: " + str(len(interactions_data_frame[
                                                                        #interactions_data_frame.relevance_count == 1])))
    #logging.debug("- - - - Dataframe length before the drop: " + str(len(interactions_data_frame)))
    #logging.debug('- - - - Number of distinct query ids : {:d}'.format(interactions_data_frame['query_ID'].nunique()))
    interactions_data_frame = interactions_data_frame[interactions_data_frame.relevance_count > 1]
    #logging.debug("- - - - Dataframe length after the drop: " + str(len(interactions_data_frame)))
    #logging.debug('- - - - Number of distinct query ids : {:d}'.format(interactions_data_frame[
    #                                                               'query_ID'].nunique()))
    interactions_data_frame.drop(columns=['relevance_count'], inplace=True)
    interactions_data_frame = interactions_data_frame.reset_index(drop=True)
    return interactions_data_frame


def training_test_set_split(interactions_data_frame, query_id_sample_threshold, percentage, test_set_size):
    # Separate from the remaining dataframe the rows for only train and those from which we take the test set
    under_sampled_only_train = interactions_data_frame[interactions_data_frame.queryIdCount <
                                                       query_id_sample_threshold/percentage]
    check_query = under_sampled_only_train['query_ID'].unique()
    to_steal_from = interactions_data_frame[interactions_data_frame.queryIdCount >=
                                            query_id_sample_threshold/percentage]
    # Separate in training and test set
    to_steal_from_group = to_steal_from.groupby(['query_ID', 'Ranking'])
    flags = (to_steal_from_group.cumcount() + 1) <= to_steal_from_group['Ranking'].transform(
        'size') * percentage
    to_steal_from = to_steal_from.assign(to_steal=flags)
    interactions_test = to_steal_from[to_steal_from.to_steal == True]
    interactions_train = to_steal_from[to_steal_from.to_steal == False]
    # If we don't achieve test size we add an entire query id group from the only train
    if len(interactions_test) < test_set_size and len(
            under_sampled_only_train[under_sampled_only_train.queryIdCount >= query_id_sample_threshold]) > 0:
        to_steal_from = under_sampled_only_train[under_sampled_only_train.queryIdCount >= query_id_sample_threshold]
        to_steal_from_group = to_steal_from.groupby('query_ID')
        to_steal_from_group_ordered = to_steal_from_group.size().sort_values(ascending=False)
        i = 0
        while len(interactions_test) < test_set_size and i < len(to_steal_from_group_ordered):
            to_move = to_steal_from[to_steal_from.query_ID == to_steal_from_group_ordered.index.values[i]]
            interactions_test = pd.concat([interactions_test, to_move], ignore_index=True, sort=False)
            under_sampled_only_train = pd.concat([under_sampled_only_train, to_move], sort=False)
            under_sampled_only_train = under_sampled_only_train.reset_index().drop_duplicates(
                under_sampled_only_train.columns, keep=False).set_index('index')
            i = i + 1
    interactions_train = pd.concat([interactions_train, under_sampled_only_train], sort=False)
    interactions_train = interactions_train.drop(columns=['queryIdCount', 'to_steal'])
    #query_group_id_keys_features = features.get_query_group_id_keys()
    #interactions_test = interactions_test.drop(columns=query_group_id_keys_features, errors='ignore')
    interactions_test = interactions_test.drop(columns=['queryIdCount', 'to_steal'])
    return interactions_train, interactions_test


def mapping_relevance_label(mapping):
    if mapping == "10":
        # Group the position values to relevance labels from 0 to 10
        d = {range(0, 2): 10, range(2, 3): 9, range(3, 4): 8, range(4, 6): 7,
             range(6, 11): 6, range(11, 21): 5, range(21, 36): 4, range(36, 56): 3,
             range(56, 81): 2, range(81, 131): 1, range(131, 201): 0}
    elif mapping == "20":
        # Group the position values to relevance labels from 0 to 20
        d = {range(1, 2): 20, range(2, 3): 19, range(3, 4): 18, range(4, 5): 17, range(5, 6): 16, range(6, 7): 15,
             range(7, 8): 14, range(8, 9): 13, range(9, 10): 12,
             range(10, 11): 11, range(11, 14): 10, range(14, 19): 9, range(19, 26): 8, range(26, 36): 7,
             range(36, 46): 6,
             range(46, 61): 5, range(61, 76): 4, range(76, 96): 3,
             range(96, 116): 2, range(116, 151): 1, range(151, 201): 0}
    else:
        # Group the position values to relevance labels from 0 to 10
        d = {range(0, 2): 10, range(2, 3): 9, range(3, 4): 8, range(4, 6): 7,
             range(6, 11): 6, range(11, 21): 5, range(21, 36): 4, range(36, 56): 3,
             range(56, 81): 2, range(81, 131): 1, range(131, 201): 0}

    return d


def training_set_builder(output_dir, dataset_name, mapping):

    newds_store = pd.HDFStore(output_dir+'/'+dataset_name+'.h5', 'r')
    newds = newds_store[dataset_name]
    newds_store.close()

    d = mapping_relevance_label(mapping)

    newds['Ranking'] = newds['Position'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))
    #newds.drop(columns=['Position'], inplace=True)

    #relevance_counts = newds.groupby("query_ID")["Ranking"].value_counts()

    # create the training set to train the model
    write_training_test_set(newds, output_dir, test_set_size=688239)
