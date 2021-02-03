import pandas as pd
import numpy as np
import statistics

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def ndcg_dataframe(data_frame, queries, relevance_label):
    # Calculate the NDCG at k on a dataframe
    query_groups = data_frame.query_ID.unique()
    ndcg_scores_list = []
    for n in query_groups:
        grouped = data_frame[data_frame[queries] == n]
        #grouped_index = np.arange(1, len(grouped) + 1)
        ndcg = ndcg_at_k(grouped[relevance_label], 10)
        ndcg_scores_list.append(ndcg)
    final_ndcg = statistics.mean(ndcg_scores_list)
    #logging.info('- - - - The final ndcg is: ' + str(final_ndcg))
    print("The final ndcg is: " + str(final_ndcg))


def keep_only_maximum_stream(data_frame, queries, ids):
    # Create a dataframe only with the maximum number of streams for each song for each query
    query_groups = data_frame[queries].unique()
    ids_groups = data_frame[ids].unique()
    column_names = data_frame.columns
    df = pd.DataFrame(columns=column_names)
    for n in query_groups:
        grouped_queries = data_frame[data_frame[queries] == n].reset_index(drop=True)
        for i in ids_groups:
            if (grouped_queries[ids] == i).any():
                grouped_ids = grouped_queries[grouped_queries[ids] == i].iloc[[0]]
                df = df.append(grouped_ids, ignore_index=True)
            else:
                continue
    return df

def streams_ordering(dataset_path, largest_streams_dataset_path):
    # Load the dataset
    newds_store = pd.HDFStore(dataset_path + "/newds_hash.h5", 'r')
    newds = newds_store['newds_hash']
    newds_store.close()

    # Group the position values to relevance labels from 0 to 20
    d = {range(1, 2): 20, range(2, 3): 19, range(3, 4): 18, range(4, 5): 17, range(5, 6): 16, range(6, 7): 15,
         range(7, 8): 14, range(8, 9): 13, range(9, 10): 12,
         range(10, 11): 11, range(11, 14): 10, range(14, 19): 9, range(19, 26): 8, range(26, 36): 7, range(36, 46): 6,
         range(46, 61): 5, range(61, 76): 4, range(76, 96): 3,
         range(96, 116): 2, range(116, 151): 1, range(151, 201): 0}

    newds['Ranking'] = newds['Position'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))

    # Check correlation between Position and Streams and Streams values counts
    print("Correlation between Position and Streams is: ", newds['Streams'].corr(newds['Position']))
    # Print the correlation matrix
    correlation_matrix_newds = newds.corr()

    #streams_count = newds['Streams'].value_counts()
    #streams_count_query_group = newds.groupby('query_ID')['Streams'].value_counts()
    #streams_sort_values = newds['Streams'].sort_values(ascending=False)

    # Check correlation between Ranking and Streams
    print("Correlation between Ranking and Streams is: ", newds['Streams'].corr(newds['Ranking']))
    # Check the largest Streams value per query group (list of the first 10)
    largest10_streams_list_query_group = newds.groupby(['query_ID'])['Streams'].nlargest(10)

    # New dataframe, sorted by Streams values per each query
    newds_sorted = newds.groupby(['query_ID']).apply(lambda x: x.sort_values(["Streams"], ascending=False)).reset_index(
        drop=True)
    # Calcutate the ndcg on  a dataframe
    ndcg_dataframe(newds_sorted, 'query_ID', 'Ranking')

    # New dataframe with the largest streams, sorted by Streams values per each query
    #dataset_largest_stream = keep_only_maximum_stream(newds_sorted, 'query_ID', 'ID')
    #dataset_largest_stream.to_csv(largest_streams_dataset_path + '/spotify_largest_streams.csv', index=False)

    dataset_largest_stream = pd.read_csv(largest_streams_dataset_path + '/spotify_largest_streams.csv')
    #print("Correlation between Position and Streams is: ", dataset_largest_stream['Streams'].corr(dataset_largest_stream['Position']))
    #correlation_matrix_df = dataset_largest_stream.corr()
    #print("Correlation between Ranking and Streams is: ", dataset_largest_stream['Streams'].corr(dataset_largest_stream['Ranking']))

    newds_largeststream_sorted = dataset_largest_stream.groupby(['query_ID']).apply(lambda x: x.sort_values(["Streams"], ascending=False)).reset_index(
        drop=True)
    #calcutate the ndcg
    ndcg_dataframe(newds_largeststream_sorted, 'query_ID', 'Ranking')


