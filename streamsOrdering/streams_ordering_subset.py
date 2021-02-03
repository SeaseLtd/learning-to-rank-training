import pandas as pd
import numpy as np
import statistics
from streamsOrdering import streams_ordering


def streams_ordering_subset(dataset_path, largest_streams_dataset_path):
    newds_store = pd.HDFStore(dataset_path + "/newds_hash.h5", 'r')
    newds = newds_store['newds_hash']
    newds_store.close()

    # Create the Subset, only with Top 21 songs
    subs = newds[newds['Position'] <= 21]

    # Group the position values to relevance labels from 0 to 20
    d = {range(1, 2): 20, range(2, 3): 19, range(3, 4): 18, range(4, 5): 17, range(5, 6): 16, range(6, 7): 15,
         range(7, 8): 14, range(8, 9): 13, range(9, 10): 12,
         range(10, 11): 11, range(11, 12): 10, range(12, 13): 9, range(13, 14): 8, range(14, 15): 7, range(15, 16): 6,
         range(16, 17): 5, range(17, 18): 4, range(18, 19): 3,
         range(19, 20): 2, range(20, 21): 1, range(21, 22): 0}

    subs['Ranking'] = subs['Position'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))

    # Check correlation between Position and Streams and Streams values counts
    print("Correlation between Position and Streams is: ", subs['Streams'].corr(subs['Position']))
    # Correlation matrix
    correlation_matrix_subds = subs.corr()
    # Check correlation between Ranking and Streams and Streams values counts
    print("Correlation between Ranking and Streams is: ", subs['Streams'].corr(subs['Ranking']))

    # New dataframe, sorted by Streams values per each query
    subs_sorted = subs.groupby(['query_ID']).apply(lambda x: x.sort_values(["Streams"], ascending=False)).reset_index(
        drop=True)
    streams_ordering.ndcg_dataframe(subs_sorted, 'query_ID', 'Ranking')

    # New dataframe (Subset) with the largest streams, sorted by Streams values per each query
    #subs_largest_stream = streams_ordering.keep_only_maximum_stream(subs_sorted, 'query_ID', 'ID')
    #subs_largest_stream.to_csv(largest_streams_dataset_path + '/spotify_largest_streams_subset.csv', index=False)

    subs_largest_stream = pd.read_csv(largest_streams_dataset_path + "/spotify_largest_streams_subset.csv")
    newsubs_sorted = subs_largest_stream.groupby(['query_ID']).apply(lambda x: x.sort_values(["Streams"], ascending=False)).reset_index(drop=True)
    streams_ordering.ndcg_dataframe(newsubs_sorted, 'query_ID', 'Ranking')
