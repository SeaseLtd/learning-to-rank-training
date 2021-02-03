import pandas as pd
from preprocess import preprocessing

def generate_id_from_columns(input_data_frame, query_features, id_column):
    str_id = input_data_frame[query_features[0]].astype(str)
    for feature in query_features[1:]:
        str_id = str_id + '_' + input_data_frame[feature].astype(str)
    input_data_frame[id_column] = pd.factorize(str_id)[0]


def preprocessing_new_query_ID(output_dir, dataset_path, encoding):
    # Load the dataset with NaN filled (spotify_NaNfilled.csv)
    newds = pd.read_csv(dataset_path)

    # Start Features Engineering
    # Split the 'Date' feature in day, month and year and extract day of the week
    split_date = newds['Date'].str.split(pat='-', expand=True).astype(int)
    newds['Date'] = pd.to_datetime(newds['Date'], format='%Y-%m-%d')
    newds['Weekday'] = newds['Date'].dt.dayofweek
    newds['Year'] = split_date[0]
    newds['Month'] = split_date[1]
    newds['Day'] = split_date[2]
    newds.drop(columns=['Date'], inplace=True)
    newds.drop(columns=['Year'], inplace=True)

    # Generate the query_ID as an hash of multiple query-level features
    query_features = ['Region', 'Day', 'Month', 'Weekday']
    generate_id_from_columns(newds, query_features, id_column='query_ID')
    unique_query_ID = newds['query_ID'].unique()
    query_ID_value_counts = newds['query_ID'].value_counts()
    newds.drop(columns=['Region'], inplace=True)
    newds.drop(columns=['Day'], inplace=True)
    newds.drop(columns=['Month'], inplace=True)
    newds.drop(columns=['Weekday'], inplace=True)

    # Leave one out encoding for 'Artists' feature
    # Fit and Transform Data
    newds = preprocessing.LOU_encoding(newds, "Artists", "Position")

    # choose which technique to encode the 'Track' feature
    if encoding == "hash":
        # Hash encoding
        newds2 = preprocessing.hash_encoding(newds, "Track")
    elif encoding == "d2v":
        # doc2vec encoding
        newds2 = preprocessing.doc2vec_encoding(newds, "Track")
    else:
        print("no encoding found")

    preprocessing.write_set(newds2, output_dir, 'newds_'+encoding)

