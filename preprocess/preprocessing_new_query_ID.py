import pandas as pd
from preprocess import preprocessing


def generate_id_from_columns(input_data_frame, query_features, id_column):
    str_id = input_data_frame[query_features[0]].astype(str)
    for feature in query_features[1:]:
        str_id = str_id + '_' + input_data_frame[feature].astype(str)
    input_data_frame[id_column] = pd.factorize(str_id)[0]


def preprocessing_new_query_ID(output_dir, dataset_path, encoding):
    # Load the dataset (Spotify)
    ds = pd.read_csv(dataset_path)

    # Rename Title column
    ds = ds.rename(columns={"Track Name": "Title"})

    # Remove the first part of the the URL
    ds['URL'] = ds['URL'].str.replace('https://open.spotify.com/track/', '')
    # Factorize URL
    ds['ID'] = pd.factorize(ds['URL'])[0]
    ds.drop(columns=['URL'], inplace=True)
    print(" \nCount total NaN at each column in a DataFrame prior the FILL NAN : \n\n",
          ds.isnull().sum())
    # Create a dictionary ID-Title and fill NaN in the Title column
    title_filled = preprocessing.fill_nan(ds, "ID", "Title", "Track")
    # Create a dictionary ID-Artist and fill NaN in the Artist column
    artist_filled = preprocessing.fill_nan(ds, "ID", "Artist", "Artists")

    data = pd.concat([title_filled, artist_filled], axis=1)
    ds.drop(columns=['Title'], inplace=True)
    ds.drop(columns=['Artist'], inplace=True)
    newds = ds.join(data.set_index(ds.index))
    print(" \nCount total NaN at each column in a DataFrame : \n\n",
          newds.isnull().sum())

    # Save the new dataset to csv
    newds.to_csv(output_dir + '/spotify_NaNfilled.csv', index=False)

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
    newds = preprocessing.lou_encoding(newds, "Artists", "Position")

    # choose which technique to encode the 'Track' feature
    if encoding == "hash":
        # Hash encoding for 'Track' feature
        newds2 = preprocessing.hash_encoding(newds, "Track")
    elif encoding == "d2v":
        # Doc2vec for 'Track' feature
        newds2 = preprocessing.doc2vec_encoding(newds, "Track")
    elif encoding == "onehot":
        newds2 = preprocessing.one_hot(newds, "Track")
    elif encoding == "binary":
        newds2 = preprocessing.binary_encoding(newds, "Track")
    else:
        newds2 = preprocessing.binary_encoding(newds, "Track")

    preprocessing.write_set(newds2, output_dir, 'newds_'+encoding)
