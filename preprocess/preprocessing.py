import pandas as pd
import string
import re
import category_encoders as ce
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def cleanData(text):
    text = re.sub('[' + string.punctuation + ']', '', str(text).lower())
    text = re.sub(r'\n', r' ', text)
    return text

def write_set(dataset, output_dir, set_name):
    # Binary save
    store = pd.HDFStore(output_dir+'/'+set_name+'.h5')
    store[set_name] = dataset
    store.close()

def LOU_encoding(dataset, variable_to_encode, target_variable):
    # Leave One Out encoding
    encoder = ce.LeaveOneOutEncoder(cols=[variable_to_encode]).fit(dataset, dataset[target_variable])
    new_dataset = encoder.transform(dataset)
    return new_dataset

def hash_encoding(dataset, variable_to_encode):
    # Hash encoding
    encoder = ce.HashingEncoder(cols=[variable_to_encode], n_components=8)
    new_dataset = encoder.fit_transform(dataset)
    return new_dataset

def one_hot(dataset, variable_to_encode):
    # One Hot encoding
    encoder = ce.OneHotEncoder(cols=[variable_to_encode])
    new_dataset = encoder.fit_transform(dataset)
    return new_dataset

def doc2vec_encoding(dataset, variable_to_encode):
    # Doc2Vec encoding
    dataset[variable_to_encode] = dataset[variable_to_encode].map(lambda x: cleanData(x))
    lines = dataset[variable_to_encode]
    token = []
    for line in lines:
        line = line.split()
        token.append(line)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(token)]
    model = Doc2Vec(documents, min_count=1)
    print(model.wv.vocab)
    vec = model.docvecs.doctag_syn0
    model.save('d2v_model')
    model = Doc2Vec.load('d2v_model')

    lines_vector = pd.DataFrame(vec)
    dataset2 = pd.concat([dataset, lines_vector.set_index(dataset.index)], axis=1)
    dataset2.drop(columns=[variable_to_encode], inplace=True)
    return dataset2


def fillnan(dataset, key, value, newvalue):
    # Check Nan
    print(" \nCount total NaN at each column in a DataFrame : \n\n",
          dataset.isnull().sum())
    data_value = dataset[[key, value]].drop_duplicates().dropna()
    url_value = data_value[key]
    track_value = data_value[value]
    dict_value = dict(zip(url_value, track_value))
    # Fill NaN using the dictionary
    data_value_filled = dataset[[key, value]]
    data_value_filled.set_index([key], drop=False, inplace=True)
    data_value_filled[key].update(pd.Series(dict_value))
    data_value_filled = data_value_filled.rename(columns={key: newvalue})
    print(" \nCount total NaN at each column in a DataFrame : \n\n",
          data_value_filled.isnull().sum())
    value_filled = data_value_filled[newvalue]
    return value_filled

def preprocessing(output_dir, dataset_path, encoding):
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
    title_filled = fillnan(ds, "ID", "Title", "Track")
    # Create a dictionary ID-Artist and fill NaN in the Artist column
    artist_filled = fillnan(ds, "ID", "Artist", "Artists")

    data = pd.concat([title_filled, artist_filled], axis=1)
    ds.drop(columns=['Title'], inplace=True)
    ds.drop(columns=['Artist'], inplace=True)
    newds = ds.join(data.set_index(ds.index))
    print(" \nCount total NaN at each column in a DataFrame : \n\n",
          newds.isnull().sum())

    # Save the new dataset to csv
    newds.to_csv(dataset_path+'/spotify_NaNfilled.csv', index=False)

    # Create the query_ID from the Region column
    newds['query_ID'] = pd.factorize(newds['Region'])[0]
    newds.drop(columns=['Region'], inplace=True)
    #print(newds['query_ID'].value_counts())
    #print(newds['Streams'].value_counts())

    # Split the 'Date' column in day, month and year and extract day of the week feature
    split_date = newds['Date'].str.split(pat='-', expand=True).astype(int)
    newds['Date'] = pd.to_datetime(newds['Date'], format='%Y-%m-%d')
    newds['Weekday'] = newds['Date'].dt.dayofweek
    newds['Year'] = split_date[0]
    newds['Month'] = split_date[1]
    newds['Day'] = split_date[2]
    newds.drop(columns=['Date'], inplace=True)

    # Leave one out encoding for 'Artists' feature
    # Fit and Transform Data
    newds = LOU_encoding(newds, "Artists", "Position")

    if encoding == "hash":
        # Hash encoding for 'Track' feature
        newds2 = hash_encoding(newds, "Track")
    elif encoding == "d2v":
        # Doc2vec for 'Track' feature
        newds2 = doc2vec_encoding(newds, "Track")
    else:
        print("no encoding found")

    write_set(newds2, output_dir, 'newds_'+encoding)






