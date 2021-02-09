import pysolr
import json

def books():
    books = json.loads(open('books.json').read())
    for book in books:
        id = book['metadata']['id']
        print("Indexing %s" % id)
        try:
            yield {'id': id,
                   'title': book['bibliography']['title'],
                   'author': book['bibliography']['author']['name'],
                   'downloads': book['metadata']['downloads'],
                   'languages': [language for language in book['bibliography']['languages']],
                   'subjects': [subject for subject in book['bibliography']['subjects']]}
        except KeyError as k:
            print(k)
            continue


if __name__ == "__main__":
    solr = pysolr.Solr('http://localhost:8983/solr/books', timeout=100)
    solr.add(books())
