import pysolr
import json


if __name__ == "__main__":
    solr = pysolr.Solr('http://localhost:8983/solr/books', timeout=100)

    books = json.loads(open('books.json').read())
    documents = []

    for book in books:
        id = book['metadata']['id']
        print("Indexing %s" % id)
        doc = {'id': id,
                   'title': book['bibliography']['title'],
                   'author': book['bibliography']['author']['name'],
                   'downloads': book['metadata']['downloads'],
                   'languages': [language for language in book['bibliography']['languages']],
                   'subjects': [subject for subject in book['bibliography']['subjects']]}
        documents.append(doc)
    solr.add(documents)
    print("finished")
