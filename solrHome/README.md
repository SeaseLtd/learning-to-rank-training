Solr Index for the [Gutemberg Project](https://www.gutenberg.org).

# Start up Solr

1. Download and unpack [Solr 8.1.1](https://www.apache.org/dyn/closer.lua/lucene/solr/8.1.1/solr-8.1.1.zip)
2. Run Solr pointing at the Solr Home directory for the Relevance Course

```
./bin/solr start -f -s /path/to/relevanceCourse/
```

In your browser, navigate to "http://localhost:8983/solr/" to confirm Solr is up and running

# Index Gutemberg Project Books

1. Verify you can open books.json - this dataset has been extracted from the Gutemberg Project by [corgis](https://think.cs.vt.edu/corgis/json/index.html)
2. Install [Python 3.6](https://www.python.org/downloads/) and the [pysolr library](https://github.com/django-haystack/pysolr) library - pip install pysolr
3. Run `python index.py` to index the books

# Confirm Solr has around 1000 books

Navigate [here](http://localhost:8983/solr/books/select?q=*:*) and confirm you get results.
