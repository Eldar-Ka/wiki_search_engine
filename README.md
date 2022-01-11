# wiki_search_engine

  ## code structure:

search_frontend.py - initiallization of the search engine localy

inverted_index_colab.py - indexer of documents which creates postings list for the search engine localy

search_frontend_vm.py - initiallization of the search engine on the VM machine over GCP

inverted_index_gcp.py - indexer of documents which creates postings list for the search engine over GCP


  ## functionality:

search()- based on body and anchor search, returns up to a 100 of the best search results for the query

get_pagerank()-Returns PageRank values for a list of provided wiki article IDs

get_pageview() - Returns the number of page views that each of the provide wiki articles had

pg_table()-Return a pandas DF pagerank of all the wiki_docs

binary_search_by_index(query, index, N=-1) - implementation of the title and anchor search

search_by_body_index(query_to_search, N=-1) -  implementation of the body search using cosine similarity for all the wiki articles

 build_inverted_index()- Builds an indexer for the entire english wiki corpus
