import math

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
# %load_ext google.colab.data_table
import bz2
from functools import partial
from collections import Counter, OrderedDict
import pickle
import heapq
from itertools import islice, count, groupby
from xml.etree import ElementTree
import codecs
import csv
import os
import re
import gzip
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# %matplotlib inline
from pathlib import Path
import itertools
from time import time
import hashlib
from inverted_index_colab import *

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from graphframes import *


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        with open("postings/index_body.pkl", 'rb') as f:
            self.index_body = pickle.loads(f.read())
        with open("postings/index_anchor.pkl", 'rb') as f:
            self.index_anchor = pickle.loads(f.read())
        with open("postings/index_title.pkl", 'rb') as f:
            self.index_title = pickle.loads(f.read())
        with open("pageview/pageviews-202108-user.pkl", 'rb') as f:
            self.pageview = dict(pickle.loads(f.read()))
        self.df = pd.read_csv('wikidumps/pr.csv')
        self.titles = pd.read_csv('wikidumps/titles.csv')
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


nltk.download('stopwords', quiet=True)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = search_by_body_index(query)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = binary_search_by_index(query, app.index_title)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = binary_search_by_index(query, app.index_anchor)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    """
    # return the selected pageRank
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for doc in wiki_ids:
        val = app.df.loc[app.df["id"] == doc]["pagerank"].values
        if len(val) == 0:
            res.append(0)
        else:
            res.append(val[0])
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    """
    # return the selected pageView
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [app.pageview.get(i, "page not exist") for i in wiki_ids]
    # END SOLUTION
    return jsonify(res)


def pg_table():
    # return pandas table of all the docs
    spark = SparkSession.builder.getOrCreate()
    pages_links = spark.read.parquet("wikidumps/multistream1_preprocessed.parquet").limit(1000).select("id",
                                                                                                       "anchor_text").rdd

    def help(id, anchor):
        dic = {}
        for elem in anchor:
            dic[elem[0]] = 1
        lst = []
        for key in dic:
            lst.append((id, key))
        return lst

    def generate_graph(pages):
        """ Compute the directed graph generated by wiki links.
        Parameters:
        -----------
          pages: RDD
            An RDD where each row consists of one wikipedia articles with 'id' and
            'anchor_text'.
        Returns:
        --------
          edges: RDD
            An RDD where each row represents an edge in the directed graph created by
            the wikipedia links. The first entry should the source page id and the
            second entry is the destination page id. No duplicates should be present.
          vertices: RDD
            An RDD where each row represents a vetrix (node) in the directed graph
            created by the wikipedia links. No duplicates should be present.
        """
        edges = pages.flatMap(lambda x: help(x[0], x[1]))
        vertices1 = edges.flatMap(lambda x: (x[0], x[1]))
        vertices = vertices1.distinct()
        vertices = vertices.map(lambda x: [x])
        return edges, vertices

    # construct the graph for a small sample of (1000) pages
    edges, vertices = generate_graph(pages_links)
    # time the actual execution
    edgesDF = edges.toDF(['src', 'dst']).repartition(4, 'src')
    verticesDF = vertices.toDF(['id']).repartition(4, 'id')
    g = GraphFrame(verticesDF, edgesDF)
    pr_results = g.pageRank(resetProbability=0.15, maxIter=10)
    pr = pr_results.vertices.select("id", "pagerank")
    pr = pr.sort(col('pagerank').desc())
    pr.repartition(1).write.csv('pr', compression="gzip")
    return pr.toPandas()

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))


def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens


def binary_search_by_index(query, index, N=-1):

    words, pls = tuple(zip(*list(map(lambda tup: tup, index.posting_lists_iter()))))
    query_to_search = tokenize(query)

    candidates = {}

    N = len(app.titles.index)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            couter = [(doc_id, freq)
                               for
                               doc_id, freq in list_of_doc]

            for doc_id, freq in couter:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + freq

    result = sorted([(doc_id[0], freq) for doc_id, freq in candidates.items()],
                    key=lambda x: x[1],
                    reverse=True)
    if N != -1:
        result = result[:N]

    print("result")
    print(result)
    result = [(id, app.titles.loc[app.titles["id"] == id]["title"].values[0]) for id, freq in result]

    return result


def search_by_body_index(query_to_search, N=-1):

    def get_candidate_documents_and_scores(query_to_search, index, words, pls):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: generator for working with posting.
        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        candidates = {}
        N = len(app.titles.index)
        for term in np.unique(query_to_search):
            if term in words:
                list_of_doc = pls[words.index(term)]
                normlized_tfidf = [(doc_id,
                                    (freq / app.titles.loc[app.titles["id"] == doc_id]["text"].values[0]) * math.log(
                                        N / index.df[term], 10))
                                   for
                                   doc_id, freq in list_of_doc]

                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

    def generate_query_tfidf_vector(query_to_search, index):
        """
        Generate a vector representing the query. Each entry within this vector represents a tfidf score.
        The terms representing the query will be the unique terms in the index.

        We will use tfidf on the query as well.
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the query.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """

        epsilon = .0000001
        total_vocab_size = len(index.df)
        Q = np.zeros((total_vocab_size))
        term_vector = list(index.df.keys())
        counter = Counter(query_to_search)
        for token in np.unique(query_to_search):
            if token in index.df.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = math.log(len(app.titles.index) / (df + epsilon), 10)  # smoothing

                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q  # TODO check return 1.0

    def generate_document_tfidf_matrix(query_to_search, index, words, pls):
        """
        Generate a DataFrame `D` of tfidf scores for a given query.
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: generator for working with posting.
        Returns:
        -----------
        DataFrame of tfidf scores.
        """

        total_vocab_size = len(index.df)
        candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                               pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), total_vocab_size))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = index.df.keys()

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key
            D.loc[doc_id][term] = tfidf  # TODO check work ?

        return D

    def get_topN_score_for_query(query_to_search, index, N=-1):
        """
        Generate a dictionary that gathers for every query its topN score.

        Parameters:
        -----------
        query_to_search: query
        index:           inverted index loaded from the corresponding files.
        N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

        Returns:
        -----------
        return: a list of  topN pairs as follows:
                                                            key: query_id
                                                            value: title
        """
        # YOUR CODE HERE
        words, pls = tuple(zip(*list(map(lambda tup: tup, index.posting_lists_iter()))))
        val = tokenize(query_to_search)

        D = generate_document_tfidf_matrix(val, index, words, pls)
        Q = generate_query_tfidf_vector(val, index)

        result = sorted([(doc_id, score) for doc_id, score in cosine_similarity(D, Q).items()],
                        key=lambda x: x[1],
                        reverse=True)
        if N != -1:
            result = result[:N]
        #print("result")
        #print(result)
        result = [(id, app.titles.loc[app.titles["id"] == id]["title"].values[0]) for id, score in result]
        return result

    def cosine_similarity(D, Q):
        """
        Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
        Generate a dictionary of cosine similarity scores
        key: doc_id
        value: cosine similarity score

        Parameters:
        -----------
        D: DataFrame of tfidf scores.

        Q: vectorized query with tfidf scores

        Returns:
        -----------
        dictionary of cosine similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: cosine similarty score.
        """
        # YOUR CODE HERE
        dic = {}
        index = list(D.index)
        tfidf = D.to_numpy()
        i = 0
        for lst in tfidf:
            dic[index[i]] = np.dot(lst, Q) / (np.linalg.norm(lst) * np.linalg.norm(Q))  # TODO check return 1.0
            i += 1

        return dic

    return get_topN_score_for_query(query_to_search, app.index_body, N)


def build_inverted_index():
    spark = SparkSession.builder.getOrCreate()
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = []

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    NUM_BUCKETS = 124

    def _hash(s):
        return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

    def token2bucket_id(token):
        return int(_hash(token), 16) % NUM_BUCKETS

    def word_count(text, id):
        """ Count the frequency of each word in `text` (tf) that is not included in
        `all_stopwords` and return entries that will go into our posting lists.
        Parameters:
        -----------
          text: str
            Text of one document
          id: int
            Document id
        Returns:
        --------
          List of tuples
            A list of (token, (doc_id, tf)) pairs
            for example: [("Anarchism", (12, 5)), ...]
        """
        tokens = [token.group() for token in RE_WORD.finditer(text.lower())]

        lst_c = Counter([token for token in tokens if token not in all_stopwords])
        lst_c = list(lst_c.items())
        lst = []

        for token in lst_c:
            lst.append((token[0], (id, token[1])))

        return lst

    def reduce_word_counts(unsorted_pl):
        """ Returns a sorted posting list by wiki_id.
        Parameters:
        -----------
          unsorted_pl: list of tuples
            A list of (wiki_id, tf) tuples
        Returns:
        --------
          list of tuples
            A sorted posting list.
        """
        return sorted(unsorted_pl)

    def calculate_df(postings):
        """ Takes a posting list RDD and calculate the df for each token.
        Parameters:
        -----------
          postings: RDD
            An RDD where each element is a (token, posting_list) pair.
        Returns:
        --------
          RDD
            An RDD where each element is a (token, df) pair.
        """
        return postings.map(lambda tup: (tup[0], len(tup[1])))

    def partition_postings_and_write(postings):
        """ A function that partitions the posting lists into buckets, writes out
        all posting lists in a bucket to disk, and returns the posting locations for
        each bucket. Partitioning should be done through the use of `token2bucket`
        above. Writing to disk should use the function  `write_a_posting_list`, a
        static method implemented in inverted_index_colab.py under the InvertedIndex
        class.
        Parameters:
        -----------
          postings: RDD
            An RDD where each item is a (w, posting_list) pair.
        Returns:
        --------
          RDD
            An RDD where each item is a posting locations dictionary for a bucket. The
            posting locations maintain a list for each word of file locations and
            offsets its posting list was written to. See `write_a_posting_list` for
            more details.
        """
        rdd = postings.map(lambda tup: (token2bucket_id(tup[0]), [tup]))
        rdd = rdd.reduceByKey(lambda a, b: a + b)
        rdd = rdd.map(lambda tup: InvertedIndex.write_a_posting_list(tup))

        return rdd

    # word counts map
    full_path = "gs://wikidata_preprocessed/*"
    parquetFile = spark.read.parquet(full_path)
    doc_text_pairs = parquetFile.select("text", "id").rdd
    word_counts = doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))
    postings = word_counts.groupByKey().mapValues(reduce_word_counts)
    # filtering postings and calculate df
    postings_filtered = postings.filter(lambda x: len(x[1]) > 50)
    w2df = calculate_df(postings_filtered)
    w2df_dict = w2df.collectAsMap()
    # partition posting lists and write out
    _ = partition_postings_and_write(postings_filtered).collect()


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
