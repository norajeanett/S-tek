from collections import Counter
from typing import Any, Dict, Iterator, List
from .sieve import Sieve
from .ranker import Ranker
from .corpus import Corpus
from .invertedindex import InvertedIndex


class SimpleSearchEngine:
    """
    Realizes a simple query evaluator that efficiently performs N-of-M matching over an inverted index.
    I.e., if the query contains M unique query terms, each document in the result set should contain at
    least N of these m terms. For example, 2-of-3 matching over the query 'orange apple banana' would be
    logically equivalent to the following predicate:

       (orange AND apple) OR (orange AND banana) OR (apple AND banana)
       
    Note that N-of-M matching can be viewed as a type of "soft AND" evaluation, where the degree of match
    can be smoothly controlled to mimic either an OR evaluation (1-of-M), or an AND evaluation (M-of-M),
    or something in between.

    The evaluator uses the client-supplied ratio T = N/M as a parameter as specified by the client on a
    per query basis. For example, for the query 'john paul george ringo' we have M = 4 and a specified
    threshold of T = 0.7 would imply that at least 3 of the 4 query terms have to be present in a matching
    document.
    """

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self.__corpus = corpus
        self.__inverted_index = inverted_index

    def evaluate(self, query: str, options: Dict[str, Any], ranker: Ranker) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing N-out-of-M ranked retrieval. I.e., for a supplied query having M
        unique terms, a document is considered to be a match if it contains at least N <= M of those terms.

        The matching documents, if any, are ranked by the supplied ranker, and only the "best" matches are yielded
        back to the client as dictionaries having the keys "score" (float) and "document" (Document).

        The client can supply a dictionary of options that controls the query evaluation process: The value of
        N is inferred from the query via the "match_threshold" (float) option, and the maximum number of documents
        to return to the client is controlled via the "hit_count" (int) option.
        """
        
        terms_count = Counter(self.__inverted_index.get_terms(query))
        terms = list(terms_count.keys())
        threshold = options.get("match_threshold")
        m = len(terms)
        n = max(1, min(m, int(threshold * m)))
        
        # Henter postings for hver term 
        postings_iterators = [self.__inverted_index.get_postings_iterator(term) for term in terms]
        postings_map = [next(posting, None) for posting in postings_iterators]
        results = []

        # Lager en while loop som kjører hvis det er postings å prosessere 
        while any(postings_map):
            doc_id = min(posting.document_id for posting in postings_map if posting is not None) #fikse?liker ikke denne
            # Teller hvor mange termer som matcher doc_id
            doc_match_count = sum(1 for posting in postings_map if posting and posting.document_id == doc_id) #ah samme
            # Hvis dokumentet matcher minst N termer, kjør
            if doc_match_count >= n:
                ranker.reset(doc_id) 
                for term_index, term in enumerate(terms):
                    posting = postings_map[term_index]
                    if posting and posting.document_id == doc_id:
                        ranker.update(term, terms_count[term], posting) 
                results.append((ranker.evaluate(), doc_id)) # Legger til dokumentene i resultatene
            # vi går videre til neste posting for de termene som matcher oppdatert dokument
            postings_map = [next(postings_iterators[term_index], None) if posting and posting.document_id == doc_id else posting for term_index, posting in enumerate(postings_map)]
        sieve = Sieve(max(1, min(100, options.get("hit_count", 10)))) # Tok fra mattermost 
        
        
        for score, doc_id in results:
            sieve.sift(score, doc_id) # Tatt fra suffix  

        # Returner resultatene i rangert rekkefølge fra Sieve, tatt fra suffixarray
        for score, doc_id in sieve.winners():
            yield {'document': self.__corpus.get_document(doc_id), 'score': score}
