# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# conda activate myenv

import math
from .ranker import Ranker
from .corpus import Corpus
from .posting import Posting
from .invertedindex import InvertedIndex


class BetterRanker(Ranker):
    """
    A ranker that does traditional TF-IDF ranking, possibly combining it with
    a static document score (if present).

    The static document score is assumed accessible in a document field named
    "static_quality_score". If the field is missing or doesn't have a value, a
    default value of 0.0 is assumed for the static document score.

    See Section 7.1.4 in https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf.
    
    Her skal du implementere noe 
    """

    # These values could be made configurable. Hardcode them for now.
    _dynamic_score_weight = 1.0
    _static_score_weight = 1.0
    _static_score_field_name = "static_quality_score"
    _static_score_default_value = 0.0

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self._score = 0.0
        self._document_id = None
        self._corpus = corpus
        self._inverted_index = inverted_index

    def reset(self, document_id: int) -> None:
        self._score = 0.0
        self._document_id = document_id

    def update(self, term: str, multiplicity: int, posting: Posting) -> None:
        assert self._document_id == posting.document_id # Sjekker om posting tilhører riktig dokument 
        
        term_frequency = posting.term_frequency # Henter ut termfrekvensene (TF)
        document_frequency = self._inverted_index.get_document_frequency(term) #Henter Dokument Frekensen (DF)
        total_documents = len(self._corpus) # henter ut totalt anntall dokumenter i korpused 
        
        # Regner ut IDF og TF-IDF scoren
        idf = math.log(total_documents / document_frequency)
        tf_idf = term_frequency * idf 
        self._score += tf_idf# Legger til TF-IDF til scoren 
        
    
    def evaluate(self) -> float:
        # Hent statisk score fra dokumentet. Hvis feltet mangler, bruk standardverdien.
        static_score = self._corpus.get_document(self._document_id).get_field(self._static_score_field_name, self._static_score_default_value ) # Et rot? ja ah
        # Returner total poengsum som er summen av TF-IDF score (self._score) og den statiske scoren addet med self._static_score_weight. 
        return self._score + (self._static_score_weight * static_score)