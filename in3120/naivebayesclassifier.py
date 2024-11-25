# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
from collections import Counter
from typing import Any, Dict, Iterable, Iterator
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus
from .document import Document


class NaiveBayesClassifier:
    """
    Defines a multinomial naive Bayes text classifier. For a detailed primer, see
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html.
    """

    def __init__(self, training_set: Dict[str, Corpus], fields: Iterable[str],
                 normalizer: Normalizer, tokenizer: Tokenizer):
        """
        Trains the classifier from the named fields in the documents in the
        given training set.
        """
        # Used for breaking the text up into discrete classification features.
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer

        # The vocabulary we've seen during training.
        self.__vocabulary = InMemoryDictionary()

        # Maps a category c to the logarithm of its prior probability,
        # i.e., c maps to log(Pr(c)).
        self.__priors: Dict[str, float] = {}

        # Maps a category c and a term t to the logarithm of its conditional probability,
        # i.e., (c, t) maps to log(Pr(t | c)).
        self.__conditionals: Dict[str, Dict[str, float]] = {}

        # Maps a category c to the denominator used when doing Laplace smoothing.
        self.__denominators: Dict[str, int] = {}

        # Train the classifier, i.e., estimate all probabilities.
        self.__compute_priors(training_set)
        self.__compute_vocabulary(training_set, fields)
        self.__compute_posteriors(training_set, fields)

    def __compute_priors(self, training_set) -> None:
        """
        Estimates all prior probabilities (log-probabilities).
        
        Her må du implementere kode
        """
        total_docs = sum(len(docs) for docs in training_set.values())
        for category, docs in training_set.items():
            self.__priors[category] = math.log(len(docs) / total_docs)
        
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def __compute_vocabulary(self, training_set, fields) -> None:
        """
        Builds up the overall vocabulary as seen in the training set.
        
        Her må du implementere kode
        """
        for category, docs in training_set.items():
            for doc in docs:
                for field in fields:
                    # Henter termer fra tekstfeltet og legger dem til i ordforrådet
                    # bruker document.py og henter ut field 
                    text = doc.get_field(field, "")
                    terms = self.__get_terms(text)
                    for term in terms:
                        self.__vocabulary.add_if_absent(term)
        
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")


    def __compute_posteriors(self, training_set, fields) -> None:
        """
        Estimates all conditional probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        
        Her må du implementere kode
        """
        for category, docs in training_set.items():
            term_counts = Counter()
            for doc in docs:
                for field in fields:
                    # Henter innhold fra feltet og normaliserer det
                    text = doc.get_field(field, "")
                    # Teller opp antall forekomster av hver term
                    term_counts.update(self.__get_terms(text))

            
            denominator = sum(term_counts.values()) + self.__vocabulary.size()
            self.__denominators[category] = denominator
            self.__conditionals[category] = {}
            for term, term_id in self.__vocabulary:
                term_count = term_counts.get(term, 0)
                # beregner sannsyneligheten
                probability = (term_count + 1) / denominator
                # Lagrer logaritmen
                self.__conditionals[category][term_id] = math.log(probability)
        
        
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")


    def __get_terms(self, buffer) -> Iterator[str]:
        """
        Processes the given text buffer and returns the sequence of normalized
        terms as they appear. Both the documents in the training set and the buffers
        we classify need to be identically processed.
        """
        tokens = self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))
        return (self.__normalizer.normalize(t) for t in tokens)

    def get_prior(self, category: str) -> float:
        """
        Given a category c, returns the category's prior log-probability log(Pr(c)).

        This is an internal detail having public visibility to facilitate testing.
        
        Denne kan du la stå
        """
        return self.__priors[category]

    def get_posterior(self, category: str, term: str) -> float:
        """
        Given a category c and a term t, returns the posterior log-probability log(Pr(t | c)).

        This is an internal detail having public visibility to facilitate testing.
        
        Implementer kode her
        """
        term_id = self.__vocabulary.get_term_id(term)
        if term_id is not None:
            # Henter betinget sannsynlighet hvis term finnes i ordforrådet
            return self.__conditionals[category].get(term_id, math.log(1 / self.__denominators[category]))
        return 0.0 # Returnerer 0 hvis termen ikke finnes
    
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")


    def classify(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Classifies the given buffer according to the multinomial naive Bayes rule. The computed (score, category) pairs
        are emitted back to the client via the supplied callback sorted according to the scores. The reported scores
        are log-probabilities, to minimize numerical underflow issues. Logarithms are base e.

        The results yielded back to the client are dictionaries having the keys "score" (float) and
        "category" (str).
        
        Her må du implementere kode. 
        """
        
        # Henter termer fra buffer som finnes i ordforrådet
        terms = self.__get_terms(buffer)
        observed_terms = [term for term in terms if self.__vocabulary.get_term_id(term) is not None]

        scores = []
        
        # Itererer gjennom hver kategori for å beregne total score
        for category in self.__priors:
            score = self.get_prior(category)
            for term in observed_terms:
                score += self.get_posterior(category, term)
            scores.append({"score": score, "category": category})

        
        scores.sort(key=lambda x: x["score"], reverse=True)

        for result in scores:
            yield result
        
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

