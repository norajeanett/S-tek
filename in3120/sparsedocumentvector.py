# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

from __future__ import annotations
from typing import Iterable, Iterator, Dict, Tuple, Optional
from math import sqrt
from .sieve import Sieve

class SparseDocumentVector:
    """
    A simple representation of a sparse document vector. The vector space has one dimension
    per vocabulary term, and our representation only lists the dimensions that have non-zero
    values.

    Being able to place text buffers, be they documents or queries, in a vector space and
    thinking of them as point clouds (or, equivalently, as vectors from the origin) enables us
    to numerically assess how similar they are according to some suitable metric. Cosine
    similarity (the inner product of the vectors normalized by their lengths) is a very
    common metric.
    
    Du skal implementere i denne filen 
    """

    def __init__(self, values: Dict[str, float]):
        # An alternative, effective representation would be as a
        # [(term identifier, weight)] list kept sorted by integer
        # term identifiers. Computing dot products would then be done
        # pretty much in the same way we do posting list AND-scans.
        self._values = dict(values) # Lurer på om denne bør endres for at den siste testen skal komme gjennom, får det ikke helt til slik jeg vil 
        
        
        #lage en for loop her slik at test_only_non_zero_elements_are_kept kjører, 

        # We cache the length. It might get used over and over, e.g., for cosine
        # computations. A value of None triggers lazy computation.
        self._length : Optional[float] = None

    def __iter__(self):
        return iter(self._values.items())

    def __getitem__(self, term: str) -> float:
        return self._values.get(term, 0.0)

    def __setitem__(self, term: str, weight: float) -> None:
        self._values[term] = weight
        self._length = None

    def __contains__(self, term: str) -> bool:
        return term in self._values

    def __len__(self) -> int:
        """
        Enables use of the built-in len/1 function to count the number of non-zero
        dimensions in the vector. It is not for computing the vector's norm.
        """
        return len(self._values)

    def get_length(self) -> float:
        """
        Returns the length (L^2 norm, also called the Euclidian norm) of the vector.
        """
        if self._length is None:
            self._length = sqrt(sum(weight ** 2 for weight in self._values.values()))
        return self._length 
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def normalize(self) -> None:
        """
        Divides all weights by the length of the vector, thus rescaling it to
        have unit length.
        """
        lenght = self.get_length()
        if lenght > 0:
            for term in self._values:
                self._values[term] /= lenght
            self._length = 1.0
        else:
            self._length = 0.0
        
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def top(self, count: int) -> Iterable[Tuple[str, float]]:
        """
        Returns the top weighted terms, i.e., the "most important" terms and their weights.
        """
        assert count >= 0
        return sorted(self._values.items(), key=lambda item: item[1], reverse=True)[:count]
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def truncate(self, count: int) -> None:
        """
        Truncates the vector so that it contains no more than the given number of terms,
        by removing the lowest-weighted terms.
        """
        assert count >= 0 
        top_terms = self.top(count)
        self._values = dict(top_terms)
        self._length = None
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def scale(self, factor: float) -> None:
        """
        Multiplies every vector component by the given factor.
        """
        
        for term in self._values:
            self._values[term] *= factor
        self._length = None
        
        
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def dot(self, other: SparseDocumentVector) -> float:
        """
        Returns the dot product (inner product, scalar product) between this vector
        and the other vector.
        """
        return sum(self._values.get(term, 0.0) * other[term] for term in self._values)
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def cosine(self, other: SparseDocumentVector) -> float:
        """
        Returns the cosine of the angle between this vector and the other vector.
        See also https://en.wikipedia.org/wiki/Cosine_similarity.
        """
        dot_product = self.dot(other)
        length_self = self.get_length()
        length_other = other.get_length()
        if length_self == 0 or length_other == 0:
            return 0.0
        return dot_product / (length_self * length_other)       
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    @staticmethod
    def centroid(vectors: Iterator[SparseDocumentVector]) -> SparseDocumentVector:
        """
        Computes the centroid of all the vectors, i.e., the average vector.
        """
        centroid_values = {}
        count = 0
        for vector in vectors:
            for term, weight in vector:
                centroid_values[term] = centroid_values.get(term, 0.0) + weight 
            count += 1
        if count > 0:
            for term in centroid_values:
                centroid_values[term] /= count
        return SparseDocumentVector(centroid_values)
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
        
    