from typing import Iterator, Dict, Any, List, Tuple, Optional
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications and some minor
    NLP extensions. The running time of this algorithm is virtually independent of the size of the dictionary,
    and linear in the length of the buffer we are searching in.

    The tokenizer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    def __init__(self, trie: Trie, normalizer: Normalizer, tokenizer: Tokenizer):
        self.__trie = trie
        self.__normalizer = normalizer  # The same as was used for trie building.
        self.__tokenizer = tokenizer  # The same as was used for trie building.

    def scan(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Scans the given buffer and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        The matches, if any, are yielded back to the client as dictionaries having the keys "match" (str),
        "surface" (str), "meta" (Optional[Any]), and "span" (Tuple[int, int]). Note that "match" refers to
        the matching dictionary entry, "surface" refers to the content of the input buffer that triggered the
        match (the surface form), and "span" refers to the exact location in the input buffer where the surface
        form is found. Depending on the normalizer that is used, "match" and "surface" may or may not differ.

        A space-normalized version of the surface form is emitted as "surface", for convenience. Clients
        that require an exact surface form that is not space-normalized can easily reconstruct the desired
        string using the emitted "span" value.

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and more.
        """
        tokens = self.__tokenizer.tokens(buffer)  # Tokeniserer inndata-bufferen
        live_states: List[Tuple[Trie, int, str]] = []  # Liste over aktive tilstander (trie-node, startposisjon, konsumert streng)

        def __consume(state: Trie, token: str, consumed: str) -> Tuple[Optional[Trie], str]:
            # Prøver å consume token direkte
            next_node = state.consume(token)
            if next_node:
                return next_node, consumed + token

            # Hvis det ikke går, prøver jeg med et mellomrom prefiksert token
            next_node = state.consume(f" {token}")
            if next_node:
                return next_node, consumed + f" {token}"
            return None, consumed

        # Iterate over hvert token og dens posisjon i bufferen
        for token, (start, end) in tokens:
            normalized_token = self.__normalizer.normalize(token)
            new_live_states = []
            matches = []  # Samler matches, soom jeg vil yield samtidig senere

            # Behandler eksisterende tilstander 
            for state, state_start, consumed_str in live_states:
                next_node, updated_consumed_str = __consume(state, normalized_token, consumed_str)

                if next_node:
                    new_live_states.append((next_node, state_start, updated_consumed_str))

                    # Hvis denne nye tilstanden er endelig, betyr det at vi har et treff
                    if next_node.is_final():
                        surface_form = buffer[state_start:end]
                        matches.append({
                            'surface': " ".join(surface_form.split()),  # Normaliserer mellomrom
                            'span': (state_start, end),
                            'match': updated_consumed_str,   # Bruk den oppdaterte "consumed" strengen
                            'meta': next_node.get_meta()  # Hent metadata assosiert med treffet
                        })

            # Sjekker om det nåværende tokenet kan starte et nytt treff i trien, ved en ny startstilstand
            initial_state, initial_consumed_str = __consume(self.__trie, normalized_token, "")
            if initial_state:
                new_live_states.append((initial_state, start, initial_consumed_str))

                # Hvis starttilstanden er endelig, legg den til i treffene
                if initial_state.is_final():
                    surface_form = buffer[start:end]
                    matches.append({
                        'surface': " ".join(surface_form.split()),  
                        'span': (start, end),
                        'match': initial_consumed_str,
                        'meta': initial_state.get_meta()
                    })

           # Oppdaterer de aktive tilstandene for neste iterasjon
            live_states = new_live_states

            # Returnerer alle treff funnet for dette tokenet samtidig
            yield from matches 
