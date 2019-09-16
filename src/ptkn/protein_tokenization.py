"""Pure-python protein expression string tokenization utilities for biomedical NLP"""
import heapq
import itertools

# Positive and negative expression indicators
VOCAB_SIGN_POS = ['+', 'pos', 'positive', 'high', 'hi', 'bright']
VOCAB_SIGN_NEG = ['-', '–', '—', '−', 'neg', 'negative', 'lo', 'low', 'dim']


def match(string, words, max_unmatched_len):
    """Recursively break string input using known vocabulary (with unknown substrings of maximum size)
    
    Generates tuples with values so far for (in this order):
        - count of unmatched characters
        - count of unmatched words 
        - count of matched words
        - list of (substring, is_known) tuples where substring is a word to be broken out and is_known
            is a boolean indicator of whether or not that word was present in the provided word list
    """
    if len(string) == 0:
        yield 0, 0, 0, []
        return
    
    # Loop through oov offsets up to at most `max_unmatched_len`
    n = min(max_unmatched_len + 1, len(string))
    matched = False
    for u in range(n):
        # Ignore u characters at beginning of current candidate and run
        # prefix search for all substrings, taking the largest one possible
        substr = string[u:]
        for i in range(1, len(substr)+1):
            prefix = substr[:i]
            tail = None if i >= len(substr) else substr[:(i+1)]
            if prefix not in words or (tail is not None and tail in words):
                continue
            # Recursion for substring starting at index u + i
            for ctcu, ctwu, ctwm, tokens in match(substr[i:], words, max_unmatched_len):
                matched = True
                tknu, tknm = [] if u == 0 else [(string[:u], False)], [(prefix, True)]
                yield ctcu + u, ctwu + len(tknu), ctwm + len(tknm), tknu + tknm + tokens 
        # Break when any oov offset results in a prefix hit 
        # (this minimizes the length of oov substrings)
        if matched:
            break
    # Return trailing unmatched substring, if necessary
    if not matched:
        yield len(string), 1, 0, [(string, False)]
            

def split(string, words, max_unmatched_len=0, n_results=1, mode='most_words'):
    """Partition a string into words using a controlled vocabulary
    
    This is a solution to the Word Break Problem that includes:
        - Unknown words of at most length `max_unmatched_len`
        - Greedy maximum length known word partitioning
        - Greedy minimum length unknown word partitioning
        
    Example:
    
        words = ["this", "is", "the", "famous", "Word", "break", "b", "r", "e", "a", "k", "br", "problem"]
        string = "WordbreakINGproblem"
        results = split(string, words, max_unmatched_len=3, mode='most_words', n_results=1)[0]
        print([w if known else '|' + w + '|' for w, known in results[-1]])
        >>> ['Word', 'br', 'e', 'a', 'k', '|ING|', 'problem']
        
        results = split(string, words, max_unmatched_len=3, mode='least_words', n_results=1)[0]
        print([w if known else '|' + w + '|' for w, known in results[-1]])
        >>> ['Word', 'break', '|ING|', 'problem']
        
    Returns:
        A list of tuples with values (in this order):
            - count of unmatched characters
            - count of unmatched words 
            - count of matched words
            - list of (substring, is_known) tuples where substring is a word to be broken out and is_known
                is a boolean indicator of whether or not that word was present in the provided word list
    """
    if mode not in ['most_words', 'least_words']:
        raise ValueError(f'Mode must be one of "most_words" or "least_words" not "{mode}"')
    # Limit max unmatched length to length of input string (default to 0 if None)
    max_unmatched_len = min(max(max_unmatched_len or 0, 0), len(string))
    mode = (-1, -1) if mode == 'least_words' else (-1, 1)
    res = []

    # Get unmatched char count, unmatched word count, matched word count, and token list (in that order)
    for ctcu, ctwu, ctwm, tokens in match(string, words, max_unmatched_len):
        ctwt = ctwu + ctwm
        # Minimize number of unmatched characters, maximize or minimize 
        # number of total tokens based on mode
        key = (ctcu * mode[0], ctwt * mode[1])
        item = (key, ctcu, ctwu, ctwm, ctwt, tokens)
        # Push onto bounded heap
        (heapq.heappush if len(res) < n_results else heapq.heappushpop)(res, item)

    # Return all requested results except for those used in sorting heap entry
    return [r[1:] for r in heapq.nlargest(n_results, res)]


class ProteinToken(object):
    
    SIGN_POS_CHAR = '⁺'
    SIGN_NEG_CHAR = '⁻'
    
    def __init__(self, token_text, sign_text, sign_value, metadata):
        self.token_text = token_text
        self.sign_text = sign_text
        self.sign_value = sign_value
        self.text = (token_text or '') + (sign_text or '')
        self.metadata = metadata
        
    @property
    def sign_value_text(self):
        """Normalized sign as text

        For example, tokens like CD69hi and CD69+++ would both return '⁺' for this property

        To change the normalized sign character, set ProteinToken.SIGN_POS_CHAR or ProteinToken.SIGN_NEG_CHAR
        """
        sign = ''
        if self.sign_value == 1:
            sign = ProteinToken.SIGN_POS_CHAR
        elif self.sign_value == -1:
            sign = ProteinToken.SIGN_NEG_CHAR
        return sign
    
    def to_string(self, show_sign_text=True):
        return ''.join([
            self.token_text or '', 
            self.sign_value_text or '', 
            '(' + self.sign_text + ')' if self.sign_text and show_sign_text else ''
        ])
    
    def __repr__(self):
        return self.to_string(show_sign_text=True)

        
class ProteinTokenizer(object):
    
    def __init__(self, vocab_pr, vocab_sign_pos=VOCAB_SIGN_POS, vocab_sign_neg=VOCAB_SIGN_NEG):
        """Tokenizer for extracting tokens with semantics common to protein expression strings in biomedical texts
        
        This is helpful for resolving expressions in noun phrases, typically in the context of single cell assays,
        such as (all of the below are from PMC articles):
           - CD4+CD25+FOXP3+
           - CD4+CD45RO/RBbright
           - CD3CD4CD8alow
           - CD4-CD8+CD3+RORγt+
           - CD3CD4CD99+IL-12
           - CD4+CD45RA+CD45RO−CD62L+CCR7+CD127+CD27+CD28+CD95+CD122+
           
        Tokenization of such strings is not possible purely based on text patterns so this functionality
        exists as a way to incorporate prior knowledge of known proteins and aliases.
        
        A dictionary must be provided containing keys with protein names/aliases for which extracted tokens
        will be associated with the values for those keys.
        
        Example:
        
            # Create the protein vocabulary
            string = 'CD4+CD45RA+CD45RO-4-1BB-CD62L+++CCR7loCD127posCD27positiveCD28hiCD95+CD122+'
            vocab_pr = [
                # 'CCR7', 'CD27' --> intentionally omitted to show that surrounding context is still enough
                'CD4', 'CD45', 'CD45RA', 'CD45RO', 'CD62L', '4-1BB', 
                'CD127', 'CD28', 'CD122', 'CD95', 'CD122']
            vocab_pr = {k: dict(name=k) for k in vocab_pr}
            tokenizer = ProteinTokenizer(vocab_pr)
            for t in tokenizer.tokenize(string):
                print(f'{t.text} [term={t.token_text}, sign={t.sign_text}, value={t.sign_value}, metadata={t.metadata}]')
                
            # Note that "+++" is resolved to indicating positive expression but no more so than "+"
            CD4+     [term=CD4, sign=+, value=1, metadata={'name': 'CD4'}]
            CD45RA+  [term=CD45RA, sign=+, value=1, metadata={'name': 'CD45RA'}]
            CD45RO-  [term=CD45RO, sign=-, value=-1, metadata={'name': 'CD45RO'}]
            4-1BB-   [term=4-1BB, sign=-, value=-1, metadata={'name': '4-1BB'}]
            CD62L+++ [term=CD62L, sign=+++, value=1, metadata={'name': 'CD62L'}]
            CCR7     [term=CCR7, sign=None, value=0, metadata=None]
            CD127pos [term=CD127, sign=pos, value=1, metadata={'name': 'CD127'}]
            CD27     [term=CD27, sign=None, value=0, metadata=None]
            CD28hi   [term=CD28, sign=hi, value=1, metadata={'name': 'CD28'}]
            CD95+    [term=CD95, sign=+, value=1, metadata={'name': 'CD95'}]
            CD122+   [term=CD122, sign=+, value=1, metadata={'name': 'CD122'}]
        
        Args:
            - vocab_pr: dict-like with keys containing protein string names and values that are any arbitrary
                metadata to be associated with matches to those names
            - vocab_sign_pos: strings indicating positive expression
            - vocab_sign_neg: strings indicating negative expression
        """
        self.vocab = vocab_pr
        self.tokens = list(vocab_pr.keys()) + vocab_sign_pos + vocab_sign_neg
        self.signs = {**{v:1 for v in vocab_sign_pos}, **{v:-1 for v in vocab_sign_neg}}
        
    def tokenize(self, string, max_unmatched_len=16):
        """Generate ProteinToken instances from provided string
        
        Args:
            - string: input string (e.g. "CD4+CD25+FOXP3+")
            - max_unmatched_len: maximum length of any OOV substring; this represents a trade-off between 
                accuracy and efficiency as the default value (16) may not be enough to catch long
                strings not in the provided dictionary, but increases it further will increase
                the size of the recursive search space
        Returns:
            ProteinToken generator
        """
        splits = split(string, self.tokens, max_unmatched_len=max_unmatched_len, n_results=1)
        if not splits:
            yield ProteinToken(string, None, 0)
            return
        
        # Get last item in first result (i.e. list of tokens)
        tokens = splits[0][-1]

        # Create groups to collapse tokens by using an integer id assigned
        # to each span of tokens where consecutive sign tokens are collapsed
        # and consecutive non-sign tokens are not 
        keys = [v[0] not in self.signs for v in tokens]  # Flag true when not sign token
        keys = itertools.accumulate([int(k) for k in keys])  # Cumulative sum
        keys = list(enumerate(keys))  # Add token indexes

        # Yield collapsed tokens
        # k -> token group id
        # g -> List of tuples like (token index, token group id)
        for k, g in itertools.groupby(keys, lambda v: v[1]):
            sn_txt, pr_txt, sn_val = None, None, 0

            # Get tokens for this group (as strings)
            strs = [tokens[v[0]][0] for v in g]

            # Determine which tokens are for signs
            mask = [t in self.signs for t in strs]
            
            # Handle sign only group
            if mask[0]:
                sn_txt = strs
            else:
                pr_txt = [strs[i] for i in range(len(strs)) if not mask[i]]
                sn_txt = [strs[i] for i in range(len(strs)) if mask[i]]

            # Resolve sign string to integer value
            if sn_txt:
                signs = set([self.signs.get(v, 0) for v in sn_txt]) - set([0])
                # Only set sign if no conflicts exist
                if len(signs) == 1:
                    sn_val = list(signs)[0]
                
            # Convert token lists to string 
            pr_txt = ''.join(pr_txt) if pr_txt else None
            sn_txt = ''.join(sn_txt) if sn_txt else None
            
            # Fetch metadta and ignore signs for OOV terms
            metadata = None
            if pr_txt:
                metadata = self.vocab.get(pr_txt, None)
                if metadata is None:
                    sn_txt, sn_val = None, 0
            yield ProteinToken(pr_txt, sn_txt, sn_val, metadata)