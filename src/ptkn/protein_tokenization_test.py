import unittest
from protein_tokenization import ProteinTokenizer, ProteinToken

VOCAB1 = [
    'CD4', 'CD45RA', 'CD45', 'CD45RO', 'CD62L', 'CCR7', 'CD127',
    'CD27', 'CD28', 'CD122', 'CD8a', 'CD8', 'CD3', 'Thy1', '4-1BB', 
    'CCR7', 'RORgt', 'CD95', 'CD122'
]

CASES = [
    # Test lots of in-vocab (mixed with a few adjacent OOV) strings and different sign expressions
    ('CD4+CD45RA+CD45RO-4-1BB-CD62L+++CCR7loCD127posCD27positiveCD28hiCD95+CD122+', [
        'CD4+x', 'CD45RA+x', 'CD45RO-x', '4-1BB-x', 'CD62L+x', 'CCR7-x', 
        'CD127+x', 'CD27+x', 'CD28=o', 'CD95=o', 'CD122=o'
    ], ['CD95', 'CD28', 'CD122']),
    
    # Test partial OOV with punctuation
    ('CD4+CD45RO/RBbright', ['CD4+x', 'CD45RO=x', '/RB=o']),
    
    # Test OOV on either end
    ('CD3CD4PBMC', ['CD3=o', 'CD4=x', 'PBMC=o'], ['CD3']),
    
    # Test partial OOV (i.e. hierarchical match)
    ('Thy1.1+OT-1+CD8+', ['Thy1=x', '.1=o','OT=o', '1=o', 'CD8+x']),
    
    # Test sign at start and terms with no sign
    ('+CD95CD4CD8negative', ['+o', 'CD95=x', 'CD4=x', 'CD8-x']),
    
    # Test sign only
    ('+', ['+o']),
    
    # Test sign only with conflict
    ('++-negative', ['=o']),
    
    # Test term with conflicting sign
    ('CD4+-', ['CD4=x']),
    
    # Test single term only
    ('CD4', ['CD4=x']),
    ('CD4neg', ['CD4-x'])
]

SIGN_CHARS = {1: '+', 0: '=', -1: '-'}
META_CHARS = {True: 'x', False: 'o'}

class ProteinTokenizationTest(unittest.TestCase):
    
    def test_cases(self):
        for c in CASES:
            v = list(set(VOCAB1) - set(c[2] if len(c) > 2 else []))
            v = {k: dict(name=k) for k in v}
            tokens = ProteinTokenizer(v).tokenize(c[0])
            
            expected = c[1]
            actual = []
            for t in tokens:
                sign_char = SIGN_CHARS[t.sign_value]
                meta_char = META_CHARS[t.metadata is not None]
                actual.append((t.token_text or '') + sign_char + meta_char)
            self.assertEquals(actual, c[1], f'Actual not equal expected for case {c[0]}')
