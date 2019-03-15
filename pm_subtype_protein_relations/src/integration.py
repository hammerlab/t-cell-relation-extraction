from bs4 import BeautifulSoup
import re

def clean_text(text):
    # Remove individual lines that have a very small number of characters
    text = '\n'.join([l for l in text.split('\n') if len(l.strip()) == 0 or len(l.strip()) >= 64])
    # Replace 2+ newlines with double space
    text = re.sub('\n{2,}', '\n\n', text)
    return text

def extract_text(xml):
    if not xml:
        return None
    soup = BeautifulSoup(xml, 'xml')
    # TODO: remove citation elements (ex: <xref ref-type="bibr" rid="B1">1</xref>)
    body = soup.find('body')
    return clean_text(body.text) if body else None

def combine_text(title, abstract, body):
    return '{}\n{}\n{}'.format(title or '', abstract or '', body or '')

def get_scispacy_pipeline(model='en_ner_jnlpba_md'):
    import spacy
    # Scispacy post-release
    nlp = spacy.load('en_core_sci_md')
    # en_ner_jnlpba_md or en_ner_craft_md are most appropriate
    ner = spacy.load(model)
    nlp.replace_pipe('ner', ner.pipeline[0][1])
    
    # The individual entity type names (e.g. CELL_TYPE, PROTEIN, etc. need to be 
    # added to the core nlp vocab in order to avoid "label not in StringStore errors")
    ner_types = sorted(list(set([typ.split('-')[-1] for typ in ner.pipeline[0][1].move_names])))
    for typ in ner_types:
        nlp.vocab.strings.add(typ)
        
    return nlp
