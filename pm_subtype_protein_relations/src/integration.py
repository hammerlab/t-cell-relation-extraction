from bs4 import BeautifulSoup
import pandas as pd
import re
import string

WS_REGEX = re.compile('\n{2,}')
TC_REGEX_1 = re.compile('<sup><xref[^>]*>.*?</sup>')
TC_REGEX_2 = re.compile('<xref[^>]*>[^<]*</xref[^>]*>')

def clean_text_whitespace(text):
    # Remove individual lines that have a very small number of characters
    text = '\n'.join([l for l in text.split('\n') if len(l.strip()) == 0 or len(l.strip()) >= 64])
    # Replace 2+ newlines with double space
    text = WS_REGEX.sub('\n\n', text)
    return text

def remove_xml_citations(text):
    # Remove citation elements
    # Example: <sup><xref ref-type="bibr" rid="CR8">8</xref>–<xref ref-type="bibr" rid="CR11">11</xref></sup>
    
    # First, attempt to remove everything in superscript citation reference as there
    # are often characters in the superscript but not the xref tags that relate to the
    # citations (hyphens and commas mainly) and these should be removed as well
    text = TC_REGEX_1.sub('', text)
    
    # Then remove any lingering citations not in superscript
    text = TC_REGEX_2.sub('', text)
    return text

def extract_text(xml, clean_whitespace=True, remove_citations=True):
    if not xml:
        return None
    # Apply transformations prior to BS4 tag strip (happens on .text call)
    if remove_citations:
        xml = remove_xml_citations(xml)
    soup = BeautifulSoup(xml, 'xml')
    body = soup.find('body')
    if not body:
        return None
    body = body.text
    # Post-tag-stripping transformations
    if clean_whitespace:
        body = clean_text_whitespace(body)
    return body

def combine_text(title, abstract, body):
    parts = ['' if pd.isnull(p) else p for p in [title, abstract, body]]
    
    def add_punc(t):
        return t + '.' if t.strip() and t.strip()[-1] not in string.punctuation else t
    
    return '\n'.join([add_punc(p) for p in parts])

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
