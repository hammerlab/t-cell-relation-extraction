"""Snorkel Overrides"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from collections import defaultdict
from snorkel.models import construct_stable_id
from snorkel.parser.parser import Parser, ParserConnection
from snorkel.parser import TextDocPreprocessor


class DocListProcessor(TextDocPreprocessor):

    def __init__(self, paths, encoding="utf-8"):
        super().__init__(None, encoding=encoding)
        self.paths = paths

    def _get_files(self, path):
        return self.paths


class SpaCyParser(Parser):

    def __init__(self, nlp):
        super().__init__(name="spaCy")
        self.model = nlp

    def connect(self):
        return ParserConnection(self)

    def parse(self, document, text):
        text = self.to_unicode(text)

        doc = self.model(text)

        position = 0
        for sent in doc.sents:
            parts = defaultdict(list)
            text = sent.text

            for i,token in enumerate(sent):
                parts['words'].append(str(token))
                parts['lemmas'].append(token.lemma_)
                parts['pos_tags'].append(token.tag_)
                parts['ner_tags'].append(token.ent_type_ if token.ent_type_ else 'O')
                parts['char_offsets'].append(token.idx)
                parts['abs_char_offsets'].append(token.idx)
                head_idx = 0 if token.head is token else token.head.i - sent[0].i + 1
                parts['dep_parents'].append(head_idx)
                parts['dep_labels'].append(token.dep_)

            # make char_offsets relative to start of sentence
            parts['char_offsets'] = [
                p - parts['char_offsets'][0] for p in parts['char_offsets']
            ]
            parts['position'] = position

            # Link the sentence to its parent document object
            parts['document'] = document
            parts['text'] = text

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Assign the stable id as document's stable id plus absolute
            # character offset
            abs_sent_offset = parts['abs_char_offsets'][0]
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
            if document:
                parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)

            position += 1

            yield parts
