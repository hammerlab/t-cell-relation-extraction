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
from snorkel.models import Document
from tcre import integration


def default_ent_fn(ent):
    return {'type': ent.label_, 'cid': 'O'}


class SpaCyParser(Parser):

    def __init__(self, nlp, ent_fn=default_ent_fn):
        super().__init__(name="spaCy")
        self.model = nlp
        self.ent_fn = ent_fn

    def connect(self):
        return ParserConnection(self)

    def parse(self, document, text):
        text = self.to_unicode(text)

        # Clip to max length to avoid error like
        # ValueError: [E088] Text of length 1290071 exceeds maximum of 1000000
        if len(text) > self.model.max_length:
            text = text[:self.model.max_length]

        doc = self.model(text)

        position = 0
        for sent in doc.sents:
            parts = defaultdict(list)
            text = sent.text
            # Map token index to entity for reverse lookup
            # Note: The token indexes are absolute, not relative to sentence start and
            # must be matched against the token.i attribute (not sentence token counter)
            entities = {token_index: ent for ent in sent.ents for token_index in range(ent.start, ent.end)}
            for i, token in enumerate(sent):
                ent = {'type': 'O', 'cid': 'O'}
                if token.i in entities:
                    ent = self.ent_fn(entities[token.i])
                parts['words'].append(str(token))
                parts['lemmas'].append(token.lemma_)
                parts['pos_tags'].append(token.tag_)
                parts['char_offsets'].append(token.idx)
                parts['abs_char_offsets'].append(token.idx)
                head_idx = 0 if token.head is token else token.head.i - sent[0].i + 1
                parts['dep_parents'].append(head_idx)
                parts['dep_labels'].append(token.dep_)
                parts['ner_tags'].append(ent['type'])
                parts['entity_types'].append(ent['type'])
                parts['entity_cids'].append(ent['cid'])

            # make char_offsets relative to start of sentence
            parts['char_offsets'] = [
                p - parts['char_offsets'][0] for p in parts['char_offsets']
            ]
            parts['position'] = position

            # Link the sentence to its parent document object
            parts['document'] = document
            parts['text'] = text

            # Assign the stable id as document's stable id plus absolute
            # character offset
            abs_sent_offset = parts['abs_char_offsets'][0]
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
            if document:
                parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)

            position += 1

            yield parts
