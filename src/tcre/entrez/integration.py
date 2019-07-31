"""Older parsing logic necessary for ensuring compatible parsing of past annotated documents"""
from bs4 import BeautifulSoup
import pandas as pd
import re
import string
import sys
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

WS_REGEX = re.compile('\n{2,}')
TC_REGEX_1 = re.compile(r'<sup><xref[^>]*>.*?</sup>')
TC_REGEX_2 = re.compile(r'<xref[^>]*>[^<]*</xref[^>]*>')


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


def parse_article(soup):
    res = {}

    # Extract IDs
    ids = soup.find('article-meta')
    ids = ids.find_all('article-id') if ids else []

    def get_id(typ):
        idt = [t for t in ids if t.get('pub-id-type') == typ]
        return idt[0].text if idt else None

    res['id_pmc'] = get_id('pmc')
    res['id_pmid'] = get_id('pmid')
    res['id_doi'] = get_id('doi')

    # Extract dates
    def parse_date(t):
        if not t or not t.find('year'):
            return None
        date_string = '{}-{}-{}'.format(
            t.find('year').text,
            t.find('month').text if t.find('month') else '01',
            t.find('day').text if t.find('day') else '01'
        )
        try:
            return pd.to_datetime(date_string)
        except Exception as e:
            logger.warning('Failed to parse date string "%s"; Reason: %s', date_string, e)
            return None

    # Dates related to transmission
    history_dates = soup.find('history')
    history_dates = history_dates.find_all('date') if history_dates else []

    def get_history_date(typ):
        dt = [t for t in history_dates if t.get('date-type') == typ]
        return parse_date(dt[0]) if dt else None

    res['date_received'] = get_history_date('received')
    res['date_accepted'] = get_history_date('accepted')

    # Earlist publication date
    pub_dates = [parse_date(t) for t in soup.find_all('pub-date')]
    pub_dates = [date for date in pub_dates if date is not None]
    res['date_pub'] = min(pub_dates) if pub_dates else None

    # Extract journal metadata
    res['journal_titles'] = '|:|'.join(list(set([t.text for t in soup.find_all('journal-title')])))
    res['journal_ids'] = '|:|'.join(list(set([t.text for t in soup.find_all('journal-id')])))

    res['title'] = soup.find('title-group').find('article-title').text
    try:
        res['abstract'] = soup.find('abstract').text
    except:
        res['abstract'] = None

    # Apply body extraction to raw xml
    res['body'] = extract_text(str(soup))
    return res


def parse_nxml(doc):
    soup = BeautifulSoup(doc, 'xml')
    return pd.DataFrame([parse_article(article) for article in soup.find_all('article')])