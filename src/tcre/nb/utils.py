"""Utilities for programmatic notebook manipulation"""
import copy
import nbformat
from nbconvert import HTMLExporter

nbs = {}
FORCE_NB_RELOAD = False


def _true(*_):
    return True


def get_nb(path, force=None):
    global nbs
    force = FORCE_NB_RELOAD if force is None else force
    if path not in nbs or force:
        with open(path, 'r') as fd:
            nb = nbformat.read(fd, as_version=4)
        nbs[path] = nb
    return nbs[path]


def get_cell_nb(nb, predicate):
    predicate = predicate or _true
    nb = copy.deepcopy(nb)
    nb['cells'] = [c for c in nb['cells'] if predicate(c)]
    return nb


def to_html(nb):
    html_exporter = HTMLExporter()
    html_exporter.template_file = 'basic'
    (body, resources) = html_exporter.from_notebook_node(nb)
    return body, resources


def get_tag_html(path, name, prefix):
    def predicate(c):
        if 'metadata' in c and 'tags' in c['metadata']:
            for t in c['metadata']['tags']:
                if t == prefix + '.' + name:
                    return True
        return False
    return to_html(get_cell_nb(get_nb(path), predicate))
