import spacy
import networkx as nx
import pandas as pd
import numpy as np
import re
import logging
import json
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)

def extract_relation(r):
    s = r.replace('/r/','')    
    return f'__{s}__'

def extract_word(w):
    s = re.search('\/c\/en\/(.*?)(\/|$)', w).group(1)
    s = s.replace('_',' ')
    return s

def create_graph(df):
    g = nx.DiGraph()
    for i,row in df.iterrows():
        start = extract_word(row['from'])
        end = extract_word(row['to'])
        r = extract_relation(row['relation'])
        
        g.add_edge(start,end, relation=r,weight=row['weight'])
    return g

def filter_tokens(tokens,tokens_cache,return_lemmas=True):
    new_tokens = []
    for token in tokens:
        if not token.is_stop and not token.is_punct and token.text.lower() not in tokens_cache:
            if return_lemmas:
                new = token.lemma_
            else:
                if token.dep_ in ['compound','amod']:
                  new = token.text.lower()
                else:
                  new = token.lemma_.lower()
            new_tokens.append(new)
            tokens_cache[token.text.lower()] = 'cached'
    return new_tokens

def get_tokens(sent,tokens_cache):
    tokens = []
    prev_start = 0
    
    for chunk in sent.noun_chunks:
        if filtered:=filter_tokens(sent[prev_start:chunk.start], tokens_cache):
          tokens.append(filtered)
        if filtered:=filter_tokens(chunk,tokens_cache, return_lemmas=False):
          tokens.append(filtered)
        prev_start = chunk.end
    if filtered:=filter_tokens(sent[prev_start:],tokens_cache):
        tokens.append(filtered)
    return tokens

def preprocess(msg_str):
    doc = nlp(msg_str)
    vectors = sent2wec.encode([s.text for s in doc.sents])
    preprocessed = []
    tokens_cache = {}
    for sent,vec in zip(doc.sents,vectors):
        tokens = get_tokens(sent,tokens_cache)        
        preprocessed.append((tokens,vec) )
    return preprocessed, sent2wec.encode(doc.text)

def get_relations(c):
    global conceptnet
    rels = []
    if c in conceptnet:
        for n, attrs in conceptnet[c].items():
            rels.append(f"{c}{attrs['relation']}{n}")
    return rels

def rel2_sent(triple):
    f,rel,t = triple.split('__')
    rel = re.sub('([A-Z]+)', r' \1', rel).strip()
    return f'{f} {rel} {t}'

def filter_relations(rels_vecs,sent_vec,limit=10):
    sims = cosine_similarity(rels_vecs,sent_vec)
    return np.argsort(sims,axis=0)[::-1][:limit].flatten()    

def flip_relation(rel):
    return rel.split(' ')[::-1]

def extract_from_msg(msg,limit=3, compare_all_text=False):
    sentence_tokens,text_vec = preprocess(msg)
    all_rels = set()
    for sent,sent_vec in sentence_tokens:
        for token in sent:
            rels = []
            if len(token) > 1:
                t = ' '.join(token)
                rels += get_relations(t)                
            for t in token:                                
                rels += get_relations(t)
            if rels:
                
                sent_rels = [rel2_sent(r) for r in rels]
                rels_vecs = sent2wec.encode(sent_rels)

                sent_vec = sent_vec.reshape(1, -1)
                idxs = filter_relations(rels_vecs,sent_vec,limit=limit)
                rels = np.array(rels)
                rels = rels[idxs]
            
                # rels = [r for r in rels if flip_relation(r) not in all_rels]
                all_rels.update(rels)
    if compare_all_text:
        all_rels = list(all_rels)
        sent_rels = [rel2_sent(r) for r in all_rels]
        rels_vecs = sent2wec.encode(sent_rels)
        text_vec = text_vec.reshape(1, -1)
        idxs = filter_relations(rels_vecs,text_vec)
        all_rels = np.array(all_rels)
        all_rels = all_rels[idxs]
    return all_rels

print('Loading extraction models...')
sent2wec = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm",disable=['ner'])
conceptnet = pd.read_csv('/scratch/lustre/home/illa7843/cn_extraction/conceptnet_en_filtered_13.csv')
conceptnet = create_graph(conceptnet)
print('ConceptNet extraction ready!')
