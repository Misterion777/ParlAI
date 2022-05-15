from parlai.core.message import Message
from parlai.tasks.blended_skill_talk.extract import extract_from_msg

TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
def split_concepts_string(concepts):
    triples = concepts.split(TOKEN_END_KNOWLEDGE)
    concepts = ""                        
    for triple in triples:
        triple = triple[len(TOKEN_KNOWLEDGE):]
        concepts += _add_knowledge_token(triple,add_space=True)        
    return concepts

def rm_knowledge(txt):
    f_idx = txt.find(TOKEN_KNOWLEDGE)
    l_idx = txt.rfind(TOKEN_END_KNOWLEDGE) + len(TOKEN_END_KNOWLEDGE)    
    if f_idx != -1 and l_idx != -1:
        return txt[:f_idx] + txt[l_idx:]
    return txt

def add_knowledge_to_act(act:Message):
    knowledge = get_knowledge(act["text"])
    act.force_set("text",act["text"]+knowledge)
    return act

def get_knowledge(text,limit=3,compare_all_text=False):        
    concepts = extract_from_msg(text,limit=limit,compare_all_text=compare_all_text)
    knowledge = ''
    for triple in concepts:      
        knowledge += _add_knowledge_token(triple,add_space=True)                                      
    return knowledge

def _add_knowledge_token(triple,add_space=False):
    f_idx = triple.find('__')
    l_idx = triple.rfind('__') + 2
    delim = ' ' if add_space else ''
    if f_idx != -1 and l_idx != -1:
        triple = f"{triple[:f_idx]}{delim}{triple[f_idx:l_idx]}{delim}{triple[l_idx:]}"
        return f'{TOKEN_KNOWLEDGE}{delim}{triple}{delim}{TOKEN_END_KNOWLEDGE}'
    return triple
