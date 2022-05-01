TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
def split_concepts(concepts):
    triples = concepts.split(TOKEN_END_KNOWLEDGE)
    concepts = ""                        
    for triple in triples:
        triple = triple[len(TOKEN_KNOWLEDGE):]
        f_idx = triple.find('__')
        l_idx = triple.rfind('__') + 2
        if f_idx != -1 and l_idx != -1:
            triple = f"{triple[:f_idx]} {triple[f_idx:l_idx]} {triple[l_idx:]}"
            concepts += f" {TOKEN_KNOWLEDGE} {triple} {TOKEN_END_KNOWLEDGE}"
    return concepts