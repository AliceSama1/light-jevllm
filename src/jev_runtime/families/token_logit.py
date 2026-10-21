import json
from ..core.errors import ServiceError, object_fields
from ..token_logit import TokenLogitAdapter
from ..decision_ir import Candidate
class TokenLogitFamily:
    def __init__(self, encode): self.adapter, self.closed = TokenLogitAdapter(encode), False
    def compile(self, request):
        if self.closed: raise ServiceError(503, 'runtime closing')
        questions=request['questions']
        if not isinstance(questions,dict) or not 1<=len(questions)<=64: raise ValueError('questions must contain 1..64 named questions')
        items=[]
        for name,q in questions.items():
            if not isinstance(name,str) or not name.strip(): raise ValueError('question IDs must be nonblank')
            object_fields(q, ('type','instructions','criteria','token_labels'), ('type','instructions'))
            instructions=q['instructions']
            if not isinstance(instructions,(str,dict,list)) or isinstance(instructions,str) and not instructions.strip(): raise ValueError('instructions must be nonblank text, object or array')
            kind=q['type']
            if kind=='noul':
                criteria=q.get('criteria',{'false':None,'true':None})
                if not isinstance(criteria,dict) or set(criteria)!={'false','true'}: raise ValueError('noul criteria must contain true and false')
            elif kind=='choice':
                criteria=q.get('criteria')
                if not isinstance(criteria,dict) or not 2<=len(criteria)<=255: raise ValueError('choice needs 2..255 criteria')
            elif kind=='score':
                levels=q.get('criteria')
                if not isinstance(levels,list) or not 2<=len(levels)<=10: raise ValueError('score needs 2..10 levels')
                criteria={str(i):v for i,v in enumerate(levels)}
            else: raise ValueError('unsupported question type')
            if any(not isinstance(k,str) or not k.strip() for k in criteria): raise ValueError('candidate IDs must be nonblank')
            labels=q.get('token_labels',{k:k for k in criteria})
            if not isinstance(labels,dict) or set(labels)!=set(criteria) or any(not isinstance(v,str) or not v for v in labels.values()): raise ValueError('token_labels must map every candidate to nonempty text')
            prompt=json.dumps({'state':request['state'],'question':instructions,'candidates':[{'id':k,'label':labels[k],'description':v} for k,v in criteria.items()]},ensure_ascii=False,allow_nan=False)+'\nAnswer with exactly one candidate label:\n'
            items.append(self.adapter.compile(name,kind,prompt,[Candidate(k,labels[k],v) for k,v in criteria.items()]))
        return items
    def readout(self, items, output):
        return {item.question.id: self.adapter.decode(item,[row[k] for k in item.candidate_ids]) for item,row in zip(items,output,strict=True)}
    def close(self): self.closed=True
