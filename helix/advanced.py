"""Quantization, prefix caching, and greedy speculative decoding."""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

@dataclass
class Int8Weight:
    values: np.ndarray
    scale: np.ndarray
    @classmethod
    def from_float(cls,weight):
        weight=np.asarray(weight,np.float32); scale=np.max(np.abs(weight),axis=1,keepdims=True)/127
        scale=np.where(scale==0,1,scale).astype(np.float32)
        return cls(np.clip(np.rint(weight/scale),-127,127).astype(np.int8),scale)
    def dequantize(self): return self.values.astype(np.float32)*self.scale
    @property
    def bytes(self): return self.values.nbytes+self.scale.nbytes

class PrefixCache:
    def __init__(self,capacity):
        if capacity<=0: raise ValueError("positive capacity required")
        self.capacity=capacity; self.entries=OrderedDict(); self.hits=0; self.lookups=0
    def put(self,tokens,value):
        key=tuple(tokens); self.entries.pop(key,None); self.entries[key]=value
        while len(self.entries)>self.capacity: self.entries.popitem(last=False)
    def longest(self,tokens):
        self.lookups+=1; matches=[k for k in self.entries if len(k)<=len(tokens) and tuple(tokens[:len(k)])==k]
        if not matches: return None
        key=max(matches,key=len); value=self.entries.pop(key); self.entries[key]=value; self.hits+=1
        return key,value
    @property
    def hit_rate(self): return self.hits/self.lookups if self.lookups else 0

def greedy(model,tokens): return int(np.argmax(model.logits(tokens)[0,-1]))

def speculative_generate(target,draft,prompt,max_new_tokens,depth=4):
    if depth<=0: raise ValueError("depth must be positive")
    tokens=list(prompt); generated=[]; accepted=0; proposed=0; target_batches=0
    while len(generated)<max_new_tokens:
        proposals=[]; draft_context=list(tokens)
        for _ in range(min(depth,max_new_tokens-len(generated))):
            token=greedy(draft,draft_context); proposals.append(token); draft_context.append(token)
        # Teacher-forced target verification in one target forward pass.
        logits=target.logits(tokens+proposals); target_batches+=1
        start=len(tokens)-1
        mismatch=False
        for index,proposal in enumerate(proposals):
            wanted=int(np.argmax(logits[0,start+index])); proposed+=1
            if proposal==wanted:
                tokens.append(proposal); generated.append(proposal); accepted+=1
            else:
                tokens.append(wanted); generated.append(wanted); mismatch=True; break
        if not mismatch and not proposals: break
    return generated,{"accepted":accepted,"proposed":proposed,"acceptance_rate":accepted/proposed if proposed else 0,"target_batches":target_batches}
