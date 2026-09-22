"""Deterministic tiny decoder and streaming engine; correctness fixture, not an LLM."""
import numpy as np
from runtime import causal_attention, rms_norm

class TinyDecoder:
    def __init__(self,vocab=32,width=16,heads=2,seed=0):
        rng=np.random.default_rng(seed); scale=width**-0.5
        self.vocab,self.width,self.heads=vocab,width,heads
        self.embed=rng.normal(0,scale,(vocab,width)).astype(np.float32)
        self.q=rng.normal(0,scale,(width,width)).astype(np.float32); self.k=rng.normal(0,scale,(width,width)).astype(np.float32)
        self.v=rng.normal(0,scale,(width,width)).astype(np.float32); self.out=rng.normal(0,scale,(width,width)).astype(np.float32)
        self.norm=np.ones(width,np.float32); self.head=self.embed.T
    def logits(self,tokens):
        x=self.embed[np.asarray(tokens,dtype=np.int64)][None]
        def split(y): return y.reshape(1,len(tokens),self.heads,-1).transpose(0,2,1,3)
        attended=causal_attention(split(x@self.q),split(x@self.k),split(x@self.v))
        merged=attended.transpose(0,2,1,3).reshape(1,len(tokens),self.width)@self.out
        return rms_norm(x+merged,self.norm)@self.head
    def generate(self,prompt,max_new_tokens,eos=None):
        tokens=list(prompt)
        for _ in range(max_new_tokens):
            token=int(np.argmax(self.logits(tokens)[0,-1])); tokens.append(token); yield token
            if eos is not None and token==eos: break

class StreamingEngine:
    def __init__(self,model): self.model=model
    def stream(self,prompt,max_new_tokens=16,eos=None):
        if not prompt: raise ValueError("prompt required")
        yield from self.model.generate(prompt,max_new_tokens,eos)
