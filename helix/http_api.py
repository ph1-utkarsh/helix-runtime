"""Minimal local JSON/NDJSON streaming serving layer."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path!="/generate": self.send_error(404); return
            try:
                length=int(self.headers.get("Content-Length","0")); row=json.loads(self.rfile.read(length))
                prompt=row["prompt"]; limit=int(row.get("max_new_tokens",16))
                if not isinstance(prompt,list) or not 1<=limit<=256: raise ValueError
                self.send_response(200); self.send_header("Content-Type","application/x-ndjson"); self.end_headers()
                for token in engine.stream(prompt,limit):
                    self.wfile.write((json.dumps({"token":token})+"\n").encode()); self.wfile.flush()
            except (ValueError,KeyError,TypeError,json.JSONDecodeError): self.send_error(400)
        def log_message(self,*args): pass
    return Handler

def serve(engine,host="127.0.0.1",port=0): return ThreadingHTTPServer((host,port),handler(engine))
