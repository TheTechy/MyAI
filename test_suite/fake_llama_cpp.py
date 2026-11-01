"""Fake llama_cpp mimicking the 0.3.33 surface verified from source."""
class LlamaGrammar:
    def __init__(self, _grammar): self._grammar = _grammar; self.origin = None
    @classmethod
    def from_string(cls, grammar, verbose=True):
        g = cls(grammar); g.origin = ("gbnf", grammar, verbose); return g
    @classmethod
    def from_json_schema(cls, json_schema, verbose=True):
        g = cls(json_schema); g.origin = ("schema", json_schema, verbose); return g

class Llama:
    instances = []
    def __init__(self, **kw):
        self.init_kwargs = kw
        self.closed = False
        self.calls = []
        self.n_tokens = 42
        self.metadata = {"tokenizer.chat_template": kw.pop("_template", "")}
        self._trained = getattr(type(self), "TRAINED", 8192)
        self._api_ok  = getattr(type(self), "API_OK", True)
        self.model = _Handle(self._trained if self._api_ok else None)
        self._model = _FakeModel(self._trained, self.model)
        Llama.instances.append(self)
    def n_ctx(self):
        n = self.init_kwargs.get("n_ctx", 512)
        # llama-cpp resolves n_ctx=0 to the model's trained maximum
        return self._trained if n == 0 else n
    def close(self): self.closed = True
    def tokenize(self, text, add_bos=True, special=False):
        # One token per word: predictable counts for tests.
        self.tokenize_calls = getattr(self, "tokenize_calls", []) + [(text, add_bos, special)]
        return [0] * len(text.decode("utf-8").split())
    def create_chat_completion(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            def gen():
                yield {"choices":[{"delta":{"content":"hel"},"finish_reason":None}]}
                yield {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}
            return gen()
        return {"choices":[{"message":{"role":"assistant","content":"hello"},
                            "finish_reason":"stop"}]}

# --- trained-context surface (mirrors 0.3.33) -------------------------------
class _FakeModel:
    def __init__(self, trained, handle): self._trained = trained; self.model = handle
    def n_ctx_train(self):
        if self._trained is None: raise RuntimeError("unavailable")
        return self._trained

def llama_model_n_ctx_train(handle):
    """Public C API path. handle carries the trained value, or raises."""
    if getattr(handle, "trained", None) is None: raise RuntimeError("unavailable")
    return handle.trained

class _Handle:
    def __init__(self, trained): self.trained = trained
