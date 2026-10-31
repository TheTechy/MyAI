import ast
ast.parse(open("core/inference/filegen.py").read()); print("ok  ast.parse")
from core.inference import filegen as F

# detect_requested_extension: needs create-intent + type; negation/no-file guards
assert F.detect_requested_extension("write me a python script") == ".py"
assert F.detect_requested_extension("create an HTML page, no JavaScript") == ".html"
assert F.detect_requested_extension("check my bash script") is None          # no create intent
assert F.detect_requested_extension("make a note, no file please") is None   # no-file guard
assert F.detect_requested_extension("what is a csv file?") is None           # no create intent
print("ok  detect_requested_extension (intent + negation + no-file)")

# make_filename: topic words near the cue, stopwords stripped
fn = F.make_filename("create an html page for my coffee shop menu", ".html")
assert fn.endswith(".html") and "coffee" in fn and "html" not in fn.replace(".html","")
print("ok  make_filename picks topic words", fn)

# wrap + correct
assert F.wrap_as_file_block("hi", "a.txt") == "[FILE:a.txt]\nhi\n[/FILE]"
corrected = F.correct_file_extensions("[FILE:notes.md]\nx[/FILE]", "save it as a text file")
assert "[FILE:notes.txt]" in corrected
print("ok  wrap + correct_file_extensions")

# suppress_prose_if_file_delivered
assert F.suppress_prose_if_file_delivered("long\nprose\nhere\nlots", 1) == ""   # long → suppressed
assert F.suppress_prose_if_file_delivered("Here you go.", 1) == "Here you go."  # short → kept
assert F.suppress_prose_if_file_delivered("anything", 0) == "anything"          # no file → kept
print("ok  suppress_prose_if_file_delivered")

# file_description
assert "text file" in F.file_description([{"filename":"a.txt"}])
assert "2 files" in F.file_description([{"filename":"a.txt"},{"filename":"b.md"}])
print("ok  file_description")

# looks_like_question
assert F.looks_like_question("Could you clarify the colour?") is True
assert F.looks_like_question("What format would you like") is True
assert F.looks_like_question("Here is your report.") is False
print("ok  looks_like_question")

# find_historical_file_request: only during a clarifying exchange
hist = [{"role":"user","content":"make me an html page"},
        {"role":"model","content":"What colour scheme would you like?"}]
assert F.find_historical_file_request(hist) == ".html"
hist2 = [{"role":"user","content":"make me an html page"},
         {"role":"model","content":"Here is your page."}]
assert F.find_historical_file_request(hist2) is None    # last model turn not a question
print("ok  find_historical_file_request (clarifying-exchange only)")

# ── prepare_files orchestration (fake deps) ─────────────────────────────────────
class Deps:
    def has_file_block(self, t): return "[FILE:" in t
    def strip_file_blocks(self, t):
        import re; return re.sub(r'\[FILE:[^\]]+\]\n?.*?\[/FILE\]','',t,flags=re.DOTALL).strip()
    def extract_and_generate(self, t):
        import re
        return [{"filename": m.group(1)} for m in re.finditer(r'\[FILE:([^\]]+)\]', t)]
d = Deps()

# (a) model emitted a block → extract, strip, suppress long prose
ans, files = F.prepare_files("Intro line\n[FILE:tips.txt]\na\nb[/FILE]\nlong trailing prose here", 
                             "write a text file of tips", None, set(), d)
assert files and files[0]["filename"]=="tips.txt"
assert "[FILE:" not in ans
print("ok  prepare_files: emitted block → extracted + stripped")

# (b) file REQUESTED but model wrote prose (no block) → WRAP into a file
ans, files = F.prepare_files("def greet(): return 'hi'", "write me a python script that greets",
                             None, set(), d)
assert files and files[0]["filename"].endswith(".py"), files
print("ok  prepare_files: file requested, no block → WRAPPED into a file")

# (c) file requested BUT user uploaded something → do NOT wrap (they want analysis)
ans, files = F.prepare_files("Your code looks fine.", "review my python script",
                             None, {"mycode.py"}, d)
assert not files and ans == "Your code looks fine."
print("ok  prepare_files: upload present → no wrap (analysis, not generation)")

# (d) model asked a clarifying question → do NOT wrap
ans, files = F.prepare_files("What framework should I use?", "make me an html page",
                             None, set(), d)
assert not files and "?" in ans
print("ok  prepare_files: clarifying question → no wrap")

# (e) uploaded file must not be re-emitted even if model blocks it
ans, files = F.prepare_files("[FILE:mycode.py]\nx[/FILE]", "tweak it", None, {"mycode.py"}, d)
assert not files
print("ok  prepare_files: uploaded filename never re-emitted")

print("\nALL GREEN")
