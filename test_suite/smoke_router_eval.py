import sys
sys.path.insert(0, "scripts")
import router_eval as E
from core.inference.router import RouteDecision, DIRECT, ROUTES

def dec(route, parsed=True, valid=True, repaired=False):
    return RouteDecision(route, parsed, valid, "", "", repaired)

# 1) A PERFECT decider (returns exactly the expected route) → 100% correct, stable.
def perfect(q):
    exp = next(e for (qq, e, g) in E.TEST_SET if qq == q)
    if exp == E.CONSISTENCY:
        return dec("direct")          # recommendations: some stable choice
    return dec(exp)
res = E.run_eval(perfect, consistency_runs=9)
assert res["correct"] == res["graded"], (res["correct"], res["graded"])
assert res["graded"] == len(E.TEST_SET) - 2      # 2 recommendation queries excluded
assert res["free_clean"] == res["n"] and res["repaired"] == 0 and res["fell_back"] == 0
assert all(top == k for (_, top, k, _) in res["consistency"])   # all stable
print(f"ok  perfect decider → {res['correct']}/{res['graded']} correct, all stable")

# 2) repaired / fell_back accounting.
def mixed(q):
    exp = next(e for (qq, e, g) in E.TEST_SET if qq == q)
    if exp == E.CONSISTENCY: return dec("web_search")
    if exp == "weather":      return dec(exp, repaired=True)      # via grammar repair
    if exp == "news":         return dec(DIRECT, parsed=False, valid=False)  # fell back
    return dec(exp)
res = E.run_eval(mixed, quick=True)
assert res["repaired"] == 3      # all three weather queries repaired
assert res["fell_back"] == 1     # the news query fell back
# news counted WRONG (fell back to direct != news); weather still correct (repair gave right route)
wrong = [r for r in res["rows"] if r[0] == "WRONG"]
assert len(wrong) == 1 and wrong[0][1] == "tools" and wrong[0][3] == DIRECT
print("ok  repaired/fell_back accounted; fallback counts as misroute")

# 3) consistency catches an UNSTABLE query.
class Flaky:
    def __init__(s): s.n = 0
    def __call__(s, q):
        s.n += 1
        return dec("weather" if s.n % 2 else "direct")   # alternates
res = E.run_eval(Flaky(), consistency_runs=4)
assert any(top != k for (_, top, k, _) in res["consistency"]), "must flag instability"
print("ok  consistency flags an unstable query")

# 4) print_report returns 0 when clean, 1 when a misroute exists.
assert E.print_report(E.run_eval(perfect, consistency_runs=3)) == 0
assert E.print_report(E.run_eval(mixed, quick=True)) == 1
print("ok  print_report exit codes (0 clean / 1 attention)")

# 5) grammar builds from the full menu (route names all present).
from core.inference.router import build_route_grammar
g = build_route_grammar()
assert all(n in g for n in ROUTES)
print("ok  eval references full menu; grammar in sync")

print("\nALL GREEN")
