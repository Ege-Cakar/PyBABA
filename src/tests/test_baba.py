import pytest
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add project root
from baba import Literal, Rule, BipolarABA

# --------- Helpers ---------
def L(name: str) -> Literal:
    return Literal(name)

def make_basic_framework():
    """The running example from our REPLs:
       a -> b -> c (support chain), and b attacks a.
    """
    a, b, c = L("a"), L("b"), L("c")
    na, nb, nc = L("¬a"), L("¬b"), L("¬c")
    A = {a, b, c}
    contr = {a: na, b: nb, c: nc}
    rules = {
        Rule(b, a),   # b ← a  (support)
        Rule(c, b),   # c ← b  (support)
        Rule(na, b),  # ¬a ← b (attack)
    }
    return BipolarABA(A, contr, rules), (a, b, c, na, nb, nc)

def make_defence_framework():
    """
    a and b mutually attack; d attacks b; support edges none (flat).
    Δ = {a} defends a (because it attacks b back).
    """
    a, b, d = L("a"), L("b"), L("d")
    na, nb, nd = L("¬a"), L("¬b"), L("¬d")
    A = {a, b, d}
    contr = {a: na, b: nb, d: nd}
    rules = {
        Rule(na, b),  # b attacks a
        Rule(nb, a),  # a attacks b
        Rule(nb, d),  # d also attacks b (extra attacker to test closures)
    }
    return BipolarABA(A, contr, rules), (a, b, d, na, nb, nd)

def as_fsets(sets_iter):
    """Convert list[set[Literal]] to a set of frozensets for order-insensitive comparison."""
    return {frozenset(s) for s in sets_iter}

def make_large_framework(n: int = 30):
    """
    Deterministic 'large' framework to smoke-test performance.
    Support chain: a1 -> a2 -> ... -> an
    Extra supports: a_i -> a_{i+2} (skip-links)
    Attacks: for every 3rd node, a_{i+1} attacks a_i  (¬a_i ← a_{i+1})
    """
    A = [L(f"a{i}") for i in range(1, n + 1)]
    NA = [L(f"¬a{i}") for i in range(1, n + 1)]
    assumptions = set(A)
    contrary = {a: na for a, na in zip(A, NA)}

    rules = set()

    # main support chain + skip links
    for i in range(n - 1):
        rules.add(Rule(A[i + 1], A[i]))          # a_{i+1} ← a_i
        if i + 2 < n:
            rules.add(Rule(A[i + 2], A[i]))      # a_{i+2} ← a_i  (skip)

    # periodic attacks
    for i in range(0, n - 1, 3):
        rules.add(Rule(NA[i], A[i + 1]))         # ¬a_i ← a_{i+1}

    F = BipolarABA(assumptions, contrary, rules)
    return F, A, NA

# --------- Validation tests ---------
def test_validation_missing_contrary():
    a = L("a")
    with pytest.raises(ValueError):
        BipolarABA({a}, {}, set())

def test_validation_duplicate_contraries():
    a, b = L("a"), L("b")
    na = L("¬x")
    with pytest.raises(ValueError):
        BipolarABA({a, b}, {a: na, b: na}, set())

def test_validation_bad_rule_head():
    a, na = L("a"), L("¬a")
    # head is "weird" not in assumptions or contraries
    with pytest.raises(ValueError):
        BipolarABA({a}, {a: na}, {Rule(L("weird"), a)})

def test_validation_bad_rule_body():
    a, na = L("a"), L("¬a")
    with pytest.raises(ValueError):
        BipolarABA({a}, {a: na}, {Rule(na, L("notA"))})

# --------- Index / adjacency sanity ---------
def test_indices_basic():
    F, (a, b, c, *_ ) = make_basic_framework()
    assert F.support_from(a) == {b}
    assert F.support_from(b) == {c}
    assert F.support_from(c) == set()
    assert F.attack_from(b) == {a}
    assert F.attack_from(a) == set()

# --------- Closure tests ---------
def test_closure_basic():
    F, (a, b, c, *_ ) = make_basic_framework()
    assert F.closure({a}) == {a, b, c}
    assert F.closure({b}) == {b, c}
    assert F.closure({c}) == {c}
    assert F.is_closed({a, b, c})
    assert not F.is_closed({a, b})

def test_closure_properties():
    F, (a, b, c, *_ ) = make_basic_framework()
    # Monotonicity: Δ ⊆ Γ ⇒ Cl(Δ) ⊆ Cl(Γ)
    cl_a = F.closure({a})
    cl_ab = F.closure({a, b})
    assert {a}.issubset({a, b})
    assert cl_a.issubset(cl_ab)

    # Idempotence: Cl(Cl(Δ)) = Cl(Δ)
    assert F.closure(cl_a) == cl_a

def test_closure_raises_on_non_assumption():
    F, (a, b, c, na, *_ ) = make_basic_framework()
    with pytest.raises(ValueError):
        F.closure({a, na})  # na is a contrary

# --------- Derivation tests ---------
def test_derives_assumptions():
    F, (a, b, c, *_ ) = make_basic_framework()
    assert F.derives({a}, c)
    assert not F.derives({b}, a)

def test_derives_contraries():
    F, (a, b, c, na, nb, nc) = make_basic_framework()
    assert F.derives({a}, na)      # a → b, b attacks a
    assert not F.derives({c}, na)  # c can't trigger ¬a
    with pytest.raises(ValueError):
        F.derives({a, nb}, na)     # nb is not an assumption

def test_derives_unknown_target():
    F, (a, *_ ) = make_basic_framework()
    weird = L("weird")
    with pytest.raises(ValueError):
        F.derives({a}, weird)

# --------- Attack tests ---------
def test_attacks_and_attacks_set():
    F, (a, b, c, na, nb, nc) = make_basic_framework()
    assert F.attacks({a}, a)          # Δ derives ¬a
    assert not F.attacks({a}, b)
    assert not F.attacks_set({a}, {b, c})
    with pytest.raises(ValueError):
        F.attacks({a}, na)            # must be an assumption

# --------- Conflict-freeness tests ---------
def test_conflict_free_basic():
    F, (a, b, c, *_ ) = make_basic_framework()
    assert F.conflict_free({a}) is False
    assert F.conflict_free({b}) is True
    assert F.conflict_free({c}) is True
    assert F.conflict_free({a, c}) is False
    assert F.conflict_free({a, b}) is False

def test_conflict_free_raises():
    F, (a, *_ ) = make_basic_framework()
    with pytest.raises(ValueError):
        F.conflict_free({a, L("¬x")})

# --------- Defence tests ---------
def test_closed_attackers_helper():
    F, (a, b, c, na, *_ ) = make_basic_framework()
    attackers = F._closed_attackers_of(a)
    assert len(attackers) == 1
    assert attackers[0] == {b, c}

def test_defends_positive():
    F, (a, b, d, na, nb, nd) = make_defence_framework()
    # Attackers of a: {b}. Cl({b}) = {b} (no supports)
    # Δ = {a} attacks b (rule ¬b ← a), so Δ defends a
    assert F.defends({a}, a) is True

def test_defends_negative():
    F, (a, b, d, na, nb, nd) = make_defence_framework()
    # Δ = {d} does NOT defend a (it doesn't attack b)
    assert F.defends({d}, b) is False

def test_defends_set():
    F, (a, b, d, na, nb, nd) = make_defence_framework()
    assert F.defends_set({a}, {a}) is True
    assert F.defends_set({a}, {a, b}) is False  # {a} doesn't defend b from {d}

def test_defends_raises():
    F, (a, b, d, na, nb, nd) = make_defence_framework()
    with pytest.raises(ValueError):
        F.defends({a, na}, a)  # na is not an assumption
    with pytest.raises(ValueError):
        F.defends({a}, na)     # alpha must be assumption


def test_defended_by_basic():
    F, (a, b, c, na, nb, nc) = make_basic_framework()
    # Δ = {b,c} defends b and c (nobody attacks them); not a
    defended = F.defended_by({b, c})
    assert defended == {b, c}

def test_is_admissible_basic():
    F, (a, b, c, *_ ) = make_basic_framework()
    assert F.is_admissible(set()) is True
    assert F.is_admissible({c}) is True
    assert F.is_admissible({b, c}) is True
    assert F.is_admissible({a}) is False
    assert F.is_admissible({b}) is False        # not closed
    assert F.is_admissible({a, b}) is False     # not conflict-free

def test_admissible_and_preferred_basic():
    F, (a, b, c, *_ ) = make_basic_framework()
    adm = as_fsets(F.admissible_extensions())
    pref = as_fsets(F.preferred_extensions())

    expected_adm = {frozenset(), frozenset({c}), frozenset({b, c})}
    expected_pref = {frozenset({b, c})}

    assert adm == expected_adm
    assert pref == expected_pref

def test_is_admissible_defence_framework():
    F, (a, b, d, *_ ) = make_defence_framework()
    assert F.is_admissible(set()) is True
    assert F.is_admissible({a}) is True
    assert F.is_admissible({d}) is True
    assert F.is_admissible({a, d}) is True
    assert F.is_admissible({b}) is False          # can't defend against both a and d
    assert F.is_admissible({a, b}) is False       # not conflict-free
    assert F.is_admissible({b, d}) is False       # not conflict-free

def test_admissible_and_preferred_defence_framework():
    F, (a, b, d, *_ ) = make_defence_framework()
    adm = as_fsets(F.admissible_extensions())
    pref = as_fsets(F.preferred_extensions())

    expected_adm = {frozenset(), frozenset({a}), frozenset({d}), frozenset({a, d})}
    expected_pref = {frozenset({a, d})}

    assert adm == expected_adm
    assert pref == expected_pref

# ---------- large framework smoke test ----------
def test_large_framework_smoke():
    F, A, NA = make_large_framework( thirty := 30 )

    # pick a few literals
    a1, a10, a20 = A[0], A[9], A[19]
    na1 = NA[0]

    # closure shouldn't blow up
    cl = F.closure({a1})
    assert len(cl) >= 10  # should reach far due to support chain/skip links

    # conflict-free check on a random-ish subset
    subset = {a1, a10, a20}
    assert isinstance(F.conflict_free(subset), bool)

    # derives/attacks shouldn't crash
    F.derives({a1}, a20)
    F.derives({a1}, na1)
    F.attacks({a1}, a1)

    # defence on something (most likely vacuous)
    F.defends({a10}, a10)

    print(F.admissible_extensions())
    print("\n")
    print(F.preferred_extensions())

    print("Large framework smoke test passed")

# large smoke test (ADD)
def test_large_framework_semantics_smoke():
    F, A, NA = make_large_framework( twenty := 20 )
    prefs = F.preferred_extensions()
    assert prefs, "Should find at least one preferred extension"
    # sanity: each preferred is admissible & maximal
    for S in prefs:
        assert F.is_admissible(S)
        assert not any(S < T for T in prefs)