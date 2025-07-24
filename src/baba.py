from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Any, Tuple, Callable, Iterable
from collections import Counter
from itertools import combinations

# Notes:
# Δ attacks β iff there exists a subset Δ′ ⊆ Δ such that Δ′ ⊢ ¯β (a deduction of the contrary of β). Since in deductions 
# you can utilize rules whos bodies are in ∆, anything that can be reached from ∆' can be utilized in the derivation of ¯β
# Thus, checking attackers among Cl(Δ) is equivalent to searching all derivations.

# ---------- Literal ----------
@dataclass(frozen=True)
class Literal:
    key: str
    payload: Any = field(default=None, compare=False)

    def __hash__(self) -> int:
        return hash(self.key)

    def __str__(self) -> str:
        return self.key

# Extendable for logic later

# ---------- Rule ----------
@dataclass(frozen=True)
class Rule:
    """Bipolar rule: head ← body   (body ∈ A; head ∈ A ∪ ¯A)."""
    head: Literal
    body: Literal

    def __post_init__(self):
        if not isinstance(self.head, Literal) or not isinstance(self.body, Literal):
            raise TypeError("Rule.head and Rule.body must be Literal instances.")

# ---------- Framework ----------
@dataclass
class BipolarABA:
    assumptions: Set[Literal]
    contrary: Dict[Literal, Literal]         # α -> ¯α
    rules: Set[Rule]

    _support_from: Dict[Literal, Set[Literal]] = field(init=False, default_factory=dict)
    _attack_from:  Dict[Literal, Set[Literal]] = field(init=False, default_factory=dict)
    _inv_contrary: Dict[Literal, Literal] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._validate_core()
        self._index_rules()

    # --- validation ---
    def _validate_core(self) -> None:
        if not isinstance(self.assumptions, set) or not all(isinstance(a, Literal) for a in self.assumptions):
            raise TypeError("assumptions must be a set[Literal]")

        # contrary total on A
        missing = self.assumptions - self.contrary.keys()
        if missing:
            raise ValueError(f"Contrary missing for assumptions: {{ {', '.join(m.key for m in missing)} }}")
        vals = list(self.contrary.values())
        dups = {v.key for v, c in Counter(vals).items() if c > 1}
        if dups:
            raise ValueError(f"Multiple assumptions share the same contrary: {dups}")

        all_contraries = set(self.contrary.values())
        for r in self.rules:
            if r.body not in self.assumptions:
                raise ValueError(f"Rule body {r.body} is not an assumption")
            if (r.head not in self.assumptions) and (r.head not in all_contraries):
                raise ValueError(f"Rule head {r.head} is neither an assumption nor a known contrary")

    # --- indexing ---
    def _index_rules(self) -> None:
        self._support_from = {a: set() for a in self.assumptions}
        self._attack_from  = {a: set() for a in self.assumptions}

        self._inv_contrary = {v: k for k, v in self.contrary.items()}  # ¯β -> β

        for r in self.rules:
            if r.head in self.assumptions:          # support rule β ← α
                self._support_from[r.body].add(r.head)
            else:                                    # attack rule ¯β ← α
                attacked = self._inv_contrary[r.head]
                self._attack_from[r.body].add(attacked)

    # --- public views ---
    def support_from(self, a: Literal) -> Set[Literal]:
        return self._support_from.get(a, set())

    def attack_from(self, a: Literal) -> Set[Literal]:
        return self._attack_from.get(a, set()) 

    def closure(self, delta: Iterable[Literal]) -> Set[Literal]:
        """
        Compute Cl(delta): start with delta, add all assumptions reachable via support rules.
        Only assumptions can appear in the result (by definition).
        """
        # Normalise & validate input
        result: Set[Literal] = set(delta)
        unknown = result - self.assumptions
        if unknown:
            raise ValueError(f"closure() called with literals not in assumptions: { {u.key for u in unknown} }")

        # BFS over support edges
        queue = list(result)
        while queue:
            a = queue.pop()
            for sup in self._support_from.get(a, ()):
                if sup not in result:
                    result.add(sup)
                    queue.append(sup)
        return result

    def is_closed(self, delta: Iterable[Literal]) -> bool:
        """Check if delta == Cl(delta)."""
        dset = set(delta)
        return dset == self.closure(dset)

    def derives(self, delta: Iterable[Literal], target: Literal) -> bool:
        """
        Δ ⊢ target ?
        - If target is an assumption: membership in Cl(Δ).
        - If target is a contrary ¯β: exists α ∈ Cl(Δ) with rule ¯β ← α.
        """
        dset = set(delta)
        if not dset.issubset(self.assumptions):
            bad = {x.key for x in dset - self.assumptions}
            raise ValueError(f"derives() given literals not in assumptions: {bad}")

        cl = self.closure(dset)

        if target in self.assumptions:
            return target in cl

        if target in self._inv_contrary:  # it's a contrary
            attacked = self._inv_contrary[target]
            return any(attacked in self._attack_from[a] for a in cl)

        raise ValueError(f"Unknown target literal {target.key}")

    def attacks(self, delta: Iterable[Literal], beta: Literal) -> bool:
        """Δ attacks β  ⇔  Δ derives ¯β."""
        if beta not in self.assumptions:
            raise ValueError(f"attacks() expects an assumption, got {beta.key}")
        return self.derives(delta, self.contrary[beta])

    def attacks_set(self, delta: Iterable[Literal], Bs: Iterable[Literal]) -> bool:
        """Δ attacks B iff it attacks at least one β ∈ B."""
        return any(self.attacks(delta, b) for b in Bs)

    def conflict_free(self, delta: Iterable[Literal]) -> bool:
        """
        Δ is conflict-free iff it does not attack any of its own members.
        (We over-approx with the whole Δ; if a subset attacks, Δ attacks too.)
        """
        dset = set(delta)
        if not dset.issubset(self.assumptions):
            bad = {x.key for x in dset - self.assumptions}
            raise ValueError(f"conflict_free() got non-assumptions: {bad}")

        cl = self.closure(dset)
        for x in dset:
            # any attacker of x reachable from Δ ?
            if any(x in self._attack_from[a] for a in cl):
                return False
        return True


    def _closed_attackers_of(self, alpha: Literal) -> list[Set[Literal]]:
        """
        Heuristic but sufficient in Bipolar ABA:
        any attack on α must end with a rule ¬α ← a  (single body).
        So consider closures of each direct attacker 'a'.
        """
        if alpha not in self.assumptions:
            raise ValueError(f"_closed_attackers_of expects an assumption, got {alpha.key}")

        direct_attackers = {a for a in self.assumptions if alpha in self._attack_from[a]}
        return [self.closure({a}) for a in direct_attackers]


    def defends(self, delta: Iterable[Literal], alpha: Literal) -> bool:
        """
        Δ defends α iff for every closed attacker B of α, Δ attacks B.
        We instantiate closed attackers as closures of single attackers (see above).
        """
        dset = set(delta)
        if alpha not in self.assumptions:
            raise ValueError(f"defends() expects an assumption for alpha, got {alpha.key}")
        if not dset.issubset(self.assumptions):
            bad = {x.key for x in dset - self.assumptions}
            raise ValueError(f"defends() got non-assumptions in delta: {bad}")

        for B in self._closed_attackers_of(alpha):
            # B is closed and (by construction) attacks alpha
            if not self.attacks_set(dset, B):
                return False
        return True

    def defends_set(self, delta: Iterable[Literal], gamma: Iterable[Literal]) -> bool:
        """Δ defends every β ∈ Γ."""
        gset = set(gamma)
        return all(self.defends(delta, b) for b in gset)

    def defended_by(self, delta: Iterable[Literal]) -> Set[Literal]:
        """{ α ∈ A | Δ defends α }"""
        dset = set(delta)
        return {a for a in self.assumptions if self.defends(dset, a)}

    def is_admissible(self, delta: Iterable[Literal]) -> bool:
        dset = set(delta)
        return self.is_closed(dset) and self.conflict_free(dset) and self.defends_set(dset, dset)

    def _enum_adm_pref(self) -> tuple[set[frozenset[Literal]], set[frozenset[Literal]]]:
        """Backtracking search with pruning. Returns (admissible_sets, preferred_sets)."""
        ordered = tuple(sorted(self.assumptions, key=lambda l: l.key))
        visited: set[frozenset[Literal]] = set()
        adm_sets: set[frozenset[Literal]] = set()
        pref_sets: set[frozenset[Literal]] = set()

        def dfs(current: Set[Literal], start_idx: int) -> None:
            cur_f = frozenset(current)
            if cur_f in visited:
                return
            visited.add(cur_f)

            # record admissible (cheap check thanks to pruning)
            if self.is_admissible(current):
                adm_sets.add(cur_f)

            extendable = False
            for i in range(start_idx, len(ordered)):
                a = ordered[i]
                if a in current:
                    continue
                new_set = self.closure(current | {a})
                # prune early
                if not self.conflict_free(new_set):
                    continue
                if not self.defends_set(new_set, new_set):
                    continue
                extendable = True
                dfs(new_set, i + 1)

            # maximal wrt ⊆ among explored = candidate preferred
            if not extendable and self.is_admissible(current):
                pref_sets.add(cur_f)

        dfs(set(), 0)

        # ensure true maximality (remove any that is subset of another)
        pref_sets = {S for S in pref_sets if not any(S < T for T in pref_sets)}
        return adm_sets, pref_sets

    def admissible_extensions(self) -> list[Set[Literal]]:
        adm, _ = self._enum_adm_pref()
        return [set(s) for s in adm]

    def preferred_extensions(self) -> list[Set[Literal]]:
        _, pref = self._enum_adm_pref()
        return [set(s) for s in pref]