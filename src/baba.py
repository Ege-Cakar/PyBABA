from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Any, Tuple, Callable, Iterable
from collections import Counter
from itertools import combinations
from copy import deepcopy

# Notes:
# Δ attacks β iff there exists a subset Δ′ ⊆ Δ such that Δ′ ⊢ ¯β (a deduction of the contrary of β). Since in deductions 
# you can utilize rules whos bodies are in ∆, anything that can be reached from ∆' can be utilized in the derivation of ¯β
# Thus, checking attackers among Cl(Δ) is equivalent to searching all derivations.

# ---------- Dataclasses ----------
@dataclass(frozen=True)
class Literal:
    key: str
    payload: Any = field(default=None, compare=False)

    def __hash__(self) -> int:
        return hash(self.key)

    def __str__(self) -> str:
        return self.key

# Extendable for logic later

@dataclass
class DerivationNode:
    literal: "Literal"
    rule: "Rule | None"                    # None for Δ-leaf
    child: "DerivationNode | None" = None  # at most one child

    # nice console view
    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        head = f"{pad}{self.literal.key}"
        if self.rule:
            head += f"   ← {self.rule.body.key}"
        if self.child:
            return head + "\n" + self.child.pretty(indent + 1)
        return head

    # Graphviz-DOT export
    def to_dot(self, parent_id: int | None = None, counter: list[int] | None = None) -> str:
        if counter is None:
            counter = [0]
        my_id = counter[0]; counter[0] += 1
        lines = [f'  n{my_id} [label="{self.literal.key.replace(chr(34), r"\"")}"];']
        if parent_id is not None:
            lines.append(f'  n{parent_id} -> n{my_id};')
        if self.child:
            lines.append(self.child.to_dot(my_id, counter))
        return "\n".join(lines)


@dataclass
class DerivationTree:
    root: DerivationNode

    def pretty(self) -> str:
        return self.root.pretty()

    def to_dot(self) -> str:
        return "digraph Derivation {\n" + self.root.to_dot() + "\n}"

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

    # generic back-tracking over admissible supersets, with extra filter (for finding extensions)
    def _enum_with_filter(
        self,
        keep_fn: Callable[[Set[Literal]], bool],
        need_maximal: bool = False,
    ) -> list[Set[Literal]]:
        """
        Enumerate all admissible sets that also satisfy `keep_fn(S)`.
        If need_maximal=True, keep only ⊆-maximal ones.
        """
        ordered = tuple(sorted(self.assumptions, key=lambda l: l.key))
        seen: set[frozenset[Literal]] = set()
        results: list[Set[Literal]] = []

        def dfs(current: Set[Literal], start_idx: int) -> None:
            cur_f = frozenset(current)
            if cur_f in seen:
                return
            seen.add(cur_f)

            if self.is_admissible(current) and keep_fn(current):
                results.append(current)

            for i in range(start_idx, len(ordered)):
                a = ordered[i]
                if a in current:
                    continue
                new_set = self.closure(current | {a})
                if not self.conflict_free(new_set):
                    continue
                if not self.defends_set(new_set, new_set):
                    continue
                dfs(new_set, i + 1)

        dfs(set(), 0)

        if need_maximal:
            maximal = [S for S in results if not any(S < T for T in results)]
            return maximal
        return results

# -------- EXTENSIONS -----------

    def is_admissible(self, delta: Iterable[Literal]) -> bool:
        dset = set(delta)
        return self.is_closed(dset) and self.conflict_free(dset) and self.defends_set(dset, dset)

    def admissible_extensions(self) -> list[Set[Literal]]:
        return self._enum_with_filter(lambda _: True)

    def preferred_extensions(self) -> list[Set[Literal]]:
        return self._enum_with_filter(lambda _: True, need_maximal=True)

    def is_complete(self, delta: Iterable[Literal]) -> bool:
        dset = set(delta)
        return (
            self.is_admissible(dset)
            and dset == self.defended_by(dset)        # fix-point condition
        )

    def complete_extensions(self) -> list[Set[Literal]]:
        return self._enum_with_filter(self.is_complete) 

    def is_set_stable(self, delta: Iterable[Literal]) -> bool:
        dset = set(delta)
        if not (self.is_closed(dset) and self.conflict_free(dset)):
            return False
        outsiders = self.assumptions - dset
        for beta in outsiders:
            if not self.attacks_set(dset, self.closure({beta})):
                return False
        return True

    def set_stable_extensions(self) -> list[Set[Literal]]:
        return self._enum_with_filter(self.is_set_stable)

    def well_founded_extension(self) -> Set[Literal] | None:
        comps = self.complete_extensions()
        if not comps:
            return None
        inter = set.intersection(*map(set, comps))
        return inter

    def ideal_extensions(self) -> list[Set[Literal]]:
        prefs = self.preferred_extensions()
        if not prefs:
            return []      # definition vacuous if no preferred
        def subset_of_all_pref(S: Set[Literal]) -> bool:
            return all(S.issubset(P) for P in prefs)
        # need ⊆-max admissible sets that are subset of every preferred
        return self._enum_with_filter(subset_of_all_pref, need_maximal=True)
    
    def build_derivation_tree(self, delta: Iterable["Literal"], goal: "Literal") -> "DerivationTree | None":
        """
        Return ONE derivation tree of `goal` from Δ, or None if goal not derivable.
        Works under Bipolar-ABA’s single-body rule restriction.
        """
        # we work with the closure once
        base = set(delta)
        if not base.issubset(self.assumptions):
            raise ValueError("Δ must contain only assumptions")

        cl = self.closure(base)

        # recursive helper
        def derive(target: Literal) -> DerivationNode | None:
            # 1. leaf?  (target directly provided by Δ)
            if target in base:
                return DerivationNode(target, None, None)

            # 2. support rule head?
            for body in cl:
                if target in self._support_from[body]:
                    child = derive(body)
                    if child:
                        return DerivationNode(target, Rule(target, body), child)

            # 3. contrary head?
            if target in self._inv_contrary:           # target = ¬β
                attacked = self._inv_contrary[target]
                for body in cl:
                    if attacked in self._attack_from[body]:
                        child = derive(body)
                        if child:
                            return DerivationNode(target, Rule(target, body), child)

            return None  # no derivation found

        root_node = derive(goal)
        return DerivationTree(root_node) if root_node else None
    #  Enumerate *all* derivation trees for a goal from Δ
    def build_all_derivation_trees(
        self,
        delta: Iterable["Literal"],
        goal: "Literal",
        max_paths: int | None = None,           # optional cut-off
    ) -> list["DerivationTree"]:
        """
        Return a list of DerivationTree objects, one for each distinct path that
        derives `goal` from Δ.  Stops early after `max_paths` (if given).
        """
        base = set(delta)
        if not base.issubset(self.assumptions):
            raise ValueError("Δ must contain only assumptions")

        cl = self.closure(base)
        trees: list[DerivationTree] = []

        def dfs(target: Literal, seen: set[Literal]) -> list[DerivationNode]:
            """Return all DerivationNode roots that derive `target`."""
            if target in seen:                                   # avoid cycles
                return []
            if max_paths is not None and len(trees) >= max_paths:
                return []

            # case 1: leaf
            if target in base:
                return [DerivationNode(target, None, None)]

            results: list[DerivationNode] = []

            # case 2: support rules
            for body in self.assumptions:
                if body in cl and target in self._support_from[body]:
                    for child in dfs(body, seen | {target}):
                        results.append(
                            DerivationNode(target, Rule(target, body), child)
                        )

            # case 3: attack rules if target is a contrary
            if target in self._inv_contrary:
                attacked = self._inv_contrary[target]
                for body in cl:
                    if attacked in self._attack_from[body]:
                        for child in dfs(body, seen | {target}):
                            results.append(
                                DerivationNode(target, Rule(target, body), child)
                            )

            return results

        roots = dfs(goal, set())
        for r in roots:
            trees.append(DerivationTree(r))
            if max_paths is not None and len(trees) >= max_paths:
                break
        return trees

    def build_dialectical_tree(
        self,
        delta: Iterable["Literal"],
        alpha: "Literal",
        semantics: str = "admissible",
        max_depth: int | None = None,
    ) -> "DialecticalTree":
        """
        Build a dialogue tree under the chosen semantics.
        semantics ∈ {"admissible", "preferred"} #TODO: Implement the others too!
        """
        drv_cls = {"admissible": AdmissibleDriver,
                "preferred": PreferredDriver}.get(semantics)
        if drv_cls is None:
            raise ValueError(f"unsupported semantics {semantics!r}")
        driver = drv_cls(self)

        Δ = set(delta)
        if alpha not in self.assumptions:
            raise ValueError("alpha must be an assumption")
        if not Δ.issubset(self.assumptions):
            raise ValueError("Δ must contain only assumptions")

        root = DialecticalNode("pro", Δ, alpha)

        def expand(node: DialecticalNode, depth: int):
            if max_depth is not None and depth >= max_depth:
                return

            if node.role == "pro":
                if node.target is None:
                    return  # nothing to defend at this level
                for B in driver.closed_attackers(node.target):
                    opp = DialecticalNode("opp", B, node.target)
                    node.children.append(opp)
                    expand(opp, depth + 1)

            else:  # node.role == "opp"
                B = node.support_set
                # minimal counter-attack: find some subset of Δ that attacks B
                attacker = next(( {x} for x in Δ if self.attacks({x}, next(iter(B))) ),
                                None)
                if attacker is None and self.attacks_set(Δ, B):
                    attacker = Δ
                if attacker:
                    pro = DialecticalNode("pro", attacker, None)
                    node.children.append(pro)
                    # optional deeper defence on each β in B
                    for beta in B:
                        child = DialecticalNode("pro", attacker, beta)
                        pro.children.append(child)
                        expand(child, depth + 2)

        expand(root, 0)

        # apply extra burden (e.g. maximality) at root
        if not driver.extra_burden(root.support_set):
            root.children.append(
                DialecticalNode("fail", set(), None)  # marks unmet burden
            )
        return DialecticalTree(root)

class _BaseDriver:
    """Abstract helper—concrete subclasses below."""
    def __init__(self, F: "BipolarABA"):
        self.F = F

    # --- required hooks ---                                     #
    def closed_attackers(self, alpha: "Literal") -> list[Set["Literal"]]:
        raise NotImplementedError

    def pro_can_answer(self, B: Set["Literal"], delta: Set["Literal"]) -> bool:
        raise NotImplementedError

    # In case we don't set it up or it's not loaded properly

    def extra_burden(self, pro_set: Set["Literal"]) -> bool:
        """Checks maximality, fixpoints, etc.  Pass-through for admissible."""
        return True


class AdmissibleDriver(_BaseDriver):
    """Default admissible semantics."""
    def closed_attackers(self, alpha):
        return self.F._closed_attackers_of(alpha)

    def pro_can_answer(self, B, delta):
        return self.F.attacks_set(delta, B)


class PreferredDriver(AdmissibleDriver):
    """⊆-maximal admissible."""
    def extra_burden(self, pro_set):
        # Δ is ⊆-maximal admissible iff no admissible superset exists.
        for a in self.F.assumptions - pro_set:
            sup = self.F.closure(pro_set | {a})
            if self.F.is_admissible(sup):
                return False
        return True

@dataclass
class DialecticalNode:
    role: str                       # "pro" or "opp"
    support_set: Set["Literal"]     # the set making the move
    target: "Literal | None"        # assumption being discussed (None if node just attacks a set)
    children: list["DialecticalNode"] = field(default_factory=list)

    # pretty‐print
    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        who = "PRO" if self.role == "pro" else "OPP"
        tgt = f" ⇒ {self.target.key}" if self.target else ""
        head = f"{pad}{who}: {{{', '.join(x.key for x in sorted(self.support_set, key=lambda l: l.key))}}}{tgt}"
        lines = [head]
        for ch in self.children:
            lines.append(ch.pretty(indent + 1))
        return "\n".join(lines)

    # DOT export
    def to_dot(self, parent_id: int | None = None, counter: list[int] | None = None) -> str:
        if counter is None:
            counter = [0]
        my_id = counter[0]; counter[0] += 1
        label = ("PRO" if self.role == "pro" else "OPP") + r"\n{" + \
                ", ".join(x.key for x in self.support_set) + "}"
        shape = "box" if self.role == "pro" else "ellipse"
        lines = [f'  n{my_id} [shape={shape}, label="{label}"];']
        if parent_id is not None:
            lines.append(f'  n{parent_id} -> n{my_id};')
        for ch in self.children:
            lines.append(ch.to_dot(my_id, counter))
        return "\n".join(lines)


@dataclass
class DialecticalTree:
    root: DialecticalNode

    def pretty(self) -> str:
        return self.root.pretty()

    def to_dot(self) -> str:
        return "digraph Dialogue {\n" + self.root.to_dot() + "\n}"