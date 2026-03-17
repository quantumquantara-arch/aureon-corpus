#!/usr/bin/env python3
"""
AUREON TRAJECTORY ENGINE + SPEECH SYNTHESIZER
==============================================
Phases 5 & 6 of the Human Speech Engine pipeline.
Author: Nadine Squires / Quantara

TRAJECTORY ENGINE (Phase 5):
    Given current state + active attractor + history,
    computes optimal next state using kappa-tau-sigma constraints.

SPEECH SYNTHESIZER (Phase 6):
    Given target state, finds speech atoms whose vectors
    sum closest to the target. Assembles into response.

HUMAN SPEECH ENGINE (Phase 7 shim):
    Drop-in replacement for LLM call.
    engine = HumanSpeechEngine('./index', './attractors.json')
    result = engine.respond(user_message)

USAGE:
    python aureon_trajectory_engine.py demo ./index --attractors ./attractors.json
    python aureon_trajectory_engine.py trajectory warmth=0.9,rapport=0.8 therapeutic --attractors ./attractors.json
"""
from __future__ import annotations
import json, math, re, time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

PHASE_DIM = 24
DIM_NAMES = [
    "valence","arousal","dominance","topic_depth","rapport",
    "formality","vulnerability","playfulness","intellectual","momentum",
    "certainty","curiosity","warmth","surprise","tension",
    "narrative","metaphor","agency","temporal_focus","scope",
    "pace","reciprocity","kappa","tau",
]

# ====================================================================
# VECTOR MATH
# ====================================================================

def _add(a, b): return [a[i]+b[i] for i in range(PHASE_DIM)]
def _sub(a, b): return [a[i]-b[i] for i in range(PHASE_DIM)]
def _scale(v, s): return [x*s for x in v]
def _mag(v): return math.sqrt(sum(x*x for x in v))
def _norm(v):
    m = _mag(v)
    return [x/m for x in v] if m > 1e-10 else [0.0]*PHASE_DIM
def _dist(a, b): return _mag(_sub(a, b))
def _clamp(v, lo=-1.0, hi=1.0): return [max(lo, min(hi, x)) for x in v]
def _mean(vecs): return [sum(v[i] for v in vecs)/len(vecs) for i in range(PHASE_DIM)] if vecs else [0.0]*PHASE_DIM


# ====================================================================
# TRAJECTORY CONSTRAINTS
# ====================================================================

@dataclass
class TrajectoryConstraints:
    """
    kappa-tau-sigma constraint set.
    kappa: spatial coherence with recent history [0,1]
    tau:   temporal responsibility / forward momentum [0,1]
    sigma: max systemic risk allowed [0,1]
    """
    kappa: float = 0.82
    tau: float = 0.80
    sigma: float = 0.20

    def decision_strength(self) -> float:
        return self.kappa * self.tau * (1.0 - self.sigma)


# ====================================================================
# TRAJECTORY ENGINE
# ====================================================================

class TrajectoryEngine:
    """
    Computes optimal next conversation state.

    Forces acting on state:
      1. Attractor pull       - toward attractor center
      2. Flow field velocity  - learned from real transcripts
      3. Kappa correction     - coherence with recent history
      4. Tau correction       - forward temporal momentum
      5. Sigma boundary       - clamp risk dimensions
      6. Target override      - optional hard steer
    """

    def __init__(self, constraints=None):
        self.constraints = constraints or TrajectoryConstraints()

    def compute(
        self,
        current_state: List[float],
        attractor_center: List[float],
        attractor_radius: float,
        flow_field: Dict[str, List[float]],
        history: List[List[float]],
        target_override: Optional[List[float]] = None,
    ) -> List[float]:
        c = self.constraints

        # 1. Attractor pull
        to_attractor = _sub(attractor_center, current_state)
        dist_to_center = _dist(current_state, attractor_center)
        pull_strength = min(0.4, dist_to_center * 0.3)
        attractor_pull = _scale(to_attractor, pull_strength)

        # 2. Flow field velocity
        grid_key = ','.join(str(int((x+1.0)/2.0*5)) for x in current_state)
        flow_velocity = flow_field.get(grid_key, [0.0]*PHASE_DIM)
        flow_contribution = _scale(flow_velocity, 0.25)

        # 3. Kappa correction - coherence with recent history
        if history:
            history_mean = _mean(history[-min(5, len(history)):])
            kappa_correction = _scale(_sub(history_mean, current_state), c.kappa * 0.15)
        else:
            kappa_correction = [0.0] * PHASE_DIM

        # 4. Tau correction - forward momentum
        tau_correction = _scale(_norm(to_attractor), c.tau * 0.2)

        # 5. Combine all forces
        delta = _add(
            _add(attractor_pull, flow_contribution),
            _add(kappa_correction, tau_correction)
        )
        next_state = _add(current_state, delta)

        # 6. Target override
        if target_override:
            next_state = _add(_scale(next_state, 0.4), _scale(target_override, 0.6))

        # 7. Sigma boundary - clamp risk dims (arousal=1, dominance=2, tension=14)
        for d in [1, 2, 14]:
            next_state[d] = min(next_state[d], 1.0 - c.sigma)

        return _clamp(next_state)

    def navigate(
        self,
        user_state: List[float],
        attractor_center: List[float],
        attractor_radius: float,
        flow_field: Dict[str, List[float]],
        history: List[List[float]],
    ) -> List[float]:
        """
        Compute where the RESPONSE should land.
        Blends user state with attractor - doesn't just echo user.
        """
        response_start = _add(_scale(user_state, 0.6), _scale(attractor_center, 0.4))
        return self.compute(
            current_state=response_start,
            attractor_center=attractor_center,
            attractor_radius=attractor_radius,
            flow_field=flow_field,
            history=history,
        )


# ====================================================================
# SPEECH SYNTHESIZER
# ====================================================================

@dataclass
class SpeechAtom:
    uid: str
    text: str
    source: str
    vector: List[float]
    distance: float = 0.0


class SpeechSynthesizer:
    """
    Greedy vector composition toward target state.
    Finds atoms that sum closest to target in 24-D phase space.
    """

    DISTANCE_THRESHOLD = 0.15
    MAX_ATOMS = 12
    MIN_ATOMS = 2

    def __init__(self, index):
        self.index = index

    def synthesize(
        self,
        target_state: List[float],
        max_atoms: int = None,
        filter_source: Optional[str] = None,
        avoid_uids: Optional[set] = None,
    ) -> Tuple[str, List[SpeechAtom], float]:
        if max_atoms is None: max_atoms = self.MAX_ATOMS
        if avoid_uids is None: avoid_uids = set()

        current_vector = [0.0] * PHASE_DIM
        selected: List[SpeechAtom] = []

        for step in range(max_atoms):
            needed = _sub(target_state, current_vector)
            current_dist = _dist(current_vector, target_state)

            if step >= self.MIN_ATOMS and current_dist < self.DISTANCE_THRESHOLD:
                break

            results = self.index.search(
                query_vector=needed, k=50, filter_source=filter_source
            )

            best = None
            for r in results:
                if r['uid'] in avoid_uids: continue
                if any(a.uid == r['uid'] for a in selected): continue
                best = r
                break
            if not best: break

            atom_vector = self._get_vector(best)
            if not atom_vector: continue

            atom = SpeechAtom(
                uid=best['uid'], text=best['text'],
                source=best['source'], vector=atom_vector,
                distance=best['distance']
            )
            selected.append(atom)
            current_vector = _add(current_vector, atom_vector)

        if not selected:
            return '', [], float('inf')

        text = self._assemble([a.text for a in selected])
        return text, selected, round(_dist(current_vector, target_state), 4)

    def _get_vector(self, result):
        idx = result.get('global_idx', -1)
        if hasattr(self.index, '_meta') and 0 <= idx < len(self.index._meta):
            return self.index._meta[idx].get('vector')
        return None

    def _assemble(self, texts):
        result = re.sub(r' +', ' ', ' '.join(t.strip() for t in texts if t.strip()))
        if result and result[-1] not in '.!?,;':
            result += '.'
        return result


# ====================================================================
# HUMAN SPEECH ENGINE - FULL INTEGRATION SHIM
# ====================================================================

class HumanSpeechEngine:
    """
    Drop-in replacement for LLM call.
    Combines: vectorizer + attractor classifier + trajectory + synthesizer.

    Usage:
        engine = HumanSpeechEngine('./index', './attractors.json')
        result = engine.respond('Tell me about dopamine')
        print(result['text'])  # assembled from real speech atoms
    """

    def __init__(self, index_dir, attractors_path=None, constraints=None):
        import importlib.util
        base = Path(__file__).parent

        def _load(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        vm = _load('aureon_atom_vectorizer', base / 'aureon_atom_vectorizer.py')
        fm = _load('aureon_faiss_builder', base / 'aureon_faiss_builder.py')
        am = _load('aureon_attractor_learner', base / 'aureon_attractor_learner.py')

        self.vectorizer = vm.AtomVectorizer()
        self.index = fm.AureonIndex(index_dir).load()
        self.synthesizer = SpeechSynthesizer(self.index)
        self.trajectory = TrajectoryEngine(constraints)

        if attractors_path and Path(attractors_path).exists():
            self.learner = am.AttractorLearner.load(attractors_path)
        else:
            self.learner = am.AttractorLearner()
            print("[HSE] Using seeded attractors (no fitted file found)")

        self._history: Deque[List[float]] = deque(maxlen=20)
        self._used_uids: set = set()
        self._turn = 0

    def respond(self, user_message, conversation_history=None, mode_override=None):
        self._turn += 1
        user_state = self.vectorizer.vectorize(user_message)
        self._history.append(user_state)

        if mode_override and mode_override in self.learner.attractors:
            mode_name = mode_override
        else:
            mode_name = self.learner.classify_state(user_state, top_n=1)[0][0]

        att = self.learner.attractors[mode_name]
        target_state = self.trajectory.navigate(
            user_state=user_state,
            attractor_center=att.center,
            attractor_radius=att.basin_radius,
            flow_field=att.flow_field,
            history=list(self._history),
        )

        text, atoms, distance = self.synthesizer.synthesize(
            target_state=target_state,
            avoid_uids=self._used_uids,
        )

        for a in atoms:
            self._used_uids.add(a.uid)
        if atoms:
            self._history.append(self.vectorizer.vectorize(text))

        return {
            'text': text,
            'target_state': target_state,
            'mode': mode_name,
            'atoms_used': len(atoms),
            'distance': distance,
            'turn': self._turn,
        }


# ====================================================================
# CLI
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='AUREON Trajectory Engine + Speech Synthesizer'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    d = sub.add_parser('demo', help='Interactive demo conversation')
    d.add_argument('index_dir')
    d.add_argument('--attractors', default=None)
    d.add_argument('--mode', default=None)

    t = sub.add_parser('trajectory', help='Show trajectory from dims to attractor')
    t.add_argument('current_dims', help='e.g. warmth=0.9,rapport=0.8')
    t.add_argument('attractor', help='Attractor name')
    t.add_argument('--attractors', required=True)

    args = parser.parse_args()

    if args.command == 'demo':
        print("[AUREON HSE] Loading engine...")
        engine = HumanSpeechEngine(args.index_dir, args.attractors)
        print("Ready. Ctrl+C to quit.\n")
        history = []
        while True:
            try:
                msg = input('You: ').strip()
                if not msg: continue
                r = engine.respond(msg, mode_override=args.mode)
                print(f"\nAUREON [{r['mode']}]: {r['text']}")
                print(f"  atoms={r['atoms_used']} dist={r['distance']} turn={r['turn']}\n")
                history.append(msg)
            except KeyboardInterrupt:
                print("\nDone.")
                break

    elif args.command == 'trajectory':
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from aureon_attractor_learner import AttractorLearner
        learner = AttractorLearner.load(args.attractors)
        state = [0.0] * PHASE_DIM
        for pair in args.current_dims.split(','):
            if '=' in pair:
                n, v = pair.strip().split('=')
                if n.strip() in DIM_NAMES:
                    state[DIM_NAMES.index(n.strip())] = float(v.strip())
        if args.attractor not in learner.attractors:
            print(f"Unknown: {args.attractor}")
            print(f"Available: {list(learner.attractors.keys())}")
            return
        att = learner.attractors[args.attractor]
        nxt = TrajectoryEngine().compute(
            state, att.center, att.basin_radius, att.flow_field, []
        )
        print("Current top dims:")
        for idx, val in sorted(enumerate(state), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"  {DIM_NAMES[idx]:20s} {val:+.3f}")
        print(f"\nNext state [{args.attractor}]:")
        for idx, val in sorted(enumerate(nxt), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"  {DIM_NAMES[idx]:20s} {val:+.3f}")
        print(f"\nDelta: {_dist(state, nxt):.4f}")


if __name__ == '__main__':
    main()
