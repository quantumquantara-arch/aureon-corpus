#!/usr/bin/env python3
"""
AUREON ATTRACTOR LEARNER — Phase 4
Author: Nadine Squires / Quantara

Learns conversation flow attractors from real transcript sequences.
Each attractor = a conversation MODE with a center in 24-D phase space.

Built-in seed modes:
    comfort_conversation, deep_exploration, debate,
    teaching_mode, therapeutic, playful_banter,
    storytelling, crisis_support

USAGE:
    python aureon_attractor_learner.py fit ./vectorized ./attractors.json
    python aureon_attractor_learner.py classify ./attractors.json warmth=0.9,rapport=0.8
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PHASE_DIM = 24
DIM_NAMES = [
    "valence","arousal","dominance","topic_depth","rapport",
    "formality","vulnerability","playfulness","intellectual","momentum",
    "certainty","curiosity","warmth","surprise","tension",
    "narrative","metaphor","agency","temporal_focus","scope",
    "pace","reciprocity","kappa","tau",
]

def _dim(name): return DIM_NAMES.index(name)
def _mean(vecs): return [sum(v[i] for v in vecs)/len(vecs) for i in range(PHASE_DIM)] if vecs else [0.0]*PHASE_DIM
def _dist(a, b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(PHASE_DIM)))
def _sub(a, b): return [a[i]-b[i] for i in range(PHASE_DIM)]
def _discretize(v, bins=5): return ','.join(str(int((x+1.0)/2.0*bins)) for x in v)


@dataclass
class Attractor:
    name: str
    center: List[float] = field(default_factory=lambda: [0.0]*PHASE_DIM)
    basin_radius: float = 0.5
    flow_field: Dict[str, List[float]] = field(default_factory=dict)
    source_count: int = 0
    fit_error: float = 0.0
    description: str = ''

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d): return cls(**d)

    def pull(self, current):
        """Compute attractor pull vector at current position."""
        to_center = _sub(self.center, current)
        dist = _dist(current, self.center)
        strength = min(1.0, dist / max(self.basin_radius, 0.01)) * 0.3
        return [x * strength for x in to_center]

    def contains(self, state):
        return _dist(state, self.center) <= self.basin_radius


class AttractorLearner:

    def __init__(self):
        self.attractors: Dict[str, Attractor] = {}
        self._seed_attractors()

    def _seed_attractors(self):
        seeds = [
            {
                'name': 'comfort_conversation',
                'description': 'Warm, relaxed, mutually supportive dialogue',
                'dims': {'valence':0.5,'warmth':0.8,'rapport':0.7,'playfulness':0.3,
                         'tension':0.1,'vulnerability':0.4,'formality':0.2,
                         'pace':0.35,'reciprocity':0.7},
                'basin_radius': 0.6,
            },
            {
                'name': 'deep_exploration',
                'description': 'Philosophical, curious, intellectually open',
                'dims': {'curiosity':0.85,'topic_depth':0.8,'intellectual':0.65,
                         'vulnerability':0.5,'formality':0.3,'tension':0.3,
                         'warmth':0.6,'metaphor':0.55,'pace':0.4},
                'basin_radius': 0.55,
            },
            {
                'name': 'debate',
                'description': 'High certainty, intellectual sparring, assertive',
                'dims': {'certainty':0.85,'intellectual':0.8,'arousal':0.6,
                         'dominance':0.65,'tension':0.55,'curiosity':0.6,
                         'warmth':0.3,'pace':0.65},
                'basin_radius': 0.5,
            },
            {
                'name': 'teaching_mode',
                'description': 'Expert explaining, structured, forward-moving',
                'dims': {'intellectual':0.8,'dominance':0.6,'certainty':0.75,
                         'topic_depth':0.7,'formality':0.55,'pace':0.55,
                         'narrative':0.4,'agency':0.65},
                'basin_radius': 0.5,
            },
            {
                'name': 'therapeutic',
                'description': 'Deep empathy, emotional safety, vulnerability welcome',
                'dims': {'warmth':0.9,'vulnerability':0.7,'rapport':0.8,'valence':0.2,
                         'pace':0.25,'formality':0.2,'reciprocity':0.75,
                         'tension':0.3,'curiosity':0.6},
                'basin_radius': 0.6,
            },
            {
                'name': 'playful_banter',
                'description': 'Witty, quick, light, affectionate',
                'dims': {'playfulness':0.85,'rapport':0.7,'arousal':0.55,'valence':0.65,
                         'formality':0.05,'pace':0.7,'reciprocity':0.8,'surprise':0.45},
                'basin_radius': 0.55,
            },
            {
                'name': 'storytelling',
                'description': 'Narrative arc, past-focused, emotionally layered',
                'dims': {'narrative':0.85,'temporal_focus':-0.5,'metaphor':0.5,
                         'arousal':0.5,'warmth':0.5,'agency':0.55,
                         'topic_depth':0.55,'pace':0.4},
                'basin_radius': 0.55,
            },
            {
                'name': 'crisis_support',
                'description': 'High vulnerability, low energy, needs grounding',
                'dims': {'vulnerability':0.9,'valence':-0.4,'arousal':0.6,'warmth':0.95,
                         'tension':0.7,'pace':0.2,'rapport':0.7,'curiosity':0.3},
                'basin_radius': 0.65,
            },
        ]
        for seed in seeds:
            center = [0.0] * PHASE_DIM
            for dim_name, val in seed['dims'].items():
                if dim_name in DIM_NAMES:
                    center[_dim(dim_name)] = val
            self.attractors[seed['name']] = Attractor(
                name=seed['name'],
                center=center,
                basin_radius=seed['basin_radius'],
                description=seed.get('description', ''),
            )

    def fit_from_sequence(self, vectors, name, description=''):
        if len(vectors) < 2:
            raise ValueError("Need at least 2 vectors")
        center = _mean(vectors)
        distances = sorted([_dist(v, center) for v in vectors])
        p95 = distances[int(len(distances) * 0.95)]
        flow_field = {}
        for i in range(len(vectors) - 1):
            key = _discretize(vectors[i])
            vel = _sub(vectors[i+1], vectors[i])
            if key in flow_field:
                ev = flow_field[key]
                flow_field[key] = [(ev[j]+vel[j])/2 for j in range(PHASE_DIM)]
            else:
                flow_field[key] = vel
        fit_error = sum(distances) / len(distances)
        attractor = Attractor(
            name=name, center=center, basin_radius=max(0.1, p95),
            flow_field=flow_field, source_count=1,
            fit_error=round(fit_error, 4), description=description,
        )
        if name in self.attractors:
            attractor = self._merge(self.attractors[name], attractor)
        self.attractors[name] = attractor
        return attractor

    def _merge(self, existing, new):
        we, wn = existing.source_count, new.source_count
        total = we + wn
        merged_center = [(existing.center[i]*we + new.center[i]*wn)/total
                         for i in range(PHASE_DIM)]
        merged_radius = (existing.basin_radius*we + new.basin_radius*wn) / total
        merged_flow = dict(existing.flow_field)
        for key, vel in new.flow_field.items():
            if key in merged_flow:
                ev = merged_flow[key]
                merged_flow[key] = [(ev[j]+vel[j])/2 for j in range(PHASE_DIM)]
            else:
                merged_flow[key] = vel
        return Attractor(
            name=existing.name, center=merged_center, basin_radius=merged_radius,
            flow_field=merged_flow, source_count=total,
            fit_error=(existing.fit_error*we + new.fit_error*wn)/total,
            description=existing.description or new.description,
        )

    def fit_from_corpus(self, vectorized_dir, verbose=True):
        vec_path = Path(vectorized_dir)
        files = sorted(vec_path.glob('**/*_vectorized.json'))
        if not files:
            print(f"[WARNING] No vectorized files in {vectorized_dir}")
            return self.attractors
        if verbose:
            print(f"[ATTRACTOR LEARNER] {len(files)} files found")
        source_vectors: Dict[str, List] = {}
        all_vectors = []
        for f in files:
            try:
                atoms = json.load(open(f, encoding='utf-8'))
                vecs = [a['vector'] for a in atoms if 'vector' in a]
                if not vecs:
                    continue
                source = atoms[0].get('source', '') if atoms else ''
                key = self._classify_source(source)
                source_vectors.setdefault(key, []).extend(vecs)
                all_vectors.extend(vecs)
                if verbose:
                    print(f"  {f.name}: {len(vecs):,} vectors -> [{key}]")
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {f.name}: {e}")
        for key, vecs in source_vectors.items():
            if len(vecs) >= 2:
                if verbose:
                    print(f"  Fitting [{key}]: {len(vecs):,} vectors")
                self.fit_from_sequence(
                    vecs, name=key,
                    description=f'Learned from {key} corpus'
                )
        if len(all_vectors) >= 2:
            if verbose:
                print(f"  Fitting [corpus]: {len(all_vectors):,} total")
            self.fit_from_sequence(
                all_vectors, name='corpus',
                description='Global corpus attractor'
            )
        if verbose:
            print(f"\n[DONE] {len(self.attractors)} attractors total")
        return self.attractors

    def _classify_source(self, source):
        s = source.lower()
        if 'huberman' in s: return 'huberman'
        elif 'rogan' in s or 'jre' in s: return 'jre'
        elif 'therapy' in s: return 'therapy'
        elif 'duncan' in s or 'dtfh' in s: return 'dtfh'
        return 'general'

    def classify_state(self, state, top_n=3):
        scores = [
            (name, round(_dist(state, att.center), 4))
            for name, att in self.attractors.items()
        ]
        scores.sort(key=lambda x: x[1])
        return scores[:top_n]

    def save(self, path):
        out = {
            'saved_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'attractor_count': len(self.attractors),
            'attractors': {name: att.to_dict()
                           for name, att in self.attractors.items()}
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(path, 'w', encoding='utf-8'), indent=2)
        print(f"[AttractorLearner] Saved {len(self.attractors)} attractors to {path}")

    @classmethod
    def load(cls, path):
        learner = cls.__new__(cls)
        learner.attractors = {}
        data = json.load(open(path, encoding='utf-8'))
        for name, d in data['attractors'].items():
            learner.attractors[name] = Attractor.from_dict(d)
        print(f"[AttractorLearner] Loaded {len(learner.attractors)} attractors")
        return learner


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='AUREON Attractor Learner — Phase 4'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    f = sub.add_parser('fit', help='Fit attractors from vectorized corpus')
    f.add_argument('vectorized_dir')
    f.add_argument('output_path')

    c = sub.add_parser('classify', help='Classify a state vector')
    c.add_argument('attractors_path')
    c.add_argument('dims', help='dim=val pairs e.g. warmth=0.9,rapport=0.8')

    args = parser.parse_args()

    if args.command == 'fit':
        learner = AttractorLearner()
        learner.fit_from_corpus(args.vectorized_dir)
        learner.save(args.output_path)
        print("\nAttractors:")
        for name, att in learner.attractors.items():
            norm = sum(x**2 for x in att.center)**0.5
            print(f"  {name:25s} norm={norm:.3f} "
                  f"radius={att.basin_radius:.3f} "
                  f"sources={att.source_count}")

    elif args.command == 'classify':
        learner = AttractorLearner.load(args.attractors_path)
        state = [0.0] * PHASE_DIM
        for pair in args.dims.split(','):
            if '=' in pair:
                n, v = pair.strip().split('=')
                if n.strip() in DIM_NAMES:
                    state[DIM_NAMES.index(n.strip())] = float(v.strip())
        print("Closest attractors:")
        for name, dist in learner.classify_state(state):
            print(f"  {name:25s} distance={dist:.4f}")


if __name__ == '__main__':
    main()
