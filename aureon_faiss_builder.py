#!/usr/bin/env python3
"""
AUREON FAISS INDEX BUILDER — Phase 3
Author: Nadine Squires / Quantara
"""
from __future__ import annotations
import json, math, struct, time
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

class FlatIndexPython:
    def __init__(self, dim=PHASE_DIM):
        self.dim = dim
        self.vectors: List[List[float]] = []

    def add(self, vectors):
        self.vectors.extend(vectors)

    def search(self, query, k=10):
        dists = sorted([(sum((query[j]-v[j])**2 for j in range(self.dim)), i)
                        for i, v in enumerate(self.vectors)])
        top = dists[:k]
        return [d for d,_ in top], [i for _,i in top]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(b'AUREON_IDX_V1')
            f.write(struct.pack('!III', 1, self.dim, len(self.vectors)))
            for v in self.vectors:
                f.write(struct.pack(f'!{self.dim}f', *v))
        print(f"[FlatIndex] Saved {len(self.vectors):,} vectors to {path}")

    @classmethod
    def load(cls, path):
        idx = cls()
        with open(path, 'rb') as f:
            magic = f.read(13)
            assert magic == b'AUREON_IDX_V1'
            _, dim, count = struct.unpack('!III', f.read(12))
            idx.dim = dim
            for _ in range(count):
                idx.vectors.append(list(struct.unpack(f'!{dim}f', f.read(dim*4))))
        print(f"[FlatIndex] Loaded {len(idx.vectors):,} vectors")
        return idx

    @property
    def ntotal(self): return len(self.vectors)


def _try_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


class AureonIndexBuilder:
    def __init__(self, index_type='flat', nlist=100):
        self.index_type = index_type
        self.nlist = nlist
        self.faiss = _try_faiss()
        self._index = None
        self._meta: List[Dict] = []

    def _make_index(self):
        if self.faiss:
            if self.index_type == 'ivf':
                q = self.faiss.IndexFlatL2(PHASE_DIM)
                idx = self.faiss.IndexIVFFlat(q, PHASE_DIM, self.nlist)
                print(f"[Builder] FAISS IVFFlat nlist={self.nlist}")
            else:
                idx = self.faiss.IndexFlatL2(PHASE_DIM)
                print("[Builder] FAISS FlatL2")
            return idx
        print("[Builder] pure-Python FlatIndex (faiss not found)")
        return FlatIndexPython(PHASE_DIM)

    def build(self, vectorized_dir, output_dir, verbose=True):
        vec_path = Path(vectorized_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        files = sorted(vec_path.glob('**/*_vectorized.json'))
        if not files:
            print(f"[ERROR] No *_vectorized.json in {vectorized_dir}")
            return {}
        if verbose:
            print(f"[AUREON INDEX BUILDER] {len(files)} files")
        self._index = self._make_index()
        t = time.time()
        total = 0
        errors = []
        for i, f in enumerate(files):
            try:
                atoms = json.load(open(f, encoding='utf-8'))
                vecs = [a['vector'] for a in atoms if 'vector' in a]
                meta = [{'uid':a.get('uid',''),'text':a.get('text','')[:200],
                         'source':a.get('source',''),'start':a.get('start',0.0),
                         'end':a.get('end',0.0),'global_idx':total+j}
                        for j,a in enumerate(atoms) if 'vector' in a]
                if self.faiss:
                    import numpy as np
                    arr = np.array(vecs, dtype='float32')
                    if self.index_type == 'ivf':
                        if not self._index.is_trained and len(arr) >= self.nlist:
                            self._index.train(arr)
                        if self._index.is_trained:
                            self._index.add(arr)
                    else:
                        self._index.add(arr)
                else:
                    self._index.add(vecs)
                self._meta.extend(meta)
                total += len(vecs)
                if verbose:
                    rate = total / max(time.time()-t, 0.001)
                    print(f"  [{i+1}/{len(files)}] {f.name} — {len(vecs):,} | {total:,} total | {rate:,.0f}/sec")
            except Exception as e:
                errors.append(f"{f.name}: {e}")
                if verbose: print(f"  [ERROR] {e}")
        elapsed = time.time() - t
        index_path = out_path / 'aureon_atoms.index'
        if self.faiss:
            self.faiss.write_index(self._index, str(index_path))
        else:
            self._index.save(str(index_path))
        meta_path = out_path / 'aureon_atoms.meta.json'
        json.dump(self._meta, open(meta_path,'w',encoding='utf-8'), separators=(',',':'))
        manifest = {'built_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    'total_atoms':total,'files_processed':len(files)-len(errors),
                    'phase_dim':PHASE_DIM,'index_type':('faiss_'+self.index_type) if self.faiss else 'python_flat',
                    'elapsed_seconds':round(elapsed,2),'errors':errors,'dim_names':DIM_NAMES}
        json.dump(manifest, open(out_path/'aureon_index.manifest','w',encoding='utf-8'), indent=2)
        if verbose: print(f"\n[DONE] {total:,} atoms in {elapsed:.1f}s")
        return manifest


class AureonIndex:
    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)
        self.faiss = _try_faiss()
        self._index = None
        self._meta: List[Dict] = []
        self._loaded = False

    def load(self):
        ip = self.index_dir / 'aureon_atoms.index'
        mp = self.index_dir / 'aureon_atoms.meta.json'
        if not ip.exists(): raise FileNotFoundError(f"Index not found: {ip}")
        if self.faiss:
            self._index = self.faiss.read_index(str(ip))
            print(f"[AureonIndex] FAISS: {self._index.ntotal:,} vectors")
        else:
            self._index = FlatIndexPython.load(str(ip))
        self._meta = json.load(open(mp, encoding='utf-8'))
        self._loaded = True
        return self

    def search(self, query_vector, k=20, filter_source=None):
        if not self._loaded: self.load()
        fetch_k = k*3 if filter_source else k
        if self.faiss:
            import numpy as np
            q = np.array([query_vector], dtype='float32')
            distances, indices = self._index.search(q, fetch_k)
            distances, indices = distances[0].tolist(), indices[0].tolist()
        else:
            distances, indices = self._index.search(query_vector, fetch_k)
        results = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self._meta): continue
            meta = self._meta[idx]
            if filter_source and filter_source.lower() not in meta.get('source','').lower(): continue
            results.append({'uid':meta['uid'],'text':meta['text'],'source':meta['source'],
                            'start':meta.get('start',0),'distance':round(dist,6),'global_idx':idx})
            if len(results) >= k: break
        return results

    def search_by_dims(self, dim_targets, k=20):
        query = [0.0]*PHASE_DIM
        for name, val in dim_targets.items():
            if name in DIM_NAMES: query[DIM_NAMES.index(name)] = val
        return self.search(query, k=k)

    @property
    def total(self): return self._index.ntotal if self._loaded else 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AUREON FAISS Index Builder')
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('vectorized_dir')
    b.add_argument('output_dir')
    b.add_argument('--type', choices=['flat','ivf'], default='flat')
    b.add_argument('--nlist', type=int, default=100)
    s = sub.add_parser('search')
    s.add_argument('index_dir')
    s.add_argument('dims')
    s.add_argument('--k', type=int, default=10)
    s.add_argument('--source', default=None)
    st = sub.add_parser('stats')
    st.add_argument('index_dir')
    args = parser.parse_args()
    if args.command == 'build':
        AureonIndexBuilder(index_type=args.type, nlist=args.nlist).build(args.vectorized_dir, args.output_dir)
    elif args.command == 'search':
        idx = AureonIndex(args.index_dir).load()
        dims = {}
        for p in args.dims.split(','):
            if '=' in p:
                n,v = p.strip().split('=')
                dims[n.strip()] = float(v.strip())
        for i,r in enumerate(idx.search_by_dims(dims, k=args.k)):
            print(f"{i+1}. [{r['distance']:.4f}] {r['text'][:100]}")
            print(f"   source: {r['source']}")
    elif args.command == 'stats':
        mp = Path(args.index_dir)/'aureon_index.manifest'
        if mp.exists():
            m = json.load(open(mp))
            for k,v in m.items():
                if k != 'dim_names': print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
