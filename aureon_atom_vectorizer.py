#!/usr/bin/env python3
"""
AUREON ATOM VECTORIZER
======================
Phase 2 of the Human Speech Engine pipeline.

Takes raw SpeechAtoms from aureon-corpus (text, uid, source, start, end)
and computes a 24-dimensional phase-space vector for each atom.

DESIGN PRINCIPLES:
    - Zero transformer dependencies. Rule-based + statistical only.
    - Every dimension has an explicit, auditable extraction method.
    - All output vectors are in [-1, 1] or [0, 1] per dimension spec.
    - Deterministic: same input always produces same vector.
    - Validated against dimension definitions in aureon_human_speech_engine.py

DIMENSION INDEX:
    0  valence          emotional positivity         [-1, +1]
    1  arousal          emotional intensity           [0, 1]
    2  dominance        assertion vs deference        [0, 1]
    3  topic_depth      surface -> profound           [0, 1]
    4  rapport          stranger -> deep trust        [0, 1]
    5  formality        casual -> formal              [0, 1]
    6  vulnerability    guarded -> exposed            [0, 1]
    7  playfulness      serious -> playful            [0, 1]
    8  intellectual     intuitive -> analytical       [0, 1]
    9  momentum         conversation energy           [0, 1]
    10 certainty        tentative -> definitive       [0, 1]
    11 curiosity        closed -> exploratory         [0, 1]
    12 warmth           cool -> warm                  [0, 1]
    13 surprise         expected -> unexpected        [0, 1]
    14 tension          resolved -> unresolved        [0, 1]
    15 narrative        factual -> story-like         [0, 1]
    16 metaphor         literal -> figurative         [0, 1]
    17 agency           passive -> active             [0, 1]
    18 temporal_focus   past -> future                [-1, +1]
    19 scope            personal -> universal         [0, 1]
    20 pace             deliberate -> rapid           [0, 1]
    21 reciprocity      monologue -> dialogue         [0, 1]
    22 kappa            internal coherence            [0, 1]
    23 tau              temporal coherence            [0, 1]

AUTHOR: Nadine Squires / Quantara
LICENSE: Proprietary
"""

from __future__ import annotations

import json
import math
import re
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ====================================================================
# DIMENSION CONSTANTS (mirrors aureon_human_speech_engine.py)
# ====================================================================

PHASE_DIM = 24

DIM_VALENCE         = 0
DIM_AROUSAL         = 1
DIM_DOMINANCE       = 2
DIM_TOPIC_DEPTH     = 3
DIM_RAPPORT         = 4
DIM_FORMALITY       = 5
DIM_VULNERABILITY   = 6
DIM_PLAYFULNESS     = 7
DIM_INTELLECTUAL    = 8
DIM_MOMENTUM        = 9
DIM_CERTAINTY       = 10
DIM_CURIOSITY       = 11
DIM_WARMTH          = 12
DIM_SURPRISE        = 13
DIM_TENSION         = 14
DIM_NARRATIVE       = 15
DIM_METAPHOR        = 16
DIM_AGENCY          = 17
DIM_TEMPORAL_FOCUS  = 18
DIM_SCOPE           = 19
DIM_PACE            = 20
DIM_RECIPROCITY     = 21
DIM_KAPPA           = 22
DIM_TAU             = 23

DIM_NAMES = [
    "valence", "arousal", "dominance", "topic_depth", "rapport",
    "formality", "vulnerability", "playfulness", "intellectual", "momentum",
    "certainty", "curiosity", "warmth", "surprise", "tension",
    "narrative", "metaphor", "agency", "temporal_focus", "scope",
    "pace", "reciprocity", "kappa", "tau",
]


# ====================================================================
# LEXICON TABLES
# ====================================================================

POS_VALENCE_WORDS = {
    "love", "wonderful", "amazing", "great", "good", "excellent", "fantastic",
    "beautiful", "happy", "joy", "joyful", "delightful", "awesome", "brilliant",
    "perfect", "magnificent", "enjoy", "enjoyed", "enjoying", "positive",
    "appreciate", "grateful", "glad", "pleased", "thrilled", "excited",
    "proud", "success", "winning", "benefit", "helpful", "effective",
    "better", "best", "right", "correct", "true", "real", "genuine",
    "healthy", "strong", "powerful", "clear", "simple", "easy", "natural",
    "free", "open", "connected", "together", "safe", "secure", "comfortable",
    "warm", "kind", "generous", "gentle", "soft", "sweet", "light",
    "hope", "hopeful", "optimistic", "confident", "sure", "certain",
    "worthy", "meaningful", "important", "valuable", "growth", "grow",
}

NEG_VALENCE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "hated", "hating",
    "sad", "unhappy", "depressed", "depression", "anxious", "anxiety",
    "fear", "afraid", "scared", "worried", "worry", "stress", "stressed",
    "pain", "painful", "hurt", "hurting", "loss", "lost", "broken",
    "fail", "failed", "failure", "mistake", "wrong", "error", "problem",
    "difficult", "hard", "struggle", "struggling", "suffer", "suffering",
    "dark", "darkness", "empty", "void", "alone", "lonely", "isolated",
    "angry", "anger", "rage", "frustrated", "frustration", "disappointed",
    "shame", "guilt", "regret", "sorry", "worse", "worst", "weak",
    "sick", "ill", "disease", "disorder", "damage", "dangerous", "threat",
    "crisis", "collapse", "destroy", "destroyed", "dead", "death", "dying",
}

HIGH_AROUSAL_WORDS = {
    "incredible", "insane", "unbelievable", "shocking", "explosive",
    "massive", "enormous", "huge", "extreme", "intense", "urgent",
    "critical", "emergency", "immediately", "now", "must", "crucial",
    "vital", "absolutely", "completely", "totally", "never", "always",
    "every", "all", "nothing", "everything", "everyone", "destroy",
    "explode", "crash", "break", "shatter", "scream", "yell", "fight",
    "battle", "war", "kill", "die", "dead", "alive", "fire", "burning",
    "revolutionary", "unprecedented", "breakthrough", "stunning", "dramatic",
}

LOW_AROUSAL_WORDS = {
    "perhaps", "maybe", "possibly", "slightly", "somewhat", "fairly",
    "rather", "quite", "generally", "usually", "often", "sometimes",
    "occasionally", "gradually", "slowly", "calmly", "quietly", "gently",
    "softly", "slightly", "minor", "small", "little", "few", "some",
    "steady", "stable", "routine", "normal", "typical", "average",
}

HIGH_DOMINANCE_WORDS = {
    "must", "should", "need", "require", "demand", "insist", "assert",
    "clearly", "obviously", "definitely", "certainly", "absolutely",
    "fact", "truth", "evidence", "proof", "know", "believe", "think",
    "argue", "claim", "state", "declare", "conclude", "decide", "lead",
    "control", "manage", "direct", "command", "authority", "power",
    "important", "significant", "critical", "essential", "fundamental",
}

LOW_DOMINANCE_WORDS = {
    "maybe", "perhaps", "might", "could", "wonder", "guess", "suppose",
    "hope", "wish", "try", "attempt", "seem", "appears", "kind of",
    "sort of", "not sure", "uncertain", "unclear", "confused", "lost",
    "help", "please", "sorry", "excuse", "apology", "forgive", "allow",
}

DEEP_TOPIC_WORDS = {
    "consciousness", "existence", "meaning", "purpose", "truth", "reality",
    "philosophy", "philosophical", "metaphysical", "essence", "nature",
    "fundamental", "underlying", "profound", "deep", "core", "root",
    "principle", "theory", "concept", "framework", "paradigm", "model",
    "mechanism", "system", "structure", "pattern", "dynamic", "relationship",
    "complexity", "emergence", "evolution", "transformation", "identity",
    "soul", "spirit", "mind", "psyche", "trauma", "healing", "wholeness",
    "integrate", "synthesis", "convergence", "coherence", "alignment",
}

SURFACE_TOPIC_WORDS = {
    "today", "yesterday", "tomorrow", "weather", "food", "coffee", "lunch",
    "traffic", "drive", "walk", "sleep", "wake", "morning", "evening",
    "weekend", "movie", "show", "game", "phone", "email", "text", "call",
    "buy", "shop", "store", "price", "cost", "sale", "deal",
}

HIGH_RAPPORT_WORDS = {
    "you know", "right", "i mean", "like", "honestly", "actually",
    "between us", "just between", "trust", "understand", "feel", "felt",
    "together", "shared", "our", "we", "us", "connect", "bond",
    "friend", "friendship", "close", "intimate", "personal", "private",
    "really", "truly", "genuinely", "care", "matter",
}

FORMAL_WORDS = {
    "therefore", "however", "furthermore", "moreover", "consequently",
    "subsequently", "nevertheless", "notwithstanding", "pursuant",
    "heretofore", "aforementioned", "whereby", "therein", "herein",
    "shall", "must", "regarding", "concerning", "pertaining", "respective",
    "indicate", "demonstrate", "establish", "constitute", "determine",
    "approximately", "sufficient", "adequate", "appropriate", "significant",
    "facilitate", "implement", "utilize", "obtain", "acquire", "provide",
}

INFORMAL_WORDS = {
    "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "yeah", "yep",
    "nope", "nah", "hey", "ok", "okay", "cool", "awesome", "wow",
    "stuff", "thing", "things", "like", "basically", "literally",
    "totally", "super", "really", "pretty", "quite", "way", "lot",
    "tons", "bunch", "dude", "man", "bro", "guys", "folks",
}

VULNERABILITY_WORDS = {
    "scared", "afraid", "fear", "worry", "uncertain", "lost", "confused",
    "hurt", "pain", "broken", "lonely", "alone", "helpless", "hopeless",
    "shame", "embarrassed", "guilty", "regret", "doubt", "insecure",
    "struggle", "suffering", "expose", "vulnerable", "open", "raw",
    "honest", "admit", "confess", "share",
}

PLAYFUL_WORDS = {
    "haha", "lol", "funny", "hilarious", "joke", "joking", "kidding",
    "just kidding", "playful", "fun", "silly", "ridiculous", "absurd",
    "wild", "crazy", "weird", "strange", "odd", "quirky", "ironic",
    "sarcastic", "tongue", "cheek", "wink", "tease", "banter",
    "laugh", "laughing", "smile", "grin", "play", "game",
}

INTELLECTUAL_WORDS = {
    "research", "study", "studies", "evidence", "data", "analysis",
    "hypothesis", "theory", "experiment", "results", "findings", "conclude",
    "therefore", "because", "since", "although", "however", "whereas",
    "mechanism", "function", "structure", "system", "process", "algorithm",
    "neuroscience", "biology", "physics", "chemistry", "mathematics",
    "cognitive", "neural", "cortex", "dopamine", "serotonin", "receptor",
    "statistically", "significantly", "correlation", "causation", "variable",
    "parameter", "dimension", "vector", "matrix", "gradient", "optimize",
}

CERTAINTY_WORDS = {
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "undoubtedly", "without question", "fact", "truth", "proven",
    "know", "certain", "sure", "confident", "convinced", "must",
    "will", "always", "never", "every", "all", "none",
}

UNCERTAINTY_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "seems", "appears",
    "think", "believe", "suppose", "guess", "wonder", "not sure",
    "unclear", "uncertain", "depends", "varies", "sometimes", "often",
    "generally", "typically", "usually", "tend", "likely", "unlikely",
}

CURIOSITY_WORDS = {
    "why", "how", "what", "where", "when", "who", "which",
    "wonder", "curious", "interesting", "fascinating", "intriguing",
    "explore", "discover", "learn", "understand", "investigate",
    "question", "ask", "imagine", "suppose",
}

WARMTH_WORDS = {
    "love", "care", "kind", "gentle", "sweet", "warm", "heart",
    "empathy", "compassion", "understand", "support",
    "here for you", "with you", "together", "hug", "comfort",
    "appreciate", "grateful", "thank", "thanks", "wonderful",
    "beautiful", "precious", "dear", "friend", "trust",
}

COLD_WORDS = {
    "cold", "distant", "detached", "indifferent", "formal", "objective",
    "neutral", "impersonal", "regardless", "irrelevant",
    "bluntly", "to the point", "moving on", "next",
}

SURPRISE_WORDS = {
    "wow", "whoa", "wait", "really", "seriously", "no way", "what",
    "unexpected", "surprised", "shocking", "astonishing", "remarkable",
    "extraordinary", "unbelievable", "incredible", "never expected",
    "didn't know", "just realized", "turns out", "actually", "in fact",
    "believe it or not", "here's the thing", "plot twist",
}

TENSION_WORDS = {
    "but", "however", "although", "yet", "still", "despite", "conflict",
    "tension", "contradiction", "paradox", "problem", "issue", "challenge",
    "concern", "worry", "risk", "danger", "threat", "uncertain", "unclear",
    "unresolved", "complicated", "complex", "difficult", "struggle",
    "versus", "against", "oppose", "disagree", "debate", "argue",
}

RESOLUTION_WORDS = {
    "solved", "resolved", "answer", "solution", "clear", "simple",
    "finally", "at last", "conclusion", "therefore", "so", "thus",
    "turns out", "in the end", "ultimately", "completed", "done",
}

NARRATIVE_WORDS = {
    "story", "happened", "then", "and then", "after", "before", "when",
    "once", "remember", "there was", "told", "said", "asked", "replied",
    "went", "came", "found", "saw", "heard", "felt", "thought",
    "started", "began", "ended", "finally", "eventually", "suddenly",
    "one day", "at that point", "in that moment", "years ago",
}

METAPHOR_WORDS = {
    "like", "as if", "as though", "imagine", "picture", "think of it as",
    "it's like", "kind of like", "similar to", "in a way", "metaphor",
    "analogy", "mirror", "reflection", "journey", "path", "road",
    "battle", "war", "fight", "light", "dark", "fire", "water", "river",
    "seed", "root", "grow", "bloom", "break", "build", "weave", "thread",
    "wave", "ocean", "mountain", "climb", "depth", "surface",
}

HIGH_AGENCY_WORDS = {
    "i", "i'll", "i'm", "i've", "i can", "i will", "i do", "i choose",
    "decide", "choose", "act", "create", "build", "make", "do",
    "take", "lead", "drive", "push", "move", "change", "transform",
    "control", "direct", "shape", "design", "plan", "execute",
    "responsible", "committed", "dedicated", "focused", "determined",
}

LOW_AGENCY_WORDS = {
    "can't", "cannot", "unable", "impossible", "helpless", "stuck",
    "trapped", "forced", "made to", "had to", "no choice", "no option",
    "victim", "suffer", "endure", "happen to", "done to me",
}

FUTURE_WORDS = {
    "will", "going to", "plan", "future", "tomorrow", "next", "soon",
    "eventually", "someday", "hope", "expect", "anticipate", "predict",
    "goal", "vision", "aspire", "intend", "shall", "would", "could",
}

PAST_WORDS = {
    "was", "were", "had", "did", "used to", "before", "ago", "once",
    "past", "history", "remember", "recall", "back then", "previously",
    "yesterday", "last year", "childhood", "grew up",
}

UNIVERSAL_WORDS = {
    "everyone", "all", "humanity", "human", "people", "society", "world",
    "universal", "collective", "culture", "civilization", "species",
    "existence", "nature", "cosmos", "universe", "always", "never",
    "throughout history", "across cultures", "fundamentally",
}

PERSONAL_WORDS = {
    "i", "me", "my", "mine", "myself", "personally", "for me",
    "in my case", "in my experience", "my life", "my story",
    "my family", "my relationship", "my work",
}

DIALOGUE_MARKERS = {
    "you", "your", "you know", "right", "don't you think", "what do you",
    "how about you", "have you", "do you", "would you", "could you",
    "question", "ask", "wonder", "curious about", "what's your",
    "tell me", "share", "together", "we", "us", "our",
}


# ====================================================================
# SOURCE CONTEXT BIASES
# ====================================================================

SOURCE_CONTEXT = {
    "huberman": {
        "base_intellectual": 0.7, "base_formality": 0.55,
        "base_topic_depth": 0.6, "base_dominance": 0.6, "base_pace": 0.55,
    },
    "jre": {
        "base_intellectual": 0.4, "base_formality": 0.15,
        "base_rapport": 0.55, "base_playfulness": 0.45, "base_pace": 0.6,
    },
    "therapy": {
        "base_vulnerability": 0.6, "base_warmth": 0.7,
        "base_topic_depth": 0.65, "base_formality": 0.35, "base_rapport": 0.65,
    },
    "dtfh": {
        "base_intellectual": 0.5, "base_metaphor": 0.55,
        "base_topic_depth": 0.65, "base_playfulness": 0.4, "base_formality": 0.2,
    },
    "default": {
        "base_intellectual": 0.35, "base_formality": 0.35, "base_topic_depth": 0.35,
    },
}


def _get_source_context(source: str) -> dict:
    s = source.lower()
    if "huberman" in s: return SOURCE_CONTEXT["huberman"]
    elif "jre" in s or "rogan" in s: return SOURCE_CONTEXT["jre"]
    elif "therapy" in s or "therapist" in s: return SOURCE_CONTEXT["therapy"]
    elif "dtfh" in s or "duncan" in s: return SOURCE_CONTEXT["dtfh"]
    else: return SOURCE_CONTEXT["default"]


# ====================================================================
# LINGUISTIC FEATURE EXTRACTORS
# ====================================================================

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()

def _bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

def _word_overlap_score(tokens, bigrams, pos_set, neg_set=None):
    all_tokens = set(tokens) | set(bigrams)
    pos_hits = len(all_tokens & pos_set)
    neg_hits = len(all_tokens & neg_set) if neg_set else 0
    total = len(tokens) + 1
    if neg_set is not None:
        return max(-1.0, min(1.0, (pos_hits - neg_hits) / (total ** 0.5)))
    else:
        return min(1.0, pos_hits / (total ** 0.4))

def _sentence_count(text): return max(1, len(re.findall(r'[.!?]+', text)))
def _avg_word_length(tokens): return sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
def _question_count(text): return len(re.findall(r'\?', text))
def _exclamation_count(text): return len(re.findall(r'!', text))
def _hedge_ratio(tokens):
    hits = sum(1 for t in tokens if t in UNCERTAINTY_WORDS)
    return min(1.0, hits / (len(tokens) + 1) * 8)
def _intensity_markers(text):
    return len(re.findall(r'\b[A-Z]{2,}\b', text)) + len(re.findall(r'[!?]{2,}', text))
def _duration_to_pace(start, end, token_count):
    duration = max(0.5, end - start)
    return min(1.0, (token_count / duration) / 4.0)
def _pronoun_profile(tokens):
    i_words = {"i", "me", "my", "mine", "myself", "i'll", "i've", "i'm", "i'd"}
    you_words = {"you", "your", "yours", "yourself", "you'll", "you've", "you're"}
    we_words = {"we", "our", "us", "ourselves", "we'll", "we've", "we're"}
    n = len(tokens) + 1
    return (
        min(1.0, sum(1 for t in tokens if t in i_words) / n * 5),
        min(1.0, sum(1 for t in tokens if t in you_words) / n * 5),
        min(1.0, sum(1 for t in tokens if t in we_words) / n * 5),
    )
def _syllable_estimate(word):
    word = word.lower().rstrip('e')
    return max(1, len(re.findall(r'[aeiou]+', word)))
def _avg_syllables(tokens):
    return sum(_syllable_estimate(t) for t in tokens) / len(tokens) if tokens else 1.0


# ====================================================================
# ATOM VECTORIZER
# ====================================================================

class AtomVectorizer:
    """
    Maps raw SpeechAtom text to a 24-D phase-space vector.
    Rule-based + statistical. No ML models. Fully deterministic.
    """

    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    def vectorize(self, text: str, source: str = "", start: float = 0.0, end: float = 5.0) -> List[float]:
        cache_key = f"{source}:{start}:{text[:50]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        tokens = _tokenize(text)
        bigrams = _bigrams(tokens)
        ctx = _get_source_context(source)
        i_ratio, you_ratio, we_ratio = _pronoun_profile(tokens)
        v = [0.0] * PHASE_DIM

        # 0 VALENCE
        v[0] = max(-1.0, min(1.0, _word_overlap_score(tokens, bigrams, POS_VALENCE_WORDS, NEG_VALENCE_WORDS)))
        # 1 AROUSAL
        arousal_lex = _word_overlap_score(tokens, bigrams, HIGH_AROUSAL_WORDS, LOW_AROUSAL_WORDS)
        v[1] = max(0.0, min(1.0, arousal_lex*0.6 + min(1.0,_intensity_markers(text)*0.3)*0.25 + min(1.0,_exclamation_count(text)*0.2)*0.15 + 0.2))
        # 2 DOMINANCE
        dom_lex = _word_overlap_score(tokens, bigrams, HIGH_DOMINANCE_WORDS, LOW_DOMINANCE_WORDS)
        v[2] = max(0.0, min(1.0, dom_lex*0.5 + i_ratio*0.3 + ctx.get("base_dominance",0.4)*0.2))
        # 3 TOPIC DEPTH
        depth_lex = _word_overlap_score(tokens, bigrams, DEEP_TOPIC_WORDS, SURFACE_TOPIC_WORDS)
        depth_syl = min(1.0, max(0.0, (_avg_syllables(tokens)-1.0)/2.0))
        depth_wl = min(1.0, max(0.0, (_avg_word_length(tokens)-3.0)/4.0))
        v[3] = max(0.0, min(1.0, depth_lex*0.4 + depth_syl*0.25 + depth_wl*0.15 + ctx.get("base_topic_depth",0.35)*0.2))
        # 4 RAPPORT
        rapport_lex = _word_overlap_score(tokens, bigrams, HIGH_RAPPORT_WORDS)
        v[4] = max(0.0, min(1.0, rapport_lex*0.3 + you_ratio*0.4 + we_ratio*0.3 + ctx.get("base_rapport",0.3)*0.2))
        # 5 FORMALITY
        formal_lex = _word_overlap_score(tokens, bigrams, FORMAL_WORDS, INFORMAL_WORDS)
        v[5] = max(0.0, min(1.0, formal_lex*0.45 + min(1.0,max(0.0,(_avg_syllables(tokens)-1.2)/1.5))*0.3 + ctx.get("base_formality",0.35)*0.25))
        # 6 VULNERABILITY
        vuln_lex = _word_overlap_score(tokens, bigrams, VULNERABILITY_WORDS)
        v[6] = max(0.0, min(1.0, vuln_lex*0.55 + max(0.0,-v[0])*0.3 + ctx.get("base_vulnerability",0.15)*0.25))
        # 7 PLAYFULNESS
        play_lex = _word_overlap_score(tokens, bigrams, PLAYFUL_WORDS)
        v[7] = max(0.0, min(1.0, play_lex*0.55 + max(0.0,0.5-v[5])*0.4 + ctx.get("base_playfulness",0.2)*0.2))
        # 8 INTELLECTUAL
        intel_lex = _word_overlap_score(tokens, bigrams, INTELLECTUAL_WORDS)
        v[8] = max(0.0, min(1.0, intel_lex*0.45 + v[3]*0.3 + ctx.get("base_intellectual",0.35)*0.25))
        # 9 MOMENTUM
        pace_raw = _duration_to_pace(start, end, len(tokens))
        v[9] = max(0.0, min(1.0, v[1]*0.4 + pace_raw*0.3 + min(1.0,_intensity_markers(text)*0.15)*0.3))
        # 10 CERTAINTY
        cert_lex = _word_overlap_score(tokens, bigrams, CERTAINTY_WORDS, UNCERTAINTY_WORDS)
        v[10] = max(0.0, min(1.0, cert_lex*0.5 + (1.0-_hedge_ratio(tokens))*0.3 + 0.2))
        # 11 CURIOSITY
        curio_lex = _word_overlap_score(tokens, bigrams, CURIOSITY_WORDS)
        v[11] = max(0.0, min(1.0, curio_lex*0.55 + min(1.0,_question_count(text)*0.35)*0.45))
        # 12 WARMTH
        warmth_lex = _word_overlap_score(tokens, bigrams, WARMTH_WORDS, COLD_WORDS)
        v[12] = max(0.0, min(1.0, warmth_lex*0.5 + max(0.0,v[0])*0.3 + ctx.get("base_warmth",0.3)*0.2))
        # 13 SURPRISE
        surp_lex = _word_overlap_score(tokens, bigrams, SURPRISE_WORDS)
        v[13] = max(0.0, min(1.0, surp_lex*0.7 + 0.1))
        # 14 TENSION
        tens_lex = _word_overlap_score(tokens, bigrams, TENSION_WORDS, RESOLUTION_WORDS)
        v[14] = max(0.0, min(1.0, tens_lex*0.55 + max(0.0,-v[0])*0.2 + 0.15))
        # 15 NARRATIVE
        narr_lex = _word_overlap_score(tokens, bigrams, NARRATIVE_WORDS)
        past_ct = len(re.findall(r'\b(was|were|had|said|went|came|told|asked|found|saw|heard|felt|thought|started|began)\b', text.lower()))
        v[15] = max(0.0, min(1.0, narr_lex*0.5 + min(1.0,past_ct*0.15)*0.5))
        # 16 METAPHOR
        meta_lex = _word_overlap_score(tokens, bigrams, METAPHOR_WORDS)
        simile_ct = len(re.findall(r'\b(like|as if|as though|just like|kind of like)\b', text.lower()))
        v[16] = max(0.0, min(1.0, meta_lex*0.45 + min(1.0,simile_ct*0.2)*0.3 + ctx.get("base_metaphor",0.25)*0.25))
        # 17 AGENCY
        agency_lex = _word_overlap_score(tokens, bigrams, HIGH_AGENCY_WORDS, LOW_AGENCY_WORDS)
        v[17] = max(0.0, min(1.0, agency_lex*0.4 + i_ratio*0.4 + 0.2))
        # 18 TEMPORAL FOCUS (signed)
        future_hits = sum(1 for t in tokens if t in FUTURE_WORDS)
        past_hits = sum(1 for t in tokens if t in PAST_WORDS)
        v[18] = max(-1.0, min(1.0, (future_hits - past_hits) / (len(tokens)+1)**0.5))
        # 19 SCOPE
        univ_hits = sum(1 for t in tokens if t in UNIVERSAL_WORDS)
        pers_hits = sum(1 for t in tokens if t in PERSONAL_WORDS)
        v[19] = max(0.0, min(1.0, 0.5 + (univ_hits - pers_hits*0.5) / (len(tokens)+1)**0.4 * 0.3))
        # 20 PACE
        v[20] = max(0.0, min(1.0, pace_raw*0.7 + ctx.get("base_pace",0.4)*0.3))
        # 21 RECIPROCITY
        recip_lex = _word_overlap_score(tokens, bigrams, DIALOGUE_MARKERS)
        v[21] = max(0.0, min(1.0, recip_lex*0.3 + min(1.0,_question_count(text)*0.25) + you_ratio*0.4 + 0.1))
        # 22 KAPPA (internal coherence)
        sc = _sentence_count(text)
        tps = len(tokens) / sc
        v[22] = max(0.0, min(1.0, (1.0-abs(tps-15)/20.0)*0.4 + (1.0-abs(v[14])*0.3)*0.3 + 0.3))
        # 23 TAU (temporal coherence / forward-serving)
        v[23] = max(0.0, min(1.0, max(0.0,v[18])*0.3 + v[9]*0.3 + v[10]*0.2 + (1.0-v[14])*0.2))

        self._cache[cache_key] = v
        return v

    def vectorize_atom(self, atom: dict) -> dict:
        vec = self.vectorize(
            text=atom.get("text",""),
            source=atom.get("source",""),
            start=atom.get("start",0.0),
            end=atom.get("end",5.0),
        )
        return {**atom, "vector": vec, "dim_names": DIM_NAMES}


# ====================================================================
# CORPUS PROCESSOR
# ====================================================================

def process_corpus(corpus_dir: str, output_dir: str, verbose: bool = True, overwrite: bool = False) -> dict:
    """
    Batch-vectorize all *_atoms.json files in corpus_dir.
    Writes *_vectorized.json to output_dir.
    """
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vectorizer = AtomVectorizer()
    stats = {"files_processed":0, "files_skipped":0, "atoms_vectorized":0, "errors":[], "elapsed_seconds":0.0}
    atom_files = sorted(corpus_path.glob("**/*_atoms.json"))
    if not atom_files:
        print(f"[WARNING] No *_atoms.json files found in {corpus_dir}")
        return stats
    if verbose:
        print(f"[AUREON VECTORIZER] {len(atom_files)} files found.")
        print(f"  Input:  {corpus_dir}")
        print(f"  Output: {output_dir}\n")
    t_start = time.time()
    for i, atom_file in enumerate(atom_files):
        out_file = output_path / (atom_file.stem + "_vectorized.json")
        if out_file.exists() and not overwrite:
            if verbose: print(f"  [{i+1}/{len(atom_files)}] SKIP {atom_file.name}")
            stats["files_skipped"] += 1
            continue
        try:
            with open(atom_file, "r", encoding="utf-8") as f:
                atoms = json.load(f)
            vectorized = [vectorizer.vectorize_atom(a) for a in atoms]
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(vectorized, f, separators=(',',':'))
            stats["files_processed"] += 1
            stats["atoms_vectorized"] += len(vectorized)
            if verbose:
                rate = stats["atoms_vectorized"] / max(time.time()-t_start, 0.001)
                print(f"  [{i+1}/{len(atom_files)}] {atom_file.name} — {len(vectorized):,} atoms | {rate:,.0f} atoms/sec")
        except Exception as e:
            err = f"{atom_file.name}: {e}"
            stats["errors"].append(err)
            if verbose: print(f"  [{i+1}/{len(atom_files)}] ERROR {err}")
    stats["elapsed_seconds"] = round(time.time()-t_start, 2)
    if verbose:
        print(f"\n[DONE] {stats['files_processed']} files | {stats['atoms_vectorized']:,} atoms | {stats['elapsed_seconds']}s")
        for e in stats["errors"]: print(f"  ERROR: {e}")
    return stats


# ====================================================================
# INSPECTION UTILITIES
# ====================================================================

def describe_vector(vector: List[float]) -> str:
    scored = sorted(enumerate(vector), key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for idx, val in scored[:5]:
        bar = '█' * int(abs(val)*20) + '░' * (20-int(abs(val)*20))
        lines.append(f"  {DIM_NAMES[idx]:20s} [{bar}] {'+' if val>=0 else '-'}{abs(val):.3f}")
    return "\n".join(lines)


def compare_atoms(atom_a: dict, atom_b: dict) -> float:
    va, vb = atom_a.get("vector",[]), atom_b.get("vector",[])
    if not va or not vb: return 0.0
    mag_a = math.sqrt(sum(x*x for x in va))
    mag_b = math.sqrt(sum(x*x for x in vb))
    if mag_a<1e-10 or mag_b<1e-10: return 0.0
    return sum(va[i]*vb[i] for i in range(len(va))) / (mag_a*mag_b)


# ====================================================================
# CLI
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AUREON Atom Vectorizer — 24-D phase vectors for speech atoms")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="Vectorize corpus")
    p.add_argument("corpus_dir")
    p.add_argument("output_dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")

    t = sub.add_parser("test", help="Test on a single sentence")
    t.add_argument("text")
    t.add_argument("--source", default="")

    ins = sub.add_parser("inspect", help="Inspect vectorized file")
    ins.add_argument("file")
    ins.add_argument("--n", type=int, default=5)

    args = parser.parse_args()

    if args.command == "process":
        process_corpus(args.corpus_dir, args.output_dir, verbose=not args.quiet, overwrite=args.overwrite)

    elif args.command == "test":
        vec = AtomVectorizer().vectorize(args.text, source=args.source)
        print(f"Input: {args.text!r}")
        print(f"Source: {args.source or 'default'}")
        for i, (name, val) in enumerate(zip(DIM_NAMES, vec)):
            bar = '█'*int(abs(val)*20) + '░'*(20-int(abs(val)*20))
            print(f"  {i:2d}  {name:20s} [{bar}] {'+' if val>=0 else '-'}{abs(val):.4f}")

    elif args.command == "inspect":
        with open(args.file, "r", encoding="utf-8") as f:
            atoms = json.load(f)
        print(f"File: {args.file} | {len(atoms):,} atoms\n")
        for atom in atoms[:args.n]:
            print(f"TEXT: {atom['text'][:80]}")
            print(describe_vector(atom['vector']))
            print()


if __name__ == "__main__":
    main()
