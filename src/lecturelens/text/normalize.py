"""Text normalisation for ASR transcripts and slide OCR, plus stemming vs. lemmatisation."""
from __future__ import annotations

import re
import unicodedata

FILLERS = {"um", "uh", "uhm", "erm", "er", "ah", "hmm", "mm", "okay so", "you know", "i mean", "like"}
SINGLE_FILLERS = {"um", "uh", "uhm", "erm", "er", "ah", "hmm", "mm"}

CONTRACTIONS = {
    "n't": " not", "'re": " are", "'s": " 's", "'ll": " will", "'ve": " have", "'m": " am", "'d": " would",
}

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

STOPWORDS = set("""a an the and or but if then else of at by for with about against between into through
during before after above below to from up down in out on off over under again further once here there
when where why how all any both each few more most other some such no nor not only own same so than too
very s t can will just don should now i me my myself we our ours ourselves you your yours yourself he him
his himself she her hers herself it its itself they them their theirs themselves what which who whom this
that these those am is are was were be been being have has had having do does did doing would could ought
let us 's going get got gonna wanna yeah okay right well also one see say said thing things actually
really basically kind sort lot""".split())


def number_to_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    if n < 1_000_000:
        rest = n % 1000
        return number_to_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + number_to_words(rest))
    return str(n)


def normalize(text: str, expand_numbers: bool = False, drop_fillers: bool = True) -> str:
    """Unicode NFKC, lowercase, expand contractions, strip punctuation, optional number expansion,
    collapse whitespace, remove ASR disfluencies."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("’", "'").replace("‘", "'")
    for k, v in CONTRACTIONS.items():
        t = t.replace(k, v)
    if expand_numbers:
        t = re.sub(r"\b\d{1,6}\b", lambda m: number_to_words(int(m.group())), t)
    t = re.sub(r"[^a-z0-9\s'-]", " ", t)
    t = re.sub(r"(?<!\w)['-]|['-](?!\w)", " ", t)
    toks = t.split()
    if drop_fillers:
        toks = [w for w in toks if w not in SINGLE_FILLERS]
        # remove immediate repetitions ("the the", "we we") — a typical spoken disfluency
        toks = [w for i, w in enumerate(toks) if i == 0 or w != toks[i - 1]]
    return " ".join(toks)


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def content_tokens(text: str, lemmatize: bool = True) -> list[str]:
    toks = [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 1 and not w.isdigit()]
    return [lemma(w) for w in toks] if lemmatize else toks


# ---- morphology ---------------------------------------------------------------------------------

_lemmatizer = None


def lemma(word: str) -> str:
    """WordNet lemmatiser (verb reading if it changes the word, else noun)."""
    global _lemmatizer
    if _lemmatizer is None:
        import nltk
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
        try:
            _lemmatizer.lemmatize("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
    # Verb reading first when it changes the word ("was" -> "be", "computing" -> "compute"); a noun-first
    # lemmatiser turns "was" into the noun "wa". Otherwise fall back to the noun lemma ("spectra" -> "spectrum").
    v = _lemmatizer.lemmatize(word, "v")
    return v if v != word else _lemmatizer.lemmatize(word, "n")


def porter_stem(word: str) -> str:
    from nltk.stem import PorterStemmer
    return PorterStemmer().stem(word)
