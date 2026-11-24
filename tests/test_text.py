import numpy as np
import pytest

from lecturelens.text.bpe import BPE
from lecturelens.text.lm import BigramLM
from lecturelens.text.normalize import content_tokens, normalize, number_to_words
from lecturelens.text.retrieval import BM25, TfidfIndex
from lecturelens.text.stats import heaps, zipf


def test_normalize_removes_disfluencies_and_contractions():
    assert normalize("Um, so we we DON'T use the... uh, Hamming window!") == \
        "so we do not use the hamming window"
    assert number_to_words(342) == "three hundred forty two"
    assert normalize("frame 25 ms", expand_numbers=True) == "frame twenty five ms"


def test_lemmatization():
    assert content_tokens("The windows were computing spectra") == ["window", "compute", "spectrum"]


def test_bpe_learns_frequent_subwords():
    words = ["autocorrelation"] * 5 + ["correlation"] * 5 + ["auto"] * 5 + ["relation"] * 3
    bpe = BPE.train(words, num_merges=40)
    assert bpe.encode_word("correlation") == ["correlation"]
    pieces = bpe.encode_word("autocorrelations")
    assert "".join(pieces) == "autocorrelations" and len(pieces) <= 4


def test_bm25_formula_and_agreement_with_rank_bm25():
    rank_bm25 = pytest.importorskip("rank_bm25")
    docs = [d.split() for d in [
        "hamming window reduces spectral leakage",
        "the cepstrum separates source and filter",
        "window length trades time and frequency resolution window",
        "bm25 is a ranking function for retrieval",
    ]]
    bm = BM25(docs, k1=1.5, b=0.75)
    # hand-computed score of doc 0 for "leakage": df=1, N=4, |d|=5, avgdl=26/4=6.5
    idf = np.log((4 - 1 + 0.5) / (1 + 0.5) + 1)
    expected = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 5 / 6.5))
    assert bm.scores(["leakage"])[0] == pytest.approx(expected)
    # with rare query terms (positive idf in both variants) rankings must agree
    q = "leakage cepstrum retrieval".split()
    ref = rank_bm25.BM25Okapi(docs, k1=1.5, b=0.75).get_scores(q)
    assert list(np.argsort(-bm.scores(q))) == list(np.argsort(-ref))


def test_tfidf_cosine_is_one_for_identical_doc():
    docs = [["a", "b", "b"], ["c", "d"]]
    idx = TfidfIndex(docs)
    s = idx.search(["a", "b", "b"])
    assert s[0] == pytest.approx(1.0) and s[1] == pytest.approx(0.0)


@pytest.mark.parametrize("sm", ["laplace", "addk", "kn"])
def test_smoothed_lm_is_a_distribution_and_beats_unseen(sm):
    train = [s.split() for s in ["the window is hamming", "the window is hann", "the signal is speech"]] * 3
    lm = BigramLM(train, smoothing=sm)
    vocab = sorted(lm.vocab - {"<s>"})
    total = sum(lm.prob(w, "window") for w in vocab)
    assert total == pytest.approx(1.0, abs=0.02)
    assert np.isfinite(lm.perplexity([["the", "window", "is", "unseenword"]]))
    assert lm.perplexity([["the", "window", "is", "hann"]]) < lm.perplexity([["hann", "is", "window", "the"]])


def test_zipf_and_heaps_on_synthetic_zipfian_text():
    rng = np.random.default_rng(0)
    toks = [f"w{r}" for r in rng.zipf(2.0, 20000)]
    _, _, s, _ = zipf(toks)
    assert 0.8 < s < 2.5
    _, _, K, beta = heaps(toks)
    assert 0 < beta < 1
