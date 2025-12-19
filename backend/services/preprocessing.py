from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

def preprocess_text_ru(text: str) -> str:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    return " ".join(token.lemma for token in doc.tokens)
