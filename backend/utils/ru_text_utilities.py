from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab

class RuTextUtilities:
    """
    Класс для предобработки русского текста: сегментация, морфологическая разметка, лемматизация.
    """
    def __init__(self):
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.morph_vocab = MorphVocab()

    def preprocess(self, text: str) -> str:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        return " ".join(token.lemma for token in doc.tokens)
