# preprocess.py
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class Preprocess:

    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.remover = StopWordRemoverFactory().create_stop_word_remover()

    def preprocess(self, text):
        # # 1 stemming
        text_stem = self.stemmer.stem(text)
        #
        # # 2 hapus stop words
        text_clean = self.remover.remove(text_stem)
        #
        # # 3 tokenization
        # # 3.1 lowercase
        lowercase = text_clean.lower()
        preprocessed_text = lowercase.translate(None, string.punctuation).split()

        return preprocessed_text


if __name__ == '__main__':
    preprocess = Preprocess()

    # sentence = "Film ini menarik untuk ditonton!"
    # output = preprocess.preprocess(sentence)
    # print output
    #
    import pandas as pd
    #
    data = pd.read_csv('Coba.csv', sep=',')

    preprocessed_documents = []
    titles = data.TOPIK.tolist()


    for title in titles:
        preprocessed = preprocess.preprocess(title)
        preprocessed_documents.append(' '.join(preprocessed))
    # #
    # # # tambahkan kolom baru di data dengan name preprocessed
    # # print cosine_preprocessed_documents
    data['preprocessed'] = preprocessed_documents
    #
    # # # save data dengan nama berbeda
    data.to_csv('Coba.csv')
