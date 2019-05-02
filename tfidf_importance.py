'''
This file is used to evaluate how good the context-based text generation model works. 
Meaning, the closeness of the given context with the generated sentence.
working: extract the noun chunks from the sentence, compute tf-idf values. based on the cluster the similarity of all the words to the context word weighted by the tf-idf values.
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from data_clean import get_clean_data
import spacy, nltk

corpus = get_clean_data()
sent_tokens = nltk.sent_tokenize(corpus)
Vectorizer = TfidfVectorizer()
tfidf_matrix = Vectorizer.fit_transform(sent_tokens)
vocab = Vectorizer.get_feature_names()
nlp_pipeline = spacy.load('en')

# get_nouns() selects only the nouns from the sentences, so that only relevant information are selected i.e, nouns
def get_nouns(sent):
    tokenized = nlp_pipeline(sent)
    noun_tokens = []
    for token in tokenized:
        if token.pos_ in['NOUN', 'PROPN']:
            noun_tokens.append(str(token))
    return ' '.join(noun_tokens)

# applies tf-idf transformation on the obtained noun chunks, thus returning tf-idf values of those words
def get_tfidf(sent):
    vals = Vectorizer.transform(sent)
    tfidf_values = {}
    for idx,data in zip(vals.indices, vals.data):
        tfidf_values[vocab[idx]] = data
    return tfidf_values

with open('testdata') as f:
    testdata = f.read()

test_sents = nltk.sent_tokenize(testdata)
for sent in test_sents:
    nouns = get_nouns(sent)
    print("nouns", nouns)
    tfidf = get_tfidf([nouns])
    print("Tfidf. . . ", tfidf)

