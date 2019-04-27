import collections, nltk, math, sys
from nltk import ngrams
from nltk.corpus import webtext
import string

corpus = '''Monty Python (sometimes known as The Pythons) were a British surreal comedy group who created the sketch comedy show Monty Python's Flying Circus,
that first aired on the BBC on October 5, 1969. Forty-five episodes were made over four series. The Python phenomenon developed from the television series
into something larger in scope and impact, spawning touring stage shows, films, numerous albums, several books, and a stage musical.
The group's influence on comedy has been compared to The Beatles' influence on music. Now you say you have already constructed the unigram model, meaning, for each word you have the relevant probability. 
Then you only need to apply the formula. I assume you have a big dictionary unigram[word] that would provide the probability of each word in the corpus. 
You also need to have a test set. If your unigram model is not in the form of a dictionary, tell me what data structure you have used, so I could adapt it to my solution accordingly.'''

'''
# unigram and bigram functions

def unigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for t in tokens:
        model[(t),] += 1
    weight = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/weight
    return model

def bigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for i in range(len(tokens))[:-1]:
        model[(tokens[i], tokens[i+1])] +=1
    weight = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/weight
    return model
'''

def get_corpus():
    winetext = webtext.raw('wine.txt')
    punct = string.punctuation +'\n\r'
    winetext = winetext.translate(str.maketrans('','',punct))
    return winetext

def entropy(sent, model):
    entropy = 0
    words = sent.split()
    for word in set(words):
        px = words.count(word)/len(words)
        entropy += -px * math.log(model[word],2)
    return entropy

def ngramize(tokens, n_gram, smoothing):
    model = collections.defaultdict(lambda: smoothing)
    ngms = ngrams(tokens, n_gram)
    for gms in ngms:
        model[gms] += 1
    weight = float(sum(model.values()))
    for gms in model:
        model[gms] = model[gms]/weight
    return model

def perplexity(testset, model, n_gram):
    testset = ngrams(testset.split(), n_gram)
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
        print("current perplexity . . . . .", pow(perplexity, 1/float(N)) )
        print("Accumulating perplexity . . .", perplexity, "word", model[word])
    if N!=0:
        return pow(perplexity, 1/float(N)) 
    else:
        print("entering in....")
        return perplexity*100

if __name__ == '__main__':
    if len(sys.argv)>0:
        n = int(sys.argv[1])
        test = ' '.join(sys.argv[2:])
    tokens = nltk.word_tokenize(get_corpus())
    model = ngramize(tokens,n,1/len(tokens))
    print("Perplexity..", perplexity(test, model,n))
