import string

puncts = string.punctuation.replace('.','')
punct = str.maketrans('','', puncts)

def get_clean_data():
    data = open('iotr.txt','r').read()
    unpunctuated = data.translate(punct)
    return ' '.join(x for x in unpunctuated.split())
