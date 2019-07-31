import markovify
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Marxify():
  with open('corpus/marx_quotes_wikiquotes.txt', errors='ignore') as fin:
    _marxModel = markovify.Text(fin.read())
  with open('corpus/engels_quotes.txt', errors='ignore') as fin:
    _engelsModel = markovify.Text(fin.read())
  with open('corpus/che_quotes.txt', errors='ignore') as fin:
    _cheModel = markovify.Text(fin.read())
  with open('corpus/communist_manifesto_gutenberg.txt', errors='ignore') as fin:
    _communistModel = markovify.Text(fin.read())
  tfidfModel = TfidfVectorizer(max_df=0.9, stop_words='english')
  with open('corpus/brown.txt') as f:
    text = f.read()
    docs = text.split('#')
    tfidfModel.fit(docs)
  
  def __init__(self, text, textWeight=3, nKeywords=5):
    self._text = text
    self._baseModel = markovify.Text(text)
    self._textWeight = textWeight
    self._nKeywords = nKeywords

    # get keywords
    tfidfVector = self.tfidfModel.transform([self._text])
    maxIndices = tfidfVector.A[0].argsort()
    self._keywords = np.array(self.tfidfModel.get_feature_names())[np.flip(maxIndices)][:self._nKeywords]

  def makeReleventSentence(self, model):
    '''
    Create a sentence with at least one keyword from the input text
    '''
    combinedModel = markovify.combine([model, self._baseModel], [1, self._textWeight])
    while True:
      sentence = combinedModel.make_short_sentence(230)
      if sentence is None:
        continue
      for keyword in self._keywords:
        if keyword in sentence[:-1].lower().split():
          return sentence

  def makeCommunistSentence(self):
    return self.makeReleventSentence(self._communistModel)

  def marxSays(self):
    return self.makeReleventSentence(self._marxModel)

  def engelsSays(self):
    return self.makeReleventSentence(self._engelsModel)

  def cheSays(self):
    return self.makeReleventSentence(self._cheModel)