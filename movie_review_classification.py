import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SKlearnClassifier
from sklearn.naive_bayes import MultinomialNB,BErnoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import LinearSVC,NuSVC
from nltk.classify import ClassifierI
import pickle

class voteclassify(ClassifierI):
	def __init__(self, *classifiers):
		self.classifiers=classifiers

	def classify(self,feature):
		votes=[]
		for cfier in self.classifiers:
			vote=cfier.classify(feature)
			votes.append(vote)
		return mode(votes)

	def credibility(self,feature):
		votes=[]
		for cfier in self.classifiers:
			vote=cfier.classify(feature)
			votes.append(vote)

		choices=votes.count(mode(votes))
		cred=choices/len(votes)
		return cred


 

document=[(list(movie_reviews.words(fileid)),category) 
for category in movie_reviews.categories() 
for fileid in movie_reviews.fileids(category)]


random.shuffle(document)

savedocs=open("pickleup.pickle","wb")
pickle.dump(document,savedocs)
savedocs.close()
"""
#print(document[1])

#Creates a document named'document' with a tuple of 2 elements - ([list of all words in the text file of moview review], category of movie review as whether positive or negative)
#alternately:
document=[]
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		document.append((list(movie_reviews.words(fileid)),category))

"""
all_words=[]
for w in movie_reviews.words():
 	all_words.append(w.lower())

#print(all_words)
all_words=nltk.FreqDist(all_words)
"""p=all_words.most_common(3000)
word_features=[]
for i in p:
	word_features.append(i[0])"""

#print(all_words["good"]) #returns how many times the word 'good' occurs

word_features=list(all_words.keys())[:3000]
#print(word_features)
def find_features(doc):
	words=set(doc)
	features={}
	for w in word_features:
		features[w]=(w in words)

	return features


print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
feature_sets=[(find_features(review),category) for (review,category) in document]
"""
c=[]
for a in p:
	c.append(p[0])

print(c)
"""
training_set=feature_sets[:1900]
testing_set=feature_sets[1900:]

classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Accuracy percentage of Original Naive BAyes Classifier:",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Accuracy percentage of MNB classifier:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("Accuracy percentage of BernoulliNB Classifier:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("Accuracy percentage of Logistic Regression Classifier:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGD_Classifier= SklearnClassifier(SGDClassifier())
SGD_Classifier.train(training_set)
print("Accuracy percentage of SGDClassifier:", (nltk.classify.accuracy(SGD_Classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("Accuracy percentage of LinearSVC classifier:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("Accuracy percentage of NuSVC classifier:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier =   voteclassify (classifier,
								  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGD_Classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print("Accuracy percentage of Voted Classifier:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
