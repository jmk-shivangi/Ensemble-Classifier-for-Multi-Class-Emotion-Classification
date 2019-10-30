#cleaning my dataset code

import pandas as pd
import numpy as np
import re, sys, os,csv 
import datetime
import warnings
from datetime import timedelta
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import normalize
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfTransformer


warnings.filterwarnings("ignore")

#freqdist = nltk.FreqDist()
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

cols = ['text','emotion']
tweets_df = pd.read_csv('l2sorted.csv',header=None,names=cols)
print tweets_df.head()
tweets_df.emotion.value_counts()
#tweets_df.drop([tid],axis=1,inplace=true)
#tweets_df.head()
num_tweets = tweets_df.shape[0]
print("Total tweets: " + str(num_tweets))

def get_wordnet_pos(treebank_tag):

	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return None # for easy if-statement 

def preprocess(text):
	#print text
	lemma =[]
	lower_case = text.lower()
	emo =['#love','#fear','#sadness','#anger','#joy','#surprise']
	for word in emo:
		if word in text:
			text=text.replace(word," ")
	text = ' '.join(text.split())
	soup=BeautifulSoup(text,'lxml')
	souped = soup.get_text()
	stripped = re.sub(combined_pat, '', souped)
	try:
		clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
	except:
		clean = stripped
	letters_only = re.sub("[^a-zA-Z]", " ", clean)
	lower_case = letters_only.lower()
	#words = tok.tokenize(lower_case)
	#print 'Apply Lemmatizer'
	#porter_stemmer = nltk.stem.PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	#words = nltk.word_tokenize(lower_case)
	tokens = tok.tokenize(lower_case)
	tagged = nltk.pos_tag(tokens)
	for word, tag in tagged:
		wntag = get_wordnet_pos(tag)
		if wntag is None:# not supply tag in case of None
			lemma.append(lemmatizer.lemmatize(word)) 
		else:
			lemma.append(lemmatizer.lemmatize(word, pos=wntag)) 
	#print lemma
	#for j,word in enumerate(words):
	  #words[j] = lemmatizer.lemmatize(word)
	#temp_df.tweet[i] = ' '.join(words)
	#return temp_df
	#return (" ".join(lemma)).strip()
	ff = " ".join(lemma)
	lemma = []
	return ff

"""def preprocess(text):
	soup=BeautifulSoup(text,'lxml')
	souped = soup.get_text()
	stripped = re.sub(combined_pat, '', souped)
	try:
		clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
	except:
		clean = stripped
	letters_only = re.sub("[^a-zA-Z]", " ", clean)
	lower_case = letters_only.lower()
	words = tok.tokenize(lower_case)
	return (" ".join(words)).strip()"""
	
print("Beginning processing of tweets at: " + str(datetime.datetime.now()))
clean_tweet_texts = []

for i in (tweets_df.text):
	clean_tweet_texts.append(preprocess(i))

				  
#clean_tweet_texts = clean(tweets_df)
	
clean_df = pd.DataFrame(clean_tweet_texts,columns=['cleanedtext'])
clean_df['target'] = tweets_df.emotion
clean_df['text'] = tweets_df.text
clean_df.head()

clean_df.to_csv('clean_tweet_l1sorted.csv',encoding='utf-8')
csv = 'clean_tweet_l1sorted.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()

y = my_df.target
X = my_df.drop(['text','target'], axis=1, inplace=True)
X = my_df
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)	
skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=None)

X_np = np.array(X)
y_np = np.array(y)

acc_ens = []
pre1_ens = []
pre2_ens = []
recall1_ens = []
recall2_ens = []
f1_1ens = []
f1_2ens = []
ck_ens = []
estimators = []
model1 =  MultinomialNB()
estimators.append(('MNB',model1))
model2 = LinearSVC(C=1.0, intercept_scaling=1, multi_class='ovr', class_weight='balanced', random_state=0, tol=0.0001, verbose=0)
estimators.append(('SVM',model2))
model3 = LogisticRegression(class_weight='balanced',multi_class='ovr')
estimators.append(('LR',model3))
model4 = SGDClassifier(alpha=0.001,class_weight='balanced', max_iter=100)
estimators.append(('SGD',model4))
print estimators

for train_index, test_index in skf:
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X_np[train_index], X_np[test_index]
	y_train, y_test = y_np[train_index], y_np[test_index]
	X_train = pd.DataFrame(X_train)
	X_test = pd.DataFrame(X_test)
	y_train = pd.DataFrame(y_train)
	y_test = pd.DataFrame(y_test)
	cvec = CountVectorizer(stop_words='english', ngram_range=(1, 2))
	#cvec = CountVectorizer(stop_words='english', ngram_range=(2, 2))
	#tvec = TfidfVectorizer(sublinear_tf=True, norm='l2', ngram_range=(1, 2), stop_words='english')
	#transformer = TfidfTransformer().fit(X_train_bow)
	#X_train_bow_tfidf = transformer.transform(X_train_bow)
	#X_test_bow_tfidf = transformer.transform(X_test_bow)
	#tvec = TfidfVectorizer(sublinear_tf=True, norm='l2', ngram_range=(2, 2), stop_words='english')
	X_train_bow = cvec.fit_transform(X_train[0].values.astype('U'))
	#print X_train_bow
	#X_train_bow = X_train_bow / X_train_bow.max(axis=0)
	X_test_bow = cvec.transform(X_test[0].values.astype('U'))	
	#X_train_tfidf = tvec.fit_transform(X_train[0].values.astype('U'))
	#X_test_tfidf = tvec.transform(X_test[0].values.astype('U'))
	#dense = X_tfidf.todense()
	#print X_train_tfidf
	#feature_names = tvec.get_feature_names() 
	#print len(feature_names)
	ensemble = VotingClassifier(estimators)
	ensemble.fit(X_train_bow, y_train)
	ens_predictions = ensemble.predict(X_test_bow)
	acc_ens.append(float(accuracy_score(ens_predictions, y_test)))
	pre1_ens.append(float(precision_score(y_test, ens_predictions, average='weighted')))
	pre2_ens.append(float(precision_score(y_test, ens_predictions, average='macro')))
	recall1_ens.append(float(recall_score(y_test, ens_predictions, average='weighted'))) 
	recall2_ens.append(float(recall_score(y_test, ens_predictions, average='macro')))
	f1_1ens.append(float(f1_score(y_test, ens_predictions, average='weighted')))
	f1_2ens.append(float(f1_score(y_test, ens_predictions, average='macro')))
	ck_ens.append(float(cohen_kappa_score(y_test, ens_predictions)))
	cm = confusion_matrix(y_test, ens_predictions)
	print cm

print "L1sorted Results Ensemble............\n" 
print "ENSEMBLE Accuracy : " , sum(acc_ens) / float(len(acc_ens))
print "ENSEMBLE Pre1 : " , sum(pre1_ens) / float(len(pre1_ens))
print "ENSEMBLE Pre2 : " , sum(pre2_ens) / float(len(pre2_ens))
print "ENSEMBLE Recall1 : " , sum(recall1_ens) / float(len(recall1_ens))
print "ENSEMBLE Recall2 : " , sum(recall2_ens) / float(len(recall2_ens))
print "ENSEMBLE F1_score1 : " , sum(f1_1ens) / float(len(f1_1ens))
print "ENSEMBLE F1_score2 : " , sum(f1_2ens) / float(len(f1_2ens))
print "ENSEMBLE Cohen Kappa : " , sum(ck_ens) / float(len(ck_ens))

	 
	
	
