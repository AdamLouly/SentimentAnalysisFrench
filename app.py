from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from flask_restful import Resource, Api
from json import dumps



app = Flask(__name__)

def polarity(inputs):
	polarity = open('static//polarity.pkl','rb')
	clf = joblib.load(polarity)
	vectorizer = open('static//cv.pkl','rb')
	cv = joblib.load(vectorizer)
	message=inputs
	data = [message]
	vect = cv.transform(data).toarray()
	my_prediction = clf.predict(vect)
	return my_prediction.astype(str)[0]

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/viz')
def viz():
	file = open("static//words.txt")
	return render_template('viz.html',words=file.read())
@app.route('/viz2')

def viz2():
	file = open("static//pie.txt")
	return render_template('viz2.html',words=file.read().split(','))
@app.route('/predict',methods=['POST'])
def predict():
	df = pd.read_csv('pola.csv', encoding="UTF-8",error_bad_lines=False,header=None, usecols=[0,1])
	df['label'] = df[0].map({"0": 0, "4": 1})

	df.columns=['id','message','label']
	df = df.dropna()

	X = df['message']
	y = df['label']
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)	
	

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)
	
@app.route('/test/', methods=['GET'])
def get_tasks():
    return jsonify({'polarity':polarity(request.args.get('mot')).astype(str)[0]})


if __name__ == '__main__':
	app.run(debug=True)