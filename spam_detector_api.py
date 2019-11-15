import pickle
from flask import Flask,render_template,request
import numpy as np

app = Flask(__name__)
model = pickle.load(open('MultinomialNB.pkl','rb'))
transformer = pickle.load(open('CountVectorizer.pkl','rb'))

@app.route('/')

def home():
	return render_template('home.html')

@app.route('/predict',methods = ['POST'])

def predict():
	
	if request.method == 'POST':	
		message = str(request.form['message'])
		if message == '':
			prediction = 'empty'
		else:
			to_predict = transformer.transform([message]).toarray()
			prediction = model.predict(to_predict)

	return render_template('result.html',prediction_value = prediction)

if __name__ == '__main__':
	app.run(debug=True)