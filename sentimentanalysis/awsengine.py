from flask import Flask,request,jsonify,render_template
import joblib
import pandas as pd
import datetime
import os

app = Flask(__name__)
model_path = os.getcwd()+r'\sentimentanalysis\models\model'
#model_path = os.getcwd()+r'/models/model'
classifier = joblib.load(model_path+r'/classifier.pkl') 

def predictfunc(review):    
      
     prediction = classifier.predict(review)
     if prediction[0]==1:
          sentiment='Positive'
     else:
          sentiment='Negative'      
     return prediction[0],sentiment

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
     
     if request.method == 'POST':
        result = request.form
        content = request.form['review']
        review = pd.Series(content)
        prediction,sentiment =predictfunc(review)      
     return render_template("predict.html",pred=prediction,sent=sentiment)

if __name__ == '__main__':
     app.run(debug = True,port=8080)