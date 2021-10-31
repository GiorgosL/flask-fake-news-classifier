#!/usr/bin/env python
# coding: utf-8
from flask import Flask, render_template, url_for, request
import pickle
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    message = request.form['message']
    my_prediction = model.predict(tv.transform([message]))

    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    model = pickle.load(open('model.pkl', 'rb'))
    tv = pickle.load(open('tv.pkl', 'rb'))
    logging.info('model and vectoriser loaded')
    logging.info('App starting')
    app.run()