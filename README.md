# PREDICTO - Predict Stock Market Prices


Stock Market Price Prediction is one of the most sought after problem statement in the field of Machine Learning and Depp Learning, due to the huge impact it could have on the World, when one could sucessfully predict the prices of the Stock for a month in advance. But also this makes it one of the most difficult problem. There are various factors that are involved in the change of Stock prices. A few of those are physical factors vs. physhological, rational and irrational behaviour, etc. , thus making the prediction of Stock prices difficult.


![screencapture-file-C-Users-kgsas-Desktop-Mini-Projects-Stock-Market-Prediction-Flask-src-templates-index-html-2021-07-14-11_47_25](https://user-images.githubusercontent.com/58512797/125575452-c041f0e4-0601-4099-8b6b-ad540a0554ee.png)

PREDICTO is a web based application that predicts the Stock prices based on Machine Learning, Statistical , and Deep Learning Algorithms. The front end consists of Flask and the Website was designed using the Mobirise application.


In, this project, we make use of various algorithms like Linear Regression, Facbook's Prophet, LSTM and ARIMA, to predict the prices of Stock for the next 100 days.

## Setting Up

1. Download or Clone the Respository

2. Create a seperate project environment - Recommended

      > conda create --name env_name python=3.8
     
      > conda activate env_name

3. Installing the Required Packages
      > pip install -r requirements.txt


3. Running the Applicaation
      > cd predicto/src
  
      > python main.py

## Results

### Linear Regression
![screencapture-127-0-0-1-5000-predict-2021-07-14-11_48_46](https://user-images.githubusercontent.com/58512797/125577054-04eb2878-329a-4358-8d6b-74d674464775.png)
### Facebook Prophet
![screencapture-127-0-0-1-5000-predict-2021-07-14-11_49_07](https://user-images.githubusercontent.com/58512797/125577040-200ef667-7fa3-47a6-8b36-086470a31421.png)
### LSTM
![screencapture-127-0-0-1-5000-predict-2021-07-14-11_52_49](https://user-images.githubusercontent.com/58512797/125577047-ff682147-5970-4a0d-9e29-2aed6227eea3.png)
### ARIMA
![screencapture-127-0-0-1-5000-predict-2021-07-14-11_53_28](https://user-images.githubusercontent.com/58512797/125577051-02864fdf-aadd-4fa7-9063-633ca759c830.png)

