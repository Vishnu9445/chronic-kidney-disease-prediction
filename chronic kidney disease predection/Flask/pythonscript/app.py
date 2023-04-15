from flask import Flask, render_template, request
import numpy as np
import pickle
app=Flask(__name__) #initialize a flask app
model = pickle.load(open('CKD.pk1', 'rb')) #loading the model
@app.route('/') #route to display the home page
def home():
    return render_template('homr.html') #rendering the home page
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')
@app.route('/predict',methods=['POST']) #route to show the predictions in a web UI
def predict():
    #reading the inputs given by the user
    input_features = [float(x) for x in request.form.values()]
    feature_values = [np.array(input_features)]

    features_name = ['blood_urea','blood_glucose_random','anemia','coronary_artery_disease','pus_cell','red_blood_cell','diabetesmellitus','pedal_edema']
    df = pd.DataFrame(feature_values, columns=features_name)

    #prediction using the loaded model file
    output = model.predict(df)

#showing the prediction results in a UI #showing the prediction results in a UI
return render_template('result.html',prediction_text=output)
if __name__ == '__main__':
     #running the app
app.run(debug=True)
