from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load('models/best_model.pkl')
le = joblib.load('models/label_encoder.pkl')

# Load datasets
description = pd.read_csv('datasets/description.csv')
precautions = pd.read_csv('datasets/precautions.csv')
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv('datasets/diets.csv')

#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    return desc, pre, med, die

symptoms = [
    "a_dry_coat", "abdominal_pain", "anorexia", "bloody_or_cloudy_urine",
    "change_in_appetite", "chronic_or_recurring_infections", "decreased_appetite",
    "dehydration", "diarrhea", "excessive_amounts_of_energy", "fever",
    "frequent_urination", "gingivitis", "increased_appetite", "increased_thirst",
    "increased_thirst_and/or_urination", "lethargy", "loss_of_appetite",
    "persistent_diarrhea", "persistent_fever", "poor_coat_condition",
    "restlessness", "seizures_or_neurological_disorders",
    "urinary_or_respiratory_infections", "sweet_smelling_breath",
    "various_eye_conditions", "vomiting", "weight_loss"
]

symptoms_dict = {symptom: idx for idx, symptom in enumerate(symptoms)}

def format_disease_name(name):
    return name.replace('_', ' ').title()

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    return desc, pre, med, die

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    input_vector = pd.DataFrame([input_vector], columns=symptoms_dict.keys())
    pred_encoded = model.predict(input_vector)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    return pred_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')

        if not selected_symptoms:
            return render_template('index.html', symptoms=symptoms, message="Please select at least one symptom.")

        predicted_disease = get_predicted_value(selected_symptoms)
        formatted_disease = format_disease_name(predicted_disease)

        desc, pre, med, die = helper(formatted_disease)

        return render_template('index.html', symptoms=symptoms,
                               predicted_disease=formatted_disease,
                               dis_des=desc,
                               my_precautions=pre[0],
                               medications=med,
                               my_diet=die,)

    return render_template('index.html', symptoms=symptoms)
@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)



