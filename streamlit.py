import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_model_and_scaler():
    try:
        with open('random_forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('standard_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('label_encoder.pkl', 'rb') as file:
            le = pickle.load(file)
        return model, scaler, le
    except FileNotFoundError:
        st.error("Model files not found. Please make sure are in the same directory.")
        return None, None, None

# Mapping dictionaries
marital_status_map = {
    1: 'single', 2: 'married', 3: 'widower', 4: 'divorced',
    5: 'facto union', 6: 'legally separated'
}

occupation_map = {
    0: 'Student', 1: 'Legislative/Executive', 2: 'Scientific specialists',
    3: 'Technicians', 4: 'Admin staff', 5: 'Services/Sellers',
    6: 'Agriculture/Fisheries', 7: 'Industry/Construction',
    8: 'Machine Operators', 9: 'Unskilled Workers',
    10: 'Armed Forces', 90: 'Other', 99: 'Blank', 101: 'Armed Forces Officers',
    102: 'Armed Forces Sergeants', 103: 'Other Armed Forces',
    112: 'Admin/Commercial Directors', 114: 'Hotel/Trade Directors',
    121: 'Science/Engineering Specialists', 122: 'Health professionals',
    123: 'Teachers', 124: 'Finance/Admin Specialists', 125: 'ICT Specialists',
    131: 'Science/Engineering Technicians', 132: 'Health Technicians',
    134: 'Legal/Social/Cultural Technicians', 135: 'ICT Technicians',
    141: 'Secretaries/Data Operators', 143: 'Finance/Admin Operators',
    144: 'Other Admin Support', 151: 'Personal service workers', 152: 'Sellers',
    153: 'Personal care workers', 154: 'Security services',
    161: 'Farmers (market)', 163: 'Subsistence farmers', 171: 'Construction workers',
    172: 'Metallurgy workers', 173: 'Artisan/Precision workers', 174: 'Electricians/Electronics',
    175: 'Processing workers', 181: 'Plant operators', 182: 'Assembly workers',
    183: 'Drivers/Operators', 191: 'Cleaners', 192: 'Unskilled in agriculture',
    193: 'Unskilled in industry', 194: 'Meal assistants', 195: 'Street vendors'
}

previous_qualification_map = {
    1: "Secondary education", 2: "Higher education - bachelor's degree",
    3: "Higher education - degree", 4: "Higher education - master's",
    5: "Higher education - doctorate", 6: "Frequency of higher education",
    9: "12th year of schooling - not completed", 10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling", 14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle", 38: "Basic education 2nd cycle",
    39: "Technological specialization", 40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical", 43: "Higher education - master (2nd cycle)"
}

binary_map = {1: 'yes', 0: 'no'}
gender_map = {1: 'male', 0: 'female'}
daytime_map = {1: 'daytime', 0: 'evening'}

nationality_map = {
    1: 'Portuguese', 2: 'German', 6: 'Spanish', 11: 'Italian',
    13: 'Dutch', 14: 'English', 17: 'Lithuanian', 21: 'Angolan',
    22: 'Cape Verdean', 24: 'Guinean', 25: 'Mozambican', 26: 'Santomean',
    32: 'Turkish', 41: 'Brazilian', 62: 'Romanian', 100: 'Moldova',
    101: 'Mexican', 103: 'Ukrainian', 105: 'Russian', 108: 'Cuban', 109: 'Colombian'
}

course_map = {
    33: 'Biofuel Production', 171: 'Animation/Multimedia', 8014: 'Social Service (evening)',
    9003: 'Agronomy', 9070: 'Communication Design', 9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering', 9130: 'Equinculture', 9147: 'Management',
    9238: 'Social Service', 9254: 'Tourism', 9500: 'Nursing', 9556: 'Oral Hygiene',
    9670: 'Ad/Marketing', 9773: 'Journalism', 9853: 'Basic Education', 9991: 'Management (evening)'
}

application_mode_map = {
    1: '1st phase - general', 2: 'Ordinance 612/93', 5: 'Special - Azores',
    7: 'Other higher courses', 10: 'Ordinance 854-B/99', 15: 'International student',
    16: 'Special - Madeira', 17: '2nd phase - general', 18: '3rd phase - general',
    26: 'Different Plan', 27: 'Other Institution', 39: 'Over 23 years old',
    42: 'Transfer', 43: 'Change course', 44: 'Tech diploma',
    51: 'Change institution/course', 53: 'Short cycle diploma',
    57: 'Change institution/course (Int.)'
}

def main():
    st.set_page_config(page_title="Student Status Prediction", page_icon="🎓", layout="wide")
    
    st.title("🎓 Student Status Prediction System")
    st.markdown("---")
    
    # Load model and scaler
    model, scaler, le = load_model_and_scaler()
    
    if model is None:
        st.stop()
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        # Gender
        gender_options = list(gender_map.values())
        gender = st.selectbox("Gender", gender_options)
        gender_value = [k for k, v in gender_map.items() if v == gender][0]
        
        # Marital Status
        marital_options = list(marital_status_map.values())
        marital_status = st.selectbox("Marital Status", marital_options)
        marital_value = [k for k, v in marital_status_map.items() if v == marital_status][0]
        
        # Nationality
        nationality_options = list(nationality_map.values())
        nationality = st.selectbox("Nationality", nationality_options)
        nationality_value = [k for k, v in nationality_map.items() if v == nationality][0]
        
        # Age at enrollment
        age_enrollment = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20)
        
        # International
        international_options = list(binary_map.values())
        international = st.selectbox("International Student", international_options)
        international_value = [k for k, v in binary_map.items() if v == international][0]
        
        # Educational special needs
        special_needs = st.selectbox("Educational Special Needs", international_options)
        special_needs_value = [k for k, v in binary_map.items() if v == special_needs][0]
        
        # Displaced
        displaced = st.selectbox("Displaced", international_options)
        displaced_value = [k for k, v in binary_map.items() if v == displaced][0]
    
    with col2:
        st.subheader("Academic Information")
        
        # Course
        course_options = list(course_map.values())
        course = st.selectbox("Course", course_options)
        course_value = [k for k, v in course_map.items() if v == course][0]
        
        # Application mode
        application_options = list(application_mode_map.values())
        application_mode = st.selectbox("Application Mode", application_options)
        application_value = [k for k, v in application_mode_map.items() if v == application_mode][0]
        
        # Daytime/Evening attendance
        attendance_options = list(daytime_map.values())
        attendance = st.selectbox("Attendance Mode", attendance_options)
        attendance_value = [k for k, v in daytime_map.items() if v == attendance][0]
        
        # Previous qualification
        qualification_options = list(previous_qualification_map.values())
        prev_qualification = st.selectbox("Previous Qualification", qualification_options)
        qualification_value = [k for k, v in previous_qualification_map.items() if v == prev_qualification][0]
        
        # Previous qualification grade
        prev_qualification_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=120.0, step=0.1)
        
        # Admission grade
        admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=120.0, step=0.1)
        
        # Scholarship holder
        scholarship_options = list(binary_map.values())
        scholarship = st.selectbox("Scholarship Holder", scholarship_options)
        scholarship_value = [k for k, v in binary_map.items() if v == scholarship][0]
    
    # Financial Information
    st.subheader("Financial Information")
    col3, col4 = st.columns(2)
    
    with col3:
        # Tuition fees up to date
        tuition_options = list(binary_map.values())
        tuition_fees = st.selectbox("Tuition Fees Up to Date", tuition_options)
        tuition_value = [k for k, v in binary_map.items() if v == tuition_fees][0]
        
        # Debtor
        debtor = st.selectbox("Debtor", tuition_options)
        debtor_value = [k for k, v in binary_map.items() if v == debtor][0]
    
    with col4:
        # GDP
        gdp = st.number_input("GDP", value=0.0, step=0.01)
        
        # Unemployment rate
        unemployment_rate = st.number_input("Unemployment Rate", value=0.0, step=0.1)
        
        # Inflation rate
        inflation_rate = st.number_input("Inflation Rate", value=0.0, step=0.1)
    
    # Parents' Information
    st.subheader("Parents' Information")
    col5, col6 = st.columns(2)
    
    with col5:
        # Mother's occupation
        mothers_occupation_options = list(occupation_map.values())
        mothers_occupation = st.selectbox("Mother's Occupation", mothers_occupation_options)
        mothers_occupation_value = [k for k, v in occupation_map.items() if v == mothers_occupation][0]
        
        # Mother's qualification
        mothers_qualification_options = list(previous_qualification_map.values())
        mothers_qualification = st.selectbox("Mother's Qualification", mothers_qualification_options)
        mothers_qualification_value = [k for k, v in previous_qualification_map.items() if v == mothers_qualification][0]
    
    with col6:
        # Father's occupation
        fathers_occupation_options = list(occupation_map.values())
        fathers_occupation = st.selectbox("Father's Occupation", fathers_occupation_options)
        fathers_occupation_value = [k for k, v in occupation_map.items() if v == fathers_occupation][0]
        
        # Father's qualification
        fathers_qualification_options = list(previous_qualification_map.values())
        fathers_qualification = st.selectbox("Father's Qualification", fathers_qualification_options)
        fathers_qualification_value = [k for k, v in previous_qualification_map.items() if v == fathers_qualification][0]
    
    st.subheader("Academic Performance")
    col7, col8 = st.columns(2)
    
    with col7:
        cu1_credited = st.number_input("1st Sem - Credited Units", min_value=0, value=0)
        cu1_enrolled = st.number_input("1st Sem - Enrolled Units", min_value=0, value=6)
        cu1_evaluations = st.number_input("1st Sem - Evaluations", min_value=0, value=6)
        cu1_approved = st.number_input("1st Sem - Approved Units", min_value=0, value=6)
        cu1_grade = st.number_input("1st Sem - Grade", min_value=0.0, max_value=20.0, value=13.0, step=0.1)
        cu1_without_eval = st.number_input("1st Sem - Without Evaluations", min_value=0, value=0)
    
    with col8:
        # Curricular units 2nd sem
        cu2_credited = st.number_input("2nd Sem - Credited Units", min_value=0, value=0)
        cu2_enrolled = st.number_input("2nd Sem - Enrolled Units", min_value=0, value=6)
        cu2_evaluations = st.number_input("2nd Sem - Evaluations", min_value=0, value=6)
        cu2_approved = st.number_input("2nd Sem - Approved Units", min_value=0, value=6)
        cu2_grade = st.number_input("2nd Sem - Grade", min_value=0.0, max_value=20.0, value=13.0, step=0.1)
        cu2_without_eval = st.number_input("2nd Sem - Without Evaluations", min_value=0, value=0)
    
    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Student Status", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'Marital_status': [marital_value],
            'Application_mode': [application_value],
            'Application_order': [1],  # Default value, adjust as needed
            'Course': [course_value],
            'Daytime_evening_attendance': [attendance_value],
            'Previous_qualification': [qualification_value],
            'Previous_qualification_grade': [prev_qualification_grade],
            'Nacionality': [nationality_value],
            'Mothers_qualification': [mothers_qualification_value],
            'Fathers_qualification': [fathers_qualification_value],
            'Mothers_occupation': [mothers_occupation_value],
            'Fathers_occupation': [fathers_occupation_value],
            'Displaced': [displaced_value],
            'Educational_special_needs': [special_needs_value],
            'Debtor': [debtor_value],
            'Tuition_fees_up_to_date': [tuition_value],
            'Gender': [gender_value],
            'Scholarship_holder': [scholarship_value],
            'Age_at_enrollment': [age_enrollment],
            'International': [international_value],
            'Curricular_units_1st_sem_credited': [cu1_credited],
            'Curricular_units_1st_sem_enrolled': [cu1_enrolled],
            'Curricular_units_1st_sem_evaluations': [cu1_evaluations],
            'Curricular_units_1st_sem_approved': [cu1_approved],
            'Curricular_units_1st_sem_grade': [cu1_grade],
            'Curricular_units_1st_sem_without_evaluations': [cu1_without_eval],
            'Curricular_units_2nd_sem_credited': [cu2_credited],
            'Curricular_units_2nd_sem_enrolled': [cu2_enrolled],
            'Curricular_units_2nd_sem_evaluations': [cu2_evaluations],
            'Curricular_units_2nd_sem_approved': [cu2_approved],
            'Curricular_units_2nd_sem_grade': [cu2_grade],
            'Curricular_units_2nd_sem_without_evaluations': [cu2_without_eval],
            'GDP': [gdp],
            'Unemployment_rate': [unemployment_rate],
            'Inflation_rate': [inflation_rate]
        })
        
        try:
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Decode prediction
            status_labels = le.classes_
            predicted_status = status_labels[prediction]
            
            # Display results
            st.success("Prediction completed successfully!")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric("Predicted Status", predicted_status)
            
            with col_result2:
                st.metric("Confidence", f"{max(prediction_proba):.2%}")
            
            # Display probability for each class
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Status': status_labels,
                'Probability': prediction_proba
            }).sort_values('Probability', ascending=False)
            
            st.bar_chart(prob_df.set_index('Status'))
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please make sure all model files are properly saved and the input features match the training data.")

if __name__ == "__main__":
    main()

