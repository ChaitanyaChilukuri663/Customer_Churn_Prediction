import pickle
import pandas as pd
import streamlit as st

# set basic page configuration
st.set_page_config(
    page_title="Churn Prediction Model",
    page_icon="🥸",
    layout="centered"
)

# load the model
model = pickle.load(open("model.pkl", 'rb'))

# create page header 
st.markdown("# :green[Customer Churn Prediction Model]")

# take inputs from user
age = st.text_input(label="Give Customer Age: ", placeholder=int)
gender = st.selectbox(label="Select Customer Gender", options=['Male', 'Female'])
location = st.selectbox(
                label="Select Custoemr Location",
                options=['Houston', 'Los Angeles', 'Miami', 'Chicago', 'New York']
            )
subscription_length_months = st.text_input(label="How many months of subscription this customer has: ", placeholder=int)
monthly_bill = st.text_input(label="What is the monthly bill of this customer: ", placeholder=float)
total_used_gb = st.text_input(label="How many GB of content consumed by this customer: ", placeholder=int)

# create an array of all these inputs
features = [{
    'Age': age,
    'Gender': gender,
    'Location': location,
    'Subscription_Length_Months': subscription_length_months,
    'Monthly_Bill': monthly_bill,
    'Total_Usage_GB': total_used_gb
}]

# convert it to pandas dataframe before passing it to model
features = pd.DataFrame(features)


if total_used_gb:
    output = model.predict(features)
    if output == 1:
        st.markdown("### :red[Churn Warning: Customer May Be Nearing Exit.]")
        st.write("With a '1' on the radar, the model is suggesting a noteworthy probability that this customer might just be contemplating a departure.")
    if output == 0:
        st.markdown("### :green[Churn Resistant: Strong Customer Loyalty Indicated.]")
        st.write("With a '0' on the scene, it's like the model is giving a thumbs-up to this customer's staying power. The chances of them leaving? Well, let's just say they're safely tucked in the 'unlikely' zone.")