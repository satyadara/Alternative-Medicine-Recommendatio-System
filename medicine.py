
import pickle
import streamlit as st
import nltk
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
 

# Load data
medicines = pd.read_json('/Users/saryugundimeda/Documents/med.py/new.json')
medicines.dropna(inplace=True)
medicines['Description'] = medicines['Description'].apply(lambda x: x.split())
medicines['Reason'] = medicines['Reason'].apply(lambda x: x.split())
medicines['Description'] = medicines['Description'].apply(lambda x: [i.replace(" ", "") for i in x])
medicines['tags'] = medicines['Description'] + medicines['Reason']

# Prepare data for recommendation
new_df = medicines[['index', 'Drug_Name', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Preprocess text data
ps = PorterStemmer()
cv = CountVectorizer(stop_words='english', max_features=5000)

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute similarity matrix
similarity = cosine_similarity(vectors)

# Define function for recommendation
def recommend(medicine):
    try:
        medicine_index = new_df[new_df['Drug_Name'] == medicine].index[0]
        distances = similarity[medicine_index]
        medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommended_medicines = [new_df.iloc[i[0]].Drug_Name for i in medicines_list]
        return recommended_medicines
    except IndexError:
        return []

# Save processed data and similarity matrix using pickle
pickle.dump(new_df.to_dict(), open('medicine_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))


# Set page configuration
st.set_page_config(page_title="MEDCARE", page_icon=None, layout='wide', initial_sidebar_state='auto')

# Set main title and tagline
st.markdown(
    """
    <div style='text-align: center; color: white;'>
        <h1>MedCare</h1>
        <p>Explore Your Options, Discover Alternatives.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load processed data and similarity matrix
medicines_dict = pickle.load(open('medicine_dict.pkl', 'rb'))
medicines = pd.DataFrame(medicines_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Define function to recommend alternative medicines
def recommend(medicine):
    try:
        medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
        distances = similarity[medicine_index]
        medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommended_medicines = [medicines.iloc[i[0]].Drug_Name for i in medicines_list]
        return recommended_medicines
    except IndexError:
        return []

# Add input box and submit button
st.markdown("<hr>", unsafe_allow_html=True)
st.write("")
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    medicine_name = st.selectbox("Enter the name of the medicine:", medicines['Drug_Name'].values)
    submit_button = st.button("Submit", key='submit_button')

# Show recommendations
if submit_button:
    if medicine_name.strip() == "":
        st.error("Please enter the name of the medicine.")
    else:
        recommendations = recommend(medicine_name)
        if recommendations:
            st.write("")
            st.markdown("<h3>Alternative medicines:</h3>", unsafe_allow_html=True)
            for med in recommendations:
                st.markdown(f"<p>{med}</p>", unsafe_allow_html=True)
        else:
            st.error("Medicine not found in the dataset.")
