# Alternative-Medicine-Recommendatio-System
It's a simple yet effective Streamlit app that helps users find alternative medicines. It works by analyzing medicine names and descriptions, then using cosine similarity to suggest similar options. Just enter a medicine name, and it quickly finds alternatives based on the data it has processed.

Features
- Easy-to-use UI – Built with Streamlit for a smooth experience.
- Alternative Medicine Suggestions – Uses cosine similarity to find similar medicines.
- Smart Text Processing – Includes preprocessing and stemming for better matches.
- Interactive Search – Users can select a medicine from a dropdown and get instant recommendations.

Tech Stack
- Python
- Streamlit
- NLP (Text Preprocessing & Stemming)
- Scikit-learn (CountVectorizer, Cosine Similarity)
- Pandas

How It Works
- The dataset contains medicine names, descriptions, and their intended use.
- Text data is preprocessed, stemmed, and vectorized using CountVectorizer.
- Cosine similarity is applied to measure how closely medicines are related.
- Users enter a medicine name and receive a list of similar alternatives.

Conclusion

MedCare makes it easy to find alternative medicines with a simple, effective approach using NLP and cosine similarity. The intuitive UI ensures a hassle-free experience. Future updates could include a larger dataset and more advanced ML models for even better recommendations.
