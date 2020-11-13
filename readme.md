# Task
Given a text transcript of a conversation between two people, we want to assign it a Topic that they were most likely talking about.

# File Structure
1) model_generation.py
 - This file takes the data as input from the metadata and tagging_test folders, reads the data, performs a preprocessing step where the data is cleaned by removing special charcters, stopwords, numbers and unwanted spaces.
 - The cleaned data is then lemmatized and vectorised after which it is passed as a training set to train the model.

2) trained_model.pkl & countvect_model.pkl
 - these are the pickled file of the trained model.

3) classify .py
 - this file loads/unpickles the trained model, takes a text file (i.e test_doc.txt) as input, preprocesses the data in it, and gives an output which tells which topic or the label the input file belongs.