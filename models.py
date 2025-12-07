import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:/Users/henry/OneDrive - Florida International University/Desktop/MedEase/Synthetic_Vs_Real_Classification/gen-lang-client-0838385118-f165ec6f8ec1.json'

from google.generativeai import list_models

models = list(list_models())
print("Available models:")
for model in models:
    print(model)