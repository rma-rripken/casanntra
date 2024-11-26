import sys
import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(current_dir)               
casanntra_dir = os.path.join(project_root, 'casanntra')  

sys.path.insert(0, casanntra_dir)

from model_builder import ModelBuilder

class TestModelBuilder(ModelBuilder):
    def __init__(self, input_names, output_names, ndays, nwindows, window_length):
        super().__init__(input_names, output_names)
        self.ndays = ndays
        self.nwindows = nwindows
        self.window_length = window_length
        self.ntime = ndays + nwindows

    def build_model(self, input_layers, input_data):
        pass

    def fit_model(self, model, fit_in, fit_out, test_in, test_out):
        pass

date_range = pd.date_range(start='2021-01-01', periods=101, freq='D')
data = {
    'datetime': date_range,
    'case': [1]*101,
    'fold': [0]*101,
    'feature1': np.arange(0, 101),
    'feature2': np.arange(1000, 1101)
}
df = pd.DataFrame(data)

ndays = 3           
nwindows = 0    
window_length = 0  
input_names = ['feature1', 'feature2']
output_names = []   

model_builder = TestModelBuilder(input_names, output_names, ndays, nwindows, window_length)

df_lagged = model_builder.create_antecedent_inputs(df,reverse=False)

print("Original DataFrame:")
print(df)
print("\nLagged DataFrame:")
print(df_lagged)