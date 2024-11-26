import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import csv
import os

print("Hello, HPC!")

file_path = os.path.join(os.path.dirname(__file__), 'test.csv')

with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Column1', 'Column2', 'Column3'])
    writer.writerow(['Random', 'Text', 'Here'])
