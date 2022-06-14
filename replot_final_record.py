
#This script will allow you to replot any of extracted final record files
#Provide the name of the obtained JSON file with flag -f

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-f','--fname', default='images/0_colorcut_cluster_5final_record.json', type=str, help='Enter the name of the JSON record you want to plot')
args = parser.parse_args()
filename = args.fname

print('NOTE: To open other file, run this script with flag -f and provide the path to the record.')
print('Working with file '+filename)

f = open(filename ,)
record = json.load(f)
f.close()

X = np.array(record['data'])[:,0]
Y = np.array(record['data'])[:,1]

fig = plt.figure(figsize=(8,6))
plt.plot(X,Y,color = record['color'])
fig.savefig(filename[:-4]+'png')

print('Check the plot at '+filename[:-4]+'png')