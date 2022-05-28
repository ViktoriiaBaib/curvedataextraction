# DB of emissivity records can be downloaded
# from https://bit.ly/3NKnmpL
#
# unzip the archive, place this file in the
# same directory and use it to work with any
# of the DB records.

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-f','--fname', default='1.json', type=str, help='Enter the name of the JSON record you want to plot')
args = parser.parse_args()
filename = args.fname

print('NOTE: To open other file, run this script with flag -f and provide the path to the record.')
print('Working with file '+filename)

f = open(filename ,)
record = json.load(f)
f.close()

X = np.array(record['data'])[:,0]
Y = np.array(record['data'])[:,1]
materials = ", ".join(record['materials'])
title = 'From '+record['authors'][0]+' et al., '+str(record['year'])+' doi='+record['doi'] + ', FIG '+record['figure_number']+'\n'+record['geometry_key']+' '+record['composition_key']+' made with '+ materials

fig = plt.figure(figsize=(8,6))
plt.plot(X,Y,color = record['color'])
plt.xlabel('Wavelength, $\mu m$')
plt.ylabel('Emissivity, a.u.')
plt.title(title)
fig.savefig(filename[:-4]+'png')

print('Check the plot at '+filename[:-4]+'png')