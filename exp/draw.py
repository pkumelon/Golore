import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('SGD.csv')

# print(df['Step'])

palette = plt.get_cmap('Set1')
color_index = 0

plt.figure(figsize=(7, 5))

loss1 = df['Golore0.25_SGDM - loss']
loss2 = df['Galore_SGDM_largebatch - loss']
loss3 = df['Galore_SGDM - loss']
loss4 = df['SGDM - loss']

plt.plot(loss1, color=palette(0), label='Golore0.25_SGDM')
plt.plot(loss2, color=palette(1), label='Galore_SGDM_largebatch')
plt.plot(loss3, color=palette(2), label='Galore_SGDM')
plt.plot(loss4, color=palette(3), label='SGDM')
legend_font = {'family': 'Arial', 'style': 'normal', 'size': 15, 'weight': 'bold'}
plt.legend(loc='upper right', prop=legend_font)
plt.grid(True)
# plt.xlim(xmin=0, xmax=100)
# plt.ylim(ymin=0,ymax=1)
plt.xlabel('iteration', weight='bold', fontsize=15)
plt.ylabel('loss', weight='bold', fontsize=15)
# plt.xticks([0,25,50,75,100], ['0', '25', '50', '75', '100'], weight='bold', fontsize=12)
# plt.yticks(weight='bold', fontsize=12)

df = pd.read_csv('Adam.csv')
plt.figure(figsize=(7, 5))

loss1 = df['Golore0.5 - loss']
loss2 = df['Galore_largebatch - loss']
loss3 = df['Galore - loss']
loss4 = df['AdamW - loss']

plt.plot(loss1, color=palette(0), label='Golore0.5')
plt.plot(loss2, color=palette(1), label='Galore_largebatch')
plt.plot(loss3, color=palette(2), label='Galore')
plt.plot(loss4, color=palette(3), label='AdamW')
legend_font = {'family': 'Arial', 'style': 'normal', 'size': 15, 'weight': 'bold'}
plt.legend(loc='upper right', prop=legend_font)
plt.grid(True)
# plt.xlim(xmin=0, xmax=100)
# plt.ylim(ymin=0,ymax=1)
plt.xlabel('iteration', weight='bold', fontsize=15)
plt.ylabel('loss', weight='bold', fontsize=15)

# plt.savefig('./xxx.pdf')

df = pd.read_csv('noB.csv')
plt.figure(figsize=(7, 5))

loss1 = df['Golore_noB - loss']
loss2 = df['Galore_noB - loss']
loss3 = df['AdamW_noB - loss']

plt.plot(loss1, color=palette(0), label='Golore_noB')
plt.plot(loss2, color=palette(1), label='Galore_noB')
plt.plot(loss3, color=palette(2), label='AdamW_noB')
legend_font = {'family': 'Arial', 'style': 'normal', 'size': 15, 'weight': 'bold'}
plt.legend(loc='upper right', prop=legend_font)
plt.grid(True)
# plt.xlim(xmin=0, xmax=100)
# plt.ylim(ymin=0,ymax=1)
plt.xlabel('iteration', weight='bold', fontsize=15)
plt.ylabel('loss', weight='bold', fontsize=15)

plt.show()
# plt.savefig('./xxx.pdf')