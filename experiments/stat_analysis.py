#!/usr/bin/env python3
import os
import numpy as np

radii = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

def load_file(path):
   fitness = np.load(path)
   return fitness

def r_robust_statistics_mean():
    #header
    print(';'.join(["radius","ratio","mean","std"]))
    
    for radius in radii:
        folder = f'Validation/ratios_r{radius}'
        for ratio in ratios:
            file = os.path.join(folder, f'r_1:{ratio}.npy')
            data = load_file(file)
            print(radius, end=';')
            print(ratio, end=';')
            print(data.mean(), end=';')
            print(data.std())
            #print(data)

 
def r_robust_statistics():
    #header
    print(';'.join(["index","radius","ratio","fitness"]))
    
    for radius in radii:
        folder = f'Validation/ratios/'
        for ratio in ratios:
            file = os.path.join(folder, f'r_{radius}:{ratio}_stack.npy')
            data = load_file(file)
            for i, v in enumerate(data):
                print(i, end=';')
                print(radius, end=';')
                print(ratio, end=';')
                print(v)
            #print(data.mean(), end=';')
            #print(data.std())
            #print(data)
 
def read_and_t_test():
    import pandas as pd
    import scipy
    df = pd.read_csv("Validation/ratios.csv", delimiter=';').drop(columns=['index'])
    print(df.groupby(['radius','ratio']).describe())

    for conf, data in df.groupby(['radius']):
        print(conf)
        green = data[(data.ratio) == 1.0]
        red = data[(data.ratio) == 0.0]
        mixed = data[(data.ratio) == 0.5]
        #print(green.fitness.describe())
        #print(red.fitness.describe())
        print(scipy.stats.ttest_ind(green.fitness,red.fitness))
        print(scipy.stats.ttest_ind(green.fitness,mixed.fitness))
        print(scipy.stats.ttest_ind(mixed.fitness,red.fitness))
        #break

if __name__ == "__main__":
   read_and_t_test()
