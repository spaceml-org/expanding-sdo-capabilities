"""
Quick script to split all the available falres events in modelling and holdout datasets
"""
import pandas as pd

filename = '/home/Valentina/expanding-sdo-capabilities/rare_events/flares_ordered.csv'
df = pd.read_csv(filename)
df = df.sample(frac=1).reset_index(drop=True)
holdout_filename = '/home/Valentina/expanding-sdo-capabilities/rare_events/flares_holdout.csv'
train_filename = '/home/Valentina/expanding-sdo-capabilities/rare_events/flares_modelling.csv'
holdout_size = int(df.shape[0]*0.30)
df[:holdout_size].to_csv(holdout_filename, index=False)
df[holdout_size:].to_csv(train_filename, index=False)