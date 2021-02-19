import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# let's try for the first
WT3 = '/tmp/rotarod_ML4/output/Day3_WT/st_ad_mm_fi_cl_190623_Day3_146m6_rotarod1_2Nov15.csv'

# now, let's
df = pd.read_csv(WT3, index_col=0)
