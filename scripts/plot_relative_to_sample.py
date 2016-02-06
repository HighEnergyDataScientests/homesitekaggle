import pandas as pd
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt


pred1 = pd.read_csv('../predictions/sample_predictions.csv')
pred2 = pd.read_csv('../predictions/pred_20160206-011715.csv')
pred2.rename(columns={"QuoteConversion_Flag": "QuoteConversion_Flag1"}, inplace=True)
pred1 = pd.merge(pred1, pred2, on="QuoteNumber")

pred1[['QuoteConversion_Flag','QuoteConversion_Flag1']].plot(style=['o','rx'])
plt.savefig('diff_plot.png')
