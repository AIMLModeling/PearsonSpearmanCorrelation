import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data_url = 'http://bit.ly/2cLzoxH'
# read data from url as pandas dataframe
gapminder = pd.read_csv(data_url)
# let us select two relevant columns
gapminder = gapminder[['gdpPercap', 'lifeExp']]
print(gapminder.head(3))

corr_pearson=gapminder.gdpPercap.corr(gapminder.lifeExp, method="pearson")
print("Pearson Correlation:", corr_pearson)

from scipy import stats
gdpPercap = gapminder.gdpPercap.values
life_exp = gapminder.lifeExp.values

(pearsoncorr,pvalue)=stats.pearsonr(gdpPercap,life_exp)
print(pearsoncorr,pvalue)

corr_spearman=gapminder.gdpPercap.corr(gapminder.lifeExp, method="spearman")
print("Spearman Correlation:", corr_spearman)

hplot = sns.distplot(gapminder['lifeExp'], kde=False, color='blue', bins=100)
plt.title('Life Expectancy', fontsize=18)
plt.xlabel('Life Exp (years)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plot_file_name="gapminder_life_expectancy_histogram.jpg"
# save as jpeg
hplot.figure.savefig(plot_file_name,
                    format='jpeg',
                    dpi=100)

sns.scatterplot('lifeExp','gdpPercap',data=gapminder)
plt.title('Non-linear relationship between GDP Percap and Life Exp', fontsize=18)
plt.ylabel('GDP Per Capita', fontsize=16)
plt.xlabel('Life Exp', fontsize=16)
