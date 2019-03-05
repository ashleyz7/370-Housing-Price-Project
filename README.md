# g6-final-proj

###Project Description###

The overall purpose of our research project is to identify some potential factors that have a relationship with the median housing price in the Seattle area. After that, we may try to use these factors to predict the Seattle median housing price in 2019, which is our outcome of interest.

To help contextualize your research, we did some background research. As one of the paper says, “property values have become an increasingly common topic of conversation in recent years”. From the research, some most influential factors are supply and demand, interest rates, economic growth, demographics, location, the potential of growth, a second bedroom, parking, home improvements. Each of them has a different level and direction of effects with the price of houses. (House Prices) According to another research paper that focuses on finding the factors that influence the real estate price in London, several of the most significant variables that may affect house price including population density, income, and GVA (which is the measure of the value of goods and services in that area) (Gu).



Our current hypothesis is that there is a significant relationship between the median sale price of an area and population of that area, percent of houses increasing in value, Median Home Value and Average Number of Days of Homes on Zillow.

We will be mainly working with the Zillow Research Dataset (https://www.zillow.com/research/data/
), which is provided by Zillow itself. The dataset includes spreadsheets of different housing relevant metrics, which can be obtained by state, metro, county, city or zip code. If necessary, we may try to use several other metrics from population housing dataset found from the Data.WA.gov website(https://www.opendatanetwork.com/dataset/data.wa.gov/tx5i-i2ja
), this dataset includes population and housing information for Washington state for 2000 and 2010.

Since there are multiple variables that we want to take into consideration, we might use the multivariate linear regression model to filter out the most relevant variables to housing price. In terms of the machine learning methods, we may use regressor models instead of classifier models, since we are not predicting categories. For example, we may try to use the KNN tree or decision tree.

The target audience would be groups of people who are planning to buy or sell houses in a few months or years, for own use or for financial investment. The area we focus on is the Seattle area, and thus, people who plan to move here, as an average buyer, might be interested in our study. In addition, house selling or investing relevant companies like Zillow might be interested in our study as well.

From our recourse, The audience would be able to learn the trend of median housing prices in Seattle area, what some of the areas would be the best to move to or investigate in, which months are the best time to buy or sell the properties, how to pick houses that have greater growth potential, etc. 

###Technical Description###

For the fianl web format, we plan to use an ipython page in markdown to compile an HTML style report.

Several challenges we might encourter include handling the missing data. In addition,we possibly would try to ransition the scope from just the Seattle area to a statewide, or even nationwide focus (a $400,000 house in Cleveland looks a lot different from a $400,000 house in Mercer Island)
 
In terms of the new technical skills, we may need to explore new models for the prediction in order to get more accurate results. For instance, the gradient boosting model seems to be a suitable one for price prediction, but we need to do more research to figure out how to train and implement it. In addition, as we may need to look for new techniques that are specifically used for predicting price. If we find new relevant dataset in the future, we may also need to learn how to implement different types of dataset (API, etc) in Python.

For our modeling approach, we will firstly do some data clean up and preparation, such as properly handling the missing values in the dataset and perhaps creating new variables based on the existed variables. Then we might make some visualizations on certain variables to see the trend, but we would ultimately use sklearn feature selection (univariate feature selection, etc.)to identify the most relevant variables. After that, we would start to machine learning modeling, possible modeling algorithms we might try to use are KNN tree and gradient boosting. We would split the testing and training data first, and then perform cross-validation and grid-research; scalers would be added depending on the situation. When finishing training the models, we would use those models to do the prediction.

As to the future challenges, since the housing prices could be greatly influenced by supply and demand, we would have to have some sort of method to account for any relevant supply-demand variables. For example, when a city has strong economic growth and job creation, housing prices will increase. More jobs bring new residents to the city which increases the demand for housing and in result increases the price of houses. If we finally decide to use two datasets together, we may need to figure out a way to find common variables in order to combine them.

Citation
“Why West Coast home prices are surging” -Kathryn Vasel https://money.cnn.com/2018/06/13/real_estate/west-coast-housing-markets/index.html

“Why are House Prices so High?” -Positive Money

https://positivemoney.org/issues/house-prices/
“House Prices.” Information About Factors That Determine Property Prices - HomeGuru, www.homeguru.com.au/house-prices.

Gu Yiyang, “What are the most important factors that influence the changes in London Real Estate Prices? How to quantify them?”,  https://arxiv.org/pdf/1802.08238.pdf
