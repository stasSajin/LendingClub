# Lending Club
Some exploratory analyses on Lending Club data.

#Click the link below to go to the report summary 
[Three Lessons](https://rawgit.com/stasSajin/LendingClub/master/03_lessons_learned/lessons_learned.html)

# Objectives
There are a ton of analyses on the web that explore the Lending Club Data. See examples in the list below:

- [Kaggle Scripts](https://www.kaggle.com/wendykan/lending-club-loan-data/scripts?sortBy=votes)
- [Data Science Academy](http://rpubs.com/jfdarre/119147)

A big focus in this report is on originality of ideas. I tried to steer away from anything that I could find that other people have done, and instead dug for insights that are not immediately obvious, such as sentiment analysis of loan descriptions and creation of pre-payment features. 

Not every analysis that I attempted was successful: some features that I created haven't provided any useful results and many other features that I created remain to be explored. Nonetheless, below is a list of results that have proven to be both useful and interesting:

Lesson 1: Borrowers who leave comments in their loan description provide lower returns than borrowers who leave the description blank. Moreover, the difference between these two groups increases as the riskiness of the loan increases.

Lesson 2: Not every defaulted and charged off loan is a bad loan

Lesson 3: An ensemble of 5 machine learning models trained and cross-validated on just 10% of the data led to an absolute 6.25% increase in cumulative returns. ml_ensemble.R has the details about the model building process.


You can find information on data cleaning and variable creation [here](https://rawgit.com/stasSajin/LendingClub/master/01_cleaning/data_cleaning.html).



