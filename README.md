# Flight Experience Sentiment Analysis Project

## 📌 Introduction

This project aims to build an intelligent system capable of **predicting the sentiment of user comments about flight experiences**. For this purpose, a dataset from Kaggle was used, containing **real tweets from passengers** about flights with different airlines.

The ultimate goal is to develop a **production-ready web application** where users can share their opinions about their flights, and the system, using natural language processing (NLP) models, automatically classifies the comments as **positive**, **negative**, or **neutral**.

This way, airlines can use the collected information to:

- Monitor customer satisfaction in real-time.
- Identify areas for service improvement.
- Make data-driven decisions.

## 📊 Dataset

The dataset was extracted from Kaggle and can be found at the following link:  
[Sentiment Analysis of Airline Tweets and Comments](https://www.kaggle.com/code/serkanp/sentiment-analysis-of-airline-tweets-and-comments)

It is a CSV file containing **15 columns** and **14,640 rows** of tweets from users sharing their flight experiences.

The key variables used in this project are:  
- **text**: The tweet text published by the user about their flight experience with a specific airline.  
- **airline_sentiment**: The target variable, which classifies the sentiment of the experience into **positive**, **negative**, or **neutral**.

These two variables are the core of the sentiment analysis modeling.

## 🧪 Model Development

In this stage, I focused on obtaining the best possible model to deploy in a production environment. To achieve this, I conducted an exploratory data analysis that helped me understand the behavior of the text and evaluate whether it was feasible to use more traditional probabilistic models, such as **Naive Bayes**.

I also performed thorough text preprocessing, selected appropriate evaluation metrics, and tested various models, comparing their performance using **fine-tuning techniques**. This process allowed me to select the most robust model in terms of **accuracy** and **generalization**.

### 📊 EDA (Exploratory Data Analysis)

In this section, I performed an exploratory analysis of the **text** variable in relation to the **airline_sentiment** label.  
The goal was to identify **statistical patterns** that help understand how the textual data is structured depending on the sentiment (positive, neutral, or negative).

Some of the key analyses included:

- **Number of words per tweet**
- **Number of unique words per tweet**
- **Average word length per tweet**
- **Total number of characters per tweet**
- Class balance across sentiment categories

This EDA step was essential to assess the **feasibility of simpler models** and guided decisions in **text cleaning**, **tokenization**, and **feature selection**.

---

### 🧠 Insights from Exploratory Analysis

From the exploratory analysis of the text data, several key insights emerge:

- **Tweet Length (in words):**  
  Negative tweets tend to be longer on average, peaking around 22–25 words. In contrast, positive and neutral tweets are generally shorter and more dispersed. This suggests that users are more verbose when expressing dissatisfaction.
  ![Words per tweet](images/1.png)

- **Number of Unique Words:**  
  Negative tweets also exhibit a higher number of unique words per tweet, indicating a richer vocabulary likely used to detail complaints. Positive and neutral tweets show a more modest and uniform distribution of unique terms.
![Unique words](images/2.png)

- **Average Word Length:**  
  Across all sentiment categories, the average word length remains fairly stable, typically between 4 and 6 characters. This implies that while users may write more when unhappy, the complexity of the vocabulary used does not vary significantly with sentiment.
  ![Average word length](images/3.png)

- **Tweet Length (in characters):**  
  Similar to word count, tweets labeled as negative often reach the character limit (140–160 characters), whereas positive and neutral tweets are generally shorter. This reinforces the idea that users elaborate more when describing negative experiences.
  ![Tweet length in characters](images/4.png)
**Conclusion:**  
These insights confirm that **tweet length and lexical diversity are strong indicators of sentiment polarity**. Negative feedback is usually more detailed, which can be leveraged in model training by including features related to text length or richness. This justifies exploring more advanced models capable of capturing nuanced language patterns beyond simple keyword-based approaches.
