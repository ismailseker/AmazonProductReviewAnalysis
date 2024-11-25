import pandas as pd
import pandas as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

import nltk

nltk.download('punt')
nltk.download('stopwords')

from nltk.corpus import stopwords

data = pd.read_csv("Amazon-Product-Reviews-Sentiment-Analysis-in-Python-Dataset.csv")

print(data.info())
data.dropna(inplace=True)

data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0
data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1

stp_words = stopwords.words('english')

def clean_review(review):
    
    cleanReview = " ".join(word for word in review.split() if word not in stp_words)
    
    return cleanReview

data['Review'] = data['Review'].apply(clean_review)
data['Sentiment'].value_counts()

consolidated=' '.join(word for word in data['Review'][data['Sentiment']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')


consolidated=' '.join(word for word in data['Review'][data['Sentiment']==1].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()

cv = TfidfVectorizer(max_features=2500)
x = cv.fit_transform(data['Review']).toarray()

from sklearn.model_selection import train_test_split

xTrain,xTest,yTrain,yTest = train_test_split(x,data['Sentiment'],test_size=0.25,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()

lr.fit(xTrain,yTrain)

pred = lr.predict(xTest)

print(accuracy_score(yTest,pred))

new_review = input("Comment: ")

new_review_cleaned = clean_review(new_review)
new_review_vectorized = cv.transform([new_review_cleaned]).toarray()

predictInput = lr.predict(new_review_vectorized)

if predictInput == 1:
    print("Positive Comment")
else:
    print("Negative Comment")





