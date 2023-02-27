
# Importing necessary libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
import string
import numpy as np
import re
from nltk.tokenize import sent_tokenize

# 1. Objective
# 
#The objective of this assignment is to extract textual data articles from the given URL and perform text analysis to compute variables that are explained below. 
# 
#Reading input file containing web links and it's attributes
df = pd.read_excel('Input.xlsx')
df.head()
df.shape
df.info()
df.isnull().sum()
df.info()

# ## 2. Data Extraction
# #### Input.xlsx
# #### For each of the articles, given in the input.xlsx file, extract the article text and save the extracted article in a text file with URL_ID as its file name.
# #### While extracting text, please make sure your program extracts only the article title and the article text. It should not extract the website header, footer, or anything other than the article text. 
# 

# ### Data Extraction for a single link
# #### - To check whether required data is getting extracted or not.

url = df['URL'][0]
content = requests.get(url).content
soup = BeautifulSoup(content,'lxml')
print(soup.prettify())
title = soup.title.text
summary = soup.find('div',class_='td-post-content')
summary = [para.text for para in summary.find_all('p')]
summary = ' '.join([str(elem) for elem in summary])
summary = summary.replace('\xa0'," ")
summary = summary.replace('\n'," ")
print(summary)

# ### Data extraction from all links
# #### - Since, the code is working fine for one link, we'll loop it for all the links

csv_file = open(r'URL_ID.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['TEXT'])

for i in range(0,len(df)):
    
    try:
        
        url = df.loc[i,'URL']
    
        content = requests.get(url).content

        soup = BeautifulSoup(content,'lxml')

        title = soup.title.text
        print(title)

        summary = soup.find('div',class_='td-post-content')
        summary = [para.text for para in summary.find_all('p')]
        summary = ' '.join([str(elem) for elem in summary])
        summary = summary.replace('\xa0'," ")
        summary = summary.replace('\n'," ")
    
    except Exception as e:
        summary = ''
    
    text = title+" "+summary
    csv_writer.writerow([text])    

csv_file.close()


# # Text Analysis

# ## 1. Sentiment Analysis

# #### Loading extracted data
# Reading the file with extracted texts
text = pd.read_csv(r'URL_ID.csv')
text.head()

# ### 1.1 Cleaning using Stop Words Lists
#importing stop words files that are provided

StopWords_Auditor = list(pd.read_csv(" StopWords/StopWords_Auditor.txt",header=None,encoding='cp1252',sep='|')[0])
StopWords_Currencies = list(pd.read_csv(" StopWords/StopWords_Currencies.txt",header=None,encoding="cp1252",on_bad_lines='skip',sep='|')[0])#
StopWords_DatesandNumbers = list(pd.read_csv(" StopWords/StopWords_DatesandNumbers.txt",header=None,encoding='cp1252',sep='|')[0])
StopWords_Generic = list(pd.read_csv(" StopWords/StopWords_Generic.txt",header=None,encoding='cp1252',sep='|')[0])
StopWords_GenericLong = list(pd.read_csv(" StopWords/StopWords_GenericLong.txt",header=None,encoding='cp1252',sep='|')[0])
StopWords_Geographic = list(pd.read_csv(" StopWords/StopWords_Geographic.txt",header=None,encoding='cp1252',sep='|')[0])
StopWords_Names = list(pd.read_csv(" StopWords/StopWords_Names.txt",header=None,encoding='cp1252',sep='|')[0])

# Converting strings to list
StopWords_Auditor = [str(word).lower() for word in StopWords_Auditor]
StopWords_Currencies = [str(word).lower() for word in StopWords_Currencies]
StopWords_DatesandNumbers = [str(word).lower() for word in StopWords_DatesandNumbers]
StopWords_Generic = [str(word).lower() for word in StopWords_Generic]
StopWords_GenericLong = [str(word).lower() for word in StopWords_GenericLong]
StopWords_Geographic = [str(word).lower() for word in StopWords_Geographic]
StopWords_Names = [str(word).lower() for word in StopWords_Names]

#creating func for removing stop words only
def remove_stopwords(text):
    txt=' '.join([word for word in text.split() if word.lower() not in StopWords_Auditor])
    txt1=' '.join([word for word in txt.split() if word.lower() not in StopWords_Currencies])
    txt2=' '.join([word for word in txt1.split() if word.lower() not in StopWords_DatesandNumbers])
    txt3=' '.join([word for word in txt2.split() if word.lower() not in StopWords_Generic])
    txt4=' '.join([word for word in txt3.split() if word.lower() not in StopWords_GenericLong])
    txt5=' '.join([word for word in txt4.split() if word.lower() not in StopWords_Geographic])
    txt6=' '.join([word for word in txt5.split() if word.lower() not in StopWords_Names])
    return txt6

# Applying function to remmove stopwords
text['TEXT_StopWords_removed'] = text['TEXT'].apply(remove_stopwords)

print(text['TEXT_StopWords_removed'])


# 1.2 Creating a dictionary of Positive and Negative words
# Importing given dictionary for reference
positive = pd.read_csv(' MasterDictionary/positive-words.txt',header=None,encoding='cp1252',skip_blank_lines=True)[0]
negative = pd.read_csv(' MasterDictionary/negative-words.txt',header=None,encoding='cp1252',skip_blank_lines=True)[0]

# Converting to list
positive = [word for word in positive]
negative = [word for word in negative]


# ### 1.3	Extracting Derived variables
# Splitting into sentences
sentences = [sent_tokenize(record) for record in text['TEXT_StopWords_removed']]
count_sentences = [len(record) for record in sentences]
count_sentences_arr = np.array(count_sentences)

# Splitting into words
words = [word for word in text['TEXT_StopWords_removed']]
count_words = [len(word) for word in words]
count_words_arr = np.array(count_words)


# #### Positive Score
positive_score = []
for record in text['TEXT_StopWords_removed']:
    score = 0
    for word in record.split():
        if( word in positive):
            score+=1
    positive_score.append(score)

positive_score_arr = np.array(positive_score)


# #### Negative Score
negative_score = []
for record in text['TEXT_StopWords_removed']:
    score = 0
    for word in record.split():
        if( word in negative):
            score+=-1
    negative_score.append(-1*score)

negative_score_arr = np.array(negative_score)


# #### Polarity Score
# ##### Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
Polarity_Score=(positive_score_arr-negative_score_arr)/((positive_score_arr+negative_score_arr)+0.000001)
print('polarity_score=', Polarity_Score)


# #### SUBJECTIVITY SCORE
# ##### Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)
Subjectivity_score=(positive_score_arr+negative_score_arr)/((count_words_arr)+ 0.000001)
print('subjectivity_score',Subjectivity_score)


# ## 2. Analysis of Readability

# #### AVG SENTENCE LENGTH
# ##### Average Sentence Length = the number of words / the number of sentences
Avg_sent_length = count_words_arr/count_sentences_arr
Avg_sent_length


# #### PERCENTAGE OF COMPLEX WORDS
# ##### Percentage of Complex words = the number of complex words / the number of words *100
count_complex_words = []
for record in text['TEXT_StopWords_removed']:
    count = 0
    for word in record.split():
        d = {}.fromkeys('aeiou',0)
        haslotsvowels = False
        for x in word.lower():
            if x in d:
                d[x] += 1
        for q in d.values():
            if q > 2:
                haslotsvowels = True
        if haslotsvowels:
            count += 1
    count_complex_words.append(count)
    
count_complex_words_arr = np.array(count_complex_words)
percentage_complex_words = count_complex_words_arr/count_words*100


# #### FOG Index
# ##### Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
Fog_Index = 0.4 * (Avg_sent_length + percentage_complex_words)
print('fog index= ',Fog_Index )


# ## 3.Average Number of Words Per Sentence
# ##### Average Number of Words Per Sentence = the total number of words / the total number of sentences
Avg_words_per_sentance = count_words_arr/count_sentences_arr
Avg_words_per_sentance


# ## 4.Complex Word Count
count_complex_words = []
for record in text['TEXT_StopWords_removed']:
    count = 0
    for word in record.split():
        d = {}.fromkeys('aeiou',0)
        haslotsvowels = False
        for x in word.lower():
            if x in d:
                d[x] += 1
        for q in d.values():
            if q > 2:
                haslotsvowels = True
        if haslotsvowels:
            count += 1
    count_complex_words.append(count)

count_complex_words_arr = np.array(count_complex_words)

# ## 5.Word count

from nltk.corpus import stopwords

nltk_stopwords = stopwords.words('english')
punc = [punc for punc in string.punctuation]

#creating func for removing stop words,punctuations and converting to words
def remove_punc_stopwords_nltk(text):
    nopunc =[char for char in text if char not in punc]
    nopunc=''.join(nopunc)
    txt=' '.join([word for word in nopunc.split() if word.lower() not in nltk_stopwords])
    return txt

text['TEXT_cleaned'] = text['TEXT_StopWords_removed'].apply(remove_punc_stopwords_nltk)

words = [word for word in text['TEXT_cleaned']]
count_words = [len(word.split()) for word in words]
count_words_arr = np.array(count_words)


# ## 6. Syllable count per word

syllable_count = []
vowels=['a','e','i','o','u']
for record in text['TEXT_StopWords_removed']:
    count=0
    for i in record:
        x=re.compile('[es|ed]$')
        if x.match(i.lower()):
            count+=0
        else:
            for j in i:
                if(j.lower() in vowels ):
                    count+=1
    syllables = count
    syllable_count.append(syllables)

syllable_count_arr = np.array(syllable_count)
syllable_count_per_word = np.round(syllable_count_arr/count_words_arr,decimals=4)
print(syllable_count_per_word)


# ## 7. Personal Pronouns

count_pronouns = []
pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us)|(?-i:Us))\b',re.I)
for record in text['TEXT']:
    pronouns = pronounRegex.findall(record)
    count_pronouns.append(len(pronouns))

count_pronouns


# ## 8. AVERAGE WORD LENGTH
# ##### AVERAGE WORD LENGTH = Sum of the total number of characters in each word/Total number of words

count_characters = np.array([len(record) for record in text['TEXT_cleaned']])

average_word_length = (count_characters/count_words_arr)

average_word_length

# Generating output file for all the analysis
Output_Data = pd.DataFrame({"URL_ID":df['URL_ID'],\
                            "URL":df['URL'],\
                            "POSITIVE SCORE":positive_score_arr,\
                            "NEGATIVE SCORE":negative_score_arr,\
                            "POLARITY SCORE":Polarity_Score,\
                            "SUBJECTIVITY SCORE":Subjectivity_score,\
                            "AVG SENTENCE LENGTH":Avg_sent_length,\
                            "PERCENTAGE OF COMPLEX WORDS":percentage_complex_words,\
                            "FOG INDEX":Fog_Index,\
                            "AVG NUMBER OF WORDS PER SENTENCE":Avg_words_per_sentance,\
                            "COMPLEX WORD COUNT":count_complex_words_arr,\
                            "WORD COUNT":count_words_arr,\
                            "SYLLABLE PER WORD":syllable_count_per_word,\
                            "PERSONAL PRONOUNS":count_pronouns,\
                            "AVG WORD LENGTH":average_word_length
                           })

Output_Data.to_excel(' Output_Data.xlsx',index=False)
    