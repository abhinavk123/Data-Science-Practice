#using stemming instead of lemmatization

from nltk.stem import PorterStemmer
ps = PorterStemmer() 

count = 0
for i in range(0,2100000,batch):
    print("Iteration number:",count/batch)
    fin = []
    for j in range(count,count+batch):
        if j%1000 ==0:
            print(j)
        Final_words = []
        if len(df.loc[j,"review"]) > 75:
            text = df.loc[j,"review"][:75]
        else:
            text = df.loc[j,"review"]
        for word in text:
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word = word.lower()
                word_Final = ps.stem(word)
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'reviewText_final'
        fin.append(str(Final_words))
    df.loc[coutn:count_batch-1,'reviewText_final'] = fin
    count+=batch
