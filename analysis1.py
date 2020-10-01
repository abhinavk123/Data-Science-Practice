REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

#removing the blank rows if any
print("removing empty rows...")
df['review'].dropna(inplace=True)

#tokenise the data samples
print("tokenising reviews...")
total=len(df)
batch = 100000
count = 0
for i in range(0,total,batch):
    print("Iteration number:",count/batch)
    tokenised =[]
    for j in range(count,count+batch):
        tokenised.append(word_tokenize(df.loc[j,"review"]))
    df.loc[count:count+batch-1,"review"] = tokenised
    count = count+batch
