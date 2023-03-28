import numpy as np 
import pandas as pd 

# use of nlp lib to extract skills from job description
def nlp(df):   

    import string
    import nltk
    from nltk.corpus import stopwords
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    %matplotlib inline
    from textblob import Word
    import nltk


    df['technical_skills'] = df['technical_skills'].apply(lambda x: " ".join(x.lower()for x in x.split()))
    ## remove tabulation and punctuation
    df['technical_skills'] = df['technical_skills'].str.replace('[^\w\s]',' ')
    ## digits
    df['technical_skills'] = df['technical_skills'].str.replace('\d+', '')

    #creating job_description aggregate and storing it into variable name jda 
    ## jda stands for job description aggregated
    jda = df.groupby(['Job_title'])['technical_skills'].sum().reset_index()
    # print("Aggregated job descriptions: ")
    # print(jda)

    #visualizing the data

    jobs_list = jda.job_title.unique().tolist()
    for job in jobs_list:

        # Start with one review:
        text = jda[jda.job_title == job].iloc[0]['technical_skills']
        # Create and generate a word cloud image:
        wordcloud = WordCloud().generate(text)
        print("\n***",job,"***\n")
        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()



    #remove stop words
    stop_words = ['junior', 'senior','experience','etc','job','work','company','technique', 'mumbai', 'global', 'risk', 'bank', 'internal', 'team', 'client', 'knowledge', 'quality', 'process', 'job', 'detail', 'enterprise', 'culture', 
                        'candidate','skill','skills','language','menu','inc','new','plus','years','development', 'process', 'work', 'tax', 'audit', 'financial', 'operational', 'experience', 
                    'technology','organization','ceo','cto','account','manager','data','scientist','mobile','competence', 'siemens',
                        'developer','product','revenue','strong']

    df['technical_skills'] = df['technical_skills'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

    ## lemmatization
    df['technical_skills'] = df['technical_skills'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    print("Preprocessed data: \n")
    print(df.head())

    ## Converting text to features 
    vectorizer = TfidfVectorizer()
    #Tokenize and build vocabulary
    X = vectorizer.fit_transform(df['technical_skills'])
    y = df.job_title

    # split data into 80% training and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 


    # Fit model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    ## Predict
    y_predicted = clf.predict(X_test)

    #evaluate the predictions
    print("Accuracy score is: ",accuracy_score(y_test, y_predicted))
    print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
    print("Confusion Matrix: ")

    print(confusion_matrix(y_test, y_predicted))
    print("Classification Report: ")
    print(classification_report(y_test, y_predicted))

    from textblob import TextBlob
    technical_skills = ['python', 'c','r', 'c++','java','hadoop','scala','flask','pandas','spark','scikit-learn',
                        'numpy','php','sql','mysql','css','mongdb','nltk','fastai' , 'keras', 'pytorch','tensorflow',
                    'linux','Ruby','JavaScript','django','react','reactjs','ai','ui','tableau']
    feature_array = vectorizer.get_feature_names()
    # number of overall model features
    features_numbers = len(feature_array)
    ## max sorted features number
    n_max = int(features_numbers * 0.1)

    ##initialize output dataframe
    output = pd.DataFrame()
    for i in range(0,len(clf.classes_)):
        print("\n****" ,clf.classes_[i],"****\n")
        class_prob_indices_sorted = clf.feature_log_prob_[i, :].argsort()[::-1]
        raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
        print("list of unprocessed skills :")
        print(raw_skills)
        
        ## Extract technical skills
        top_technical_skills= list(set(technical_skills).intersection(raw_skills))[:6]
        #print("Top technical skills",top_technical_skills)
        
        output = output.append({'job_title':clf.classes_[i],
                            'technical_skills':top_technical_skills},
                        ignore_index=True)

    job_data = df.merge(output,on='job_title')
    job_data['technical_skills'] = job_data['technical_skills'].apply(lambda x: ",".join(x))
    dividing_into_class(job_data)


#Dividing the Dataset into Four Classes (i.e. class 1, class 2, class 3, class 4) by using KMeans Clustering


def dividing_into_class(data):
#importing Kmeans from Sklearn.cluster library

    from sklearn.cluster import KMeans
    #n_cluster value is 4 
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data[['employees', 'followers']])
    class_ = kmeans.predict(data[['employees', 'followers']])
    print(class_)
    data['job class']= class_
    return data



df = pd.read_csv(r'C:\Users\MUHAMMED ASU\OneDrive\Desktop\ML project\ml project\full_data.csv')
job_data = nlp(df)
job_data.to_csv('final_job_data.csv')