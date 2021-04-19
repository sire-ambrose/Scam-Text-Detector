import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from  sklearn.model_selection import train_test_split
import re
import numpy as np
import pandas as pd
import streamlit as st
st.write("# Scam App")

file=pd.read_csv('fraud_csv1.txt', encoding='latin', error_bad_lines=False)

X_text='send your payment uba 08033926857'

def ego(ste):
    st=ste.split()
    #print(st)
    for i in st:
        #print(i)
        egho1 = re.search(r'\dk', i)
        if egho1:
            st[st.index(i)]='egho'
        else:
            pass
        egho2 = re.search(r'\dK', i)
        if egho2:
            st[st.index(i)]='egho'
        else:
            pass
        call_num1 = re.search(r'8\d\d\d\d\d\d\d\d\d', i)
        if call_num1 :
            
            #print('Yes')
            st[st.index(i)]='phone_number'
        else:
            pass
        call_num2 = re.search(r'9\d\d\d\d\d\d\d\d\d', i)
        if call_num2 :
            st[st.index(i)]='phone_number'
        else:
            pass
        call_num3 = re.search(r'7\d\d\d\d\d\d\d\d\d', i)
        if call_num3 :
            st[st.index(i)]='phone_number'
        else:
            pass
        call_num4 = re.search(r'http', i)
        if call_num4 :
            st[st.index(i)]='web_link'
        else:
            pass
    #result=' '.join(st)
    return ' '.join(st)

def egho_call_num(sheet):
    sheet_array=np.array(sheet)
    try:
        m=sheet_array.shape[0]
        for j in range(m):
            #print(sheet_array[j][0])
            #print(ego(sheet_array[j][0]))
            sheet_array[j][0]=ego(sheet_array[j][0])
        sheet=pd.DataFrame(sheet_array, columns=['message', 'label'])
    except:
        sheet=ego(sheet)
    return sheet

def process_train(file):
    file1=egho_call_num(file)
    count_vector=CountVectorizer(ngram_range=(1,1), lowercase=True, stop_words='english')
    X=count_vector.fit_transform(file1.iloc[:,0])
    X_train=X.toarray()
    y=list(file1.iloc[:,1])
    x_train,x_test,y_train, y_test= train_test_split(X_train,y, test_size=0.2, random_state=2)
    return x_train,x_test,y_train, y_test, count_vector
x_train,x_test,y_train, y_test, count_vector=process_train(file)

def process_input(X_test, count_vector):
    X_test1=egho_call_num(X_test)
    X_test2=count_vector.transform([X_test1])
    X_test3=X_test2.toarray()
    return X_test3
X_test=process_input(X_text, count_vector)

from sklearn.naive_bayes import MultinomialNB
NB=MultinomialNB()
NB.fit(x_train,y_train)
y_pred=NB.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
acc=accuracy_score(y_test, y_pred)*100
con=confusion_matrix(y_test, y_pred)
print('accuracy :',acc)
print('confusion matrix : \n',con)
text=st.text_area(label='Paste The Message')
user=open('dataset.txt', 'a')
user.write(text+'\n')
user.close()
#print(text)
if st.button('Submit'):
    if text != str():
        se=text
        se=process_input(se, count_vector)
        if NB.predict(se)[0]==1 :
            st.write('## SCAM')
            st.write('The text is '+str(round(NB.predict_proba(se)[0][1]*100))+'%'+' scam')
        else :
            st.write('## NOT SCAM')
            st.write('The text is '+str(round(NB.predict_proba(se)[0][0]*100))+'%'+' not scam')
st.write('\n\n\n\n\n\n')
st.write('### Contact Developer : ')
st.write('[Facebook](https://www.facebook.com/profile.php?id=100005064735483)')
st.write('[Github ](https://github.com/sire-ambrose)')
st.write('[Linkedin](https://www.linkedin.com/in/ambrose-ikpele-61643419a)')