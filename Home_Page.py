import streamlit as st
import pandas as pd
import seaborn as snsg
from wordcloud import WordCloud

from LTN_Library_Functions import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import datetime
from datetime import date,datetime
import streamlit as st
import io

from underthesea import word_tokenize
import glob
from wordcloud import WordCloud,STOPWORDS
import openpyxl 

# !pip install import-ipynb
import import_ipynb
import re
from underthesea import text_normalize
from pyvi import ViTokenizer

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from wordcloud import WordCloud
from pyvi import ViTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline

import string
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestCentroid

from sklearn.svm import LinearSVC, SVC
from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score
from sklearn.metrics import  precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

#----------------------------------------------------------------------------------------------------
# Support voice
import datetime
import re
import urllib.request as urllib2
from time import strftime

#################
##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

#----------------------------------------------------------------------------------------------------
def tokenize_vietnamese_text(text):
    tokens = ViTokenizer.tokenize(text)
    return tokens

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                               u"\U00002500-\U00002BEF"  # Chinese characters
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text

def replace_variations_of_khong(text):
    variations = ['ko', 'k', 'khong', 'kô', 'hok', 'hông', 'hem', 'k0', 'khg', 'chẳng', 
                  'chả', 'đéo', 'đếch', 'cóc', 'đách', 'éo', 'đâu có', 'đâu']
    for variation in variations:
        text = re.sub(r'\b{}\b'.format(variation), 'không', text)
    return text

def process_special_word(text):
    special_words = ['không', "tuy", "tuy là", "dù là"]
    new_text = ''
    text_lst = text.split()
    i = 0
    while i <= len(text_lst) - 1:
        word = text_lst[i]
        if word in special_words:
            next_idx = i + 1
            if next_idx <= len(text_lst) - 1:
                word = word + '_' + text_lst[next_idx]
            i = next_idx + 1
        else:
            i = i + 1
        new_text = new_text + word + ' '
    return new_text.strip()

def simple_text_clean(dataframe):
    stop_words = stopwords_lst
    
    dataframe['Comment_new'] = dataframe['Comment']

    # Normalize text
    dataframe['Comment_new'] = dataframe['Comment_new'].apply(text_normalize)

    # Remove HTTP links
    dataframe['Comment_new'] = dataframe['Comment_new'].replace(
        r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '',
        regex=True)

    # Remove end of line characters
    dataframe['Comment_new'] = dataframe['Comment_new'].replace(r'[\r\n]+', ' ', regex=True)

    # Remove numbers, only keep letters
    dataframe['Comment_new'] = dataframe['Comment_new'].replace('[\w]*\d+[\w]*', '', regex=True)

    # Remove punctuation
    dataframe['Comment_new'] = dataframe['Comment_new'].replace('[^\w\s]', ' ', regex=True)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        dataframe['Comment_new'] = dataframe['Comment_new'].replace(char, ' ')

    # Remove multiple spaces with one space
    dataframe['Comment_new'] = dataframe['Comment_new'].replace('[\s]{2,}', ' ', regex=True)

    # Some lines start with a space, remove them
    dataframe['Comment_new'] = dataframe['Comment_new'].replace('^[\s]{1,}', '', regex=True)

    # Some lines end with a space, remove them
    dataframe['Comment_new'] = dataframe['Comment_new'].replace('[\s]{1,}$', '', regex=True)

    # Convert to lower case
    dataframe['Comment_new'] = dataframe['Comment_new'].str.lower()
    
        # Replace variations of 'không'
    dataframe['Comment_new'] = dataframe['Comment_new'].apply(replace_variations_of_khong)

    # Remove emojis
    dataframe['Comment_new'] = dataframe['Comment_new'].apply(remove_emoji)

    # Tokenize Vietnamese text
    dataframe['Comment_new'] = dataframe['Comment_new'].apply(tokenize_vietnamese_text)
        # Process special words
    
    dataframe['Comment_new'] = dataframe['Comment_new'].apply(process_special_word)

    # Remove rows that are empty
    dataframe = dataframe[dataframe['Comment_new'].str.len() > 0]

    # Remove stop words
    def remove_stopwords(text):
        text_split = text.split()
        text = [word for word in text_split if word not in stop_words]
        return ' '.join(text)

    dataframe['Comment_new'] = dataframe['Comment_new'].apply(remove_stopwords)

    return dataframe

#----------------------------------------------------------------------------------------------------
def is_open(time_range):
    try:
        start_time, end_time = time_range.split(' - ')
        if start_time.lower() == 'nan' or end_time.lower() == 'nan':
            return False
        start_time = datetime.strptime(start_time, '%H:%M').time()
        end_time = datetime.strptime(end_time, '%H:%M').time()
        if start_time <= current_time <= end_time:
            return True
        if start_time > end_time:  # Trường hợp thời gian qua nửa đêm
            return current_time >= start_time or current_time <= end_time
        return False
    except ValueError:
        return False
#----------------------------------------------------------------------------------------------------
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input, na=False)]

    return df

#----------------------------------------------------------------------------------------------------
def plot_wordcloud(df_col, stopword_list):
    text = ' '.join(df_col.dropna().to_list())
    
    word_cloud = WordCloud(width = 1000, height = 800, background_color ='White',
                        stopwords = stopword_list, min_font_size = 14)
    word_cloud.generate(text)
    plt.figure(figsize = (15 , 9))
    plt.imshow(word_cloud)
    plt.axis("off")
    st.pyplot(plt)
    words_df_dict = word_cloud.words_
    return words_df_dict
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#### HV : LÊ THỐNG NHỨT - NGUYỄN THỊ TƯỜNG VI
#### XÂY DỰNG GUI : Project 3 Sentiment Analysis
#### ĐỒ ÁN TN : Data science 
#----------------------------------------------------------------------------------------------------
# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import datetime
from datetime import date,datetime
import streamlit as st
import io

from underthesea import word_tokenize
import glob
from wordcloud import WordCloud,STOPWORDS
import openpyxl 

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
# !pip install import-ipynb
import import_ipynb
#from NTTV_Library_Functions import *
#----------------------------------------------------------------------------------------------------
# Support voice
import time
import sys
import ctypes
import datetime
import json
import re
import webbrowser
import smtplib
import requests
import urllib
import urllib.request as urllib2
from time import strftime
#import pyaudio
#----------------------------------------------------------------------------------------------------
# Part 1: Build project
#----------------------------------------------------------------------------------------------------
# Load data
print('Loading data.....')
df = df()

# Load model
print('Loading model.....')
# Đọc model LogisticRegression()
model = load_model_Sentiment()
# Đọc model CountVectorizer()
cv = load_model_cv()

# Data pre - processing
#df.drop(['Unnamed: 0'],axis=1,inplace=True)
# remove duplicate
df.drop_duplicates(inplace=True)
# remove missing values
df.dropna(inplace=True)

# split data into train and test
print('Split data into train and test...........')
X=df['comment']
y=df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_cv = cv.transform(X_test)
y_pred = model.predict(X_test_cv)
cm = confusion_matrix(y_test, y_pred)

##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

#----------------------------------------------------------------------------------------------------
def text_transform(comment):
    # # Xử lý tiếng việt thô
    comment = process_text(comment, emoji_dict, teen_dict, wrong_lst)
    # Chuẩn hóa unicode tiếng việt
    comment = covert_unicode(comment)
    # Kí tự đặc biệt
    comment = process_special_word(comment)
    # postag_thesea
    comment = process_postag_thesea(comment)
    #  remove stopword vietnames
    comment = remove_stopword(comment, stopwords_lst)
    return comment 

###########################################################################################################################################################################################################################################
# Tạo các tab
tab1, tab2 = st.tabs(["Lê Thống Nhứt", "Nguyễn Thị Tường Vy"])

# Nội dung cho tab 1 (Lê Thống Nhứt)
with tab1:
    #st.header("Lê Thống Nhứt")
    #st.write("Ứng dụng Streamlit!")
    # Phần xử lý ảnh logo (có thể thay đổi nếu cần)
    try:
        from PIL import Image
        img = Image.open('IMG/logo_ttth.jpg')  
        st.image(img, caption='TRUNG TÂM TIN HỌC - ĐH KHTN')
    except FileNotFoundError:
        st.error("")    

    st.subheader('Sentiment Analysis on ShopeeFood (Projec 3)')

    # Tạo tab bên trong tab1
    subtab1, subtab2, subtab3 = st.tabs(["Tổng quan.", "Xây dựng mô hình.", 'Dự đoán mới.'])

    # Nội dung cho subtab1 (Tổng quan)
with subtab1:
 #----------------------------------------------------------------------------------------------------
    st.subheader('Tổng quan.')
    # st.write('''**Yêu cầu** :
    st.write('''
    - ShopeeFood là một kênh phối hợp với các nhà hàng/quán ăn kinh doanh thực phẩm online.
    - Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhà hàng/ quán ăn hiểu được khách hàng rõ hơn, biết họ đánh giá về mình như thế nào để cải thiện hơn trong dịch vụ/ sản phẩm
    - Đánh giá và nhận xét do người mua cho một sản phẩm là một thông tin quan trọng đối với đối tác thương mại điện tử lớn như Shopee. Những đánh giá sản phẩm này giúp người bán hiểu nhu cầu của khách hàng và nhanh chóng điều chỉnh các dịch vụ của mình để mang lại trải nghiệm tốt hơn cho khách hàng trong đơn hàng tiếp theo
    ''')
    st.write('''**Mục tiêu/ Vấn đề** : Xây dựng mô hình dự đoán giúp nhà hàng/ quán ăn có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ (tích cực, tiêu cực hay trung tính), điều này giúp cho nhà hàng hiểu được tình hình kinh doanh, hiểu được ý kiến của khách hàng từ đó giúp nhà hàng cải thiện hơn trong dịch vụ, sản phẩm.
    ''')
    st.write('''
    **Hướng dẫn chi tiết** :
    - Hiểu được vấn đề
    - Import các thư viện cần thiết và hiểu cách sử dụng
    - Đọc dữ liệu được cung cấp
    - Thực hiện EDA (Exploratory Data Analysis – Phân tích Khám phá Dữ liệu) cơ bản ( sử dụng Pandas Profifing Report )
    - Tiền xử lý dữ liệu : Làm sạch, tạo tính năng mới , lựa chọn tính năng cần thiết....
    ''')
    st.write('''
    **Bước 1** : Business Understanding

    **Bước 2** : Data Understanding ==> Giải quyết bài toán Sentiment analysis trong E-commerce bằng thuật toán nhóm Supervised Learning - Classification : Naive Bayes, KNN, Logictic Regression...

    **Bước 3** : Data Preparation/ Prepare : Chuẩn hóa tiếng việt, viết các hàm xử lý dữ liệu thô...

    **Xử lý tiếng việt** : ''')

    st.write('''
    **1.Tiền xử lý dữ liệu thô** :''')

    st.write('''
    - Chuyển text về chữ thường
    - Loại bỏ các ký tự đặc biệt nếu có
    - Thay thế emojicon/ teencode bằng text tương ứng
    - Thay thế một số punctuation và number bằng khoảng trắng
    - Thay thế các từ sai chính tả bằng khoảng trắng
    - Thay thế loạt khoảng trắng bằng một khoảng trắng''')
    
    st.write('''**2.Chuẩn hóa Unicode tiếng Việt** :''')
    st.write('''**3.Tokenizer văn bản tiếng Việt bằng thư viện underthesea** :''')
    st.write('''**4.Xóa các stopword tiếng Việt** :''')
    st.write('''**Bước 4&5: Modeling & Evaluation/ Analyze & Report**''')
    st.write('''**Xây dựng các Classification model dự đoán**''')
    
    st.write('''
    - Naïve Bayes\n
    - Logistic Regression\n
    - Tree Algorithms…\n
    - Thực hiện/ đánh giá kết quả các Classification model\n
    - R-squared\n
    - Acc, precision, recall, f1,…''')
    
    st.write('''**Kết luận**''')
    st.write('''**Bước 6: Deployment & Feedback/ Act**''')
    st.write('''Đưa ra những cải tiến phù hợp để nâng cao sự hài lòng của khách hàng, thu hút sự chú ý của khách hàng mới''')
    
    st.subheader('Giáo viên hướng dẫn')
    st.write('''
    **Cô : Khuất Thùy Phương**
    ''')
    st.subheader('Học viên thực hiện')
    st.write('''
    **HV : Lê Thống Nhứt & Nguyễn Thị Tường Vy**
    ''')
#----------------------------------------------------------------------------------------------------

##############################################################################################################
   # Nội dung cho subtab2 (Xây dựng mô hình)
    
with subtab2:
 #----------------------------------------------------------------------------------------------------
    st.subheader('Xây dựng mô hình.')
    st.write('#### Tiền xử lý dữ liệu')
    st.write('##### Hiển thị dữ liệu')
    st.table(df.head())
    # plot bar chart for sentiment
    st.write('##### Biểu đồ Bar cho biểu thị tình cảm')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df['Sentiment'].value_counts().index, df['Sentiment'].value_counts().values)
    ax.set_xticks(df['Sentiment'].value_counts().index)
    ax.set_xticklabels(['Tiêu cực', 'Tích cực'])
    ax.set_ylabel('Số lượng bình luận')
    ax.set_title('Biểu đồ Bar cho biểu thị tình cảm')
    st.pyplot(fig)

    ## Negative
    st.write('##### Wordcloud Cho bình luận tiêu cực')
    neg_ratings=df[df.Sentiment==0]
    neg_words=[]
    for t in neg_ratings.comment:
        neg_words.append(t)
    neg_text=pd.Series(neg_words).str.cat(sep=' ')
    ## instantiate a wordcloud object
    wc =WordCloud(
        background_color='black',
        max_words=200,
        stopwords=stopwords_lst,
        width=1600,height=800,
        max_font_size=200)
    wc.generate(neg_text)
    ## Display the wordcloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc,interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)    

    ## Positive
    st.write('##### Wordcloud Cho bình luận tích cực')
    pos_ratings=df[df.Sentiment==1]
    pos_words=[]
    for t in pos_ratings.comment:
        pos_words.append(t)
    pos_text=pd.Series(pos_words).str.cat(sep=' ')
    ## instantiate a wordcloud object
    wc =WordCloud(
        background_color='black',
        max_words=200,
        stopwords=stopwords_lst,
        width=1600,height=800,
        max_font_size=200)
    wc.generate(pos_text)
    ## Display the wordcloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc,interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)    

    st.write('#### Xây dựng mô hình và đánh giá:')
    st.write('##### Confusion matrix')
    st.table(cm)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)    

    st.write('##### Classification report')
    st.table(classification_report(y_test, y_pred, output_dict=True))
  
    st.write('##### Accuracy')
    # show accuracy as percentage with 2 decimal places
    st.write(f'{accuracy_score(y_test, y_pred) * 100:.2f}%')
#----------------------------------------------------------------------------------------------------    
    # Thêm tab "Dự đoán"
with subtab3:
    
    st.subheader('Dự đoán mới')
    st.write('''
    Nhập vào một bình luận và mô hình sẽ dự đoán tình cảm của bình luận. 
    ''')

    menu = ["Nhập bình luận", "Tải tệp Excel", "Tải tệp CSV"]
    choice = st.selectbox("Menu",menu)
    if choice == "Nhập bình luận":
        comment = st.text_input('Nhập vào một bình luận')
        if st.button('Dự đoán'):
            if comment != '':
                comment = text_transform(comment)
                comment = cv.transform([comment])
                y_predict = model.predict(comment)

                if y_predict[0] == 1:
                    st.write('Tình cảm của bình luận là tích cực')
                else:
                    st.write('Tình cảm của bình luận là tiêu cực')
            else:
                st.write('Nhập vào một bình luận')
    elif choice == "Tải tệp Excel":
        st.write('Bạn chọn upload excel')
        uploaded_file = st.file_uploader("Bạn vui lòng chọn file: ")
        if uploaded_file is not None:
            # check file type not excel
            if uploaded_file.name.split('.')[-1] != 'xlsx':
                st.write('File không đúng định dạng, vui lòng chọn file excel')

            elif uploaded_file.name.split('.')[-1] == 'xlsx':

                # load data excel
                df_upload = pd.read_excel(uploaded_file)

                # predict sentiment of review
                # list result
                list_result = []
                for i in range(len(df_upload)):
                    comment = df_upload['review_text'][i]
                    comment = text_transform(comment)
                    comment = cv.transform([comment])
                    y_predict = model.predict(comment)
                    list_result.append(y_predict[0])

                # apppend list result to dataframe
                df_upload['sentiment'] = list_result
                df_after_predict = df_upload.copy()
                # change sentiment to string
                y_class = {0: 'Tình cảm của bình luận là tiêu cực', 1: 'Tình cảm của bình luận là tích cực'}
                df_after_predict['sentiment'] = [y_class[i] for i in df_after_predict.sentiment]

                # show table result
                st.subheader("Result & Statistics :")
                # get 5 first row
                st.write("5 bình luận đầu tiên: ")
                st.table(df_after_predict.iloc[:, [0, 1]].head())
                # st.table(df_after_predict.iloc[:,[0,1]])

                # show wordcloud
                st.subheader("Biểu đồ Wordcloud trích xuất đặc trưng theo nhóm sentiment: ")
                cmt0 = df_after_predict[df_after_predict['sentiment'] == 'Tình cảm của bình luận là tiêu cực']
                cmt1 = df_after_predict[df_after_predict['sentiment'] == 'Tình cảm của bình luận là tích cực']
                cmt0 = cmt0['review_text'].str.cat(sep=' ')
                cmt1 = cmt1['review_text'].str.cat(sep=' ')
                wc0 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
                wc0.generate(cmt0)
                wc1 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
                wc1.generate(cmt1)
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].imshow(wc0, interpolation='bilinear')
                ax[0].axis('off')
                ax[0].set_title('Tình cảm của bình luận là tiêu cực')
                ax[1].imshow(wc1, interpolation='bilinear')
                ax[1].axis('off')
                ax[1].set_title('Tình cảm của bình luận là tích cực')
                st.pyplot(fig)

                # show plot bar chart of sentiment
                st.subheader("Biểu đồ cột thể hiện số lượng bình luận theo nhóm sentiment: ")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax = sns.countplot(x='sentiment', data=df_after_predict)
                st.pyplot(fig)

                # download file excel
                st.subheader("Tải tệp excel kết quả dự đoán: ")
                output = io.BytesIO()
                writer = pd.ExcelWriter(output)
                df_after_predict.to_excel(writer, sheet_name='Sheet1', index=False)
                writer.save()
                output.seek(0)

                st.download_button('Download', data=output, file_name='result_excel.xlsx',
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    elif choice == "Tải tệp CSV" :
        st.write('Bạn chọn upload csv')
        uploaded_file = st.file_uploader("Bạn vui lòng chọn file: ")
        if uploaded_file is not None:
            # check file type if not csv
            if uploaded_file.name.split('.')[-1] != 'csv':
                st.write('File không đúng định dạng, vui lòng chọn file csv')
            elif uploaded_file.name.split('.')[-1] == 'csv':

                # load data csv
                df_upload = pd.read_csv(uploaded_file)
                #df_upload.drop('Unnamed: 0', axis=1, inplace=True)

                df_upload.drop('ID', axis=1, inplace=True)

                # predict sentiment of review
                # list result
                list_result = []
                for i in range(len(df_upload)):
                    comment = df_upload['review_text'][i]
                    comment = text_transform(comment)
                    comment = cv.transform([comment])
                    y_predict = model.predict(comment)
                    list_result.append(y_predict[0])

                # apppend list result to dataframe
                df_upload['sentiment'] = list_result
                df_after_predict = df_upload.copy()
                # change sentiment to string
                y_class = {0: 'Tình cảm của bình luận là tiêu cực', 1: 'Tình cảm của bình luận là tích cực'}
                df_after_predict['sentiment'] = [y_class[i] for i in df_after_predict.sentiment]

                # show table result
                st.subheader("Result & Statistics :")
                # get 5 first row
                st.write("5 bình luận đầu tiên: ")
                st.table(df_after_predict.iloc[:, [0, 1]].head())
                # st.table(df_after_predict.iloc[:,[0,1]])

                # show wordcloud
                st.subheader("Biểu đồ Wordcloud trích xuất đặc trưng theo nhóm sentiment: ")
                cmt0 = df_after_predict[df_after_predict['sentiment'] == 'Tình cảm của bình luận là tiêu cực']
                cmt1 = df_after_predict[df_after_predict['sentiment'] == 'Tình cảm của bình luận là tích cực']
                cmt0 = cmt0['review_text'].str.cat(sep=' ')
                cmt1 = cmt1['review_text'].str.cat(sep=' ')
                wc0 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
                wc0.generate(cmt0)
                wc1 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
                wc1.generate(cmt1)
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].imshow(wc0, interpolation='bilinear')
                ax[0].axis('off')
                ax[0].set_title('Tình cảm của bình luận là tiêu cực')
                ax[1].imshow(wc1, interpolation='bilinear')
                ax[1].axis('off')
                ax[1].set_title('Tình cảm của bình luận là tích cực')
                st.pyplot(fig)

                # show plot bar chart of sentiment
                st.subheader("Biểu đồ cột thể hiện số lượng bình luận theo nhóm sentiment: ")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax = sns.countplot(x='sentiment', data=df_after_predict)
                st.pyplot(fig)

                # download file csv
                st.subheader("Tải tệp csv kết quả dự đoán: ")
                output = io.BytesIO()
                df_after_predict.to_csv(output, index=False)
                output.seek(0)
                st.download_button('Download', data=output, file_name='result_csv.csv', mime='text/csv')
 

        # ... (Phần còn lại của code trong phần "Dự đoán mới")

 
 

##########################################################################################################################################################################
# Nội dung cho tab 2 (Nguyễn Thị Tường Vy) - Tương tự như tab 1
with tab2:
    #st.header("Nguyễn Thị Tường Vy")
    #st.write("Ứng dụng Streamlit!")
    # Phần xử lý ảnh logo (có thể thay đổi nếu cần)
    try:
        from PIL import Image
        img = Image.open('IMG/logo_ttth.jpg')  # Hoặc sử dụng một ảnh khác
        st.image(img, caption='TRUNG TÂM TIN HỌC - ĐH KHTN')
    except FileNotFoundError:
        st.error("")    

    st.subheader('Sentiment Analysis on ShopeeFood (Projec 3)')


    # Tạo tab bên trong tab2
    subtab1, subtab2, subtab3, subtab4 = st.tabs(['Tổng quan Project', 'Xây dựng mô hình', 'Dự đoán bình luận tiêu cực hay tích cực', 'Thông tin nhà hàng'])

    # Nội dung cho subtab1 (Tổng quan)
with subtab1:
    st.subheader('Tổng quan Project')

    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image('IMG/image-3437.png')
    
    # st.write('''**Yêu cầu** :
    st.write('''
    - ShopeeFood là một kênh phối hợp với các nhà hàng/quán ăn kinh doanh thực phẩm online.
    - Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhà hàng/ quán ăn hiểu được khách hàng rõ hơn, biết họ đánh giá về mình như thế nào để cải thiện hơn trong dịch vụ/ sản phẩm
    - Đánh giá và nhận xét do người mua cho một sản phẩm là một thông tin quan trọng đối với đối tác thương mại điện tử lớn như Shopee. Những đánh giá sản phẩm này giúp người bán hiểu nhu cầu của khách hàng và nhanh chóng điều chỉnh các dịch vụ của mình để mang lại trải nghiệm tốt hơn cho khách hàng trong đơn hàng tiếp theo
    ''')
    st.write('''**Mục tiêu/ Vấn đề** : Xây dựng mô hình dự đoán giúp nhà hàng/ quán ăn có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ (tích cực, tiêu cực hay trung tính), điều này giúp cho nhà hàng hiểu được tình hình kinh doanh, hiểu được ý kiến của khách hàng từ đó giúp nhà hàng cải thiện hơn trong dịch vụ, sản phẩm.
    ''')
    st.write('''
    **Hướng dẫn chi tiết** :
    - Hiểu được vấn đề
    - Import các thư viện cần thiết và hiểu cách sử dụng
    - Đọc dữ liệu được cung cấp
    - Thực hiện EDA (Exploratory Data Analysis – Phân tích Khám phá Dữ liệu) cơ bản ( sử dụng Pandas Profifing Report )
    - Tiền xử lý dữ liệu : Làm sạch, tạo tính năng mới , lựa chọn tính năng cần thiết....
    ''')
    st.write('''
    **Bước 1** : Business Understanding

    **Bước 2** : Data Understanding ==> Giải quyết bài toán Sentiment analysis trong E-commerce bằng thuật toán nhóm Supervised Learning - Classification : Naive Bayes, KNN, Logictic Regression...

    **Bước 3** : Data Preparation/ Prepare : Chuẩn hóa tiếng việt, viết các hàm xử lý dữ liệu thô...

    **Xử lý tiếng việt** : ''')

    st.write('''
    **1.Tiền xử lý dữ liệu thô** :''')

    st.write('''
    - Chuyển text về chữ thường
    - Loại bỏ các ký tự đặc biệt nếu có
    - Thay thế emojicon/ teencode bằng text tương ứng
    - Thay thế một số punctuation và number bằng khoảng trắng
    - Thay thế các từ sai chính tả bằng khoảng trắng
    - Thay thế loạt khoảng trắng bằng một khoảng trắng''')
    
    st.write('''**2.Chuẩn hóa Unicode tiếng Việt** :''')
    st.write('''**3.Tokenizer văn bản tiếng Việt bằng thư viện underthesea** :''')
    st.write('''**4.Xóa các stopword tiếng Việt** :''')
    st.write('''**Bước 4&5: Modeling & Evaluation/ Analyze & Report**''')
    st.write('''**Xây dựng các Classification model dự đoán**''')
    
    st.write('''
    - Naïve Bayes\n
    - Logistic Regression\n
    - Tree Algorithms…\n
    - Thực hiện/ đánh giá kết quả các Classification model\n
    - R-squared\n
    - Acc, precision, recall, f1,…''')
    
    st.write('''**Kết luận**''')
    st.write('''**Bước 6: Deployment & Feedback/ Act**''')
    st.write('''Đưa ra những cải tiến phù hợp để nâng cao sự hài lòng của khách hàng, thu hút sự chú ý của khách hàng mới''')
    
    st.subheader('Giáo viên hướng dẫn')
    st.write('''
    **Cô : Khuất Thùy Phương**
    ''')
    st.subheader('Học viên thực hiện')
    st.write('''
    **HV : Lê Thống Nhứt & Nguyễn Thị Tường Vy**
    ''') 

with subtab2:
    st.subheader('Xây dựng mô hình')
        #----------------------------------------------------------------------------------------------------
    Restaurants = pd.read_csv('dataset/1_Restaurants.csv') 
    Reviews = pd.read_csv('dataset/2_Reviews.csv')
    
    # Phần xem tổng quan dữ liệu
    st.subheader('Data Overview')
    if 'Restaurants' in locals() and 'Reviews' in locals():
        st.subheader('1_Restaurants.csv')
        col1, col2 = st.columns(2)
        col1.metric(label="Rows", value=Restaurants.shape[0])
        col2.metric(label="Columns", value=Restaurants.shape[1])
        st.write("#### Restaurants Samples")
        st.dataframe(Restaurants.sample(5))

        st.subheader('2_Reviews.csv')
        col1, col2 = st.columns(2)
        col1.metric(label="Rows", value=Reviews.shape[0])
        col2.metric(label="Columns", value=Reviews.shape[1])
        st.write("#### Reviews Samples")
        st.dataframe(Reviews.sample(5))
        
        st.write("#### Exploratory Data Analysis")
        st.write("""##### Rating """)
        st.image('IMG/rating.png')
        data = {
        "count": [29959],
        "min": [0.0],
        "25%": [6.0],
        "50%": [7.6],
        "75%": [8.8],
        "max": [10.0],
        "mean": [7.089149],
        "median": [7.6],
        "mode": [10.0],
        "std": [2.353754],
        "range": [10.0],
        "iqr": [2.8],
        "var": [5.540158],
        "skew": [-0.956055],
        "kurtosis": [0.329342]
    }
        st.title('Rating Statistical Summary Table')
        st.table(pd.DataFrame(data, index=["Rating"]))
        st.markdown('''#### Nhận xét:
=> các đánh giá tích cực chiếm ưu thế trong mẫu dữ liệu này.

Biến có oulier
                    ''')

        st.write("""##### Phân bố tích cực và tiêu cực (Rating trên và dưới 5)""")
        st.image('IMG/ratingclassify.png')
        
        ## Negative
        st.write('##### Wordcloud Cho bình luận tiêu cực')
        st.image('IMG/wcdislike.png')
        st.image('IMG/wc2_hist.png')
               
        ## Positive
        st.write('##### Wordcloud Cho bình luận tích cực')
        st.image('IMG/wclike.png')
        st.image('IMG/wchist.png')

        
        st.subheader('Build model')
        st.markdown('Biến đổi văn bản thành các đặc trưng TF-IDF, dùng RandomOverSampler upsample traindata, dùng các mô hình classifier để phân lớp tiêu cực và tích cực')
        st.markdown('''| Model                | Train Score | Test Score | Precision | Recall | F1-Score | AUC   |
|----------------------|-------------|------------|-----------|--------|----------|-------|
| Logistic Regression | 91.74%      | 90.29%     | 0.92      | 0.90   | 0.91     | 0.957 |
| Naive Bayes         | 93.02%      | 90.55%     | 0.92      | 0.91   | 0.91     | 0.950 |
| KNN                  | 91.61%      | 90.30%     | 0.92      | 0.90   | 0.91     | 0.957 |
| Decision Tree       | 91.52%      | 90.32%     | 0.92      | 0.90   | 0.91     | 0.957 |

                    ''')
        st.markdown('''Tóm lại, mô hình Naive Bayes có vẻ hoạt động tốt nhất trong bốn mô hình này, với mức độ UAC cao và tốt khi data không cân bằng (cả khi không xử lý và sau khi xử lý cân bằng).
                    ''')
              
        st.subheader('Evaluation')
        st.markdown('''
                    
    Tạo pipeline hoàn chỉnh với upsampling
                        
    pipe_line2_oversampling = ImbPipeline([
    ('preprocessor', preprocessor),  # Xử lý trước dữ liệu
    
    ('oversampler', RandomOverSampler()),  # Upsampling dữ liệu
    
    ('Naive Bayes', MultinomialNB())  # Mô hình phân loại])
                    ''')
        st.image('IMG/pipline.png')
        col1, col2 = st.columns(2)
        col1.metric(label="Train score", value= "93.02%")
        col2.metric(label="Test score", value= "90.55%")
        st.image('IMG/cm.png')
        st.markdown("""
|             | precision | recall | f1-score | support |
|-------------|-----------|--------|----------|---------|
| 0           | 0.67      | 0.87   | 0.76     | 865     |
| 1           | 0.97      | 0.91   | 0.94     | 4216    |
| accuracy    |           |        | 0.91     | 5081    |
| macro avg   | 0.82      | 0.89   | 0.85     | 5081    |
| weighted avg| 0.92      | 0.91   | 0.91     | 5081    |
""")
        st.write('ROC Curse trên tập test')
        st.image('IMG/roc.png')
        st.markdown('Mô hình chạy tốt trên tập dữ liệu, mô hình không bị underfiting / overfiting')
#----------------------------------------------------------------------------------------------------  

    # Nội dung cho subtab2 (Dự đoán bình luận tiêu cực hay tích cự)
with subtab3:
    st.subheader('Dự đoán bình luận tiêu cực hay tích cự')
        
#----------------------------------------------------------------------------------------------------
    st.image('IMG/Untitled design.png')
    st.subheader('Dự đoán bình luận tiêu cực hay tích cực')
    st.write('''
    Hướng dẫn: Chọn phương thức nhập liệu, nhập liệu và dự đoán
    ''')
    from joblib import load
    # Đọc pipeline từ file
    model  = load('model/pipeline_model.joblib')
    # menu = ["Nhập bình luận", "Tải tệp Excel", "Tải tệp CSV", "Bình luận bằng giọng nói", "Nói chuyện với chatGPT"]
    # menu = ["Nhập bình luận", "Tải tệp Excel", "Tải tệp CSV"]
    menu = ["Nhập một bình luận", "Nhập nhiều dòng dữ liệu trực tiếp", 
            "Đăng tệp Excel", "Đăng tệp CSV"]
    choice = st.selectbox("Phương thức nhập liệu",menu)
    if choice == "Nhập một bình luận":
        comment = st.text_input('Nhập vào một bình luận', key='comment_input_1')
        if st.button('Dự đoán', key='predict_button'):
            if comment != '':
                new_df = pd.DataFrame([comment], columns =['Comment'])
                new_df = simple_text_clean(new_df)
                new_df['words'] = [len(x.split(' ')) for x in new_df['Comment']]
                new_df['length'] = [len(x) for x in new_df['Comment']]
                predictions = model.predict(new_df)
                
                if predictions[0] == 1:
                    st.write('Tình cảm của bình luận là tích cực')
                else:
                    st.write('Tình cảm của bình luận là tiêu cực')
            else:
                st.write('Nhập vào một bình luận')
    elif choice == "Nhập nhiều dòng dữ liệu trực tiếp":
        st.subheader("Nhập nhiều dòng dữ liệu trực tiếp")        
        
        num_lines = st.slider('Chọn số dòng để nhập:', min_value=2, max_value=5, value=2)
        new_df = pd.DataFrame(columns=["Comment"])
        default_comments = [
    "Ngon bổ rẻ",
    "Xấu dở mắc",
    "Ngon bổ rẻ",
    "Xấu dở mắc",
    "Ngon bổ rẻ"
]
        for i in range(num_lines):
            default_value = default_comments[i] if i < len(default_comments) else ""
            comment = st.text_area(f"Nhập ý kiến {i+1}:", value=default_value)
            new_df = new_df.append({"Comment": comment}, ignore_index=True)
        st.dataframe(new_df)     
        new_df = simple_text_clean(new_df)
        new_df['words'] = [len(x.split(' ')) for x in new_df['Comment']]
        new_df['length'] = [len(x) for x in new_df['Comment']]
        predictions = model.predict(new_df)
                

        # apppend list result to dataframe
        new_df['predictions'] = predictions
        df_after_predict = new_df.copy()
        # change sentiment to string
        # Mapping for sentiment
        sentiment_map = {0: 'Tiêu cực', 1: 'Tích cực'}
        df_after_predict['Sentiment'] = df_after_predict['predictions'].map(sentiment_map)

                # show table result
        st.subheader("Result & Statistics :")
        # get 5 first row
        st.write("5 bình luận đầu tiên: ")
        st.table(df_after_predict.head())
        st.dataframe(filter_dataframe(df_after_predict[['Comment', 'Sentiment']]))
                # st.table(df_after_predict.iloc[:,[0,1]])

                # show wordcloud
        st.subheader("Biểu đồ Wordcloud trích xuất đặc trưng theo nhóm sentiment: ")
        cmt0 = df_after_predict[df_after_predict['Sentiment'] == 'Tiêu cực']
        cmt1 = df_after_predict[df_after_predict['Sentiment'] == 'Tích cực']
        cmt0 = cmt0['Comment_new'].str.cat(sep=' ')
        cmt1 = cmt1['Comment_new'].str.cat(sep=' ')
        wc0 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
        wc0.generate(cmt0)
        wc1 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
        wc1.generate(cmt1)
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(wc0, interpolation='bilinear')
        ax[0].axis('off')
        ax[0].set_title('Tình cảm của bình luận là tiêu cực')
        ax[1].imshow(wc1, interpolation='bilinear')
        ax[1].axis('off')
        ax[1].set_title('Tình cảm của bình luận là tích cực')
        st.pyplot(fig)

        # show plot bar chart of sentiment
        st.subheader("Biểu đồ cột thể hiện số lượng bình luận theo nhóm sentiment: ")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.countplot(x='Sentiment', data=df_after_predict)
        st.pyplot(fig)

                # download file excel
        st.subheader("Tải tệp csv kết quả dự đoán: ")
        
        csv = df_after_predict.to_csv(index=False).encode('utf-8')
    
        st.download_button('Download',
                           data=csv, file_name='result_csv.csv', mime='text/csv')
          
    
    elif choice == "Đăng tệp Excel":
        st.write('Bạn chọn upload excel')
        uploaded_file = st.file_uploader("Bạn vui lòng chọn file chứa dữ liệu nhận xét")
        if uploaded_file is not None:
            # check file type not excel
            if uploaded_file.name.split('.')[-1] != 'xlsx':
                st.write('File không đúng định dạng, vui lòng chọn file excel')

            elif uploaded_file.name.split('.')[-1] == 'xlsx':

                # load data excel
                df_upload = pd.read_excel(uploaded_file, header=0)
                st.dataframe(df_upload)
                
                # Check if 'Comment' column exists
                if 'Comment' not in df_upload.columns:
                    st.write('Dữ liệu không hợp lệ, cột nào của bạn chứa comment?')
                    comment_column = st.selectbox("Chọn cột chứa comment", df_upload.columns)
                    df_upload = df_upload.rename(columns={comment_column: 'Comment'})
                
                # predict sentiment of review
                # list result
                new_df = df_upload
                new_df = simple_text_clean(new_df)
                new_df['words'] = [len(x.split(' ')) for x in new_df['Comment']]
                new_df['length'] = [len(x) for x in new_df['Comment']]
                predictions = model.predict(new_df)
                

                # apppend list result to dataframe
                df_upload['predictions'] = predictions
                df_after_predict = df_upload.copy()
                # change sentiment to string
                # Mapping for sentiment
                sentiment_map = {0: 'Tiêu cực', 1: 'Tích cực'}
                df_after_predict['Sentiment'] = df_after_predict['predictions'].map(sentiment_map)

                # show table result
                st.subheader("Result & Statistics :")
                # get 5 first row
                st.write("5 bình luận đầu tiên: ")
                st.table(df_after_predict.head())
                st.dataframe(filter_dataframe(df_after_predict[['Comment', 'Sentiment']]))
                # st.table(df_after_predict.iloc[:,[0,1]])

                # show wordcloud
                st.subheader("Biểu đồ Wordcloud trích xuất đặc trưng theo nhóm sentiment: ")
                cmt0 = df_after_predict[df_after_predict['Sentiment'] == 'Tiêu cực']
                cmt1 = df_after_predict[df_after_predict['Sentiment'] == 'Tích cực']
                cmt0 = cmt0['Comment_new'].str.cat(sep=' ')
                cmt1 = cmt1['Comment_new'].str.cat(sep=' ')
                wc0 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
                wc0.generate(cmt0)
                wc1 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
                wc1.generate(cmt1)
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].imshow(wc0, interpolation='bilinear')
                ax[0].axis('off')
                ax[0].set_title('Tình cảm của bình luận là tiêu cực')
                ax[1].imshow(wc1, interpolation='bilinear')
                ax[1].axis('off')
                ax[1].set_title('Tình cảm của bình luận là tích cực')
                st.pyplot(fig)

                # show plot bar chart of sentiment
                st.subheader("Biểu đồ cột thể hiện số lượng bình luận theo nhóm sentiment: ")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax = sns.countplot(x='Sentiment', data=df_after_predict)
                st.pyplot(fig)

                # download file csv
                st.subheader("Tải tệp csv kết quả dự đoán: ")
                output = io.BytesIO()
                df_after_predict.to_csv(output, index=False)
                output.seek(0)
                st.download_button('Download', data=output, file_name='result_csv.csv', mime='text/csv')

    elif choice == "Đăng tệp CSV" :
        st.write('Bạn chọn upload csv')
        uploaded_file = st.file_uploader("Bạn vui lòng chọn file: ")
        if uploaded_file is not None:
            # check file type if not csv
            if uploaded_file.name.split('.')[-1] != 'csv':
                st.write('File không đúng định dạng, vui lòng chọn file csv')
            elif uploaded_file.name.split('.')[-1] == 'csv':

                # load data csv
                df_upload = pd.read_csv(uploaded_file)
                df_upload.drop('Unnamed: 0', axis=1, inplace=True)

                # predict sentiment of review
                # Check if 'Comment' column exists
                if 'Comment' not in df_upload.columns:
                    st.write('Dữ liệu không hợp lệ, cột nào của bạn chứa comment?')
                    comment_column = st.selectbox("Chọn cột chứa comment", df_upload.columns)
                    df_upload = df_upload.rename(columns={comment_column: 'Comment'})
                
                # predict sentiment of review
                # list result
                new_df = df_upload
                new_df = simple_text_clean(new_df)
                new_df['words'] = [len(x.split(' ')) for x in new_df['Comment']]
                new_df['length'] = [len(x) for x in new_df['Comment']]
                predictions = model.predict(new_df)

                # apppend list result to dataframe
                df_upload['predictions'] = predictions
                df_after_predict = df_upload.copy()
                # change sentiment to string
                # Mapping for sentiment
                sentiment_map = {0: 'Tiêu cực', 1: 'Tích cực'}
                df_after_predict['sentiment'] = df_after_predict['predictions'].map(sentiment_map)
                # show table result
                st.subheader("Result & Statistics :")
                # get 5 first row
                st.write("5 bình luận đầu tiên: ")
                st.table(df_after_predict.head())
                st.dataframe(filter_dataframe(df_after_predict[['Comment', 'sentiment']]))

                # show wordcloud
                st.subheader("Biểu đồ Wordcloud trích xuất đặc trưng theo nhóm sentiment: ")
                cmt0 = df_after_predict[df_after_predict['sentiment'] == 'Tiêu cực']
                cmt1 = df_after_predict[df_after_predict['sentiment'] == 'Tích cực']
                cmt0 = cmt0['Comment_new'].str.cat(sep=' ')
                cmt1 = cmt1['Comment_new'].str.cat(sep=' ')
                wc0 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
                wc0.generate(cmt0)
                wc1 = WordCloud(background_color='black', max_words=200, stopwords=stopwords_lst, width=1600,
                                height=800, max_font_size=200)
                wc1.generate(cmt1)
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].imshow(wc0, interpolation='bilinear')
                ax[0].axis('off')
                ax[0].set_title('Tình cảm của bình luận là tiêu cực')
                ax[1].imshow(wc1, interpolation='bilinear')
                ax[1].axis('off')
                ax[1].set_title('Tình cảm của bình luận là tích cực')
                st.pyplot(fig)

                # show plot bar chart of sentiment
                st.subheader("Biểu đồ cột thể hiện số lượng bình luận theo nhóm sentiment: ")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax = sns.countplot(x='sentiment', data=df_after_predict)
                st.pyplot(fig)

                # download file csv
                st.subheader("Tải tệp csv kết quả dự đoán: ")
                output = io.BytesIO()
                df_after_predict.to_csv(output, index=False)
                output.seek(0)
                st.download_button('Download', data=output, file_name='result_csv.csv', mime='text/csv')

#----------------------------------------------------------------------------------------------------


    # Nội dung cho subtab2 (Dự đoán bình luận tiêu cực hay tích cự)
with subtab4:
    st.subheader('Thông tin nhà hàng')
        
    df = pd.read_csv('dataset/data1.csv')
    df2 = pd.read_csv('dataset/data2.csv')
    cols = ['IDRestaurant', 'Restaurant', 'Address', 'Time', 'Price', 'District','RatingNo', 'avgRating']
    st.subheader('Thông tin nhà hàng')
    
    st.subheader('Dữ liệu nhà hàng')
    col1, col2 = st.columns(2)
    col1.metric(label="Số cửa hàng", value=df2.shape[0])
    col2.metric(label="Số quận", value=df2['District'].nunique())
    st.write("#### Restaurants Samples")
    st.dataframe(df2[cols].sample(5))
    st.image('IMG/wc_rests.png')
    st.image('IMG/no-quan.png')
    st.image('IMG/price_quan.png')
    st.image('IMG/like-dislike-years.png')
    
    
    from datetime import datetime
    current_time = st.time_input("Nhập giờ")
    # Lọc các cửa tiệm còn mở cửa
    df2['is_open1'] = df2['Time1'].apply(is_open)
    df2['is_open2'] = df2['Time2'].apply(is_open)
    df2['is_open3'] = df2['Time3'].apply(is_open)
    open_stores = df2[(df2['is_open1']|df2['is_open2']|df2['is_open3'])] 

    st.write('Có ' + str(len(open_stores))+ ' cửa hàng còn mở cửa')
    st.dataframe(filter_dataframe(open_stores[cols]))
      
    # Thống kê số lượng comment và số lượng "like" theo năm của cửa hàng 953
    default_ID = 953
    selected_ID = st.number_input('Nhập Restaurant ID', min_value=df2.IDRestaurant.min(), max_value=df2.IDRestaurant.max(), value=default_ID, step=1)
    name = df[df['IDRestaurant']==selected_ID]['Restaurant'].unique()
    st.write('Thông tin nhà hàng')
    st.table(df2[df2['IDRestaurant']==selected_ID][cols])
    
     
    if len(name) > 0:
        name = name[0]
        comment_count = df[df['IDRestaurant']==selected_ID].groupby('Year').size()
        like_count = df[(df['Rating_class'] == 'Like')&(df['IDRestaurant']==selected_ID)].groupby('Year').size()
        dislike_count = df[(df['Rating_class'] == 'Dislike')&(df['IDRestaurant']==selected_ID)].groupby('Year').size()
        
        # Vẽ biểu đồ
        if not comment_count.empty and not like_count.empty and not dislike_count.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(comment_count.index, comment_count.values, marker='o', linestyle='-', label='Tổng số lượng comment')
            plt.plot(like_count.index, like_count.values, marker='o', linestyle='-', label='Số lượng "like"', color='green')
            plt.plot(dislike_count.index, dislike_count.values, marker='o', linestyle='-', label='Số lượng "dislike"', color='red')
            plt.title(f'Số lượng comment, "like" và dislike theo năm của cửa hàng\n {name}')
            plt.xlabel('Năm')
            plt.ylabel('Số lượng')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.write(f"Không có dữ liệu để vẽ biểu đồ cho nhà hàng {name}")
        
        comment_count2 = df[df['IDRestaurant']==953].groupby('Month').size()
        like_count2 = df[(df['Rating_class'] == 'Like')&(df['IDRestaurant']==953)].groupby('Month').size()
        dislike_count2 = df[(df['Rating_class'] == 'Dislike')&(df['IDRestaurant']==953)].groupby('Month').size()
        if not comment_count.empty and not like_count.empty and not dislike_count.empty:
            # Vẽ biểu đồ
            plt.figure(figsize=(10, 6))
            plt.plot(comment_count2.index, comment_count2.values, marker='o', linestyle='-', label='Tổng số lượng comment')
            plt.plot(like_count2.index, like_count2.values, marker='o', linestyle='-', label='Số lượng "like"', color='green')
            plt.plot(dislike_count2.index, dislike_count2.values, marker='o', linestyle='-', label='Số lượng "dislike"', color='red')
            plt.title(f'Số lượng comment, "like" và dislike theo tháng của cửa hàng\n {name}')
            plt.xlabel('Năm')
            plt.ylabel('Số lượng')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt)
            
        st.write('Những nhận xét tiêu cực')     
        plot_wordcloud(df[df.Rating_label==0]['Comment_new'], stopwords_lst) 
        st.write('Những nhận xét tích cực')    
        plot_wordcloud(df[df.Rating_label==1]['Comment_new'], stopwords_lst)    
    
    else:
        st.write("Nhà hàng không tồn tại hoặc không có dữ liệu cho nhà hàng này.")
        




