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
from Library_functions import *
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

#----------------------------------------------------------------------------------------------------
# Part 2: Build app
#----------------------------------------------------------------------------------------------------
# Overview

st.image('logo_ttth.jpg')

st.title("Trung tâm tin học - ĐH KHTN")
st.header('Data Science')
st.subheader('Sentiment Analysis on ShopeeFood')

# st.video('https://www.youtube.com/watch?v=q3nSSZNOg38&list=PLFTWPHJsZXVVnckL0b3DYmHjjPiRB5mqX')

menu = ['Tổng quan', 'Xây dựng mô hình', 'Dự đoán']
choice = st.sidebar.selectbox('Menu', menu)

#----------------------------------------------------------------------------------------------------
if choice == 'Tổng quan':
#----------------------------------------------------------------------------------------------------
    st.subheader('Tổng quan')
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
elif choice == 'Xây dựng mô hình':
#----------------------------------------------------------------------------------------------------
    st.subheader('Xây dựng mô hình')
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
elif choice == 'Dự đoán':
#----------------------------------------------------------------------------------------------------
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
 
