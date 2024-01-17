import os
import PyPDF2
import nltk
from gensim import corpora, models
import re

folder_path = r"C:\Users\kkhha\OneDrive\Desktop\null\pdf\pp"

documents = []
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()
            documents.append(text)
    except PyPDF2.utils.PdfReadError:
        print(f"not found file path: {file_path}")

stop_words = [
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could',
    "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's",
    'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm",
    "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't",
    'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't",
    'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't",
    'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",
    'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
    'yourselves'
]

texts_without_stopwords = []
for document in documents:
    words = document.lower().split()
    words_without_stopwords = [word for word in words if word not in stop_words]
    texts_without_stopwords.append(words_without_stopwords)

dictionary = corpora.Dictionary(texts_without_stopwords)
corpus = [dictionary.doc2bow(text) for text in texts_without_stopwords]

lda_model = models.LdaModel(corpus, num_topics=6, id2word=dictionary, passes=2)

num_topics_to_print = 6
for index, topic in lda_model.print_topics(num_topics=num_topics_to_print):
    topic_text = re.sub(r'\b(?:{})\b'.format('|'.join(stop_words)), '', topic)
    print(f"Topic {index + 1}: {topic_text}")

new_document = "new text"
new_text = new_document.lower().split()  
new_text_without_stopwords = [word for word in new_text if word not in stop_words]
new_bow = dictionary.doc2bow(new_text_without_stopwords)
topic_distribution = lda_model.get_document_topics(new_bow)
print(topic_distribution)
