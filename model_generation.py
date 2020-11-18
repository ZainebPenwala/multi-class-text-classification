import re
import pandas as pd 
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle


def pre_process(data):
    
    # cleaning the data by removing special characters, spaces and numbers
    pattern = r'[0-9]'
    parse = re.sub(pattern, '' ,data).replace('.','').replace('_','').replace('Agent', '').replace('Customer','')
    clean = re.sub('[\(\[].*?[\)\]]','', parse).replace('{','').replace('}','')

    # tokenization
    tokens = word_tokenize(clean)

    # removing stopwords
    stop_words = set(stopwords.words('english'))
    text_without_stopwords = [t for t in tokens if t not in stop_words]
    # print(text_without_stopwords)

    # lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)
    lemmatized_text = ' '.join(lemmatize_words(text_without_stopwords))

    return lemmatized_text

with open('metadata/mapping_conv_topic.train.txt') as f:
    lines = f.readlines()
    # print(lines)

df = pd.DataFrame(columns=['tag', 'body'])

tag = []
body = []

for line in lines:
    topic = line.split('"')[1:-1]
    # print(label)
    file_num = line.split('"')[0].replace(' ','')
    path = 'tagging_test/trans.' + file_num + '.txt'
    # print(path)
    data = open(path).read()
    cleaned_data = pre_process(data)
    tag.append(topic)
    body.append(cleaned_data)
    

df['tag'] = tag
df['body'] = body

df['tag'] = df['tag'].str[0]
# print(df)

# generating a csv which contains labels and cleaned/ processed features
# df.to_csv('train_data.csv', index=False)

col = ['tag', 'body']
df = df[col]

df = df[pd.notnull(df['body'])]

df.columns = ['tag', 'body']

df['category_id'] = df['tag'].factorize()[0]
category_id_df = df[['tag', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'tag']].values)
# print(df.head)


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.body).toarray()
labels = df.category_id
# print(labels)
# print(features.shape)

# building and training model using navie bayesian

X_train, X_test, y_train, y_test = train_test_split(df['body'], df['tag'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

filename = "countvect_model.pkl"
with open(filename, 'wb') as file:
    pickle.dump(count_vect, file)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

# print(clf.predict(count_vect.transform(["hi credit card oh well husband gotten problem credit card n't handle well tend run maximum ask jeez 're 're really bad u um matter fact 've gotten rid credit card except MasterCard Visa pretty much keep maxed oh going say sound like like pretty many see meant individual store yeah cre- probably twenty credit card um-hum um-hum um American Express American Express Gold Optima different department store um two Visas two MasterCards oh gee mean time anybody would know send application know preapproved whatever went took really ended getting u real serious trouble oh sure um see credit card people offer um-hum um-hum um use send annual fee also right exactly American Express probably wo- worst um Gold card ca n't remember much n't even long gave um-hum know Green card like thirty five forty five dollar year yeah 'm forty five right um lot mean pay full every month exactly know 's really resented fact charging card sent back mine except one free uh-huh um also limited one one MasterCard one one Visa store accept actually got Discover card since pay back um-hum yeah Discover one one group ended um closing uh um really n't choice though mean ended ended actually losing credit card um ended going credit counseling service oh um-hum um-hum um husband n't handle credit well um-hum yeah um automatically start service close account oh see 'm still paying account account closed yeah yeah would prefer keep way 're paid um-hum um would prefer one MasterCard one Visa 's yeah yeah actually pay card every month life paid think 's really good way handle way paid every month never worry well much pay guy know month know buy afford mean handle like would like check cash um-hum right um-hum um-hum um 's lot easier keep check well also sense 're giving loan month yeah know smarter would amount saving get interest n't um-hum yeah thought adding know X percent price buy ca n't accept yeah think point time economy way um think um 's going get even worse um-hum um 'm 'm glad 're starting pay debt o- um know s- s- mean started last year um-hum um-hum probably still another year go 're completely hole um-hum think way whole economy 's going right 's good debt 's right 's right yeah yeah would would scary feeling know know juggling payment different people um-hum guess never 's know 'm rich anything 's uh mental concept yeah yeah well whe- 's 're getting rid whole credit card cycle really get mind set got really good juggling money basically robbing Peter pay Paul scary good uh-huh uh-huh thought right mean got stop eventually 'd end catching catching hit face right right um sound like know learned 're coming right yeah 've learned lot um 've learned credit card extremely dangerous hand um-hum um-hum um-hum husband 's 's 's way um-hum n't n't think well w- buy credit n't concept much money 're spending um-hum bill come um-hum a- sudden look bill go oh gosh spent much um-hum fact still two card use judiciously yeah pretty much use emergency type thing like um transmission fell car uh yeah paid new transmission um-hum um thing like um t- try use know purchase incidental type thing like know gas stuff like um-hum um-hum clothing sale find pain yeah yeah 've 've gotten 've gotten lot away credit card i- pay check lot um-hum well sound good know like said sound like 're 've really got control i- admire um uh mind set even getting know past thirty day i- would 's ideally would like um-hum um-hum um-hum yeah like said happened could n't son 's tuition came due guess know really counted quite point um yeah n't like uncomfortable know mean um-hum imagine thought lot sixty dollar interest something couple month 'm going jeez 's outrageous know retrospect n't much mean lot people pay lot yeah oh yeah know still mean figured n't need item cost sixty dollar um-hum think buy sixty dollar exactly grocery week yeah yeah yeah i- think money 've spent interest um-hum credit card 's incredible use feel badly could take income tax 've never able take income tax 'm 'm fairly newly married 've married um le two year um-hum never owned home anything never deduction right right really always money thrown away yeah jeez thrown away never really thought uh good yeah good kind got stuck back mind never um-hum um became real problem sudden came insurmountable problem yeah suspect know thinking looking friend number credit card use know amount know buy guess probably lot similar situation know n't talk yeah people n't like talk money feel uncomfortable think um-hum um-hum um-hum know mean guess lot people personal personal um-hum yeah especially n't feel 're handling quite right somebody might make fun would yeah i- think average American probably pretty heavily debt um-hum including like mortgage mean mortgage understandable debt um-hum um 's roof head right would much rather home renting like 're mean 're basically know kind throwing seven hundred dollar month away um-hum 're renting house um-hum know right way credit card situation 's nobody world 's going give u mortgage house right right suppose prove paying next year 'll good shape yeah yeah 's 's going 's going take quite um hopefully within five year 'll home um-hum um-hum um 'm really counting real soon um-hum um-hum 's 's even harder lesson affected yeah"])))

# saving or pickling the trained model

filename = "trained_model.pkl"
with open(filename, 'wb') as file:
    pickle.dump(clf, file)
