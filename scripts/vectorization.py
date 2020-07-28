'''
Attempting more complex analysis on categories which are not prone to phrase detection
Using Spacy's en_core_web_lg model for vector embeddings

'''

import spacy
import pickle
import pandas as pd
import numpy as np

import xgboost as xgb

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

nlp = spacy.load("en_core_web_lg")

folds = ['train', 'dev', 'test']
label_sentence_map = {}

for fold in folds:
    label_sentence_map[fold] = pickle.load(open('objects/starter_label_sentence_map_' + fold + '.pkl', 'rb'))

rel_classes = ['Causal_Oversimplification', 'Red_Herring', 'Obfuscation,Intentional_Vagueness,Confusion', 'Straw_Men']


target_class = rel_classes[0]

def get_dataframe(target_classes, folds=['train'], max_samples=100):
    '''
    Make feature and target label dataframes for the target classes and folds, with max_samples samples for each target/fold combo
    '''
    data = {}
    for fold in folds:
        for t,target in enumerate(target_classes):
            for i,positive in enumerate(label_sentence_map[fold][target][:max_samples]):
                proc = nlp(positive)
                vec = proc.vector
                id = "P_{}_{}_{}".format(target,fold,i)
                data[id] = [vec, target]

        for i,negative in enumerate(label_sentence_map[fold]['Null_Label'][:max_samples]):
            proc = nlp(negative)
            vec = proc.vector
            id = "N_{}_{}".format(fold,i)
            data[id] = [vec, 'Not Propaganda']

    n_features = 300 # spacy vector feature count

    df = pd.DataFrame.from_dict(data, orient='index', columns=['Vec', 'Label'])

    features = ['F{}'.format(i) for i in range(n_features)]
    df[features] = pd.DataFrame(df.Vec.tolist(),index=df.index)
    df.drop('Vec', axis=1, inplace=True)

    X = df.loc[:, features].values
    y = df.loc[:,['Label']].values

    return X,y


X_train, y_train = get_dataframe(rel_classes, folds=['train'])
X_test,y_test = get_dataframe(rel_classes, folds=['test', 'dev'], max_samples = 15)

for target_class in rel_classes + ['Null_Label']:
    print(f"Train support for class {target_class}: {len(y_train[y_train == target_class])}")


cv_params = {'min_child_weight': [1,3,5], 'max_depth':[1,3,5,7,9]}    # grid search params
fix_params = {'objective': 'multi:softmax', 'learning_rate':.2, 'n_estimators':500, 'num_class':len(np.unique(y_train))} # fixed params (previously searched)

csv = GridSearchCV(xgb.XGBClassifier(**fix_params), cv_params, scoring='precision_macro', cv = 3) # optimizing for precision
csv.fit(X_train, y_train)

print(csv.best_params_)

bp = csv.best_params_
for key in fix_params.keys():
    bp[key] = fix_params[key]

model = xgb.XGBClassifier(**bp)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

'''

# model = xgb.XGBClassifier(objective='multi:softmax', learning_rate=.2, max_depth=1, min_child_weight=4,n_estimators=500, num_class=len(np.unique(y_train)))

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principal = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principal, columns = ['principal component 1',
                                                        'principal component 2'])
df.reset_index(inplace=True, drop=True)
finalDf = pd.concat([principalDf, df[['Label']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1]
colors = ['g','r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)


ax.legend(targets)
ax.grid()
plt.show()
'''

'''
for test_class in rel_classes:
    print(f"Evalating class {test_class}")
    rel = label_sentence_map['train'][test_class][:50]
    nonrel = label_sentence_map['train']['Null_Label'][:50]

    test = label_sentence_map['test'][test_class] + label_sentence_map['dev'][test_class]
    val = label_sentence_map['test']['Null_Label'] + label_sentence_map['dev']['Null_Label']

    for i,sent in enumerate(test):
        sent = nlp(sent)
        rel_scores = []
        nonrel_scores = []

        for j,check in enumerate(rel):
            check = nlp(check)
            rel_scores.append(check.similarity(sent))

        for j,check in enumerate(nonrel):
            check = nlp(check)
            nonrel_scores.append(check.similarity(sent))

        #print(f"Recall target: {round(max(rel_scores),3)}. Null: {round(max(nonrel_scores), 3)}.")
        print(f"\tRecall Correct? {max(rel_scores) > max(nonrel_scores)}")

    for i,sent in enumerate(val[:10]):
        sent = nlp(sent)
        rel_scores = []
        nonrel_scores = []

        for j,check in enumerate(rel):
            check = nlp(check)
            rel_scores.append(check.similarity(sent))

        for j,check in enumerate(nonrel):
            check = nlp(check)
            nonrel_scores.append(check.similarity(sent))

        #print(f"Precision target: {round(max(rel_scores),3)}. Null: {round(max(nonrel_scores), 3)}.")
        print(f"\tPrecision Correct? {max(rel_scores) < max(nonrel_scores)}")
'''
