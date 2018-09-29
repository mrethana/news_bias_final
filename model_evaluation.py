from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import plotly.plotly as py
import plotly.graph_objs as go
from d2v_func import *
import itertools
from imblearn.over_sampling import SMOTE, ADASYN
from pactools import simulate_pac
from pactools.grid_search import GridSearchCVProgressBar
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

Bigram_DBOW= Doc2Vec.load("D2V_models/d2v.model")
Bigram_DMM = Doc2Vec.load("D2V_models/d2v_DMM.model")
Bigram_DMC = Doc2Vec.load("D2V_models/d2v_DMC.model")
Bigram_DBOW200 = Doc2Vec.load("D2V_models/d2v_dbow200.model")
Trigram_DBOW = Doc2Vec.load("D2V_models/TRI_d2v_dbow100.model")
Trigram_DMC = Doc2Vec.load("D2V_models/TRI_d2v_DMC.model")
Trigram_DMM = Doc2Vec.load("D2V_models/TRI_d2v_DMM.model")
All_models = [Bigram_DBOW,Bigram_DMC,Bigram_DMM, Bigram_DBOW200, Trigram_DBOW, Trigram_DMC, Trigram_DMM]


model_name_match = {'Bigram_DMC':(Bigram_DMC),
 'Trigram_DMM':(Trigram_DMM),
 'Trigram_DBOW_DMM':(Trigram_DBOW,Trigram_DMM),
 'Trigram_DMC':(Trigram_DMC),
 'Bigram_DBOW_DMM':(Bigram_DMC,Bigram_DMM),
 'Bigram_DBOW_200':(Bigram_DBOW200),
 'Bigram_DBOW_DMC':(Bigram_DBOW,Bigram_DBOW),
 'Bigram_DMM':(Bigram_DMM),
 'Trigram_DBOW':Trigram_DBOW,
 'Bigram_DBOW':Bigram_DBOW}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return cm


def get_confusion_metrics(y_actual, y_pred):
    class_names = sorted(list(set(y_actual)))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_actual, y_pred)
    np.set_printoptions(precision=2)

    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    plt.figure()
    metrics = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    print('Classification Report')
    cnf_matrix=classification_report(y_actual, y_pred, target_names=class_names)
    print (cnf_matrix)
    return metrics, cnf_matrix

def get_all_vectors_and_data(broader):
    if broader == False:
        all_vectors_df = pd.read_csv('Archive_CSV/all_vectors.csv',index_col=0)
        full_df = pd.read_csv('Archive_CSV/ALL_rows_scraped.csv', index_col=0)
        return full_df, all_vectors_df
    else:
        all_vectors_df = pd.read_csv('../Archive_CSV/all_vectors.csv',index_col=0)
        full_df = pd.read_csv('../Archive_CSV/ALL_rows_scraped.csv', index_col=0)
        return full_df, all_vectors_df

def get_true_vectors(model_name, all_vectors_df, full_df):
    true_vectors = all_vectors_df[(all_vectors_df.model_name == model_name)]
    true_vectors = true_vectors.dropna(axis = 1)
    true_vectors = true_vectors.reset_index(drop=True)
    true_vectors['labels'] = full_df.label
    holder = true_vectors.pop('model_name') # remove column modelname and store it in holder
    true_vectors['model_name'] = holder
    return true_vectors



def pull_corresponding_classifier_model(model_name, model_name_dict):
    model = joblib.load(open(model_name_dict[model_name][0], 'rb'))
    return model

def pull_corresponding_classifier_grid_search(model_name, model_name_dict):
    df = pd.read_csv(model_name_dict[model_name][1], index_col=0)
    return df

def infer_vecs_for_val(d2v_model, d2v_model_name, model_name_dict):
    val_df = pd.read_csv('Archive_CSV/validation_set.csv',index_col=0)
    full_df, all_vectors_df = get_all_vectors_and_data(False)
    length_dict = all_vectors_df.set_index('model_name')['vector_size'].to_dict()
    size = length_dict[d2v_model_name]
    val_df.text = val_df.text.astype(str)
    vectors = infer_vectors(d2v_model,val_df.text,size)
    return val_df, vectors

def infer_vecs_for_val_hybrid(d2v_model, d2v_model_2,model_name_dict):
    val_df = pd.read_csv('Archive_CSV/validation_set.csv',index_col=0)
    val_df.text = val_df.text.astype(str)
    vectors = infer_vectors_concat(d2v_model,d2v_model_2,val_df.text,200)
    return val_df, vectors


def evaluate_classifier_model(model_name, model_name_dict):
    d2v_model = model_name_match[model_name]
    if type(d2v_model) == tuple:
        val_df,true_vectors = infer_vecs_for_val_hybrid(d2v_model[0],d2v_model[1], model_name_dict)
    else:
        val_df,true_vectors = infer_vecs_for_val(d2v_model, model_name, model_name_dict)
    best_model = pull_corresponding_classifier_model(model_name, model_name_dict)
    print(model_name)
    print('Best Model Parameters')
    display(best_model)
    print('Validation Test Score: ' +str(round(best_model.score(true_vectors, val_df.labels),2)))
    pred = best_model.predict(true_vectors)
    best = get_confusion_metrics(val_df.labels, pred)
    grid_search = pull_corresponding_classifier_grid_search(model_name, model_name_dict)
    return plot_grid_search(grid_search)

def evaluate_classifier_model_hybrid(model_name, model_name_dict, d2v_model, d2v_model_2):
    val_df,true_vectors = infer_vecs_for_val_hybrid(d2v_model, model_name, model_name_dict)
    best_model = pull_corresponding_classifier_model(model_name, model_name_dict)
    print(model_name)
    print('Best Model Parameters')
    display(best_model)
    print('Validation Test Score: ' +str(round(best_model.score(true_vectors, val_df.labels),2)))
    pred = best_model.predict(true_vectors)
    best = get_confusion_metrics(val_df.labels, pred)
    grid_search = pull_corresponding_classifier_grid_search(model_name, model_name_dict)
    print(pred)
    return plot_grid_search(grid_search)


def plot_grid_search(grid_df):
    mean_train = go.Scatter(
        x=grid_df.index,
        y=grid_df.mean_test_score,
        name = 'Mean Test Score',
        fill= None,
        mode='lines',
        line=dict(
            color='red',
        )
    )
    mean_test = go.Scatter(
        x=grid_df.index,
        y=grid_df.mean_train_score,
        name = "Mean Train Score",
        fill='tonexty',
        mode='lines',
        line=dict(
            color='blue',
        )
    )

    layout = go.Layout(
        showlegend=True,
        annotations=[
            dict(
                x=grid_df.index[(grid_df.mean_test_score == grid_df.mean_test_score.max())][0],
                y=grid_df.mean_test_score.max(),
                xref='x',
                yref='y',
                text='Max Avg Test Score',
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            )
        ]
    )


    data = [mean_train, mean_test]

    # data = [trace0, trace1]


    fig = dict(data=data, layout=layout)
    return py.iplot(fig, filename='plot from API (14)')

def plot_best_grid_search(models, model_name_dict):
    master_df = pd.DataFrame()
    for model in models:
        df = pull_corresponding_classifier_grid_search(model, model_name_dict)
        df = df[(df.rank_test_score == 1)]
        df['model_name'] = model
        master_df = master_df.append(df.head(1), ignore_index=True)
    master_df = master_df.reset_index(drop=True)
    master_df.index = master_df.model_name
    show_df = master_df[['mean_train_score','mean_test_score','std_train_score','std_test_score','params']]
    display(show_df)
    return plot_grid_search(master_df)

def fix_DT_overfit(models, model_name_dict):
    master_df = pd.DataFrame()
    for model in models:
        df = pull_corresponding_classifier_grid_search(model, model_name_dict)
        df = df[(df.param_max_depth.notnull())] #remove overfit
        df['model_name'] = model
        df = df.sort_values(by=['mean_test_score'],ascending=False)
        master_df = master_df.append(df.head(1), ignore_index=True)
    master_df = master_df.reset_index(drop=True)
    master_df.index = master_df.model_name
    show_df = master_df[['mean_train_score','mean_test_score','std_train_score','std_test_score','params']]
    display(show_df)
    return plot_grid_search(master_df)

Logistic_dict = {'Bigram_DBOW': ('Classification_models/Logistic_Regression/best_log_reg_TRIDMC.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full.csv'),
                  'Bigram_DBOW_200':('Classification_models/Logistic_Regression/best_log_reg_DBOW200.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full_DBOW200.csv'),
                   'Bigram_DMC':('Classification_models/Logistic_Regression/best_log_reg_DMC.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full_DMC.csv'),
                   'Bigram_DBOW_DMC':('Classification_models/Logistic_Regression/best_log_reg_DBOWDMC.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full_DBOWDMC.csv'),
                   'Bigram_DBOW_DMM':('Classification_models/Logistic_Regression/best_log_reg_DBOWDMM.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full_DBOWDMM.csv'),
                   'Bigram_DMM':('Classification_models/Logistic_Regression/best_log_reg_DMM.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full_DMM.csv'),
                   'Trigram_DBOW':('Classification_models/Logistic_Regression/best_log_reg_TRIDBOW.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full_TRIDMCDBOW.csv'),
                   'Trigram_DBOW_DMM':('Classification_models/Logistic_Regression/best_log_reg_TRIDMMDBOW.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full_TRIDMMDBOW.csv'),
                   'Trigram_DMC':('Classification_models/Logistic_Regression/best_log_reg_TRIDMC.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full_TRIDMC.csv'),
                   'Trigram_DMM':('Classification_models/Logistic_Regression/best_log_reg_TRIDMM.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full_TRIDMM.csv')
                   ,







                  }

Decision_Tree_dict = {'Bigram_DMC': ('Classification_models/Decision_Tree/Decision_Tree_Bigram_DMC.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Bigram_DMC.csv'),
 'Trigram_DMM': ('Classification_models/Decision_Tree/Decision_Tree_Trigram_DMM.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Trigram_DMM.csv'),
 'Trigram_DBOW_DMM': ('Classification_models/Decision_Tree/Decision_Tree_Trigram_DBOW_DMM.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Trigram_DBOW_DMM.csv'),
 'Trigram_DMC': ('Classification_models/Decision_Tree/Decision_Tree_Trigram_DMC.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Trigram_DMC.csv'),
 'Bigram_DBOW_DMM': ('Classification_models/Decision_Tree/Decision_Tree_Bigram_DBOW_DMM.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Bigram_DBOW_DMM.csv'),
 'Bigram_DBOW_200': ('Classification_models/Decision_Tree/Decision_Tree_Bigram_DBOW_200.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Bigram_DBOW_200.csv'),
 'Bigram_DBOW_DMC': ('Classification_models/Decision_Tree/Decision_Tree_Bigram_DBOW_DMC.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Bigram_DBOW_DMC.csv'),
 'Bigram_DMM': ('Classification_models/Decision_Tree/Decision_Tree_Bigram_DMM.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Bigram_DMM.csv'),
 'Trigram_DBOW': ('Classification_models/Decision_Tree/Decision_Tree_Trigram_DBOW.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Trigram_DBOW.csv'),
 'Bigram_DBOW': ('Classification_models/Decision_Tree/Decision_Tree_Bigram_DBOW.pkl',
  'Classification_models/Decision_Tree/grid_search_Decision_Tree_Bigram_DBOW.csv')}

all_classifier_dictionaries = [Logistic_dict,Decision_Tree_dict]

models = ['Bigram_DMC',
 'Trigram_DMM',
 'Trigram_DBOW_DMM',
 'Trigram_DMC',
 'Bigram_DBOW_DMM',
 'Bigram_DBOW_200',
 'Bigram_DBOW_DMC',
 'Bigram_DMM',
 'Trigram_DBOW',
 'Bigram_DBOW']

def perform_SMOTE(text_vectors):
    print(text_vectors.labels.value_counts())
    X_resampled, y_resampled = SMOTE().fit_sample(text_vectors[text_vectors.columns[0:text_vectors.vector_size[0]]], text_vectors.labels)
    print(pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled

def resampled_SMOTE(model_name):
    full_df, all_vectors_df = get_all_vectors_and_data(True)
    print('Pulled Data')
    vectors = get_true_vectors(model_name,all_vectors_df,full_df)
    X_resampled, y_resampled = perform_SMOTE(vectors)
    return X_resampled, y_resampled

def train_learning_model(learning_model,hyperparameters,all_d2v_models,classifier_string):
    path_dictionary = {}
    for d2v_model in all_d2v_models:
        print('Training '+ d2v_model)
        X_resampled, y_resampled = resampled_SMOTE(d2v_model)
        learning_model = learning_model
        clf = GridSearchCVProgressBar(learning_model, hyperparameters, cv=10, verbose=0)
        best_model = clf.fit(X_resampled,y_resampled)
        grid_df = pd.DataFrame(best_model.cv_results_)
        best_model_path = '../Classification_models/'+classifier_string+'/'+classifier_string+'_'+d2v_model+'.pkl'
        grid_df_path = '../Classification_models/'+classifier_string+'/'+'grid_search_'+classifier_string+'_'+d2v_model+'.csv'
        grid_df.to_csv(grid_df_path)
        joblib.dump(best_model.best_estimator_,best_model_path)
        print('Saved Model!')
        path_dictionary[d2v_model] = (best_model_path,grid_df_path)
        print('Saved Grid Search!')
    return path_dictionary
