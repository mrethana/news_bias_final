from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import plotly.plotly as py
import plotly.graph_objs as go
from d2v_func import *
import itertools


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

def get_all_vectors_and_data():
    all_vectors_df = pd.read_csv('Archive_CSV/all_vectors.csv',index_col=0)
    full_df = pd.read_csv('Archive_CSV/ALL_rows_scraped.csv', index_col=0)
    return full_df, all_vectors_df

def get_true_vectors(model_name, all_vectors_df, full_df):
    true_vectors = all_vectors_df[(all_vectors_df.model_name == model_name)]
    true_vectors = true_vectors.dropna(axis = 1)
    true_vectors = true_vectors.reset_index(drop=True)
    true_vectors['labels'] = full_df.label
    holder = true_vectors.pop('model_name') # remove column b and store it in holder
    true_vectors['model_name'] = holder
    return true_vectors

def pull_corresponding_logreg_model(model_name):
    model = joblib.load(open(model_name_dict[model_name][0], 'rb'))
    return model

def pull_corresponding_logreg_grid_search(model_name):
    df = pd.read_csv(model_name_dict[model_name][1], index_col=0)
    return df

def evaluate_logreg_model(model_name):
    full_df, all_vectors_df = get_all_vectors_and_data()
    true_vectors = get_true_vectors(model_name, all_vectors_df, full_df)
    best_model = pull_corresponding_logreg_model(model_name)
    print(model_name)
    print('Best Model Parameters')
    display(best_model)
    print('Validation Test Score: ' +str(round(best_model.score(true_vectors[true_vectors.columns[0:true_vectors.vector_size[0]]], true_vectors.labels),2)))
    size = true_vectors.vector_size[0]
    pred = best_model.predict(true_vectors[true_vectors.columns[0:size]])
    best = get_confusion_metrics(true_vectors.labels, pred)
    grid_search = pull_corresponding_logreg_grid_search(model_name)
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

def plot_best_grid_search(models):
    master_df = pd.DataFrame()
    for model in models:
        df = pull_corresponding_logreg_grid_search(model)
        df = df[(df.rank_test_score == 1)]
        df['model_name'] = model
        master_df = master_df.append(df, ignore_index=True)
    master_df = master_df.reset_index(drop=True)
    master_df.index = master_df.model_name
    show_df = master_df[['mean_train_score','mean_test_score','std_train_score','std_test_score','param_C','param_class_weight','param_multi_class','param_penalty','param_solver']]
    display(show_df)
    return plot_grid_search(master_df)

model_name_dict = {'Bigram_DBOW': ('Classification_models/Logistic_Regression/best_log_reg_TRIDMC.pkl','Classification_models/Logistic_Regression/logistic_grid_search_full.csv'),
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
