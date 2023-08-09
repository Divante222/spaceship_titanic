import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def prepare_data(df):
    the_columns = ['HomePlanet', 'Destination']

    boolean_cols = ['VIP', 'CryoSleep', 'Transported']

    data = df.dropna(subset = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'RoomService']).copy()

    for columns in data.select_dtypes(include=['float64']).columns.tolist():
        data[columns] = data[columns].fillna(data[columns].mean())




    dummy = pd.get_dummies(data[the_columns])
    data = pd.concat([data, dummy], axis=1)

    for col in boolean_cols:
        
        data[col] = data[col].astype('bool').astype('int')

    data = data.drop(columns = ['HomePlanet', 'Name', 'Cabin', 'Destination'])

    return data


def scale_data(df):
    scaler = RobustScaler()

    scaler.fit(df.select_dtypes(include=['float64']))

    x_train_scaled = scaler.transform(df.select_dtypes(include=['float64']))
    scaled_df = pd.DataFrame(x_train_scaled, columns = df.select_dtypes(include=['float64']).columns.tolist())

    return pd.concat([scaled_df, df.select_dtypes(exclude=['float64']).reset_index(drop=True)], axis =1)


def new_visual_univariate_findings(df):
    '''
    This function displays all of my histplots during the univariate analysis
    '''
    columns = df.select_dtypes(include=['float64']).columns
    num_cols = len(columns)
    num_rows, num_cols_subplot = divmod(num_cols, 3)
    
    if num_cols_subplot > 0:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

    for i, col in enumerate(columns):
        row_idx, col_idx = divmod(i, 3)
        sns.histplot(df[col], ax=axes[row_idx, col_idx])
        axes[row_idx, col_idx].set_title(f'Histogram of {col}')

    # Delete unused subplots if necessary
    if num_cols_subplot > 0:
        for i in range(num_cols_subplot, 3):
            fig.delaxes(axes[num_rows - 1, i])

    plt.tight_layout()
    plt.show()


def new_visual_univariate_findings_boxplots(df):
    '''
    This function displays all of our histplots during the univariate analysis
    '''
    count = 0
    for col in df.select_dtypes(include=['float64']).columns:                   

        num_cols = len(df.select_dtypes(include=['float64']).columns)
        num_rows, num_cols_subplot = divmod(num_cols, 3)
        
        if num_cols_subplot > 0:
            num_rows += 1

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
        if count < 1:
            for i, col in enumerate(df.select_dtypes(include=['float64']).columns):
                
                row_idx, col_idx = divmod(i, 3)
                
                sns.boxplot(df[col], ax=axes[row_idx, col_idx])
                
                
                axes[row_idx, col_idx].set_title(f'Histogram of {col}')
            
            

            plt.tight_layout()
            plt.show()


def comparison_of_means(data):
    master_df = pd.DataFrame(columns = ['t_statistic', 'p_value', 'col'])

    for num, col in enumerate(data.select_dtypes(include = ['float64']).columns):
        t_statistic, p_value = stats.ttest_ind(data[col][data['Transported'] == 0], data[col][data['Transported'] == 1])
        master_df.loc[num] = [t_statistic, p_value, col]

    master_df['Moving Forward'] = master_df['p_value'].apply(moving_forward)

    return master_df


def chi2_test(train, y_train):
    '''
    Runs a chi2 test on all items in a list of lists and returns a pandas dataframe
    '''

    columns_list = train.select_dtypes(include = ['int64', 'uint8']).columns.tolist()

    for col in columns_list:
        value_list = [str(value) for value in train[col]]
        train[col] = np.array(value_list, dtype=object)

    chi_df = pd.DataFrame({
        'feature': [],
        'chi2': [],
        'p': [],
        'degf': [],
        'expected': []
    })

    for iteration, col in enumerate(columns_list):
        observed = pd.crosstab(train[col], y_train)
        chi2, p, degf, expected = stats.chi2_contingency(observed, correction=False)

        chi_df.loc[iteration + 1] = [col, chi2, p, degf, expected]

    chi_df['Moving Forward'] = chi_df['p'].apply(moving_forward)

    return chi_df


def split_data(df):

    the_columns = df.columns.tolist()
    the_columns.remove('Transported')

    X = df[the_columns]
    y = df.Transported

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=123)
    return X_train, X_test, y_train, y_test


def modeling(X_train, X_test, y_train, y_test, means_list, chi2_list):
    X_train = X_train[means_list + chi2_list]
    X_test = X_test[means_list + chi2_list]


    parameters_DTree = {
        'max_depth':range(1,21),
        'min_samples_leaf':range(1,11),
        'criterion': ["gini", "entropy", "log_loss"]
    }

    parameters_rf = {'max_depth':range(1,21),
                    "min_samples_leaf":range(1,21),
                    "criterion": ['gini', 'entropy', 'log_loss']
                    }

    parameters_knn = {
        'n_neighbors':range(1,21),
        'weights':['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size':range(1,31),
        'p': range(1,5)

    }

    parameters_lr = {
        'penalty':['l1', 'l2', 'elasticnet'],
        'dual':[True, False], 
        'C':range(1,21),

    }


    dt = DecisionTreeClassifier(random_state=123)
    rf = RandomForestClassifier(random_state=123)
    knn = KNeighborsClassifier()
    lr = LogisticRegression(random_state = 123)

    the_parameters = [parameters_DTree, parameters_rf, parameters_knn, parameters_lr]

    models = ['DecisionTreeClassifier','RandomForestClassifier','KNeighborsClassifier','LogisticRegression']
    master_df = pd.DataFrame()

    for number, tree in enumerate([dt, rf, knn, lr]):
        grid = GridSearchCV(tree, the_parameters[number], cv=5)
        grid.fit(X_train, y_train)

        for p, score in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
            p['score'] = score
            p['model'] = models[number]
        new_df = pd.DataFrame(pd.DataFrame(grid.cv_results_['params']).sort_values('score', ascending=False))
        master_df = pd.concat([master_df, new_df])
    return master_df


def moving_forward(p_val):
    '''
    This function returns whether or not a p value is less than alpha
    '''
    if p_val < .05:
        return 'Yes'
    else:
        return 'No'
    
def percentages_visuals(df, y_train):
    columns = df.select_dtypes(include=['int64', 'uint8']).columns.to_list()

    num_cols = len(columns)
    num_rows, num_cols_subplot = divmod(num_cols, 2)

    if num_cols_subplot > 0:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 6))

    for i, col in enumerate(columns):
        row_idx, col_idx = divmod(i, 2)

        contingency_table = pd.crosstab(df[col], y_train)

        # Calculate the total number of observations
        total_observations = contingency_table.sum().sum()

        # Convert the counts to percentages
        contingency_table_percent = (contingency_table / total_observations) * 100

        # Create a stacked bar chart in the current subplot
        ax = contingency_table_percent.plot(kind='bar', stacked=True, ax=axes[row_idx, col_idx])
        ax.set_title(f"Stacked Bar Chart of {col} vs. Target")
        ax.set_xlabel(col)
        ax.set_xticklabels(['No', 'Yes'])
        ax.set_ylabel("Percentage")
        ax.legend(title='Target', labels=['No', 'Yes'])

    plt.tight_layout()
    plt.show()


def percentages(df, y_train):
    '''
    sanity check, created pandas crosstabs to insure the percentages are correct for the stacked bar charts
    '''
    columns = df.select_dtypes(include=['int64', 'uint8']).columns.to_list()
    
    for col in columns:
        contingency_table = pd.crosstab(df[col], y_train)

        # Calculate the total number of observations
        total_observations = contingency_table.sum().sum()

        # Convert the counts to percentages
        contingency_table_percent = (contingency_table / total_observations) * 100

        print(contingency_table_percent)