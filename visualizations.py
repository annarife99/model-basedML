import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def samples_from_posterior(samples,f_path):
    sns.distplot(samples)       
    # Alternate plotting due to deprecated distplot-function
    #sns.histplot(data=posterior_samples["alpha"], kde=True, stat='density', color=next(palette), element="step")
    #sns.histplot(data=posterior_samples["sigma"], kde=True, stat='density', color=next(palette), element="step")
    plt.savefig(f_path+'samples_from_posterior.png')


def plot_numerical_variable(beta_samples,f_path): 
    color = sns.color_palette("Paired")
    list_labels = ["temp", "atemp", "humidity", "windspeed"]
    for i in range(4) : 
        sns.kdeplot(beta_samples[:,0,i], color = color[i], label = list_labels[i], fill = True)
    plt.legend()
    plt.savefig(f_path+'numerical_var.png')


def plot_categorical_variables(X_train, beta_samples,f_path):
    fig, ax = plt.subplots(5,1, figsize = (10,20))
    list_labels = ["season", "holiday", "workingday", "weather", "time_range"]
    list_seasons = ["spring", "summer", "fall", "winter"]
    list_weather = ["Clear, Few clouds, Partly cloudy, Partly cloudy", "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist", "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", ""]
    prev_categories = 0 

    for i in range(5) : 
        categories = len(np.unique(X_train[list_labels[i]]))
        samples = beta_samples[:,0,4+prev_categories:4+prev_categories + categories]
        prev_categories += categories 
        color = sns.color_palette("Paired");
        list_categories = [x for x in range(categories)]
        for j in range(samples.shape[1]) : 
            sns.kdeplot(samples[:,j], color = color[j], label = list_categories[j], fill = True, ax = ax[i])
            ax[i].legend()
            ax[i].set_title(list_labels[i])

    plt.savefig(f_path+'categorical_var.png')


def compare_yhat_ytrue(y_hat,Y_train_regression,f_path):
    sns.kdeplot(Y_train_regression, fill = True, label = "True labels distribution")
    sns.kdeplot(y_hat, fill = True, label = "Predicted labels distribution")
    plt.legend()
    plt.savefig(f_path+'yhat_ytrue.png')