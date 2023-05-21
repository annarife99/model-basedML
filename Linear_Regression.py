import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
from sklearn import linear_model
import numpy as np 
import torch
import itertools
import os, datetime

import pyro
import pyro.distributions as dist 
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam
from pyro.infer import Predictive

sns.set_style("whitegrid") 
np.random.seed(42)

from preprocessing import preprocessing_Xdata
from utils import extract_datetime, one_hot_encode, compute_error, save_results_figures


def select_variables(X_train, variables=["season", "holiday", "workingday", "weather", "time_range",  "temp", "atemp", "humidity", "windspeed"]):
    X_train_regression = X_train[variables]
    for i in ["season", "holiday", "workingday", "weather"] : 
        X_train_regression[i] = X_train_regression[i].astype("category")
    
    X_train_regression = one_hot_encode(X_train_regression)
    return X_train_regression


def simple_linear_regression(X_train_reg,Y_train):
    regr = linear_model.Ridge()
    regr.fit(X_train_reg, Y_train)
    y_hat = regr.predict(X_train_reg)
    corr, mae, rae, rmse, r2 = compute_error(Y_train, y_hat)
    print("CorrCoef: %.3f\nMAE: %.3f\nRMSE: %.3f\nR2: %.3f" % (corr, mae, rmse, r2))

    return regr, y_hat


def model(X, obs=None):
    alpha = pyro.sample("alpha", dist.Normal(0., 1.))  # Prior for the bias/intercept
    beta  = pyro.sample("beta", dist.Normal(torch.zeros(X.shape[1]), 
                                            torch.ones(X.shape[1])).to_event())  # Priors for the regression coeffcients 
    with pyro.plate("data"):
        y = pyro.sample("y", dist.Poisson(torch.exp(alpha + X.matmul(beta))), obs=obs)
        
    return y


def probabilistic_linear_regression(X_train_reg,Y_train,n_steps=500):
    y_train_regression_torch = torch.tensor(Y_train).float()

    X_train_regression_np = np.array(X_train_reg)
    X_train_regression_torch = torch.tensor(X_train_regression_np).float()

    # Define guide function
    guide = AutoDiagonalNormal(model)

    # Reset parameter values
    pyro.clear_param_store()

    # Setup the optimizer
    lr=0.001
    adam_params = {"lr": lr}
    optimizer = ClippedAdam(adam_params)

    # Setup the inference algorithm
    elbo = Trace_ELBO(num_particles=1)
    svi = SVI(model, guide, optimizer, loss=elbo)

    # Do gradient steps
    for step in range(n_steps):
        elbo = svi.step(X_train_regression_torch, y_train_regression_torch)
        if step % 500 == 0:
            print("[%d] ELBO: %.1f" % (step, elbo))

    
    predictive = Predictive(model, guide=guide, num_samples=1000, return_sites=("alpha", "beta"))
    samples = predictive(X_train_regression_torch, y_train_regression_torch)

    alpha_samples = samples["alpha"].detach().numpy()
    beta_samples = samples["beta"].detach().numpy()
    y_hat = np.round(np.mean(np.exp(alpha_samples.T + np.dot(X_train_regression_np, beta_samples[:,0].T)), axis=1))
    # convert back to the original scale
    preds = y_hat # no need to do any conversion here because the Poisson model received untransformed y's

    corr, mae, rae, rmse, r2 = compute_error(Y_train, preds)
    print('anna6')
    print("CorrCoef: %.3f\nMAE: %.3f\nRMSE: %.3f\nR2: %.3f" % (corr, mae, rmse, r2))
    info={"CorrCoef":corr,"MAE":mae, "RMSE":rmse, "R2":r2, "n_steps":n_steps,"lr":lr}

    return alpha_samples,beta_samples,y_hat, info



if __name__== "__main__":
    data_train= pd.read_csv("train.csv")
    data_test=pd.read_csv("test.csv")

    X_train= data_train.iloc[:,:-1]
    Y_train= data_train['count']    

    #Prepare data for processing
    X_train= preprocessing_Xdata(X_train) 

    #Non-probablistic approach
    X_train_reg= select_variables(X_train)
    simple_reg,y_hat_sr= simple_linear_regression(X_train_reg,Y_train)

    #Probabilitic approach
    alpha_samples,beta_samples,y_hat,errors= probabilistic_linear_regression(X_train_reg,Y_train)
    print("Training finished")

    save_results_figures(alpha_samples,beta_samples,y_hat,errors,Y_train,X_train)
    print("Results saved")



