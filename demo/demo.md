---
title: User guide for DDSIMCA package
author: Sergey Kucheryavskiy
date: 01/01/2026
---

# User guide for `ddsimca` package

This guide is mainly based on [DD-SIMCA tutorial paper](https://doi.org/10.1002/cem.3556) and most of the code below reproduces outcomes and figures shown in the paper. Therefore it is higly recommended to download and read the paper first (it is freely available for everyone) and then come back to this document.

Before you start, make sure that you have installed the `ddsimca` package. If not, just uncomment and run the following code:


```python
#!pip install ddsimca
```

It will automatically install all necessary packages, including `prcv`, which implements Procrustes cross-validation, to be used later.


## Training DD-SIMCA model and detection of outliers

[Download](https://mda.tools/ddsimca/Oregano.zip) zip archive with the Oregano dataset used in the tutorial.  It consists of several CSV files, just unzip them all to the same folder where you have this document.

The following code loads training set from a CSV file and shows first five rows and five columns of the set:

```python
import pandas as pd
import matplotlib.pyplot as plt
data_train = pd.read_csv("Target_Train.csv", index_col=0)

print(data_train.iloc[:5, :5])
```
As you can see, the data rows have object labels (hence we used `index_col=0` when loaded the data). The first column of the dataset contains class labels. For the training set it should only contain target class label (in our case `"Oregano"`), if there are more labels or no labels at all you will get an error when trying to create a model. The rest of the data frame consists of NIR spectra, already preprocessed.

Now we ready to train DD-SIMCA model and look at summary info:

```python
from ddsimca import ddsimca
m = ddsimca(data_train, ncomp = 10)
m.summary()
```
As you can see, the value `10` is the total number of components to use in the model. The optimal number can be discovered and set later, by default it is the same as the total number.

Also by default the data values were mean centered but not standardized. This can be changed by providing extra arguments, check help for `ddsimca()` method for more details.

You can visualize the number of degrees of freedom and the eigenvalues vs. number of PCs using plots (eigenvalues can also be shown log transformed, check help for the method):


```python
plt.figure(figsize = (12, 5))

ax1 = plt.subplot(1, 2, 1)
m.plotDoF(ax1, dof = "Nh")
m.plotDoF(ax1, dof = "Nq")

ax2 = plt.subplot(1, 2, 2)
m.plotEigenvals(ax2)
```

As well as show plots with the PCA loadings:

```python
plt.figure(figsize = (12, 5))

ax1 = plt.subplot(1, 2, 1)
m.plotLoadings(ax1, comp = (1, 2), type = "p")

ax2 = plt.subplot(1, 2, 2)
m.plotLoadings(ax2, comp = (1,), type = "l", color = "blue")
m.plotLoadings(ax2, comp = (2,), type = "l", color = "red")
ax2.legend()
ax2.set_xlabel("Wavenumber, cm-1")
```

Here parameter `type` tells how to show the loadings values, `"p"` stands for points (scatter plot), and `"l"` stands for lines. The parameter `comp` should be a tuple with two values for scatter plot and with one value for line plot, as it is shown above.

The model object does not have any results, it only contains values and statistics needed for applying this model to any dataset (e.g. loadings, values for centering and scaling, parameters of distance distribution, etc.). In order to get the results, you need to apply this model to a dataset. Here is how to do it for the training set:

```python
r_train_c = m.predict(data_train)
r_train_c.summary()
```
As you can see, most of the parameters, like significance level for extremes, `alpha`, and outliers, `gamma`, type of estimators for distance limits (`lim_type`) are set to default values (`0.05`, `0.01`, and `"classic"` correspondingly). If you want to change any of them, simply provide the proper values as arguments of the method `predict()`.

For example, let's create the result object using robust estimators:

```python
r_train_r = m.predict(data_train, lim_type = "robust")
```
Now let's check the acceptance plot for both results object (obtained using classic and robust estimators) in order to find any outliers. The figure below shows plots similar to (A) and (B) from Figure 2 of the paper.

```python
plt.figure(figsize = (12, 5))

ax1 = plt.subplot(1, 2, 1)
r_train_c.plotAcceptance(ax1, ncomp = 2, show_labels = True)

ax2 = plt.subplot(1, 2, 2)
r_train_r.plotAcceptance(ax2, ncomp = 2, show_labels = True)
```
We can also show these two plots using log transformed coordinates, like in Figure 3 of the paper:

```python
plt.figure(figsize = (12, 5))

ax1 = plt.subplot(1, 2, 1)
r_train_c.plotAcceptance(ax1, ncomp = 2, do_log = True, show_labels = True)

ax2 = plt.subplot(1, 2, 2)
r_train_r.plotAcceptance(ax2, ncomp = 2, do_log = True, show_labels = True)
```

Apparently, as described in the paper, we need to remove the sample `Drg12` first, and then `Drg13`. Let's do this step by step and reproduce plots (C) and (D) of Figure 2. Note, that every time we remove an outlier we need to re-train the model.

First remove `Drg12` and reproduce the plot (C):

```python
data_train_new = data_train.drop("Drg12")

m_new = ddsimca(data_train_new, 10)
r_train_new = m_new.predict(data_train_new, lim_type = "robust")

plt.figure(figsize = (5, 5))
ax = plt.subplot(1, 1, 1)
r_train_new.plotAcceptance(ax, ncomp = 2, show_labels = True)
```

Now let's remove `Drg13` and reproduce plot (D) of the Figure 2. Note, that in this case we get back to classic estimators, as from the paper we know that there are no more outliers in the data. Otherwise it can be a good idea to use robust estimators again as well as to check this plot for different number of components.

```python
data_train_final = data_train_new.drop("Drg13")

m_final = ddsimca(data_train_final, 10)
r_train_final = m_final.predict(data_train_final)

plt.figure(figsize = (5, 5))
ax = plt.subplot(1, 1, 1)
r_train_final.plotAcceptance(ax, 2)
```
Finally, here are sensitivity and extremes plots made for the training set, similar to what is shown in Figure 4 in the paper. There is one difference, in the paper we used A = 20 components in the model and here we used 10 (to make summary outcomes shorter). Therefore sensitivity plot below is shown for 10 first components only.

Here we use a versatile method `plotFoM()` which can show a plot for any of the five figures of merit: sensitivity (`"sens"`), specificity (`"spec"`), selectivity (`"sel"`), accuracy (`"acc"`) and  efficiency (`"eff"`).


```python
plt.figure(figsize = (12, 5))

ax1 = plt.subplot(1, 2, 1)
r_train_final.plotFoM(ax1, fom = "sens", show_ci = True)

ax2 = plt.subplot(1, 2, 2)
r_train_final.plotExtremes(ax2, ncomp = 2)
```
In case if FoM plot is made for sensitivity, you have a possibility to add 95% confidence interval computed based on the expected sensitivity (1 - alpha) and the number of objects in the dataset, it is shown as semi-transparent rectangle on the plot above. Similar to what is shown in the paper and what is implemented in the web-application.

## Validation and optimization

The best validation strategy is to use an independent validation set. However, if you want to keep it for the final testing of your model (or for fine tuning) you can employ [Procrustes cross-validation](https://github.com/svkucheryavski/pcv), PCV. PCV is a procedure for generation of validation set based on training set and cross-validation resampling. The code below does this based on PCA version of the method, implemented in package `prcv`:

```python
from prcv.methods import pcvpca

# get matrix with predictors from the training set
X_train = data_train_final.iloc[:, 1:].values

# generate matrix with PV-set using 20 PCs, mean centering and cross-Validation
# systematic splits (venetian blinds) to 4 segments
X_pv = pcvpca(X_train, ncomp = 20, center = True, scale = False, cv = {"type": "ven", "nseg": 4})

# create data frame from the generated data
data_pv = pd.DataFrame(X_pv, index = data_train_final.index)
data_pv.insert(0, "Class", data_train_final.Class)
data_pv.columns = data_train_final.columns

data_pv.iloc[:5, :5]
```

Now let's apply the model to the PV-set and show a combined sensitivity plot for the training set and for the PV-set, hence reproducing plot (A) from Figure 4:

```python
r_pv = m_final.predict(data_pv)

plt.figure(figsize = (5, 5))
ax = plt.subplot(1, 1, 1)
r_train_final.plotFoM(ax, fom = "sens", label = "train", show_ci = True)
r_pv.plotFoM(ax, fom = "sens", label = "pv", color = "tab:red")
```
It looks like 3 components is optimal in this case. Let's now load the test set, which consists only of the target class members and apply the model to this set as well. Then we will show sensitivity plot for the test set, reproducing plot (B) of Figure 4.

```python
data_test_target = pd.read_csv("Target_Test.csv", index_col=0)

r_test_target = m_final.predict(data_test_target)

plt.figure(figsize = (5, 5))
ax = plt.subplot(1, 1, 1)
r_test_target.plotFoM(ax, fom = "sens", show_ci = True)
```

Finally, you can also specify the optimal number of components for a model as well as for any result object. In this case, every time you make an acceptance plot (or do any other actions which depends on the number of components in the model) this value will be used as the default one:

```python
m_final.select_ncomp(3)
r_test_target.select_ncomp(3)
```


## Predictions

Predictions can be made using data frames with or without reference class labels. In the first case the result object will contain all necessary figures of merits (sensitivity for members, specificity and selectivity for non-members, accuracy and efficiency if dataset contains both members and strangers). If reference classes are not provided, the model will just make predictions (accepted/rejected).

Let's load dataset with reference classes (only non-target objects) and then remove column with class names, thus creating a data set without reference classes:

```python
data_test_nontarget = pd.read_csv("NonTarget_Non_Or.csv", index_col = 0)
data_new_nontarget = data_test_nontarget.iloc[:, 1:]
```

Let's check what is inside the datasets:

```python
data_test_nontarget.iloc[:5, :5]
```
```python
data_new_nontarget.iloc[:5, :5]
```
Now let's apply the model to both sets (remember that they contain the same objects, but one has column with reference class labels and second one does not have thus column). Then check the acceptance plot:

```python
r_test_nontarget = m_final.predict(data_test_nontarget)
r_new_nontarget = m_final.predict(data_new_nontarget)

plt.figure(figsize = (12, 5))

ax1 = plt.subplot(1, 2, 1)
r_test_nontarget.plotAcceptance(ax1, do_log = True, show_labels = True)

ax2 = plt.subplot(1, 2, 2)
r_new_nontarget.plotAcceptance(ax2, do_log = True, show_labels = True)

```
As you can see, in the first case the model indeed treated the objects as from non-target classes and shows corresponding roles (alien and external in this case) on the plot. While in the second case it splits samples to accepted (in) and rejected (out).

Let's see how different the summary information is:

```python
r_test_nontarget.summary()
```

```python
r_new_nontarget.summary()
```
The main difference is that for the second dataset there are no columns with figures of merits, true negatives and false positives.

Now let's load data which has reference classes and objects of both target and non-target classes:

```python
data_test_all = pd.read_csv("All_Test.csv", index_col = 0)
data_test_all.iloc[:5, :5]
```

There are several differences here. First of all, the acceptance plot can now be shown only for members, only for strangers, or for all samples (default option). In the latter case, the acceptance plot will color group object points by classes instead of roles until you change this by providing explicit value for parameter `show`:


```python
r_test_all = m_final.predict(data_test_all)

plt.figure(figsize = (13, 4))

ax1 = plt.subplot(1, 3, 1)
r_test_all.plotAcceptance(ax1, do_log = True)

ax2 = plt.subplot(1, 3, 2)
r_test_all.plotAcceptance(ax2, do_log = True, show_set = "strangers")

ax3 = plt.subplot(1, 3, 3)
r_test_all.plotAcceptance(ax3, do_log = True, show_set = "members")
```
Also, in this case all figures of merit, including accuracy and efficiency, are available:

```python
r_test_all.summary()
```

And they can be plotted together:

```python
plt.figure(figsize = (5, 5))

ax = plt.subplot(1, 1, 1)
r_test_all.plotFoM(ax, fom = "sens")
r_test_all.plotFoM(ax, fom = "spec")
r_test_all.plotFoM(ax, fom = "eff")

```



## Extra plots, features, and details


It is also possible to show every distance separately for given number of components (this will work for any result objects):

```python
plt.figure(figsize = (10, 14))

ax1 = plt.subplot(3, 1, 1)
r_test_all.plotDistance(ax1, ncomp = 2, distance = "h")

ax2 = plt.subplot(3, 1, 2)
r_test_all.plotDistance(ax2, ncomp = 2, distance = "q")

ax3 = plt.subplot(3, 1, 3)
r_test_all.plotDistance(ax3, ncomp = 2, distance = "f", show_crit = True)
```

As well as to show PCA scores plot:

```python
plt.figure(figsize = (12, 5))

ax1 = plt.subplot(1, 2, 1)
r_test_all.plotScores(ax1, comp = (1, 2), type = "p", show_labels = True)

ax2 = plt.subplot(1, 2, 2)
r_test_all.plotScores(ax2, comp = (1,), type = "l", color = "blue")
r_test_all.plotScores(ax2, comp = (2,), type = "l", color = "red")
ax2.legend()
```

Similar to web-application you can also make plot with expected vs. observed alien objects and selectivity vs. sensitivity plot (if there are non-target class objects in the dataset):

```python
plt.figure(figsize = (15, 4))

ax1 = plt.subplot(1, 3, 1)
r_test_all.plotAliens(ax1)

ax2 = plt.subplot(1, 3, 2)
r_test_all.plotSelectivity(ax2)

ax2 = plt.subplot(1, 3, 3)
r_test_all.plotSelectivity(ax2, ncomp = 1, color = "tab:blue", label = "A = 1")
r_test_all.plotSelectivity(ax2, ncomp = 3, color = "tab:green", label = "A = 3")
r_test_all.plotSelectivity(ax2, ncomp = 5, color = "tab:orange", label = "A = 5")
ax2.set_title("Selectivity")
ax2.legend()
```

Similar to distance, extremes, and acceptance plots, the plots above are made for the optimal number of components we pre-selected earlier (`A = 3`). You can change this by providing value using `ncomp` argument to the plotting methods.

Any result object can be also converted to a data frame, where every object is represented by several columns, such as reference class (if provided), decision (in/out), role, as well as distance values. By default these values will be computed for optimal number of components, but you can force and provide which number of components to use for creating the data frame:

```python
rdf = r_test_all.as_df(ncomp = 2)
rdf.head()
```

You can get access to all computed values and do whatever you want. For example in case of model, you can get loadings, vectors for centering and scaling, classic and robust parameters of distance distribution (for h, q and f-distances), etc. Here are some examples:

```python
# loadings
m_final.V[:5, :5]
```

The distance parameters are saved as dictionaries `hParams`, `qParams` and `fParams`. Each dictionary has two fields, `classic` and `robust`. Each field contains another tuple with scaling factors (e.g. `h0`) and number of degrees of freedom (e.g. `Nh`) computed for each component in the model. Here is an example for score distance parameters computed using classic estimates:

```python
m_final.hParams["classic"]
```
Here is another example showing how to get `q0` and `Nq` values for robust estimator and 2 PCs:

```python
q0, Nq = m_final.qParams["robust"]
A = 2
print(f"A = {A}: q0 = {q0[A - 1]:.3f}, Nq = {Nq[A - 1]}")
```
In case of result object you can get all outcomes, including critical values, distances, calculations related to Type II error (if non-target class objects are provided) and many other things by getting access to data frame outcomes:

```python
r_test_all.outcomes.head()
```
You can also get access to matrices (`nrows` x `ncomp`) with h-, q- and f-distances, and matrix with decisions:

```python
r_test_all.H[:3, :5]
```
```python
r_test_all.Q[:3, :5]
```
```python
r_test_all.F[:3, :5]
```

```python
r_test_all.D[:3, :5]
```
Here is for example decisions obtained for each object using A = 3 components:

```python
r_test_all.D[:, 2]
```
As well as matrix with roles, which are coded by integer values: 0 for `regular`, 1 for `extreme`, 2 for `outlier`, 3 for `alien` and 4 for `external`. The first three are assigned to target class members and the last two — to objects from non-target classes and to unknown objects:

```python
r_test_all.R[:, 2]
```
Here is a simple way to get the role names only for members:

```python
import numpy as np
r_names = np.array(["regular", "extreme", "outlier", "alien", "external"])
r_names[r_test_all.R[:, 2]]
```
