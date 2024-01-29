# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import LinearRegression as linr
import LogisticRegression as logr
from MiniBatchSGD import MiniBatchSGD, MiniBatchSGDMomentum, LogisticRegressionSGD, LinearRegressionSGD
import seaborn as sns
from sklearn import preprocessing
np.random.seed(19680801)

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

###################################### DATA PREPROCESSING #####################################

# DATASET 1 : ENERGY EFFICIENCY

# X1 Relative Compactness
# X2 Surface Area
# X3 Wall Area
# X4 Roof Area
# X5 Overall Height
# X6 Orientation
# X7 Glazing Area
# X8 Glazing Area Distribution
# y1 Heating Load
# y2 Cooling Load

df1 = pd.read_excel("ENB2012_data.xlsx")


#Drop samples with malformed features
df1.dropna(inplace=True)

#Basic statistics and info
print(df1.describe())
print(df1.info())

#Plot outputs as a function of each input
fig1 = plt.figure(figsize=(16,8))
i=1
for col in df1.iloc[:,:8].columns:
    ax = plt.subplot((240+i))
    ax.scatter(df1[col], df1['Y1'], marker='x')
    ax.scatter(df1[col], df1['Y2'], marker='o')
    ax.legend(['Heating Load', 'Cooling Load'])
    ax.set_xlabel(col)
    i+=1

#Pairwise correlation between features and labels
corr = df1.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corr, cmap='RdYlGn', annot=True, ax=ax)

plt.show()
    
#Split into features and labels then normalize inputs
X_energy = df1.iloc[:, :8]
Y_energy = df1.iloc[:, 8:10]



# DATASET 2 : QUALITATIVE BANKRUPTCY

# X1 Industrial Risk: {P,A,N}
# X2 Management Risk: {P,A,N}
# X3 Financial Flexibility: {P,A,N}
# X4 Credibility: {P,A,N}
# X5 Competitiveness: {P,A,N}
# X6 Operating Risk: {P,A,N}
# X7 Class: {B,NB}
df2 = pd.read_csv("Qualitative_Bankruptcy.data.txt", header=None)
df2.columns = ["Industrial Risk", "Management Risk", "Financial Flexibility", "Credibility", "Competitiveness", "Operating Risk", "Class"]


#Drop samples with malformed features
df2.dropna(inplace=True)

#Basic statistics and info
print(df2.describe())
print(df2.info())

# Encode categorical feature data using one hot encoding
X = df2.iloc[:,0:6]
encX = OneHotEncoder()
encX.fit(X)
X_bank = pd.DataFrame(encX.transform(X).toarray())

# Binarize labels
Y = df2.iloc[:, 6]
encY = LabelEncoder()
encY.fit(Y)
Y_bank = pd.Series(encY.transform(Y))

# CountPlot for all the features
fig, ax =plt.subplots(3,3)
fig.set_size_inches(12.7, 9.27)
sns.countplot(x=df2['Industrial Risk'],hue= df2['Class'], ax=ax[0][0])
sns.countplot(x=df2["Management Risk"],hue= df2['Class'], ax=ax[0][1])
sns.countplot(x=df2["Financial Flexibility"],hue= df2['Class'], ax=ax[0][2])
sns.countplot(x=df2["Credibility"],hue= df2['Class'], ax=ax[1][0])
sns.countplot(x=df2["Competitiveness"],hue= df2['Class'], ax=ax[1][1])
sns.countplot(x=df2["Operating Risk"],hue= df2['Class'], ax=ax[1][2])
sns.countplot(x=df2["Class"], ax=ax[2][0])


plt.show()


###################################### EXPERIMENTS #####################################

MSE_Y1 = lambda p, y: .5*np.mean((p["Y'1"] - y['Y1'])**2)
MSE_Y2 = lambda p, y: .5*np.mean((p["Y'2"] - y['Y2'])**2)

# -------------------- EXPERIMENT 1 AND 2: Linear and Logistic Regression with train/test split of 80/20 -------------------------

#----------------Linear regression

#Train/test split: 80/20
X_df1_train, X_df1_test, Y_df1_train, Y_df1_test = train_test_split(X_energy, Y_energy, test_size=0.2, random_state=0)

#Train the model: 
LR_lin = linr.LinearRegression()
LR_lin.fit(X_df1_train, Y_df1_train)

#Display weights
print("\n Feature weights:")
print(LR_lin.w)

#Predict target values for test dataset
predictions = LR_lin.predict(X_df1_test)

#Evaluate performance by computing mean square error for Y1 and Y2
MSE1 = MSE_Y1(predictions, Y_df1_test)
MSE2 = MSE_Y2(predictions, Y_df1_test) 
print("\n Mean Square Error:")
print("MSE Y1: "+str(MSE1))
print("MSE Y2: "+str(MSE2) + "\n")

#Find best l2 reg coefficient for linear base:

l2_reg = [0, 0.1, 1, 10]
for r in l2_reg:
    predictions = linr.LinearRegression(l2_reg=r).fit(X_df1_train, Y_df1_train).predict(X_df1_train)
    print("For l2_reg = " + str(r))
    print("MSE Y1: "+str(MSE_Y1(predictions, Y_df1_train)))
    print("MSE Y2: "+str(MSE_Y2(predictions, Y_df1_train))+"\n")

# Training and testing with L2 regularization and linear base
#Train the model: 
LR_lin = linr.LinearRegression(l2_reg=1)
LR_lin.fit(X_df1_train, Y_df1_train)

#Display weights
print("\n Feature weights:")
print(LR_lin.w)

#Predict target values for test dataset
predictions = LR_lin.predict(X_df1_test)

#Evaluate performance by computing mean square error for Y1 and Y2
print("\n Mean Square Error:")
print("MSE Y1: "+str(MSE_Y1(predictions, Y_df1_test)))
print("MSE Y2: "+str(MSE_Y2(predictions, Y_df1_test)))


### Polynomial Base: finding best l2_reg parameter and training and testing
for r in l2_reg:
    predictions = linr.LinearRegression(l2_reg=r, base='P').fit(X_df1_train, Y_df1_train).predict(X_df1_train)
    print("For l2_reg = " + str(r))
    print("MSE Y1: "+str(MSE_Y1(predictions, Y_df1_train)))
    print("MSE Y2: "+str(MSE_Y2(predictions, Y_df1_train))+"\n")


#Train the model:
LR_poly = linr.LinearRegression(l2_reg=1, base='P')
LR_poly.fit(X_df1_train, Y_df1_train)

#Predict target values for test dataset
pred_poly = LR_poly.predict(X_df1_test)

#Feature weights
print("Feature weights:")
print(LR_poly.w)

print("\n Mean Square Error:")
print("MSE Y1: "+str(MSE_Y1(pred_poly, Y_df1_test)))
print("MSE Y2: "+str(MSE_Y2(pred_poly, Y_df1_test)))


#---------------- Logistic Regression

X_train_df2, X_test_df2, y_train_df2, y_test_df2 = train_test_split(X_bank, Y_bank, test_size=0.2, random_state=0, stratify=Y_bank)

logR = logr.LogisticRegression()
logR.fit(X_train_df2, y_train_df2)

predictions = logR.predict(X_test_df2)
score = np.mean(predictions == y_test_df2)

print(f"----TEST SET ACCURACY: {score}----------")



#-----------------------------------------------------------Experiment 3: Growing training set sizes--------------------------------------------------

#-----------------------Linear Regression

# Create an instance of the LinearRegression model
LR_lin = linr.LinearRegression(l2_reg=1)

# Define a list of test sizes
test_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Create lists to store the training and testing MSE scores
train_MSE1= []
train_MSE2= []
test_MSE1 = []
test_MSE2 = []

# Loop over the test sizes
for test_size in test_sizes:
    # Split the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_energy, Y_energy, test_size=test_size, random_state=0)
    
    # Fit the model to the training data
    LR_lin.fit(X_train, y_train)
    
    # Predict the target values for the training data
    y_train_pred = LR_lin.predict(X_train)
    
    # Predict the target values for the test data
    y_test_pred = LR_lin.predict(X_test)
    
    # Evaluate the model's performance on the training set
    train_MSE1.append(MSE_Y1(y_train_pred, y_train))
    train_MSE2.append(MSE_Y2(y_train_pred, y_train))
    
    # Evaluate the model's performance on the test set
    test_MSE1.append(MSE_Y1(y_test_pred, y_test))
    test_MSE2.append(MSE_Y2(y_test_pred, y_test))

# Plot the training and testing MSEs
plt.plot(test_sizes, train_MSE1, label='Y1 Training')
plt.plot(test_sizes, test_MSE1, label='Y1 Testing')
plt.plot(test_sizes, train_MSE2, label='Y2 Training')
plt.plot(test_sizes, test_MSE2, label='Y2 Testing')
plt.xlabel('Test Size')
plt.ylabel('MSE')
plt.legend()
plt.show()


#---------------------Logistic Regression

# Test different learning rates to find the best one
lrs = [0.001, 0.0050, 0.01, 0.05, 0.1, 1, 10]    # Learning rates to test
best_lr = lrs[0]    # Initialize best learning rate to first learning rate in list
best_score = 0    # Initialize best score to zero
for lr in lrs:  # Loop over learning rates
    model = logr.LogisticRegression(lr=lr)   # Initialize model with current learning rate
    y_pred = model.fit(X_train_df2, y_train_df2)    # Fit model to training data
    y_pred = model.predict(X_train_df2)     # Predict on training data
    score = np.mean(y_pred == y_train_df2)  # Calculate accuracy on training data
    
    print(f"------RATE {lr} ------------- SCORE {score}")   # Print results
    
    if score > best_score:  # If current score is better than best score, update best score and best learning rate
        best_lr = lr    # Update best learning rate
        best_score = score      # Update best score


##### TRAIN AND PREDICT ######
model = logr.LogisticRegression()    # Initialize model with best learning rate
model.fit(X_train_df2, y_train_df2)     # Fit model to training data

predictions = model.predict(X_test_df2)    # Predict on test data
score = np.mean(predictions == y_test_df2)      # Calculate accuracy on test data

print(f"----TEST SET ACCURACY: {score}----------")  # Print results







# list to store the accuracy scores on train and test sets
train_scores = []
test_scores = []

# Loop over different sizes of training data
for sample_size in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:

    # Get the sample of the training data
    sample_index = np.random.choice(X_train_df2.index, int(sample_size*len(X_train_df2)), replace=False)
    X_train_sample = X_train_df2.loc[sample_index, :]
    y_train_sample = y_train_df2.loc[sample_index]

    # Fit the logistic regression model on the sample
    model = logr.LogisticRegression(lr=0.01, max_iters=1e4, l2_reg=0, epsilon=1e-4, add_bias=True, verbose=False)
    model.fit(X_train_sample, y_train_sample)

    # Predict on training and test sets
    y_pred_train = model.predict(X_train_sample)
    y_pred_test = model.predict(X_test_df2)

    # Calculate and store the accuracy scores
    train_scores.append(np.mean(y_pred_train == y_train_sample))
    test_scores.append(np.mean(y_pred_test == y_test_df2))
    
# Plot the results
plt.plot([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], train_scores, label='train')
plt.plot([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], test_scores, label='test')

plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#-------------------------------------------------------Experiment 4: Different Batch sizes ----------------------


# Normalize Energy Efficiency data for mini batch SGD experiments
X_energy = df1.iloc[:, :8]
X_energy = pd.DataFrame(preprocessing.normalize(X_energy))  # Normalize data

Y_energy = df1.iloc[:, 8:10]
X_train_scaled, X_test_scaled, Y_df1_train, Y_df1_test = train_test_split(X_energy, Y_energy, test_size=0.2,
                                                                          random_state=0)

#################################################################### Linear regression

# Create MiniBatchSGD batch sizes
batch_sizes = [10, 25, 50, 80, 120, 154]  # The size of the test set is 154

c = []


# Define L2 cost function
def l2_cost_fn(x, y, w):
    w = np.asarray(w)[0]
    y = y.to_numpy()
    return .5 * np.mean((x @ w - y) ** 2)


print("Linear Regression, Y1: ")
for size in batch_sizes:
    # Create an instance of the LogisticRegressionSGC model
    lin_reg = LinearRegressionSGD()

    optimizer = MiniBatchSGD(batch_size=size, max_iters=300)

    # Fit the model to the training data
    lin_reg.fit(X_train_scaled, Y_df1_train["Y1"], optimizer)

    # For plotting training curve purposes
    w_hist = optimizer.w_history

    x_copy = X_train_scaled.copy()
    x_copy.insert(0, 'X0', np.ones(X_train_scaled.shape[0]))
    x_copy = x_copy.to_numpy()

    # Append the cost value for
    c.append([l2_cost_fn(x_copy, Y_df1_train["Y1"], [w_hist[j]]) for j in range(0, len(w_hist))])

    # Predict the target values for the test data
    y_df1_pred = lin_reg.predict(X_test_scaled)

    # Evaluate the model's performance
    MSE_SGD_Y1 = .5 * np.mean((y_df1_pred["Y"] - Y_df1_test['Y1']) ** 2)

    print("For batch size ", size, " the error of the model is ", MSE_SGD_Y1, ".")

# Plot convergence speed graph on Y1.
fig = plt.figure()

for i, b in enumerate(batch_sizes):
    print(c[i])
    plt.plot(c[i], marker='.', alpha=.998, label=f'Batch size = {b}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Convergence Speed on Y1')
plt.legend()
plt.show()

c = []

print("Linear Regression, Y2: ")
for size in batch_sizes:
    # Create an instance of the LogisticRegressionSGC model
    lin_reg = LinearRegressionSGD()

    optimizer = MiniBatchSGD(batch_size=size, max_iters=300)

    # Fit the model to the training data
    lin_reg.fit(X_train_scaled, Y_df1_train["Y2"], optimizer)

    # For plotting training curve purposes
    w_hist = optimizer.w_history

    x_copy = X_train_scaled.copy()
    x_copy.insert(0, 'X0', np.ones(X_train_scaled.shape[0]))
    x_copy = x_copy.to_numpy()

    # Append the cost value for
    c.append([l2_cost_fn(x_copy, Y_df1_train["Y2"], [w_hist[j]]) for j in range(0, len(w_hist))])

    # Predict the target values for the test data
    y_df1_pred = lin_reg.predict(X_test_scaled)

    # Evaluate the model's performance
    MSE_SGD_Y2 = .5 * np.mean((y_df1_pred["Y"] - Y_df1_test['Y2']) ** 2)

    print("For batch size ", size, " the error of the model is ", MSE_SGD_Y2, ".")

# Plot convergence speed graph on Y2.
fig = plt.figure()

for i, b in enumerate(batch_sizes):
    print(c[i])
    plt.plot(c[i], marker='.', alpha=.998, label=f'Batch size = {b}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Convergence Speed on Y2')
plt.legend()
plt.show()

######################################################## Logistic regression

# Create MiniBatchSGD batch sizes
batch_sizes = [5, 10, 20, 35, 50]  # The size of the test set is 50


# Define cross entropy cost function
def cost_fn(x, y, w):
    N, D = x.shape
    w_df = pd.DataFrame(w).T
    w = w_df.to_numpy()
    z = np.dot(x, w)
    J = np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))
    return J


c = []

print("Logistic Regression")
for size in batch_sizes:
    optimizer = MiniBatchSGD(batch_size=size, max_iters=300)

    # Create an instance of the LogisticRegressionSGC model
    log_reg = LogisticRegressionSGD()

    # Fit the model to the training data
    log_reg.fit(X_train_df2, y_train_df2, optimizer)

    w_hist = optimizer.w_history

    x_copy = X_train_df2.copy()
    x_copy.insert(0, 'X0', np.ones(X_train_df2.shape[0]))
    x_copy = x_copy.to_numpy()

    c.append([cost_fn(x_copy, y_train_df2.to_numpy(), [w_hist[j]]) for j in range(0, len(w_hist))])

    # Predict the target values for the test data
    y_pred_df2 = log_reg.predict(X_test_df2)

    # Evaluate the model's accuracy
    accuracy = np.mean(y_pred_df2 == y_test_df2)

    print("For batch size ", size, "the accuracy of the model is ", accuracy, ".")

# Plot convergence speed graph.
fig = plt.figure()

for i, b in enumerate(batch_sizes):
    print(c[i])
    plt.plot(c[i], marker='.', alpha=.998, label=f'Batch size = {b}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Convergence Speed')
plt.legend()
plt.show()

#-------------------------------------------------------Experiment 4: Different Learning Rates ----------------------

########################################################################### Linear Regression

learning_rates = [0.1, .01, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

c = []

print("Linear Regression, Y1: ")
for i, lr in enumerate(learning_rates):
    modelSGD = LinearRegressionSGD()

    # Create MiniBatchSGD optimizer
    optimizer = MiniBatchSGD(learning_rate=lr, batch_size=10, max_iters=300)

    modelSGD.fit(X_train_scaled, Y_df1_train["Y1"], optimizer)

    # For plotting training curve purposes
    w_hist = optimizer.w_history

    x_copy = X_train_scaled.copy()
    x_copy.insert(0, 'X0', np.ones(X_train_scaled.shape[0]))
    x_copy = x_copy.to_numpy()

    c.append([l2_cost_fn(x_copy, Y_df1_train["Y1"], [w_hist[j]]) for j in range(0, len(w_hist))])

    # Predict target values for test dataset
    predSGD = modelSGD.predict(X_test_scaled)

    # Evaluate performance
    MSE_SGD_Y1 = .5 * np.mean((predSGD["Y"] - Y_df1_test['Y1']) ** 2)

    print("For learning rate ", lr, " the error of the model is ", MSE_SGD_Y1, ".")

# Plot learning curves
fig = plt.figure()

for i, lr in enumerate(learning_rates):
    print(c[i])
    plt.plot(c[i], marker='.', alpha=.998, label=f'learning rate = {lr}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('training curves')
plt.legend()
plt.show()

c = []

print("Linear Regression, Y2: ")
for i, lr in enumerate(learning_rates):
    modelSGD = LinearRegressionSGD()

    # Create MiniBatchSGD optimizer
    optimizer = MiniBatchSGD(learning_rate=lr, batch_size=10, max_iters=300)

    modelSGD.fit(X_train_scaled, Y_df1_train["Y2"], optimizer)

    # For plotting training curve purposes
    w_hist = optimizer.w_history

    x_copy = X_train_scaled.copy()
    x_copy.insert(0, 'X0', np.ones(X_train_scaled.shape[0]))
    x_copy = x_copy.to_numpy()

    c.append([l2_cost_fn(x_copy, Y_df1_train["Y2"], [w_hist[j]]) for j in range(0, len(w_hist))])

    # Predict target values for test dataset
    predSGD = modelSGD.predict(X_test_scaled)

    # Evaluate performance
    MSE_SGD_Y2 = .5 * np.mean((predSGD["Y"] - Y_df1_test['Y2']) ** 2)

    print("For learning rate ", lr, " the error of the model is ", MSE_SGD_Y2, ".")

# Plot learning curve
fig = plt.figure()

for i, lr in enumerate(learning_rates):
    print(c[i])
    plt.plot(c[i], marker='.', alpha=.998, label=f'learning rate = {lr}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('training curves')
plt.legend()
plt.show()

######################################################################### Logistic Regression

lrs = [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]

c = []

print("Logistic Regression")
for lr in lrs:
    # Create MiniBatchSGD optimizer
    optimizer = MiniBatchSGD(learning_rate=lr, batch_size=30, max_iters=300)

    model = LogisticRegressionSGD()

    model.fit(X_train_df2, y_train_df2, optimizer)

    # For plotting training curve purposes
    w_hist = optimizer.w_history

    x_copy = X_train_df2.copy()
    x_copy.insert(0, 'X0', np.ones(X_train_df2.shape[0]))
    x_copy = x_copy.to_numpy()
    c.append([cost_fn(x_copy, y_train_df2.to_numpy(), [w_hist[j]]) for j in range(0, len(w_hist))])

    y_pred = model.predict(X_test_df2)
    score = np.mean(y_pred == y_test_df2)

    print("For learning rate ", lr, "the accuracy of the model is ", score, ".")

# Plot learning curves
fig = plt.figure()

for i, lr in enumerate(lrs):
    print(c[i])
    plt.plot(c[i], marker='.', alpha=.998, label=f'learning rate = {lr}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('training curves')
plt.legend()
plt.show()

#-------------Experiment 6: Analytical Lineaer regression vs Mini batch SGD linear regression ----------------------

# Does not need code, is explained in the report.


#------------------------------------ Extra Experiment: Mini batch SGD with Momentum ----------------------


#################################################################### Linear Regression

learning_rates = [0.1, .01, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

c = []

print("Linear Regression, Y1: ")
for i, lr in enumerate(learning_rates):
    modelSGD = LinearRegressionSGD()

    # Create MiniBatchSGD optimizer
    optimizer = MiniBatchSGDMomentum(learning_rate=lr, batch_size=10, max_iters=500)

    modelSGD.fit(X_train_scaled, Y_df1_train["Y1"], optimizer)

    # For plotting training curve purposes
    w_hist = optimizer.w_history

    x_copy = X_train_scaled.copy()
    x_copy.insert(0, 'X0', np.ones(X_train_scaled.shape[0]))
    x_copy = x_copy.to_numpy()

    c.append([l2_cost_fn(x_copy, Y_df1_train["Y1"], [w_hist[j]]) for j in range(0, len(w_hist))])

    # Predict target values for test dataset
    predSGD = modelSGD.predict(X_test_scaled)

    # Evaluate performance by computing mean square error for Y1 and Y2
    MSE_SGD_Y1 = .5 * np.mean((predSGD["Y"] - Y_df1_test['Y1']) ** 2)

    print("For learning rate ", lr, " the error of the model is ", MSE_SGD_Y1, ".")

# Plot learning curves
fig = plt.figure()

for i, lr in enumerate(learning_rates):
    print(c[i])
    plt.plot(c[i], marker='.', alpha=.998, label=f'learning rate = {lr}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('training curves')
plt.legend()
plt.show()

c = []

print("Linear Regression, Y2: ")
for i, lr in enumerate(learning_rates):
    modelSGD = LinearRegressionSGD()

    # Create MiniBatchSGD optimizer
    optimizer = MiniBatchSGDMomentum(learning_rate=lr, batch_size=10, max_iters=500)

    modelSGD.fit(X_train_scaled, Y_df1_train["Y2"], optimizer)

    # For plotting training curve purposes
    w_hist = optimizer.w_history

    x_copy = X_train_scaled.copy()
    x_copy.insert(0, 'X0', np.ones(X_train_scaled.shape[0]))
    x_copy = x_copy.to_numpy()

    c.append([l2_cost_fn(x_copy, Y_df1_train["Y2"], [w_hist[j]]) for j in range(0, len(w_hist))])

    # Predict target values for test dataset
    predSGD = modelSGD.predict(X_test_scaled)

    # Evaluate performance
    MSE_SGD_Y2 = .5 * np.mean((predSGD["Y"] - Y_df1_test['Y2']) ** 2)

    print("For learning rate ", lr, " the error of the model is ", MSE_SGD_Y2, ".")

# Plot learning curves
fig = plt.figure()

for i, lr in enumerate(learning_rates):
    print(c[i])
    plt.plot(c[i], marker='.', alpha=.998, label=f'learning rate = {lr}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('training curves')
plt.legend()
plt.show()

###################################################################### Logistic Regression

lrs = [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]

c = []

print("Logistic Regression")
for lr in lrs:
    # Create MiniBatchSGD optimizer
    optimizer = MiniBatchSGDMomentum(learning_rate=lr, batch_size=30, max_iters=500)

    model = LogisticRegressionSGD()

    model.fit(X_train_df2, y_train_df2, optimizer)

    # For plotting training curve purposes
    w_hist = optimizer.w_history

    x_copy = X_train_df2.copy()
    x_copy.insert(0, 'X0', np.ones(X_train_df2.shape[0]))
    x_copy = x_copy.to_numpy()
    c.append([cost_fn(x_copy, y_train_df2.to_numpy(), [w_hist[j]]) for j in range(0, len(w_hist))])

    y_pred = model.predict(X_test_df2)
    score = np.mean(y_pred == y_test_df2)

    print("For learning rate ", lr, "the accuracy of the model is ", score, ".")

# Plot learning curves
fig = plt.figure()

for i, lr in enumerate(lrs):
    print(c[i])
    plt.plot(c[i], marker='.', alpha=.998, label=f'learning rate = {lr}')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('training curves')
plt.legend()
plt.show()

