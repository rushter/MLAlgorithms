import csv
import numpy as np

def slr(X,Y):
    # The role of this function is to 
    # estimate or find out the beta_0 
    # and beta_1 value of the best 
    # fit line.

    # Eruption Duration Values (Input)
    # X = [3.1,3.2,2.6,......]

    # Waiting Time Values (Output)
    # Y = [88,65,75,......]

    X_bar = np.mean(X)
    Y_bar = np.mean(Y)

    # Xi-X_bar, Yi-Y_bar
    Xi_Xbar = []
    Xi_Xbar_2 =[]
    Xi_Xbar_mul_Yi_Ybar = []

    for i in range(0,len(X)):
        Xi_Xbar.append( X[i] - X_bar )
        Xi_Xbar_2.append((X[i] - X_bar)*(X[i] - X_bar))
        Xi_Xbar_mul_Yi_Ybar.append((X[i] - X_bar) * (Y[i] - Y_bar))


    final_numerator = np.sum(Xi_Xbar_mul_Yi_Ybar)

    final_denominator = np.sum(Xi_Xbar_2)

    beta_1 = final_numerator/final_denominator

   # print(beta_1)

    beta_0 = Y_bar - beta_1 * X_bar

    #print(beta_0)
    return [beta_0,beta_1]


def predict(x_predict, beta_values):                    # slr implementation 
    y_predict = beta_values[0] + beta_values[1]*x_predict
    
    return y_predict

    
def rss(actual_y, predicted_y):                         # calculating RSS value to calculate RSE
    rss_value = 0
    for i in range(len(actual_y)):
        error2 = (actual_y[i]-predicted_y[i])*(actual_y[i]-predicted_y[i])
        rss_value += error2
#     print("\nRss value is coming out to be ", rss_value)
    return rss_value

def rse(actual_y, predicted_y):                             # Calculating RSE value
    rss_val = rss(actual_y,predicted_y)
    n = len(actual_y)
    rse_val = (rss_val/n-2)**0.5
    
    
    print("\nRse value is ",rse_val)

def rsquare(actual_y, predicted_y):                                 # Calculating R-Square value
    tss = 0
    m = np.mean(actual_y)
    for k in range(len(actual_y)):
        var = (actual_y[k]-m)*(actual_y[k]-m)
        tss += var
    rss_valu = rss(actual_y, predicted_y)    
    r2 = (tss-rss_valu)/tss
    
    print('\nR Square value is ', r2)






f = open("geyser.csv",'r')

reader = csv.reader(f)

X = []
Y = []

for row in reader:
    if row[1] != "eruptions":
        X.append(float(row[1]))
        Y.append(float(row[2]))

f.close()

betas = slr(X,Y)

ypredicted = []
for xp in X:
    ypredicted.append(predict(xp,betas))
rse(Y,ypredicted)
rsquare(Y,ypredicted)
