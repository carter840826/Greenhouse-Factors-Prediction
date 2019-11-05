import pandas as pd
import numpy as np
from sklearn import cross_validation, ensemble, preprocessing, metrics
import matplotlib.pyplot as plt  

#leading
t = 474




train = pd.read_csv('out to in_train1.csv', index_col = 0)
train_X = pd.DataFrame([train['aveb'],train['pret'],train['preb']]).T
train_y = train['avet']

test = pd.read_csv('prediction_480.csv', index_col = 0)
test_X = pd.DataFrame([test['preds2'],test['pret'],test['preb']]).T
test_y = test['avet']

bag = ensemble.BaggingRegressor(n_estimators = 100)
bag_fit = bag.fit(train_X, train_y)


# leading time 1 hr
preds = bag.predict(test_X)


test_X['aveth'] = test['aveth']
test_X['preaveth'] = test['preaveth']
test_X['prepreaveth'] = test['prepreaveth']
test_X['temp_1hr'] = preds
test_X.to_csv('preds480.csv')


sim = pd.read_csv('preds480.csv', index_col = 0)


# leading time 2 hr
for i in range(1,len(test_y)):
    j = i-1
    sim_X = pd.DataFrame([sim['preds2'][i],sim['temp_1hr'][i-1],sim['preds2'][i-1]]).T
    preds[i] = bag.predict(sim_X)
    
    

test_X['temp_2hr'] = preds
test_X.to_csv('preds480.csv')

sim = pd.read_csv('preds480.csv', index_col = 0)


#leading time 3 hr 以上


for p in range(3,t+1):
    for i in range(p-1,len(test_y)):
        q = p-1
        r = p-2
        sim_X = pd.DataFrame([sim['preds2'][i],sim['temp_1hr'][i-1],sim['preds2'][i-1]]).T
        preds[i] = bag.predict(sim_X)

        

    test_X['temp_'+str(p)+'hr'] = preds
    test_X.to_csv('preds480.csv')

    sim = pd.read_csv('preds480.csv', index_col = 0)






#===============================
#繪圖


sim = pd.read_csv('preds480.csv', index_col = 0)
p1 = test['avet']
p2 = np.zeros(len(test_y))
p3 = np.zeros(len(test_y))

#需修改比較長度
p2[0] = test['avet'][0]
for f in range(1,len(test_y)):
            p2[f] = sim['temp_'+str(f)+'hr'][f]

test_X['temp_prediction'] = p2
test_X.to_csv('preds480.csv')
#,sim['sim_th_13hr'][f],sim['sim_th_14hr'][f],sim['sim_th_15hr'][f],sim['sim_th_16hr'][f],sim['sim_th_17hr'][f],sim['sim_th_18hr'][f],sim['sim_th_19hr'][f],sim['sim_th_20hr'][f],sim['sim_th_21hr'][f],sim['sim_th_22hr'][f],sim['sim_th_23hr'][f],sim['sim_th_24hr'][f]

fig,ax = plt.subplots()
x = np.arange(0,len(test_y),1)  
plt.plot(x,p1,label = 'Measured data')  
plt.plot(x,p2,color='red',linestyle='--',label='Machine learning')
plt.legend(loc='lower right')
plt.xlim(0,50)
plt.ylim(10,20)
#plt.text(0,0.4,'leading time '+str(t)+' hr')
plt.ylabel('Soil temperature (°C)')
plt.xlabel('Time (hr)')
plt.savefig('D:paper\Figure_ML.temp.preds480.png',dpi=600)
plt.show()





