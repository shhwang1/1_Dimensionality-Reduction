import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def backward_selection(args):
    
    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    
    initial_list = []
    threshold_out = 0.0
    feature_list = X_data.columns.tolist()

    sv_per_step = [] ## 각 스텝별로 선택된 변수들
    adj_r_squared_list = [] ## 각 스텝별 수정된 결정계수
    steps = [] ## 스텝
    step = 0
    included = feature_list

    for num in range(len(feature_list)-1):
        # changed=False
        model = sm.OLS(y_data, sm.add_constant(pd.DataFrame(X_data[included]))).fit(disp=0)
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:] # P-value of each variable
        worst_pval = pvalues.max()	# choose variable with best p-value
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
        
        step += 1
        steps.append(step)        
        adj_r_squared = sm.OLS(y_data, sm.add_constant(pd.DataFrame(X_data[included]))).fit(disp=0).rsquared_adj
        adj_r_squared_list.append(adj_r_squared)
        sv_per_step.append(included.copy())
        
        # if not changed:
        #     break

    fig = plt.figure(figsize=(10,10))
    fig.set_facecolor('white')

    font_size = 15
    plt.xticks(steps,[f'step {s}\n'+'\n'.join(sv_per_step[i]) for i,s in enumerate(steps)], fontsize=7)
    plt.plot(steps, adj_r_squared_list, marker='o')

    plt.ylabel('adj_r_squared',fontsize=font_size)
    plt.grid(True)
    plt.show()

    # return included,step,steps,adj_r_squared_list,sv_per_step
