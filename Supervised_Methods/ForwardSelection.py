import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def forward_selection(args):
    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = args.split_size, shuffle=True, random_state = args.seed)

    variables = X_train.columns.tolist()
    y = y_train
    
    forward_variables = []

    sl_enter = args.sl_enter # 선택 임계치
    sl_remove = args.sl_remove # 제거 임계치
    sv_per_step = [] # 각 스텝별 선택된 변수들
    adj_r_squared_list = [] # 각 스텝별 수정된 결정계수
    
    steps = []
    step = 0

    while len(variables) > 0:
        remainder = list(set(variables) - set(forward_variables))
        pval = pd.Series(index=remainder) # 변수의 p-value
        
        for col in remainder:
            X = X_train[forward_variables+[col]]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit(disp = 0)
            pval[col] = model.pvalues[col]

        min_pval = pval.min()
        
        if min_pval < sl_enter: # p-value 값이 기준 값 보다 작으면 포함
            forward_variables.append(pval.idxmin())
            while len(forward_variables) > 0:
                selected_X = X_train[forward_variables]
                selected_X = sm.add_constant(selected_X)
                selected_pval = sm.OLS(y, selected_X).fit(disp=0).pvalues[1:]
                max_pval = selected_pval.max()

                if max_pval >= sl_remove:
                    remove_variable = selected_pval.idxmax()
                    forward_variables.remove(remove_variable)

                else:
                    break
            
            step += 1
            steps.append(step)
            adj_r_squared = sm.OLS(y, sm.add_constant(X_train[forward_variables])).fit(disp=0).rsquared_adj
            adj_r_squared_list.append(adj_r_squared)
            sv_per_step.append(forward_variables.copy())
        
        else:
            break

    fig = plt.figure(figsize=(10,10))
    fig.set_facecolor('white')

    font_size = 15

    plt.xticks(steps,[f'step {s}\n'+'\n'.join(sv_per_step[i]) for i,s in enumerate(steps)], fontsize=font_size)
    plt.plot(steps, adj_r_squared_list, marker='o')
    plt.ylabel('adj_r_squared',fontsize=font_size)
    plt.grid(True)
    plt.show()

