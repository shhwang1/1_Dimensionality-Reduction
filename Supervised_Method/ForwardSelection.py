import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def forward_selection(args):
    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    variables = X_data.columns.tolist()
    
    forward_variables = []

    sl_enter = args.sl_enter # selection threshold
    sl_remove = args.sl_remove # elimination threshold
    sv_per_step = [] # Variables selected in each steps
    adj_r_squared_list = [] # Adjusted R-Square in each steps
    
    steps = []
    step = 0

    while len(variables) > 0:
        remainder = list(set(variables) - set(forward_variables))
        pval = pd.Series(index=remainder) # P-value
        
        for col in remainder:
            X = X_data[forward_variables+[col]]
            X = sm.add_constant(X)
            model = sm.OLS(y_data, X).fit(disp = 0)
            pval[col] = model.pvalues[col]

        min_pval = pval.min()
        
        if min_pval < sl_enter: # include it if p-value is lower than threshold
            forward_variables.append(pval.idxmin())
            while len(forward_variables) > 0:
                selected_X = X_data[forward_variables]
                selected_X = sm.add_constant(selected_X)
                selected_pval = sm.OLS(y_data, selected_X).fit(disp=0).pvalues[1:]
                max_pval = selected_pval.max()

                if max_pval >= sl_remove:
                    remove_variable = selected_pval.idxmax()
                    forward_variables.remove(remove_variable)

                else:
                    break
            
            step += 1
            steps.append(step)
            adj_r_squared = sm.OLS(y_data, sm.add_constant(X_data[forward_variables])).fit(disp=0).rsquared_adj
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

