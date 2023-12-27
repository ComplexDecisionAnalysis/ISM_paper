import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_boolean_matrix(matrix):
    matrix[matrix != 0] = 1
    return(matrix)

def find_FRM(matrix):
    matrix_pre = get_boolean_matrix(matrix+np.eye(len(matrix))) # k == 1, (A + I)^k
    k = 2
    matrix_aft = get_boolean_matrix(np.linalg.matrix_power(get_boolean_matrix(matrix+np.eye(len(matrix))), k)) # k == 2, (A + I)^k
    while not (matrix_pre == matrix_aft).all():
        k += 1
        matrix_pre, matrix_aft = matrix_aft, get_boolean_matrix(np.linalg.matrix_power(get_boolean_matrix(matrix+np.eye(len(matrix))), k)) # k, k + 1
    return matrix_pre

def warshall_matrix(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if int(matrix[j][i]) == 1:
                for k in range(n):
                    matrix[j][k] = matrix[i][k] or matrix[j][k]
    return matrix

irm = np.array([[1,0,1,0,1,0,0,0,1,1,0,0],
                [0,1,0,1,0,0,1,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,1,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0,0],
                [0,1,0,1,0,0,1,0,0,0,0,1],
                [0,0,0,0,0,0,0,1,0,1,0,0],
                [1,0,1,0,1,0,0,0,1,1,0,0],
                [0,0,1,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,1,1,0],
                [0,0,0,1,1,0,0,0,0,0,0,1],
                ])


g = nx.from_numpy_array(irm, create_using = nx.DiGraph)
n = len(irm)
edgelist = [[f_node, t_node] for f_node,t_node,_ in nx.to_edgelist(g)]
ssim = []
for i in range(len(irm)):
    row = []
    for j in range(i, len(irm)):
        if [i, j] in edgelist:
            value = "X" if [j, i] in edgelist else "V"
        else:
            value = "A" if [j, i] in edgelist else "O"
        row.append(value)
    row = [""] * (n - len(row)) + row
    ssim.append(row)
print("Structural Self-interaction Matrix (SSIM)")
print(pd.DataFrame(ssim)) # Initial Reachaility Matrix
print("\n\n")


frm = warshall_matrix(irm) # same as find_FRM
# frm = find_FRM(irm)
# frm = get_boolean_matrix(irm+np.eye(len(irm)))
print("Final Reachability Matrix (M)")
print(pd.DataFrame(frm, columns = [*range(1,13)], index = [*range(1,13)]).astype(int))
print("\n\n")


matrix_m = frm - np.eye(len(frm))
sfrm = get_boolean_matrix(matrix_m - np.linalg.matrix_power(matrix_m, 2))
print("Streamed Final Reachability Matrix (SFRM)")
print(pd.DataFrame(sfrm, columns = [*range(1,13)], index = [*range(1,13)]).astype(int))
print("\n\n")

tfm = pd.DataFrame(get_boolean_matrix(irm + sfrm), columns = [*range(1,13)], index = [*range(1,13)]).astype(int)
antecedent = []
reachability = []
intersection = []
for i in range(len(tfm)):
    row = tfm.iloc[i, :].values
    row = np.where(row==1)[0] + 1
    column = tfm.iloc[:, i].values
    column = np.where(column==1)[0] + 1
    intersect = set(row).intersection(column)
    antecedent.append(row)
    reachability.append(column)
    intersection.append(intersect)

lp_df = pd.DataFrame({"name":[*range(1,len(tfm)+1)], "Reachability Set":reachability, "Antecedent Set":antecedent, "Intersection Set":intersection}, )
lp_df["Dependency Power"] = lp_df["Reachability Set"].apply(lambda x: len(x))
lp_df["Driving Power"] = lp_df["Antecedent Set"].apply(lambda x: len(x))
print("Begin level partitioning...")

print(lp_df)
print("\n\n")

def renew_df(lp_df):
    lp_df["Intersection Set"] = lp_df.apply(lambda x: set(x["Reachability Set"]).intersection(set(x["Antecedent Set"])), axis = 1)
    lp_df["Dependency Power"] = lp_df["Reachability Set"].apply(lambda x: len(x))
    lp_df["Driving Power"] = lp_df["Antecedent Set"].apply(lambda x: len(x))
    return lp_df

def del_df(lp_df, node_lst):
    tem = lp_df[["name","Reachability Set","Antecedent Set"]]
    tem = tem[tem["name"].apply(lambda x: True if x not in node_lst else False)]
    tem.loc[:,"Reachability Set"] = tem["Reachability Set"].apply(lambda x: [i for i in x if i not in node_lst])
    tem.loc[:,"Antecedent Set"] = tem["Antecedent Set"].apply(lambda x: [i for i in x if i not in node_lst])
    tem = renew_df(tem)
    return tem

print("Level Partitioning Iterations 1")
lvl = 1
dict_lvl = {}
closed = []
tem = lp_df[lp_df["Intersection Set"].apply(lambda x: True if len(x) == 1 else False)].reset_index(drop = True)
nodes = tem.loc[tem["Dependency Power"] == tem["Dependency Power"].max(), "name"].values
for node in nodes:
    dict_lvl[node] = lvl
lp_df = del_df(lp_df, nodes)
print(lp_df)
print("\n\n")


print("Level Partitioning Iterations 2")
lvl = 2
tem = lp_df[lp_df["Intersection Set"].apply(lambda x: True if len(x) == 1 else False)].reset_index(drop = True)
nodes = tem.loc[tem["Dependency Power"] == tem["Dependency Power"].max(), "name"].values
for node in nodes:
    dict_lvl[node] = lvl
lp_df = del_df(lp_df, nodes)
lp_df
print(lp_df)
print("\n\n")

print("Level Partitioning Iterations 3")
lvl = 3
nodes1 = lp_df.loc[lp_df["Dependency Power"] == lp_df["Dependency Power"].max(), "name"].values
closed.append(nodes1)
tem = lp_df[lp_df["Intersection Set"].apply(lambda x: True if len(x) == 1 else False)].reset_index(drop = True)
nodes2 = tem.loc[tem["Dependency Power"] == tem["Dependency Power"].max(), "name"].values
nodes = np.append(nodes1, nodes2)
nodes = np.append(nodes, [8, 11])
for node in nodes:
    dict_lvl[node] = lvl
lp_df = del_df(lp_df, nodes)
print(lp_df)
print("\n\n")


print("Level Partitioning Iterations 4")
lvl = 4
tem = lp_df[lp_df["Intersection Set"].apply(lambda x: True if len(x) == 1 else False)].reset_index(drop = True)
nodes = tem.loc[tem["Dependency Power"] == tem["Dependency Power"].max(), "name"].values
for node in nodes:
    dict_lvl[node] = lvl
lp_df = del_df(lp_df, nodes)
print(lp_df)


print("Level Partitioning Iterations 5")
lvl = 5
nodes1 = lp_df.loc[lp_df["Dependency Power"] == lp_df["Dependency Power"].max(), "name"].values
closed.append(nodes1)
tem = lp_df[lp_df["Intersection Set"].apply(lambda x: True if len(x) == 1 else False)].reset_index(drop = True)
nodes2 = tem.loc[tem["Dependency Power"] == tem["Dependency Power"].max(), "name"].values
nodes = np.append(nodes1, nodes2)
for node in nodes:
    dict_lvl[node] = lvl


def get_edgelist(matrix):
    from_n, to_n = np.where(matrix == 1)
    return [*zip(from_n, to_n)]


dict_node = {3: [3, -1],
             10: [3, -2],
             1: [2, -3],
             5: [3, -3],
             9: [4, -3],
             4: [6, -3],
             8: [1, -3],
             11: [5, -3],
             12: [3, -4],
             2: [3, -5],
             7: [4, -5],
             6: [5, -5]}
edgelist = list(set(get_edgelist(tfm)))
for cl in closed:
    if len(cl)%2 == 1:
        keep = cl[int((len(cl)+1)/2)-1]
    else:
        keep = cl[int((len(cl))/2)-1]
    res = [i for i in cl if i != keep]

    [edgelist.remove(edge) for edge in edgelist if (len(set(edge) & set([i-1 for i in res])) > 0) and (keep-1 not in edge)]
    
name = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]
dict_name = {i:j for i,j in zip(range(1,len(name)+1), name)}

fig, ax = plt.subplots() 
for i in dict_node:
    label = i
    x,y = dict_node[label]
    ax.scatter(x, y, s = 750, linewidth = 1, facecolor = "white", edgecolor = "black", zorder = 5)
    ax.text(x, y, dict_name[label], ha = "center", va = "center", fontfamily = "Times New Roman", fontsize = 16, zorder = 10)

for edge in edgelist:
    pfrom = dict_node[edge[0]+1]
    pto = dict_node[edge[1]+1]
    dx = pto[0] - pfrom[0]
    dy = pto[1] - pfrom[1]
    if - pfrom[1] + pto[1] <= 1:
        ax.annotate("", xy = pto, xytext = pfrom, arrowprops=dict(facecolor='black', shrink=0.1),
                horizontalalignment='left',
                verticalalignment='bottom')

xmin,xmax = ax.get_xlim()
gap = xmax - xmin
ax.set_xlim([xmin - gap/10, xmax + gap/10])

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_color("none")
ax.spines['bottom'].set_color("none")
ax.spines['left'].set_color("none")
ax.spines['right'].set_color("none")
ax.tick_params(axis='both', which = "both", length = 0)

fig.tight_layout()
plt.show()
