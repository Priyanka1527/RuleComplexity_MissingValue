#-------Importing Libraries-------#
import pandas
import itertools
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from collections import defaultdict

#-------Method to find the counts of Rule and Condiion-------#
def findRuleComplexity(lines):
    rule_count = 0
    condition_count = 0
    for item in lines:
        item = item[1:len(item)-2]
        item = item.replace(", ","\n")
        item = item.split("\n")

        rule_count += len(item)
        for rule in item:
            left,right = rule.split("->")
            conditions = left.split("&")
            length = len(conditions)
            condition_count += length
    return rule_count,condition_count

#-------When no missing attributes are present-------#
f1 = open('../data/Iris/Rules/Iris_main.txt', "r")

data = {}
data["Lower," + "?"] = list(f1)
data["Middle," + "?"] = data["Lower," + "?"]
data["Upper," + "?"] = data["Lower," + "?"]
data["Lower," + "-"] = data["Lower," + "?"]
data["Middle," + "-"] = data["Lower," + "?"]
data["Upper," + "-"] = data["Lower," + "?"]

Iris_0 = {}
for key, value in data.items():
    rule,condition = findRuleComplexity(value)
    Iris_0[key] = [rule]
    Iris_0[key].append(condition)

#-------5% missing values-------#
f1 = open('../data/Iris/Rules/Iris_5_lost_lower.txt', "r")
f2 = open('../data/Iris/Rules/Iris_5_lost_middle.txt', "r")
f3 = open('../data/Iris/Rules/Iris_5_lost_upper.txt', "r")
f4 = open('../data/Iris/Rules/Iris_5_attcon_lower.txt', "r")
f5 = open('../data/Iris/Rules/Iris_5_attcon_middle.txt', "r")
f6 = open('../data/Iris/Rules/Iris_5_attcon_upper.txt', "r")

data = {}
data["Lower," + "?"] = list(f1)
data["Middle," + "?"] = list(f2)
data["Upper," + "?"] = list(f3)
data["Lower," + "-"] = list(f4)
data["Middle," + "-"] = list(f5)
data["Upper," + "-"] = list(f6)

Iris_5 = {}
for key, value in data.items():
    rule,condition = findRuleComplexity(value)
    Iris_5[key] = [rule]
    Iris_5[key].append(condition)

#-------10% missing values-------#
f1 = open('../data/Iris/Rules/Iris_10_lost_lower.txt', "r")
f2 = open('../data/Iris/Rules/Iris_10_lost_middle.txt', "r")
f3 = open('../data/Iris/Rules/Iris_10_lost_upper.txt', "r")
f4 = open('../data/Iris/Rules/Iris_10_attcon_lower.txt', "r")
f5 = open('../data/Iris/Rules/Iris_10_attcon_middle.txt', "r")
f6 = open('../data/Iris/Rules/Iris_10_attcon_upper.txt', "r")

data = {}
data["Lower," + "?"] = list(f1)
data["Middle," + "?"] = list(f2)
data["Upper," + "?"] = list(f3)
data["Lower," + "-"] = list(f4)
data["Middle," + "-"] = list(f5)
data["Upper," + "-"] = list(f6)

Iris_10 = {}
for key, value in data.items():
    rule,condition = findRuleComplexity(value)
    Iris_10[key] = [rule]
    Iris_10[key].append(condition)

#-------15% missing values-------#    
f1 = open('../data/Iris/Rules/Iris_15_lost_lower.txt', "r")
f2 = open('../data/Iris/Rules/Iris_15_lost_middle.txt', "r")
f3 = open('../data/Iris/Rules/Iris_15_lost_upper.txt', "r")
f4 = open('../data/Iris/Rules/Iris_15_attcon_lower.txt', "r")
f5 = open('../data/Iris/Rules/Iris_15_attcon_middle.txt', "r")
f6 = open('../data/Iris/Rules/Iris_15_attcon_upper.txt', "r")

data = {}
data["Lower," + "?"] = list(f1)
data["Middle," + "?"] = list(f2)
data["Upper," + "?"] = list(f3)
data["Lower," + "-"] = list(f4)
data["Middle," + "-"] = list(f5)
data["Upper," + "-"] = list(f6)

Iris_15 = {}
for key, value in data.items():
    rule,condition = findRuleComplexity(value)
    Iris_15[key] = [rule]
    Iris_15[key].append(condition)

#-------20% missing values-------# 
f1 = open('../data/Iris/Rules/Iris_20_lost_lower.txt', "r")
f2 = open('../data/Iris/Rules/Iris_20_lost_middle.txt', "r")
f3 = open('../data/Iris/Rules/Iris_20_lost_upper.txt', "r")
f4 = open('../data/Iris/Rules/Iris_20_attcon_lower.txt', "r")
f5 = open('../data/Iris/Rules/Iris_20_attcon_middle.txt', "r")
f6 = open('../data/Iris/Rules/Iris_20_attcon_upper.txt', "r")

data = {}
data["Lower," + "?"] = list(f1)
data["Middle," + "?"] = list(f2)
data["Upper," + "?"] = list(f3)
data["Lower," + "-"] = list(f4)
data["Middle," + "-"] = list(f5)
data["Upper," + "-"] = list(f6)

Iris_20 = {}
for key, value in data.items():
    rule,condition = findRuleComplexity(value)
    Iris_20[key] = [rule]
    Iris_20[key].append(condition)

#-------25% missing values-------# 
f1 = open('../data/Iris/Rules/Iris_25_lost_lower.txt', "r")
f2 = open('../data/Iris/Rules/Iris_25_lost_middle.txt', "r")
f3 = open('../data/Iris/Rules/Iris_25_lost_upper.txt', "r")
f4 = open('../data/Iris/Rules/Iris_25_attcon_lower.txt', "r")
f5 = open('../data/Iris/Rules/Iris_25_attcon_middle.txt', "r")
f6 = open('../data/Iris/Rules/Iris_25_attcon_upper.txt', "r")

data = {}
data["Lower," + "?"] = list(f1)
data["Middle," + "?"] = list(f2)
data["Upper," + "?"] = list(f3)
data["Lower," + "-"] = list(f4)
data["Middle," + "-"] = list(f5)
data["Upper," + "-"] = list(f6)

Iris_25 = {}
for key, value in data.items():
    rule,condition = findRuleComplexity(value)
    Iris_25[key] = [rule]
    Iris_25[key].append(condition)

#-------30% missing values-------# 
f1 = open('../data/Iris/Rules/Iris_30_lost_lower.txt', "r")
f2 = open('../data/Iris/Rules/Iris_30_lost_middle.txt', "r")
f3 = open('../data/Iris/Rules/Iris_30_lost_upper.txt', "r")
f4 = open('../data/Iris/Rules/Iris_30_attcon_lower.txt', "r")
f5 = open('../data/Iris/Rules/Iris_30_attcon_middle.txt', "r")
f6 = open('../data/Iris/Rules/Iris_30_attcon_upper.txt', "r")

data = {}
data["Lower," + "?"] = list(f1)
data["Middle," + "?"] = list(f2)
data["Upper," + "?"] = list(f3)
data["Lower," + "-"] = list(f4)
data["Middle," + "-"] = list(f5)
data["Upper," + "-"] = list(f6)

Iris_30 = {}
for key, value in data.items():
    rule,condition = findRuleComplexity(value)
    Iris_30[key] = [rule]
    Iris_30[key].append(condition)

#-------35% missing values-------# 
f1 = open('../data/Iris/Rules/Iris_35_lost_lower.txt', "r")
f2 = open('../data/Iris/Rules/Iris_35_lost_middle.txt', "r")
f3 = open('../data/Iris/Rules/Iris_35_lost_upper.txt', "r")
f4 = open('../data/Iris/Rules/Iris_35_attcon_lower.txt', "r")
f5 = open('../data/Iris/Rules/Iris_35_attcon_middle.txt', "r")
f6 = open('../data/Iris/Rules/Iris_35_attcon_upper.txt', "r")

data = {}
data["Lower," + "?"] = list(f1)
data["Middle," + "?"] = list(f2)
data["Upper," + "?"] = list(f3)
data["Lower," + "-"] = list(f4)
data["Middle," + "-"] = list(f5)
data["Upper," + "-"] = list(f6)

Iris_35 = {}
for key, value in data.items():
    rule,condition = findRuleComplexity(value)
    Iris_35[key] = [rule]
    Iris_35[key].append(condition)

#-------Rule Count-------#
dd = defaultdict(list)

for d in (Iris_0, Iris_5, Iris_10, Iris_15, Iris_20, Iris_25, Iris_30, Iris_35): 
    for key, value in d.items():
        dd[key].append(value[0])

#------Condition Count-------#

cc = defaultdict(list)

for d in (Iris_0, Iris_5, Iris_10, Iris_15, Iris_20, Iris_25, Iris_30, Iris_35): 
    for key, value in d.items():
        cc[key].append(value[1])

df = pd.DataFrame.from_dict(dd)
df2 = pd.DataFrame.from_dict(cc)

#-------Plotting-------#
x = np.arange(0, 40, 5)
df['x'] = x
df2['x'] = x

#-------Rule Count-------#
ax = plt.gca()
ax.grid(True)
plt.gcf().set_size_inches(10, 5)

tck1 = splrep(df['x'], df['Lower,?'])
xnew1 = np.linspace(0, 35)
ynew1 = splev(xnew1, tck1)
plt.plot(xnew1, ynew1)

tck2 = splrep(df['x'], df['Middle,?'])
xnew2 = np.linspace(0, 35)
ynew2 = splev(xnew2, tck2)
plt.plot(xnew2, ynew2)

tck3 = splrep(df['x'], df['Upper,?'])
xnew3 = np.linspace(0, 35)
ynew3 = splev(xnew3, tck3)
plt.plot(xnew3, ynew3)

tck4 = splrep(df['x'], df['Lower,-'])
xnew4 = np.linspace(0, 35)
ynew4 = splev(xnew4, tck4)
plt.plot(xnew4, ynew4)

tck5 = splrep(df['x'], df['Middle,-'])
xnew5 = np.linspace(0, 35)
ynew5 = splev(xnew5, tck5)
plt.plot(xnew5, ynew5)

tck6 = splrep(df['x'], df['Upper,-'])
xnew6 = np.linspace(0, 35)
ynew6 = splev(xnew6, tck6)
plt.plot(xnew6, ynew6)

plt.title("Iris", fontsize=15)
plt.xlabel('Missing Attributes (%)', fontsize=15)
plt.ylabel('Rule Count', fontsize=15)
plt.plot(df['x'], df['Lower,?'], 'o', color='blue', markerfacecolor='blue')
plt.plot(df['x'], df['Middle,?'], 's', color='red', markerfacecolor='red')
plt.plot(df['x'], df['Upper,?'], '^', color='green', markerfacecolor='green')
plt.plot(df['x'], df['Lower,-'], 'x', color='magenta', markerfacecolor='magenta')
plt.plot(df['x'], df['Middle,-'], 'D', color='aqua', markerfacecolor='aqua')
plt.plot(df['x'], df['Upper,-'], 'P', color='orange', markerfacecolor='orange')
plt.legend()
plt.savefig('../Results/Iris-RuleTest.png')
#------------END---------------#