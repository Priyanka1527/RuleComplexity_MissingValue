#-------Importing Libraries-------#
import time
import pandas as pd
import numpy as np
import pylab as pl
from pandas.api.types import is_numeric_dtype
from functools import reduce
from itertools import groupby
from collections import OrderedDict

#-------Definition of all the Functions-------#

#-------Discretization-------#
def discretize(numeric_col):
    #Sorting the values of numeric column
    
    list1 = df[numeric_col].tolist()
    list1 = [ elem for elem in list1 if elem != '?']
    list1 = [float(x) for x in list1]
    sort_col = sorted(list1)
   
    point_list = set(sort_col)
    point_list = sorted(list(point_list))
    print(point_list)
    
    #Finding average between each two points
    avg_list = []
    for i in range(len(point_list)-1):
        avg = (point_list[i] + point_list[i+1])/2
        avg_list.append(round(float(avg),3))
    print(avg_list)
 
    #Performing the discretization and adding the cases
    for i in avg_list:
        case = str(numeric_col) + "," + str(point_list[0]) + ".." + str(i)
        case2 = str(numeric_col) + "," + str(i) + ".." + str(point_list[-1])
    
        case_list.append(case)
        case_list.append(case2)

#-------Computing Lower Approximation-------#
def lowerApproximation(charac_set,concept):
    #set to contain lower approximations
    lower = set()
    
    #Check for each element of the concept
    for item in concept:
        key = 'K_%d' % (item)
        set_value = charac_set[key]
  
        if set_value.issubset(set(concept)):
            lower = lower.union(set_value)
          
    return lower

#-------Computing Upper Approximation-------#
def upperApproximation(charac_set,concept):
    #set to contain lower approximations
    upper = set()
    
    #Check for each element of the concept
    for item in concept:
        key = 'K_%d' % (item)
        set_value = charac_set[key]
        upper = upper.union(set_value)
          
    return upper

#-------Computing Concept Probabilistic Approximation-------#
def probabilisticApproximation(concept):
    prob = []
    for index, row in prob_approx.iterrows():
        if len(row['charset_value']) != 0:
            probability_conditional = len(row['charset_value'].intersection(set(concept))) / len(row['charset_value'])
        else:
            probability_conditional = 0.0
            
        prob.append(round(probability_conditional,2))
    return prob

#-------Computing Middle Approximation-------#
def findMiddleApprox(concept):
    probapprox = set()
    cond_prob = probabilisticApproximation(concept) #Need to put the goal
    prob_approx['cond_probability'] = cond_prob
    for index, row in prob_approx.iterrows():
        part1,part2 = row['charset_name'].split("_")
       
        if row['cond_probability'] >= 0.50:
            if int(part2) in concept:
                probapprox = probapprox.union(row['charset_value'])
    return probapprox   

#-------Finding goal interescts for MLEM2-------#
def findGoalIntersect(goal):
    goalIntersect = []
    
    for index, row in df3.iterrows():
        #List containing intersection of (a,v) pairs and goal
        goalIntersect.append(set(row['att_val']).intersection(set(goal)))
          
    #Check if goal_intersect column exists
    if 'goal_intersect' in df3:
        df3['goal_intersect'] = goalIntersect
    else:
        #Insert new column with the recent iteration
        df3.insert(2, 'goal_intersect', goalIntersect)

#-------Updating goal interescts for MLEM2-------#
def updateGoalIntersect(goal):
    for index, row in df3.iterrows():
        if row['goal_intersect'] != set():
            row['goal_intersetct'] = set(row['att_val']).intersection(set(goal))

#-------Finding a case for MLEM2-------#
def findCases(df3):
    case_to_be = []
    #Find the cases with maximum goal coverage
    m = max(df3['goal_intersect'], key=len)
    possible_cases = [i for i, j in enumerate(df3['goal_intersect'].tolist()) if j == m]
    
    #Index of the case covering max goal and having min no. of elements
    new_df = df3.iloc[possible_cases,:]
    
    m1 = min(new_df['att_val'], key=len)
    
    for index,row in new_df.iterrows():
        if row['att_val'] == m1:
            case_to_be.append(index)
   
    return case_to_be[0]

#-------Combining intervals for MLEM2-------#
def combineInterval(test_condition):
    
    test_num = [] #This will contain the conditions having interval
    test_str = [] #This will contain the conditions having no interval
    
    #Loop through to seprate contions having intervals and no intervals
    for item in test_condition:
        if ".." in item:
            test_num.append(item)
        else:
            test_str.append(item)
   
    #Group the conditions having interval based on same attributes
    grouped = [list(g) for k, g in groupby(test_num, lambda s: s.partition(',')[0])]
    
    final_list = []
    
    #Actually combining the intervals
    for list1 in grouped:
        greatest = 0
        smallest = 0
        for item in list1:
            part1,part2 = item.split(",")
            start,stop = part2.split("..")
            start = float(start)
            stop = float(stop)

            if greatest == 0 and smallest == 0:
                greatest = start
                smallest = stop

            if start > greatest:
                greatest = float(start)

            if stop < smallest:
                smallest = float(stop)

        con_tmp = part1 + "," + str(greatest) + ".." + str(smallest)
        final_list.append(con_tmp)  
            
    actual_condition = final_list + test_str
    
    return actual_condition

#-------Logic to drop condition for MLEM2-------#
def dropCondition(condition,current_goal):
    
    for item in range(0,len(condition)):
        temp_att_val = []
        temp_cond = condition.copy() #use list.copy() as equal operator simply copies over the reference
        temp_cond.remove(condition[item])
       
        #temp_cond contains the elements after removing the current element
        for i in temp_cond:
            if i is not None:
                location = df3.index[df3['Cases'] == i].tolist()
                element = df3['att_val'].loc[location[0]]
                temp_att_val.append(set(element))
          
        #temp_att_val contains the actual value set of the corresponding cases
        #Find the intersection if the set has more than one element, otherwise no need
        if len(temp_att_val) > 1:
            intersection = set.intersection(*temp_att_val)
        
            #if the set still remains a subset of the original goal after removing current element
            #then set the current element to None as we want to drop this later
            if intersection.issubset(current_goal):
                condition[item] = None
    
    condition = [x for x in condition if x is not None]
   
    return condition 

#-------Actual MLEM2 Algorithm-------#
def stepAlgo(df3,selected_case,current_goal,B,condition,concept_curr):
    
    rule_set = []
    original_goal = current_goal.copy()
    
    
    while current_goal != set():
        #Check if the selected case is a subset of the current goal
        #List of current case
        A = df3['att_val'].loc[selected_case] 

        if B == set():
            #Copy over the current set elements to B
            for i in range(len(A)):
                B.add(A[i])

        #Elements of intersection of current and previous set
        A = set(A).intersection(B)
        B = A.copy()
        
        print("Intersection set: ", A)
        #Check if intersecting elements are subset of Goal
        if A.issubset(original_goal):
            print("SUBSET")
            #Current goal is updated after discarding the already covered goal by new rule
            if len(A) != 0:
                current_goal = set(current_goal) - df3['goal_intersect'].loc[selected_case]
            else:
                current_goal = set()
            
            print("Current goal: ", current_goal)
            #Extract the current case
            curr_case = df3['Cases'].loc[selected_case]
            #Add the conditions of a Rule
            condition.append(curr_case)
            
            print("Condition before drop or combine: ", condition)
            #Check for possibility of dropping conditions
            if len(condition) > 1:
                condition = dropCondition(condition,original_goal)
            
            #Combine the interval
            if len(condition) > 1:
                condition = combineInterval(condition)
            

            #Join conditions
            cond = ""
            for item in condition:
                cond = cond + "(" + str(item) + ")" + " & "

            cond = cond[:-2] + "->"
            rule = cond + " (" + concept + "," + concept_curr + ")"
            rule_set.append(rule)
            
            #Reset everythng and continue for covering rest of the goal
            condition = []
            B = set()
            findGoalIntersect(current_goal)
            selected_case = findCases(df3)
            
        #If not a subset of current goal
        else:
            print("NOT")
            #Assign empty set for the selected case for next iteration
            df3['goal_intersect'].loc[selected_case] = set()
            print("need to be set to NULL", df3['Cases'].loc[selected_case])
           
            #Extract the current case
            curr_case = df3['Cases'].loc[selected_case]
            #Add the case to the condition list
            condition.append(curr_case)
            
            #Check for Range overlapping of the remaining cases
            if ".." in curr_case:
                second_part = curr_case.split(',')[1]
                start = float(second_part.split('..')[0])
                end = float(second_part.split('..')[1])
                for index, row in df3.iterrows():
                    if ".." in row['Cases'] and row['Cases'] == curr_case:
                        part2 = row['Cases'].split(',')[1]
                        start1 = float(part2.split('..')[0])
                        end1 = float(part2.split('..')[1])

                        #Assign blank set for cases with overlapping ranges
                        if set((pl.frange(start,end))).issubset(pl.frange(start1,end1)) == True:
                            if start == start1 and end <= end1:
                                row['goal_intersect'] = set()
                                print("also need to be set to NULL", row['Cases'])
                                
            updateGoalIntersect(A.intersection(current_goal))
            selected_case = findCases(df3)
                                   
    return rule_set

#-------Reading Input Dataset-------#
df = pd.read_csv('../data/Iris/Iris-35-lost.csv')

#--------Find the Decision and unique Concepts-------#
df_headers = list(df)
concept = df_headers[-1]
concept_list = df[concept].unique()

#-------Calculating cases by concepts and making list of Goals-------#
#universal list containing all cases
U = [] 
temp_list = []
goal_list = []
for item in concept_list:
    for index, row in df.iterrows():
        U.append(index+1)
        if row[concept] == item:
            temp_list.append(index)
    goal_list.append(temp_list)
    temp_list = []

#-------Find out all the attributes in the dataset-------#
attributes = list(df)

#-------Find the numeric column-------#
numeric_col = df.select_dtypes(include=[np.number]).columns.tolist()

#-------Build all the cases-------#
case_list = []

#Discretization considering upto 2 decimal point
for item in attributes[:-1]:
        discretize(item)

case_list = list(OrderedDict.fromkeys(case_list))
#-------Calculating cases by concepts and making sets-------#
temp_list = []
goal_list = []
for item in concept_list:
    for index, row in df.iterrows():
        if row[concept] == item:
            temp_list.append(index+1)
    goal_list.append(temp_list)
    temp_list = []

#-------Calculating blocks for attribute-value pairs-------#
temp_list = []
att_val_list = []
for item in case_list:
    a,b = item.split(",") #a = attribute and b = value
  
    if ".." in b:
        start,end = b.split("..")
        for index, row in df.iterrows():
            if row[a] is not '?':
                if ".." not in row[a]:
                    if float(row[a]) >= float(start) and float(row[a]) <= float(end):
                        temp_list.append(index+1)
                else:
                    if row[a] == b:
                         temp_list.append(index+1)
                    
        
        att_val_list.append(temp_list)
        temp_list = []
        
    else:
        for index, row in df.iterrows():
            if type(row[a]) == list:
                tmp_list = row[a]
                for case in tmp_list:
                    if float(case) == float(b):
                        temp_list.append(index+1)
                        
            if row[a] == b:
                temp_list.append(index+1)
       
        att_val_list.append(temp_list)
        temp_list = []

#-------Creating data for case and att-value list-------#
data = {'Cases': case_list, 'att_val': att_val_list}
df2 = pd.DataFrame(data)

#-------Here starts the Approximation calculations-------#
attributes = list(df)
U = set(U)

case_list = []
#Loop through all the attributes except last - Concept
for item in attributes[:-1]:
    print(item)
    
    #check for non numeric columns
    if not is_numeric_dtype(df[item]):
        temp = df[item].unique()
        for i in temp:
            if i == '?' or i == '-':
                continue
            else:
                case = item + "," + i
                case_list.append(case)

temp_list = []
att_val_list = []
for item in case_list:
    a,b = item.split(",") #a = attribute and b = value
    for index, row in df.iterrows():
        if type(row[a]) == list:
            tmp_list = row[a]
            for case in tmp_list:
                if case == b:
                    temp_list.append(index+1)

        if row[a] == b:
            temp_list.append(index+1)

    att_val_list.append(temp_list)
    temp_list = []

#-------reating dictionary combining case_list and att_val list--------#
block = dict(zip(case_list, att_val_list))

#-------Building Characteristic Sets-------#
dic = {}
for index, row in df.iterrows():
    tmp_set = set()
    final_union = []
    char_list = []
    char_list_2 = []
    final_union_set = []
  
    for cols in attributes[:-1]:
        #If the value for corresponding attribute is a list then create all of the att-value pairs
        if type(df.loc[index,cols]) == list:
            print("When values are list") 
            for item in df.loc[index,cols]:
                block_key = cols + "," + item
                char_list.append(block_key) #char_list has all att-val cases
                print(char_list)
                
            union_set = set()
            #Compute union of att-concept value case
            for item in char_list:
                union_set = union_set.union(set(block[item]))
            
            print("Union Set: ", union_set)
            final_union.append(union_set)
        
        else:
            print("When value is single")
            block_key = cols + "," + str(df.loc[index,cols])
            char_list_2.append(block_key) #char_list_2 has all single cases
            print(char_list_2)
   
    #Compute instersection for this current row for Characteristics set
    
    print("final_union: ", final_union)
    if len(final_union):
        final_union_set = list(reduce(set.intersection, [set(item) for item in final_union]))
        
    
    print("Final Union Set: ", final_union_set)
    
    for item in char_list_2:
        if item in block:
            print("When item is found as key in the block")
            if tmp_set == set():
                #Copy over the current set elements to B
                for i in range(len(block[item])):
                    tmp_set.add(block[item][i])
                    
            tmp_set = tmp_set.intersection(set(block[item]))
            print(tmp_set)
        
        #If item not in block
        else:
            print("When item is not found as key in the block")
            print(tmp_set)
            print(U)
            if tmp_set == set():
                tmp_set = U
                
            tmp_set = tmp_set.intersection(U)
            print(tmp_set)
    
    print("Final tmp_set: ", tmp_set)
    final_union_set = set(final_union_set)
    
    if final_union_set == set():
        tmp_set = tmp_set
    else:
        tmp_set = tmp_set.intersection(final_union_set)
        
    print("This is the final value: ", tmp_set)
        
    key = ('K_%d' % (index+1))
    print(key)
    dic[key] = tmp_set
    print("\n")

#-------Form the Approximations-------#
lower_approximations = {}
for item in goal_list:
    #Key is the string converted list so as to add as dictionary key
    lower_approximations[str(item)] = lowerApproximation(dic,item)

lower_goal_list=list(lower_approximations.values())

upper_approximations = {}
for item in goal_list:
    #Key is the string converted list so as to add as dictionary key
    upper_approximations[str(item)] = upperApproximation(dic,item)

upper_goal_list=list(upper_approximations.values())

first_column = list(dic.keys())
second_column = list(dic.values())
prob_approx = pd.DataFrame(
    {'charset_name': first_column,
     'charset_value': second_column
    })

middle_approximations = {}
for item in goal_list:
    #Key is the string converted list so as to add as dictionary key
    middle_approximations[str(item)] = findMiddleApprox(item)

middle_goal_list=list(middle_approximations.values())

df3=df2

#concept_list and goal_list has 1:1 mapping
final_rules = []
start_time = time.time()

#-------Running algorithm for all the goals - Middle Approximation/concepts-------#
for i in range(0,len(middle_goal_list)):
    SPECIAL = U - middle_goal_list[i]
    findGoalIntersect(list(middle_goal_list[i]))
    
    condition = []
    B = set()
    selected_case = findCases(df3)
    
    rule_set = stepAlgo(df3,selected_case,middle_goal_list[i],B,condition,concept_list[i])
    
    #Now do for the SPECIAL cases
    findGoalIntersect(list(SPECIAL))
    condition_SPECIAL = []
    B_SPECIAL = set()
    selected_case_SPECIAL = findCases(df3)
    rule_set2 = stepAlgo(df3,selected_case_SPECIAL,SPECIAL,B_SPECIAL,condition_SPECIAL,"SPECIAL")
    
    final_rules.append(rule_set)
    final_rules.append(rule_set2)
    print("End of goal")
    
elapsed_time = time.time() - start_time

print("Time to run the algorithm (for Middle Approximation): ", round(elapsed_time, 3), "Sec")
print("\n")
print(*final_rules, sep='\n')

with open('../data/Iris/Rules/Iris_test_lost_middle.txt', 'w') as f:
    for item in final_rules:
        f.write("%s\n" % item)

#-------Running algorithm for all the goals - Lower Approximation/concepts-------#

#concept_list and goal_list has 1:1 mapping
final_rules = []
start_time = time.time()

#Running algorithm for all the goals - Lower Approximation/concepts
for i in range(0,len(lower_goal_list)):
    SPECIAL = U - lower_goal_list[i]
    findGoalIntersect(list(lower_goal_list[i]))
    condition = []
    B = set()
    selected_case = findCases(df3)
    
    rule_set = stepAlgo(df3,selected_case,lower_goal_list[i],B,condition,concept_list[i])
    
    #Now do for the SPECIAL cases
    findGoalIntersect(list(SPECIAL))
    condition_SPECIAL = []
    B_SPECIAL = set()
    selected_case_SPECIAL = findCases(df3)
    rule_set2 = stepAlgo(df3,selected_case_SPECIAL,SPECIAL,B_SPECIAL,condition_SPECIAL,"SPECIAL")
    
    final_rules.append(rule_set)
    final_rules.append(rule_set2)
    print("End of goal")
    
elapsed_time = time.time() - start_time


print("Time to run the algorithm (for Lower Approximation): ", round(elapsed_time, 3), "Sec")
print("\n")
print(*final_rules, sep='\n')

with open('../data/Iris/Rules/Iris_test_lost_lower.txt', 'w') as f:
    for item in final_rules:
        f.write("%s\n" % item)

#-------Running algorithm for all the goals - Upper Approximation/concepts-------#

#concept_list and goal_list has 1:1 mapping
final_rules = []
start_time = time.time()

#Running algorithm for all the goals - Lower Approximation/concepts
for i in range(0,len(upper_goal_list)):
    SPECIAL = U - upper_goal_list[i]
    findGoalIntersect(list(upper_goal_list[i]))
    condition = []
    B = set()
    selected_case = findCases(df3)
    
    rule_set = stepAlgo(df3,selected_case,upper_goal_list[i],B,condition,concept_list[i])
    
    #Now do for the SPECIAL cases
    findGoalIntersect(list(SPECIAL))
    condition_SPECIAL = []
    B_SPECIAL = set()
    selected_case_SPECIAL = findCases(df3)
    rule_set2 = stepAlgo(df3,selected_case_SPECIAL,SPECIAL,B_SPECIAL,condition_SPECIAL,"SPECIAL")
    
    final_rules.append(rule_set)
    final_rules.append(rule_set2)
    print("End of goal")
    
elapsed_time = time.time() - start_time

print("Time to run the algorithm (for Upper Approximation): ", round(elapsed_time, 3), "Sec")
print("\n")
print(*final_rules, sep='\n')

with open('../data/Iris/Rules/Iris_test_lost_upper.txt', 'w') as f:
    for item in final_rules:
        f.write("%s\n" % item)
#-----------------------END-------------------------#








