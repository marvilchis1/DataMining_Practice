#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:49:19 2020

@author: abel
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


w_d = '/Users/abel/Documents/Mineria_De_Datos/Code/Practice/'
i_f = w_d + 'survey_results_public.csv'
data = pd.read_csv(i_f, encoding = 'utf-8')


""" Statistical Functions """
def Percentile(lista, perc):
    aux = np.array(lista)    
    return np.percentile(aux, perc)
   
def FiveSummary(lista):
    lista.sort()
    
    mini = lista[0]
    maxi = lista[-1]    
    median = Median(lista)
    fq = Percentile(lista, 25)
    tq = Percentile(lista, 75)
    mean = Mean(lista)

    print('Min: ', mini)
    print('Max: ', maxi)
    print('First Quartile: ', fq)
    print('Median: ', median)
    print('Third Quartile: ', tq)
    print('Mean: ', mean)
    
    stdd = StandardDeviation(lista)
    print('Standard Desviation:', stdd)

def Median(lista):
    median = Percentile(lista,50)
    return median

def Mean(lista):
    mean = sum(lista)/len(lista)
    return mean

def StandardDeviation(lista):
    mean = Mean(lista)
    var = sum([(i - mean)**2 for i in lista]) / (len(lista) - 1)
    sdv = var**(1/2)
    return sdv

def PearsonCorrelation(list1, list2):
    mean1 = Mean(list1)
    mean2 = Mean(list2)    
    aux1 = aux2 = aux3 = 0    

    for i in range(0, len(list1)):
        aux1 += (list1[i] - mean1) * (list2[i] - mean2)   
        aux2 += (list1[i] - mean1) ** 2  
        aux3 += (list2[i] - mean2) ** 2
    
    pc = aux1 / ( (aux2 ** (1/2)) * (aux3 ** (1/2)) )
    
    return pc

""" Graphing Functions"""
def Boxplot(lista, xlabel, ylabel, title):
    plt.figure()
    plt.boxplot(x=lista, notch=True, sym = '')
    #plt.xticks([1],label, size = 'medium', color = 'k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def Barplot(data, col, title = ' '):
    f_data = data[ data[col].notnull() ]
    labels = UniqueValues(data, col)
    
    freqtable = {}
    
    for lab in labels:
        freqvalue = sum(f_data[col].apply(IsIn, args = (lab,)))
        freqtable[lab] = freqvalue
        
    h = np.arange(len(freqtable))
    plt.figure(figsize = (14,7))
    plt.bar(height = list(freqtable.values()), x = h)
    plt.xticks(h, freqtable.keys(), rotation = 90)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel('Quantity')

def Histogram(data, filtercol1, filtercol2, numrows = None , numcols = 1, xlabel = ' ', ylabel = ' ', separatedplots = False):
    f_data = data[ data[filtercol1].notnull() & data[filtercol2].notnull() ]    
    labels = UniqueValues(f_data, filtercol2)

        
    if xlabel is None:
        xlabel = filtercol2
    if ylabel is None:
        ylabel = filtercol1
    
    plt.figure(figsize = (14,8))
    
    if numrows is None:
        numrows = len(labels)

    for it, lab in enumerate(labels):
        filteredvalues = f_data[filtercol2].apply(IsIn, args = (lab,))
        n_data = f_data[filteredvalues]
        if separatedplots:
            plt.figure()
        plt.subplot(numrows, numcols, it+1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(lab[:10])
        plt.hist(n_data[filtercol1], bins = 10)
        plt.tight_layout()
    plt.show()

def PearsonGraph(list1, list2, xlabel = ' ', ylabel = ' '):
    x = list2
    y = list1
    
    plt.figure(figsize = (10,10), facecolor = 'w')
    plt.scatter(x=x, y=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    

""" Filtering Functions"""
# Busca si una palabra esta dentro de un string 
def IsIn(line, token):
    return line.find(token) != -1

# Listado con los datos unicos de una columna 
def UniqueValues(data, col):
    # Se crea una nueva lista depurando los valores vacios
    f_data = data[ data[col].notnull() ]
    # Se identifican aquellos datos que son unicos
    uniques = list(f_data[col].unique())
    # Primero se junta cada valor de la lista en un string con ; como separador
    # una vez obtenido el string, con split se separa cada valor separado por ;
    # posteriormente con set solo se meten en una lista los valores unicos
    return list(set ( ';'.join(uniques).split(';') ))


"""Exercises"""
# Exercise 1. Compute the five-number summary, the boxplot, the mean, and the standard deviation for the annual salary per gender.
def SalaryGender(data, gender):
    f_data = data[ data['Gender'].notnull() & data['ConvertedComp'].notnull() ]    
    salario = f_data['ConvertedComp'].tolist()
    genero = f_data['Gender'].tolist()    
    gen = []
    for s,g in zip(salario, genero):
        if ( IsIn(g,gender) ):
            #gen.append((s,genre) )
            gen.append(s)    
    return gen

print("\n**************** Exercise 1 *****************")
gentable = UniqueValues(data, 'Gender')

for gender in gentable:
    aux = []
    print("\n"+gender)
    aux = SalaryGender(data, gender)
    FiveSummary(aux)
    Boxplot(aux, gender, 'Annual Salary', 'Annual Salary per Gender')

# Exercise 2. Compute the five-number summary, the boxplot, the mean, and the standard deviation for the annual salary per ethnicity.
def SalaryEthnicity(data, ethnicity):
    f_data = data[ data['Ethnicity'].notnull() & data['ConvertedComp'].notnull() ]   
    salario = f_data['ConvertedComp'].tolist()
    etnia = f_data['Ethnicity'].tolist()    
    etn = []
    for s,e in zip(salario, etnia):
        if ( IsIn(e, ethnicity) ):
            etn.append(s)

    return etn

print("\n**************** Exercise 2 *****************")
etntable = UniqueValues(data, 'Ethnicity')

for ethnicity in etntable:
    aux = []
    print("\n"+ethnicity)
    aux = SalaryEthnicity(data, ethnicity)
    FiveSummary(aux)
    Boxplot(aux, ethnicity, 'Salary', 'Salary per Ethnicity')

# 3. Compute the five-number summary, the boxplot, the mean, and the standard deviation for the annual salary per developer type.
def SalaryDevType(data, devtype):
    f_data = data[ data['DevType'].notnull() & data['ConvertedComp'].notnull() ]   
    salario = f_data['ConvertedComp'].tolist()
    devtipo = f_data['DevType'].tolist()
    dev = []
    
    for s,d in zip(salario, devtipo):
        if ( IsIn(d, devtype)):
            dev.append(s)

    return dev

print("\n**************** Exercise 3 *****************")
devtable = UniqueValues(data, 'DevType')

for devtype in devtable:
    aux = []
    print("\n"+devtype)
    aux = SalaryDevType(data, devtype)
    FiveSummary(aux)
    Boxplot(aux, devtype, 'Salary', 'Salary per Developer Type')

# 4. Compute the median, mean and standard deviation of the annual salary per country.
def SalaryCountry(data, country):
    f_data = data[ data['Country'].notnull() & data['ConvertedComp'].notnull() ]
    salario = f_data['ConvertedComp'].tolist()
    pais = f_data['Country'].tolist()
    cont = []
    
    for s,c in zip(salario, pais):
        if (IsIn(c, country)):
            cont.append(s)
    
    return cont
    
print("\n**************** Exercise 4 *****************")
conttable = UniqueValues(data, 'Country')

for country in conttable:
    aux = []
    aux = SalaryCountry(data, country)
    if len(aux) <= 1: 
        continue
    print("\n"+country)
    print('Median: '+ str( Median(aux) ))
    print('Mean: '+str( Mean(aux) ))
    print('Standard Deviation: '+str( StandardDeviation(aux) ))    
    
# 5. Obtain a bar plot with the frequencies of responses for each developer type.
def BarplotDevType(data):
    Barplot(data, 'DevType', 'Frequency of developer type')

BarplotDevType(data = data)
    
# 6. Plot histograms with 10 bins for the years of experience with coding per gender. 
def HistogramExperienceGender(data):
    f_data = data[ data['YearsCode'].notnull() & data['Gender'].notnull() ]
    f_data['YearsCode'].replace('Less than 1 year', '0.5', inplace=True)
    f_data['YearsCode'].replace('More than 50 years', '51', inplace=True)
    f_data['YearsCode'] = f_data['YearsCode'].astype('float64')
    Histogram(f_data, 'YearsCode', 'Gender', xlabel = 'Experience Years', ylabel = 'People', numrows = 1, numcols = 3) 

HistogramExperienceGender(data = data)

# 7. Plot histograms with 10 bins for the average number of working hours per week, per developer type.
def HistogramWorkDevType(data):
    f_data = data[ data['WorkWeekHrs'].notnull() & data['DevType'].notnull() ]
    Histogram(f_data, 'WorkWeekHrs', 'DevType', xlabel = 'Week per hrs', ylabel = 'Dev Type', numrows = 6, numcols = 4)

HistogramWorkDevType(data = data)

# 8. Plot histograms with 10 bins for the age per gender.
def HistogramAgeGender(data):
    f_data = data[ data['Age'].notnull() & data['Gender'].notnull() ]
    Histogram(f_data, 'Age', 'Gender', xlabel = 'Years', ylabel = 'People', numrows = 1, numcols = 3)

HistogramAgeGender(data = data)

# 9. Compute the median, mean and standard deviation of the age per programming language.
def AgeProgLanguage(data, language):
    f_data = data[ data['LanguageWorkedWith'].notnull() & data['Age'].notnull() ]
    edad = f_data['Age'].tolist()
    lenguaje = f_data['LanguageWorkedWith'].tolist()
    lan = []
    
    for e,l in zip(edad, lenguaje):
        if (IsIn(l, language)):
            lan.append(e)
    
    return lan

print("\n**************** Exercise 9 *****************")
lantable = UniqueValues(data, 'LanguageWorkedWith')

aux = []
for language in lantable:
    aux = AgeProgLanguage(data, language)
    #if len(aux) <= 1:
    #    continue
    print("\n"+language)
    print('Median: '+ str( Median(aux) ))
    print('Mean: '+str( Mean(aux) ))
    print('Standard Deviation: '+str( StandardDeviation(aux) ))
   
# 10. Compute the correlation between years of experience and annual salary.
def CorrelationExperienceSalary(data):
    f_data = data[ data['ConvertedComp'].notnull() & data['YearsCode'].notnull() ]
    f_data['YearsCode'].replace('Less than 1 year', '0.5', inplace = True)
    f_data['YearsCode'].replace('More than 50 years', '51', inplace = True)
    f_data['YearsCode'] = f_data['YearsCode'].astype('float64')
    salario = f_data['ConvertedComp'].tolist()
    experiencia = f_data['YearsCode'].tolist()

    result = PearsonCorrelation(salario, experiencia) 
    print('Correlation between years of experience and annual salary: ' + str(result))
    PearsonGraph(salario, experiencia, 'Experience', 'Annual Salary')

print("\n**************** Exercise 10 *****************")
CorrelationExperienceSalary(data = data)

# 11. Compute the correlation between the age and the annual salary.
def CorrelationAgeSalary(data):
    f_data = data[ data['ConvertedComp'].notnull() & data['Age'].notnull() ]
    salario = f_data['ConvertedComp'].tolist()
    edad = f_data['Age'].tolist()

    result = PearsonCorrelation(salario, edad) 
    print('\nCorrelation between age and annual salary: ' + str(result))
    PearsonGraph(salario, edad, 'Age', 'Annual Salary')

print("\n**************** Exercise 11 *****************")
CorrelationAgeSalary(data = data)
    
# 12. Compute the correlation between educational level and annual salary. In this case, replace the string of 
# the educational level by an ordinal index (e.g. Primary/elementary school = 1, Secondary school = 2, and so on).
EdLevelDict = { 
     'I never completed any formal education' : 0,
     'Primary/elementary school' : 1,
     'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)' : 2,
     'Associate degree'  : 3,
     'Professional degree (JD, MD, etc.)' : 4,
     'Bachelor’s degree (BA, BS, B.Eng., etc.)' : 5,
     'Some college/university study without earning a degree' : 6,
     'Other doctoral degree (Ph.D, Ed.D., etc.)' : 7,
     'Master’s degree (MA, MS, M.Eng., MBA, etc.)': 8,
 }

def SetEdLevel(level):
    return EdLevelDict[level]

def CorrelationEdLevelSalary(data):
    f_data = data[ data['ConvertedComp'].notnull() & data['EdLevel'].notnull() ]
    f_data['EdLevel'] = f_data['EdLevel'].apply(SetEdLevel)
    
    salario = f_data['ConvertedComp'].tolist()
    educacion = f_data['EdLevel'].tolist()
    
    result = PearsonCorrelation(salario, educacion)
    print('\nCorrelation between years of expenrience and annual salary:' + str(result))
    print('\n')
    PearsonGraph(salario, educacion, 'Educational Level', 'Annual Salary')

print("\n**************** Exercise 12 *****************")
CorrelationEdLevelSalary(data = data)
  
# 13. Obtain a bar plot with the frequencies of the different programming languages.
def BarplotDevLanguage(data):
    Barplot(data, 'LanguageWorkedWith', 'Frequency of programming language')

print("\n**************** Exercise 13 *****************")
BarplotDevLanguage(data = data)    
    