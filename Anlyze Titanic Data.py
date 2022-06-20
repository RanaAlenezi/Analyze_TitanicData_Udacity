#!/usr/bin/env python
# coding: utf-8

# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




# Read in the dataset, create dataframe
titanic_data = pd.read_csv('titanic_data.csv')
# Print the first few records to review data and format
titanic_data.head()


# In[15]:


# Print the last few records to review data and format
titanic_data.tail()


# In[12]:


# Identify and remove duplicate entries
titanic_data_duplicates = titanic_data.duplicated()
print ('Number of duplicate entries is/are {}'.format(titanic_data_duplicates.sum()))

# Let us just make sure this is working
duplicate_test = titanic_data.duplicated('Age').head()
print ('Number of entries with duplicate age in top entires are {}'.format(duplicate_test.sum()))
titanic_data.head()


# In[16]:


# Create new dataset without unwanted columns
titanic_data_cleaned = titanic_data.drop(['PassengerId','Name','Ticket','Cabin','Fare','Embarked'], axis=1)
titanic_data_cleaned.head()


# In[14]:


# Calculate number of missing values
titanic_data_cleaned.isnull().sum()


# In[20]:



# Review some of the missing Age data
missing_age_bool = pd.isnull(titanic_data_cleaned['Age'])
titanic_data_cleaned[missing_age_bool].head()


# In[21]:


# Determine number of males and females with missing age values
missing_age_female = titanic_data_cleaned[missing_age_bool]['Sex'] == 'female'
missing_age_male = titanic_data_cleaned[missing_age_bool]['Sex'] == 'male'

print ('Number for females and males with age missing are {} and {} respectively'.format(
missing_age_female.sum(),missing_age_male.sum()))
# Taking a look at the datatypes
titanic_data_cleaned.info()


# In[18]:


# Looking at some typical descriptive statistics
titanic_data_cleaned.describe()


# In[19]:


# Age min at 0.42 looks a bit weird so give a closer look
titanic_data_cleaned[titanic_data_cleaned['Age'] < 1]


# In[22]:


# Taking a look at some survival rates for babies
youngest_to_survive = titanic_data_cleaned[titanic_data_cleaned['Survived'] == True]['Age'].min()
youngest_to_die = titanic_data_cleaned[titanic_data_cleaned['Survived'] == False]['Age'].min()
oldest_to_survive = titanic_data_cleaned[titanic_data_cleaned['Survived'] == True]['Age'].max()
oldest_to_die = titanic_data_cleaned[titanic_data_cleaned['Survived'] == False]['Age'].max()

print ('Youngest to survive: {} \nYoungest to die: {} \nOldest to survive: {} \nOldest to die: {}'.format(
youngest_to_survive, youngest_to_die, oldest_to_survive, oldest_to_die))


# In[61]:


# Returns survival rate/percentage of sex and class
def survival_rate(pclass, sex):
    """
    Args:
        pclass: class value 1,2 or 3
        sex: male or female
    Returns:
        survival rate as percentage.
    """
    grouped_by_total = titanic_data_cleaned.groupby(['Pclass', 'Sex']).size()[pclass,sex].astype('float')
    grouped_by_survived_sex =         titanic_data_cleaned.groupby(['Pclass','Survived','Sex']).size()[pclass,1,sex].astype('float')
    survived_sex_pct = (grouped_by_survived_sex / grouped_by_total * 100).round(2)
    
    return survived_sex_pct


# In[71]:




# Get the actual numbers grouped by class, suvival and sex
groupedby_class_survived_size = titanic_data_cleaned.groupby(['Pclass','Survived','Sex']).size()

# Print - Grouped by class, survival and sex
print (groupedby_class_survived_size)
print ('Class 1 - female survival rate: {}%'.format(survival_rate(1,'female')))
print ('Class 1 - male survival rate: {}%'.format(survival_rate(1,'male')))
print ('-----')
print ('Class 2 - female survival rate: {}%'.format(survival_rate(2,'female')))
print ('Class 2 - male survival rate: {}%'.format(survival_rate(2,'male')))
print ('-----')
print ('Class 3 - female survival rate: {}%'.format(survival_rate(3,'female')))
print ('Class 3 - male survival rate: {}%'.format(survival_rate(3,'male')))

# Graph - Grouped by class, survival and sex
g = sns.factorplot(x="Sex", y="Survived", col="Pclass", data=titanic_data_cleaned, 
                   saturation=.5, kind="bar", ci=None, size=5, aspect=.8)
# Fix up the labels
(g.set_axis_labels('', 'Survival Rate')
     .set_xticklabels(["Men", "Women"])
     .set_titles("Class {col_name}")
     .set(ylim=(0, 1))
     .despine(left=True, bottom=True))

# Graph - Actual count of passengers by survival, group and sex
g = sns.factorplot('Survived', col='Category', data=titanic_data_age_cleaned, kind='count', size=7, aspect=.8)


# Fix up the labels
(g.set_axis_labels('Suvivors', 'No. of Passengers')
    .set_xticklabels(["False", "True"])
    .set_titles('{col_name}')
)
titles = ['Men', 'Women']
for ax, title in zip(g.axes.flat, titles):
    ax.set_title(title)


# In[72]:


# Let us first identify and get rid of records with missing Age
print ('Number of men and woman with age missing are {} and {} respectively'
      .format(missing_age_female.sum(),missing_age_male.sum()))
# Drop the NaN values. Calculations will be okay with them (seen as zero) but will throw off averages and counts
titanic_data_age_cleaned = titanic_data_cleaned.dropna()
# Find total count of survivors and those who didn't
number_survived = titanic_data_age_cleaned[titanic_data_age_cleaned['Survived'] == True]['Survived'].count()
number_died = titanic_data_age_cleaned[titanic_data_age_cleaned['Survived'] == False]['Survived'].count()

# Find average of survivors and those who didn't
mean_age_survived = titanic_data_age_cleaned[titanic_data_age_cleaned['Survived'] == True]['Age'].mean()
mean_age_died = titanic_data_age_cleaned[titanic_data_age_cleaned['Survived'] == False]['Age'].mean()
# Display a few raw totals
print ('Total number of survivors {} \nTotal number of non survivors {} \nMean age of survivors {} \nMean age of non survivors {} \nOldest to survive {} \nOldest to not survive {}' .format(number_survived, number_died, np.round(mean_age_survived), 
        np.round(mean_age_died), oldest_to_survive, oldest_to_die))
# Graph - Age of passengers across sex of those who survived
g = sns.factorplot('Survived', col='Category', data=titanic_data_age_cleaned, kind='count', size=7, aspect=.8)
# Fix up the labels
(g.set_axis_labels('Suvivors', 'Age of Passengers')
    .set_xticklabels(["False", "True"])
)


# In[31]:


# Create Cateogry column and categorize people
titanic_data_age_cleaned.loc[
    ( (titanic_data_age_cleaned['Sex'] == 'female') & 
    (titanic_data_age_cleaned['Age'] >= 18) ),
    'Category'] = 'Woman'

titanic_data_age_cleaned.loc[
    ( (titanic_data_age_cleaned['Sex'] == 'male') & 
    (titanic_data_age_cleaned['Age'] >= 18) ),
    'Category'] = 'Man'

titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Age'] < 18),
    'Category'] = 'Child'
# Get the totals grouped by Men, Women and Children, and by survival
print (titanic_data_age_cleaned.groupby(['Category','Survived']).size())
# Graph - Compare survival count between Men, Women and Children
g = sns.factorplot('Survived', col='Category', data=titanic_data_age_cleaned, kind='count', size=7, aspect=.8)
# Fix up the labels
(g.set_axis_labels('Suvivors', 'No. of Passengers')
    .set_xticklabels(['False', 'True'])
)
titles = ['Men', 'Women', 'Children']
for ax, title in zip(g.axes.flat, titles):
    ax.set_title(title)


# In[29]:


# Determine number of woman that are not parents
titanic_data_woman_parents = titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Category'] == 'Woman') &
    (titanic_data_age_cleaned['Parch'] > 0)]

# Determine number of woman over 20 that are not parents
titanic_data_woman_parents_maybe = titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Category'] == 'Woman') &
    (titanic_data_age_cleaned['Parch'] > 0) & 
    (titanic_data_age_cleaned['Age'] > 20)]

titanic_data_woman_parents.head()


# In[30]:


titanic_data_woman_parents_maybe.head()


# In[74]:


# Separate out children with parents from those with nannies 
titanic_data_children_nannies =titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Category'] == 'Child') &
    (titanic_data_age_cleaned['Parch'] == 0)]

titanic_data_children_parents =titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Category'] == 'Child') &
    (titanic_data_age_cleaned['Parch'] > 0)]

# Determine children with nannies who survived and who did not
survived_children_nannies = titanic_data_children_nannies.Survived.sum()
total_children_nannies = titanic_data_children_nannies.Survived.count()
pct_survived_nannies = ((float(survived_children_nannies)/total_children_nannies)*100)
pct_survived_nannies = np.round(pct_survived_nannies,2)
survived_children_nannies_avg_age = np.round(titanic_data_children_nannies.Age.mean())

# Display results
print ('Total number of children with nannies: {}\nChildren with nannies who survived: {}\nChildren with nannies who did not survive: {}\nPercentage of children who survived: {}%\nAverage age of surviving children: {}'
.format(total_children_nannies, survived_children_nannies, 
        total_children_nannies-survived_children_nannies, pct_survived_nannies, survived_children_nannies_avg_age))

# Verify counts (looked a bit too evenly divided)
titanic_data_children_nannies[titanic_data_children_nannies['Survived'] == 1]


# In[77]:


# Determine children with parents who survived and who did not
survived_children_parents = titanic_data.Survived.sum()
total_children_parents = titanic_data.Survived.count()
pct_survived_parents = ((float(survived_children_parents)/total_children_parents)*100)
pct_survived_parents = np.round(pct_survived_parents,2)
survived_children_parents_avg_age = np.round(titanic_data_children_parents.Age.mean())

# Display results
print ('Total number of children with parents: {}\nChildren with parents who survived: {}\nChildren with parents who did not survive: {}\nPercentage of children who survived: {}%\nAverage age of surviving children: {}'.format(total_children_parents, survived_children_parents, 
        total_children_parents-survived_children_parents, pct_survived_parents,survived_children_parents_avg_age))


# In[ ]:




