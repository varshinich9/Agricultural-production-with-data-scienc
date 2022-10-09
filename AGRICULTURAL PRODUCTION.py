#!/usr/bin/env python
# coding: utf-8

# In[12]:


#data manipulation
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt
import seaborn as sns

#for interCTIVITY
from ipywidgets import interact


# In[13]:


df=pd.read_csv("C:/Users/varsh/Downloads/data.csv")


# In[15]:


df.shape


# In[19]:


df.head()


# In[21]:


#detecting null value
df.isnull().sum()
#NO MISSING VALUE=clean data set
#fillany function with mean,median or mode
#numerical mode
#categorial mean
#huge number of outliners median


# In[23]:


df['label'].value_counts()
#return a unique value of the label#


# In[28]:


#summary for all crops
print("average ratio of nitrogen in soil is:{0:.2f}".format(df['N'].mean()))
print("average ratio of phospherous in soil is:{0:.2f}".format(df['P'].mean()))
print("average ratio of potassium in soil is:{0:.2f}".format(df['K'].mean()))
print("average ratio of temperature in soil is:{0:.2f}".format(df['temperature'].mean()))
print("average ratio of humidity   in soil is:{0:.2f}".format(df['humidity'].mean()))
print("average ratio of ph  in soil is:{0:.2f}".format(df['ph'].mean()))
print("average ratio of rainfall  in soil is:{0:.2f}".format(df['rainfall'].mean()))


# In[37]:


#lets check the summary for each crop
@interact
def summary(crops=list(df['label'].value_counts().index)):
    x=df[df['label']==crops]
    print("---------------------------------------------------")
    print("stat for nitrogen")
    print("mini nitrogen required:",x['N'].min())
    print("avg nitrogen required:",x['N'].mean())
    print("max nitrogen required:",x['N'].max())
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("stat for phospherous")
    print("mini phospertous required:",x['P'].min())
    print("avg phospherous required:",x['P'].mean())
    print("max phospherous required:",x['P'].max())
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("stat for potassium")
    print("mini potassium required:",x['K'].min())
    print("avg potassium required:",x['K'].mean())
    print("max potassium required:",x['K'].max())
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("stat for temperature")
    print("mini temperature required:",x['temperature'].min())
    print("avg temperature required:",x['temperature'].mean())
    print("max temperature required:",x['temperature'].max())
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("stat for humidity")
    print("mini humidity required:",x['humidity'].min())
    print("avg humidity required:",x['humidity'].mean())
    print("max humidity required:",x['humidity'].max())
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("stat for ph")
    print("mini ph required:",x['ph'].min())
    print("avg ph required:",x['ph'].mean())
    print("max ph required:",x['ph'].max())
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("stat for rainfall")
    print("mini rainfall required:",x['rainfall'].min())
    print("avg rainfall required:",x['rainfall'].mean())
    print("max rainfall required:",x['rainfall'].max())
    print("---------------------------------------------------")
        


# In[101]:


# leta comapre the avg requirement for each crop
@interact
def compare(conditions=['N','P','K','temperature','humidity','ph','rainfall']):
    print("Avg conditions for",conditions,"is {0:.2f}".format(df[conditions].mean()))
    print("------------------------------------------------")
    print("RICE:{0:.2f}".format(df[(df['label']=='rice')][conditions].mean()))
    print("MAIZE:{0:.2f}".format(df[(df['label']=='maize')][conditions].mean()))
    print("jute:{0:.2f}".format(df[(df['label']=='jute')][conditions].mean()))
    print("blackgram:{0:.2f}".format(df[(df['label']=='blackgram')][conditions].mean()))
    print("cotton:{0:.2f}".format(df[(df['label']=='cotton')][conditions].mean()))
    print("coconut:{0:.2f}".format(df[(df['label']=='coconut')][conditions].mean()))
    print("papaya:{0:.2f}".format(df[(df['label']=='papaya')][conditions].mean()))
    print("orange:{0:.2f}".format(df[(df['label']=='orange')][conditions].mean()))
    print("apple:{0:.2f}".format(df[(df['label']=='apple')][conditions].mean()))
    print("muskmelon:{0:.2f}".format(df[(df['label']=='muskmelon')][conditions].mean()))
    print("watermelon:{0:.2f}".format(df[(df['label']=='watermelon')][conditions].mean()))
    print("grapes:{0:.2f}".format(df[(df['label']=='grapes')][conditions].mean()))
    print("mango:{0:.2f}".format(df[(df['label']=='mango')][conditions].mean()))
    print("banana:{0:.2f}".format(df[(df['label']=='banana')][conditions].mean()))
    print("pomogranate:{0:.2f}".format(df[(df['label']=='pomegranate')][conditions].mean()))
    print("lentil:{0:.2f}".format(df[(df['label']=='lentil')][conditions].mean()))
    print("blackgram:{0:.2f}".format(df[(df['label']=='blackgram')][conditions].mean()))
    print("mungbean:{0:.2f}".format(df[(df['label']=='mungbean')][conditions].mean()))
    print("mothbean:{0:.2f}".format(df[(df['label']=='mothbeans')][conditions].mean()))
    print("pegionpeas:{0:.2f}".format(df[(df['label']=='pigeonpeas')][conditions].mean()))
    print("kidneybeans:{0:.2f}".format(df[(df['label']=='kidneybeans')][conditions].mean()))
    print("chickpea:{0:.2f}".format(df[(df['label']=='chickpea')][conditions].mean()))
    print("coffee:{0:.2f}".format(df[(df['label']=='coffee')][conditions].mean()))


# In[51]:


#anamolies unusual findings
@interact
def compare(conditions=['N','P','K','temperature','humidity','ph','rainfall']):
    print("crops that require more than avg rainfall",conditions,"\n")
    print(df[df[conditions] > df[conditions].mean()]['label'].unique())
    print("----------------------------")
    print("crops that require less than avg rainfall",conditions,"\n")
    print(df[df[conditions] <= df[conditions].mean()]['label'].unique())
    


# # DISTRIBUTION

# In[89]:


import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (15,7)




plt.subplot(2,4,1)
sns.distplot(df['N'],color='lightgrey')
plt.xlabel("RATIO OF NITROGEN", fontsize=12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(df['P'],color='yellow')
plt.xlabel("RATIO OF PHOSPHEROUS", fontsize=12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(df['K'],color='blue')
plt.xlabel("RATIO OF POTASSIUM", fontsize=12)
plt.grid()

plt.subplot(2,4,4)
sns.distplot(df['rainfall'],color='red')
plt.xlabel("RATIO OF RAINFALL", fontsize=12)
plt.grid()

plt.subplot(2,4,5)
sns.distplot(df['humidity'],color='lightpink')
plt.xlabel("RATIO OF HUMIDITY", fontsize=12)
plt.grid()

plt.subplot(2,4,6)
sns.distplot(df['ph'],color='lightgreen')
plt.xlabel("RATIO OF PH", fontsize=12)
plt.grid()

plt.subplot(2,4,7)
sns.distplot(df['temperature'],color='orange')
plt.xlabel("RATIO OF TEMPERATURE", fontsize=12)
plt.grid()


# In[68]:


#MORE ANALOGY
print("few patterns")
print("crops which requires very high ratio of nitrogen in soil:",df[df['N']>120]['label'].unique())
print("crops which requires very high ratio of phospherious  in soil:",df[df['P']>100]['label'].unique())
print("crops which requires very high ratio of potassium in soil:",df[df['K']>200]['label'].unique())
print("crops which requires very high ratio temperature in soil:",df[df['temperature']>40]['label'].unique())
print("crops whioch requires very high ratio of humidity in soil:",df[df['humidity']>20]['label'].unique())
print("crops which requires very high ratio of rainfall in soil:",df[df['rainfall']>200]['label'].unique())
print("crops which requires very high ratio of ph in soil:",df[df['ph']>9]['label'].unique())
print("crops which requires very low ratio of ph in soil:",df[df['ph']<4]['label'].unique())


# # seasonal distribution

# In[67]:


print("SUMMER CROPS")
print(df[(df['temperature']>30)&(df['humidity']>50)]['label'].unique())
print("------------------------------------------------")
print("WINTER CROPS")
print(df[(df['temperature']>20)&(df['humidity']>30)]['label'].unique())
print("--------------------------------------------------")
print("RAINY CROPS")
print(df[(df['rainfall']>200)&(df['humidity']>30)]['label'].unique())


# In[72]:


from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# In[73]:


x=df.loc[:,['N','P','K','temperature','humidity','ph','rainfall']].values


# In[74]:


x.shape


# In[75]:


df.shape


# In[77]:


x_data=pd.DataFrame(x)
x_data.head()


# In[81]:


plt.rcParams['figure.figsize'] = (10,4)
wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)
#plot
plt.plot(range(1,11),wcss)
plt.title("THE ELBOW METHOD")
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.show()


# In[83]:


#implemetining k mean algo to form clusters
km=KMeans(n_clusters=4,init="k-means++",max_iter=300,n_init=10,random_state=0)
y_means=km.fit_predict(x)

#results
a=df['label']
y_means=pd.DataFrame(y_means)
z=pd.concat([y_means,a],axis=1)
z=z.rename(columns={0:'cluster'})


# In[88]:


#clusters of each crops
print("crops in cluster 1:",z[z['cluster']==0]['label'].unique())
print("------------------------------------------------------")
print("crops in cluster 2:",z[z['cluster']==1]['label'].unique())
print("------------------------------------------------------")
print("crops in cluster 3:",z[z['cluster']==2]['label'].unique())
print("------------------------------------------------------")
print("crops in cluster 4:",z[z['cluster']==3]['label'].unique())


# In[93]:


#building a predictive model suitable for the region using ML logistic regression
y=df['label']
x=df.drop(['label'],axis=1)



# In[97]:


#spliting int o traing and testin
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape
x_test.shape


# In[98]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[100]:


from sklearn.metrics import classification_report

ce=classification_report(y_test,y_pred)
print(ce)


# In[103]:


df.head()


# In[107]:


prediction=model.predict((np.array([[90,40,45,20,84,7,200]])))
print("the suggested crop for this climatic conditions is:",prediction)


# In[ ]:




