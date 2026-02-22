import pandas as pd 
df=pd.read_csv("Student_Performance.csv")
print(df.columns)
df=pd.get_dummies(df,columns=['Extracurricular Activities'])
# print(df[['Hours Studied','Performance Index']].corr())
# print(df[['Previous Scores','Performance Index']].corr())#linear boht jyada
df=df.drop_duplicates(keep='first')


x=df[['Hours Studied','Previous Scores']]
m,n=x.quantile(0.25)
o,p=x.quantile(0.75)
iqr1=o-m
iqr2=p-n 
print(x.shape)
a,b=x.mean()
x=x[(x['Hours Studied']>m-1.5*iqr1) & (x['Hours Studied']<o+1.5*iqr1) & (x['Previous Scores']>n-1.5*iqr2) & (x['Previous Scores']<p+1.5*iqr2)]
print(x.shape)
