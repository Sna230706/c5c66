import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier ,plot_tree
import matplotlib.pyplot as plt
df = pd.read_excel("Online Retail.xlsx")
df = df.dropna(subset=["CustomerID"])
df = df[df['Quantity']>0]
df['TotalPrice']=df['Quantity']*df['UnitPrice']
customer = df.groupby('CustomerID').agg(
 LastPurchase=('InvoiceDate', 'max'),
 Frequency=('InvoiceNo', 'nunique'),
 Monetary=('TotalPrice','sum')
)
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
customer['Recency'] = (snapshot_date - customer['LastPurchase']).dt.days
customer.drop(columns=['LastPurchase'],inplace = True)

customer['customer_segment'] = pd.qcut(customer['Monetary'],q=3,labels=["High","Medium","low"])
X = customer[['Recency','Frequency']]
y = customer['customer_segment']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,random_state=42)
id3_tree = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=42)
id3_tree.fit(X_train,y_train)
id3_tree.score(X_train,y_train)
id3_tree.score(X_test, y_test)
plt.figure(figsize=(16,8))
plot_tree(id3_tree,
 feature_names=X.columns,
 class_names=id3_tree.classes_,
 filled=True,
 rounded=True)
plt.title("Customer Segmentation Decision Tree (ID3/Entropy)")
plt.show()