---
Typora-root-url: ../../assets/MLpics
---

## 习题

- 6.1：试证明样本空间中任意点x到超平面（w,b）的距离为$r=\frac{\vert w^Tx+b \vert}{\vert \vert w \vert \vert}$

  <img src="/T92.png" style="zoom:50%;" />

  - 如上图所示，$x=x_0+r$

    $r=r_0 \frac{w}{\vert \vert w \vert \vert}$

    $x=x_0+r_0 \frac{w}{\vert \vert w \vert \vert}$

    $w^Tx_0+b=0，x_0=x-r_0 \frac{w}{\vert \vert w \vert \vert}$

    $w^Tx-w^Tr_0 \frac{w}{\vert \vert w \vert \vert}+b=0$

    $r_0=\frac{\vert w^Tx+b \vert}{\vert \vert w \vert \vert}$

- 6.2：试使用LIBSVM，在西瓜数据集3.0上分别用线性核和高斯核训练一个SVM，并比较其支持向量的差别

  ```python
  def SVM():
      X,y=ReadExcel('./alpha_data.xlsx')
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)
      y_train=y_train.T.tolist()[0]
      y_test=y_test.T.tolist()[0]
      prob=svm_problem(y_train,X_train)
      # t=2高斯核，t=0线性核
      param=svm_parameter('-t 2 -c 35')
      model=svm_train(prob,param)
      p_label,p_acc,p_val=svm_predict(y_test,X_test,model)
      print(p_acc)
      return p_label
  ```

  - 高斯核和线性核训练出来的支持向量一致，同时线性核以及高斯核训练的SVM准确率最高均在0.67。

- 6.3 选择两个UCI数据集，分别用线性核核高斯核训练一个SVM，并与BP神经网络和C4.5决策树进行实验比较

  ```python
  # SVC
  #线性核，核函数系数，软间隔分类器
  svc=SVC(kernel='linear',gamma="auto",C=1)
  svc.fit(X_train,y_train)
  print("The accuracy of svc on training set under linear",format(svc.score(X_train,y_train)))
  print("The accuracy of svc on test set under linear",format(svc.score(X_test,y_test)))
  
  #高斯核
  svc=SVC(kernel='rbf',gamma="auto",C=1)
  svc.fit(X_train,y_train)
  print("The accuracy of svc on training set under rbf",format(svc.score(X_train,y_train)))
  print("The accuracy of svc on test set under rbf",format(svc.score(X_test,y_test)))
  
  # BP神经网络
  mlp=MLPClassifier(random_state=0,max_iter=1000,alpha=1).fit(X_train,y_train)
  print("The accuracy of BP neural network on training set",format(mlp.score(X_train,y_train)))
  print("The accuracy of BP neural network on test set",format(mlp.score(X_test,y_test)))
  
  # C4.5决策树
  tree=DecisionTreeClassifier(random_state=0,max_depth=4).fit(X_train,y_train)
  print("The accuracy of C4.5 tree on training set",format(tree.score(X_train,y_train)))
  print("The accuracy of C4.5 tree on test set",format(tree.score(X_test,y_test)))
  ```

  - 结果

    ```C++
    The accuracy of svc on training set under linear 0.9910714285714286
    The accuracy of svc on test set under linear 1.0
    The accuracy of svc on training set under rbf 0.9642857142857143
    The accuracy of svc on test set under rbf 1.0
    
    The accuracy of BP neural network on training set 0.9732142857142857
    The accuracy of BP neural network on test set 1.0
    
    The accuracy of C4.5 tree on training set 1.0
    The accuracy of C4.5 tree on test set 0.9736842105263158
    ```

    - 对比，在训练集上准确率相差不大，而在测试集上决策树表现相对不算优异

- 6.4：试讨论线性判别分析与线性核支持向量机在何种条件下等价

  - 线性判别分析能够解决 n 分类问题
  -  而 SVM只能解决二分类问题，如果要解决 n 分类问题要通过 OvR来迂回解决.
  - 线性判别分析能将数据以同类样例间低方差和不同样例中心之间大间隔来投射到一条直线上, 但是如果样本线性不可分, 那么线性判别分析就不能有效进行, 支持向量机也是
  - 综上, 等价的条件是
    - 数据有且仅有 2 种, 也就是说问题是二分类问题
    - 数据是线性可分的

- 6.5：试述高斯核SVM与RBF神经网络之间的联系

  - 若将隐层神经元数设置为训练样本数，且每个训练样本对应一个神经元中心，则以高斯径向基函数为激活函数的RBF网络恰与高斯核SVM的预测函数相同。
  - RBF的径向基函数和高斯核SVM均采用高斯核
  - 神经网络是最小化累计误差，将参数 w 作为惩罚项；而SVM相反，主要是最小化参数，将误差作为惩罚项。

- 6.6：试析SVM对噪声敏感的原因

  - SVM的基本形态是一个硬间隔分类器，它要求所有样本都满足硬间隔约束（函数间隔要大于1）
  - 若数据集中存在噪声点并且满足线性可分，那么SVM为了将噪声点划分为正确，远离该点的类靠近，使得划分超平面的几何间距变小，泛化性能差
  - 若数据集由于噪声点无法满足线性可分，就需要使用核方法，会得到更加复杂的模型，泛化能力差（过拟合）
  - 因此需要软间隔SVM来解决这些问题

- 6.7：试给出式（6.52）的完整KKT条件

  <img src="/T93.png" style="zoom:50%;" />

- 6.8：以西瓜数据集3.0的“密度”为输入，“含糖率”为输出，试使用LIBSVM训练一个SVR\

  ```python
  def SVR():
      X,y=ReadExcel('./alpha_data.xlsx')
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)
      y_train=y_train.tolist()
      # t=2高斯核，t=0线性核
      # s=3 epsilon-SVR
      model=svm_train(y_train,X_train,'-t 0 -s 3')
      p_label,p_acc,p_val=svm_predict(y_test,X_test,model)
      print(p_acc)
  ```

  结果

  ```C
  Mean squared error = 0.00833156 (regression)
  Squared correlation coefficient = 0.00832679 (regression)
  (0.0, 0.008331563045225086, 0.008326794756839036)
  ```

- 6.9：试使用核技巧推广对率回归，产生“核对率回归”

  <img src="/T94.png" style="zoom:50%;" />

