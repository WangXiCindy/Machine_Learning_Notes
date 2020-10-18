---
Typora-root-url: ../../assets/MLpics
---

## 习题

- 4.1 试证明对于不含冲突数据（即特征向量完全相同但标记不同）的训练集，必存在与训练集一致（即训练误差为0）的决策树

  - 不含冲突数据
  - 则相同属性特征的样本进入同一叶子结点
  - 必存在与训练集一致的决策树

- 4.2 试析使用“最小训练误差”作为决策树划分选择准则的缺陷

  - 过拟合

- 4.3 试变成实现基于信息熵进行划分选择的决策树算法，并为表4.3中数据生成一棵决策树

  <img src="/T54.png" style="zoom:50%;" />

  ```python
  # 根据书中伪代码写function即可
  def treeGenerate(D,A,root,lastNode,lastA):
      flag, category = same_category(D)
      if flag:
          if category==1:
              lastNode[lastA] = '好瓜'  
          else:
              lastNode[lastA] = '坏瓜'
          return
      
      if empty_attribute(A) or sameOnA(D,A):
          lastNode[lastA]=findMost(D)
          return
      
      best_a=findBestA(D,A)
  
      root[best_a]={}
      for av in pd.unique(data.loc[:,best_a]):
          Dv = getDv_a(D,best_a,av)
          if Dv.shape[0]==0:
              root[best_a][av]=findMost(D)
          else:
              A_ = A.drop(best_a)
              root[best_a][av] = {}
              lastA = av
  
              treeGenerate(Dv,A_,root[best_a][av],root[best_a],lastA)
  ```

  ​	结果如下：

  ​	<img src="/T62.png" style="zoom:50%;" />

- 4.4 试实现基于基尼指数进行划分选择的决策树算法，为表4.2中数据生成预剪枝、后剪枝决策树，并与未剪枝决策树进行比较

  - 未剪枝

    ```python
    def treeGenerate(D,A,root,lastNode,lastA):
        flag, category = same_category(D)
        if flag:
            if category==1:
                lastNode[lastA] = '好瓜'  
            else:
                lastNode[lastA] = '坏瓜'
            return
        
        if empty_attribute(A) or sameOnA(D,A):
            lastNode[lastA]=findMost(D)
            return
        
        best_a=findBestA(D,A)
    
        root[best_a]={}
        for av in pd.unique(data.loc[:,best_a]):
            Dv = getDv_a(D,best_a,av)
            if Dv.shape[0]==0:
                root[best_a][av]=findMost(D)
            else:
                A_ = A.drop(best_a)
                root[best_a][av] = {}
                lastA = av
                
                treeGenerate(Dv,A_,root[best_a][av],root[best_a],lastA)
    ```

  ​	

  - 预剪枝

    ```python
    # 预剪枝
    def prePruning(D,Dtest,A,root,lastNode,lastA):
        # 如果基尼指数为0，即D中样本全属于同一类别，返回
        flag, category = same_category(D)
        if flag:
            if category==1:
                lastNode[lastA] = '好瓜'  
            else:
                lastNode[lastA] = '坏瓜'
            return
        
        if empty_attribute(A) or sameOnA(D,A):
            lastNode[lastA]=findMost(D)
            return
        best_a=findBestA(D,A)
        
        accCnt=calAccNum(D,Dtest)
        
        # 如果不划分的正确率更高，则不划分
        if cmpAcc(D,Dtest,accCnt,best_a,1):
            root[best_a]={}
        else:
            lastNode[lastA]=findMost(D)
            return
        
        for av in pd.unique(data.loc[:,best_a]):
    
            Dv = getDv_a(D,best_a,av)
            
            Dv_test=getDv_a(Dtest,best_a,av)
            
            if Dv.shape[0]==0:
                root[best_a][av]=findMost(D)
            else:
                A_ = A.drop(best_a)
                root[best_a][av] = {}
                lastA = av
                
                prePruning(Dv,Dv_test,A_,root[best_a][av],root[best_a],lastA)
    ```

    

  - 后剪枝

    ```python
    # 后剪枝
    def postPruning(D,Dtest,A,root,lastNode,lastA):
        
        flag, category = same_category(D)
        if flag:
            if category==1:
                lastNode[lastA] = '好瓜'  
            else:
                lastNode[lastA] = '坏瓜'
            return
        
        if empty_attribute(A) or sameOnA(D,A):
            lastNode[lastA]=findMost(D)
            return
        
        best_a=findBestA(D,A)
        if lastA!=None:
            lastlastA=lastA
        else:
            lastlastA=None
            
        root[best_a]={}
        for av in pd.unique(data.loc[:,best_a]):
            Dv = getDv_a(D,best_a,av)
            Dv_test=getDv_a(Dtest,best_a,av)
            if Dv.shape[0]==0:
                root[best_a][av]=findMost(D)
            else:
                A_ = A.drop(best_a)
                root[best_a][av] = {}
                lastA = av
    
                postPruning(Dv,Dv_test,A_,root[best_a][av],root[best_a],lastA)
        
    
        # 针对叶子结点开始剪枝
        
        accCnt=calAccNum(D,Dtest)# 计算剪枝的正确率
    
        # 如果不划分的正确率更高，则不划分
        if cmpAcc(D,Dtest,accCnt,best_a,0)==False:
            #lastNode[lastA]=findMost(D)
            lastNode[lastlastA]=findMost(D)
    
            return
    ```

    

  - 结果

    <img src="/T63.png" style="zoom:70%;" />

- 4.5 试编程实现基于对率回归进行划分选择的决策树算法，并为表4.3中数据生成一棵决策树

  ```python
  # 对率回归核心算法
  def sigmoid(Z):
      return 1.0/(1+np.exp(-Z))
  
  def gradDescent(data,label,eta=0.1,n_iters=500):
      m,n=data.shape
      label=label.reshape(-1,1)
      
      beta=np.ones((n,1))
      
      for i in range(n_iters):
          y_sig=sigmoid(data.dot(beta))
          m=y_sig-label  #计算误差值
          beta=beta-data.transpose().dot(m)*eta   #误差反传更新参数
          
      return beta
  ```

  ```python
  # 决策树生成算法
  def treeGenerate(D,root,lastNode,lastBeta):
      flag, category = same_category(D)
      if flag:
          if category==1:
              lastNode[lastBeta] = '好瓜'  
          else:
              lastNode[lastBeta] = '坏瓜'
          return
      
      if len(D[0])==1:
          lastNode[lastBeta]=findMost(D)
          return
      
      bestBeta=gradDescent(D[:, :-1], D[:, -1])
  
  
      nodeTxt=""
  
      for i in range(len(bestBeta)):
          if i==0:
              continue
          else:
              nodeTxt+="w"+str(i)+" "+str(bestBeta[i][0])+' \n '
              
      nodeTxt+="<=" + str(-bestBeta[0][0])
      
      root[nodeTxt]={nodeTxt:{}}
      #print(root[nodeTxt],'\n')
      
      Dv_b1,Dv_b2=getDv_b(D,bestBeta)
      class1="是"
      class2="否"
      # 根据beta进行数据集分割
      root[nodeTxt][class1] = {}
      root[nodeTxt][class2] = {}
      treeGenerate(Dv_b1,root[nodeTxt][class1],root[nodeTxt],class1)
      
      treeGenerate(Dv_b2,root[nodeTxt][class2],root[nodeTxt],class2)
  ```

- 4.7 图4.2是一个递归算法，若面临巨量数据，则决策树的层数会很深，使用递归方法易导致栈溢出。试使用“队列”数据结构，以参数MaxDepth控制树的最大深度，写出与图4.2等价，但不使用递归的决策树生成算法

  - 如题意，使用队列保存结点
  - 初始化一个队列，并将头结点放入队列中。
  - 用一个while循环，当队列为空时停止。
  - 让一个节点出队列作为当前节点
    - 如果当前节点中的数据都为一类，则把该节点设置为叶子节点。
    - 如果数据集只剩下类，也把当前节点设置为叶子节点，findMaxLabel
  - 如果当前节点的深度小于MaxSize，则继续划分。否则得到叶子节点。











