---
Typora-root-url: ../../assets/MLpics
---

## 部分习题解释

- 表格

  ![T1](/T1.png)

- 1.1	版本空间

  - 好瓜{色泽=青绿，根蒂=蜷缩，敲声=浊响}	坏瓜{色泽=乌黑，根蒂=稍蜷，敲声=沉闷}
  - 好瓜{色泽=青绿，根蒂=\*，敲声=\*}	坏瓜{色泽=乌黑，根蒂=\*，敲声=\*}
  - 好瓜{色泽=\*，根蒂=蜷缩，敲声=\*}	坏瓜{色泽=\*，根蒂=稍蜷，敲声=\*}
  - 好瓜{色泽=\*，根蒂=\*，敲声=浊响}	坏瓜{色泽=\*，根蒂=\*，敲声=沉闷}
  - 好瓜{色泽=\*，根蒂=蜷缩，敲声=浊响}	坏瓜{色泽=\*，根蒂=稍蜷，敲声=沉闷}
  - 好瓜{色泽=青绿，根蒂=\*，敲声=浊响}	坏瓜{色泽=乌黑，根蒂=\*，敲声=沉闷}
  - 好瓜{色泽=青绿，根蒂=蜷缩，敲声=\*}	坏瓜{色泽=乌黑，根蒂=稍蜷，敲声=\*}

- 1.2

  - 包含3种属性，假设空间大小为3x4x4+1=49
  - 考虑冗余（A=a与A=*等价
    - 具体的假设：2x3x3=18
    - 1属性泛化：2x3+3x3+2x3=21
    - 2属性泛化：2+3+3=8
    - 3属性泛化：1
    - 不考虑冗余/空集：kmax=48
    - 考虑冗余：kmax=18

- 1.3

  - 归纳偏好选择：一致性比例越高越好
    - 两个数据属性越接近，则分为同一类
    - 若相同属性出现了两种不同的分类
      - 则认为它属于与他最邻近的几个数据的属性
      - 同时去掉所有具有相同属性而不同分类的数据，可能会丢失部分信息
