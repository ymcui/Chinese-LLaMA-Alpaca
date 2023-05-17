# 代码编程

## 分数表

|         | 总分 |  #1  |  #2  |  #3  |  #4  |  #5  |  #6  |  #7  |  #8  |  #9  | #10  |
| ------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **7B**  |  27  |  1   |  2   |  2   |  1   |  3   |  8   |  2   |  5   |  2   |  1   |
| **13B** |  49  |  5   |  4   |  3   |  6   |  2   |  9   |  3   |  7   |  2   |  8   |

*总分计算 = 样例得分之和 / 样本数量 × 10*

## 测试样例

**1、请用python实现快速排序的代码**

7B:

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    left = []
    right = []

    for i in range(0,len(arr)):

        #如果中间元素不在左右两边，则直接返回原数组。

        if (i + 1.5) > len(arr) - 0.5 and arr[i] != arr[-1]:
            break

        else:

            #将左半部分插入到右半部分中

            while True:
                j = random.randint(1,len(left))

                if not sorted(left[:-1],reverse=True) == \
                        [j+1] :
                    continue

                left.append(random.choice(list(left)))
```

13B:

```python
def quick_sort(array):
    if len(array) <= 1:
        return array

    pivot = array[0] #选取第一个元素作为枢轴，并交换位置

    left, middle, right = [], [], []

    for item in array[1:]:

      if (item < pivot):
          left.append(item)

         elif (item > pivot):
            right.append(item)

              else:
                middle.append(item)

               sortedArray = [sortedArray[:]] + sortLists([middle])

                  print("Sorted Array")
                  print(sortedArray)

                  return sortedArray

if __name__ == '__main__':
    arr = range(101)
    result = quick_sort(arr)
    print('Original List', arr)
    print('Sorted List', result)
```



**2、请你用python写一下Dijkstra算法的代码**

7B:

```python
def dijkstr(weight, source):  # Dijkstra Algorithm in Python
    stack = []                      # Stack for storing nodes and their distance from origin node

    def dfs_helper(node, dist):       # Recursive function to compute shortest path between two nodes
        if weight == 0 or (dist + 1) > weight:   # Check whether current node satisfies condition of minimum cost
            return None               # Return none when satisfied else recurse on next neighbor

        stack.append((node, dist))           # Add current node with its distance as first element into list
        while len(stack) != 2 :              # Iterate over all neighbors until no more reachable nodes are found
            currNode, curDist = heapq.heapify(0), 0          # Initialize min-pivoting priority queue

            for nbr in sorted(list(neighbors[source])):  # Traverse through each adjacent node
                if not stack: break;  # If the predecessor is already visited then stop traversal
                successor = stack.pop(0)[-1]  # Get previous vertex's successor
                newDistance = max([curDist+1+weight*distance(currNode, succVertex)] ,0)  # Calculate maximum possible distance by considering only one edge at a time and keeping track of previously computed distances from source node

                dfs_helper(succVertex,newDistance)    # Recursive call to compute shortest path between two nodes (src -> dest)
```

13B:

```python
import networkx as nx   #导入网络图库
G = nx.DiGraph()    #创建一个空节点-边向量网格
weighted_graph = {0:1, 1:0}      #定义初始状态和距离矩阵
nx.add_edge(1, 0)        #添加第一个顶点与第二个顶点之间的连线
for i in range(1, 4):     #遍历所有顶点
    for j in range(2, 5):       #遍历每个顶点与其相邻的顶点之间
        if weighted_graph[i][j] == None or (float(weighted_ graph[i][j]) > max_distance and float(weighted_graph[i][j]) != none)):
            print("在第", len(G), "个顶点中找到最短路径")
            x, y = G.get_node(1)(), G.get_node(2)()
            while((y is not None and y != x)) :
                print("%d -> %d" % (x, y)), \
                    ("%.2f" % round(max([g[k][l].__dict__.items(0)[-1][-2], g[k][l + 2].__dict__.items[-1][-1][-2]] * 1.25, 3.2))))
                nx.remove_vertex(x)   #删除当前节点
                node = next(iter(nx.all_neighbors(x)[0]))      #下一个邻居
                assert(type(node) is int), "%s不是一个整数，无法添加到网络图集中。" % str(node))
                nx.add_edge(x, node)       #将新的连线加入网格
        else:
           break
print("Dijkstra算法已完成计算！")
```





**3、请你给出在PyTorch下定义一个带ReLU激活函数的全连接层的代码**

7B:

```python
class ReLuFullConvolution(nn.Module):  # This class defines full connection layer with ReLU activation function on PyTorch platform.
    def __init__(self, num_channels=32, kernel_size=16, stride=0, padding='SAME'):
        super().__init___()


    def forward(self, inputs):  # Forward propagation of input tensor to output vector (concatenated).
        x = tf.torch.flatten(3)([inputs])  # Flattens all channels into one flattened vector for easy manipulation later in code.
        features = self._relu_(x[:, :1], x[:, 2])  # Performs ReLU operation and extracts first two channel's values as new feature vectors respectively.
                              # The third value remains unchanged because it is already zero by design.

        outputs = torch.cat(0, [features[0][1:-1] , 0])  # Concatenates the extracted feature vectors together again.
```

13B:

```python
class ReLuActivation(nn.Module):  # Define our module with Relu activation function
    def __init__(self, input_size=None):   # Set up initial parameters for this layer
        super().__init__()      # Initialize parent class properties and methods
         self.relu = nn.LeakyRelu(2)     # Leaky relu weights of 2 (alpha)
                                # We can also pass in any other parameter to alter leakiness
             if input_size != None:       # If size was passed into constructor set it here as well
               assert type(input_size) == int(), "Input Size should be integer"
           else:
              assert len(self.weight.data.shape[1]) > 1,"Weights must have more than one dimension."
          endif;

    @register("fc")  # Register output feature map connection layers
    def forward(self, x):  # Override the default implementation from PyTorch's modules API
        out = F.conv2d(1, 0, 1.0/len(x), kernel_stride=0)(x*kernel_scale)+F.max_pooling(2)*kernel_scale*(-1))  # Convolutional Layer with ReLU activation function
        return [out]   # Return a list containing our single layer representation
```





**4、请写出计算两个数的最大公约数(GCD)的Python代码**

7B:

```python
def gcd(a, b):  # GCF函数，求解一个给定数字和另一个数字的最小公倍。
    if (b == 0 or not(int(a)/float(b)) and int(a) != float(a)):   # 如果输入不为实数或不能被整除且不是整数，则返回0（零）作为结果。
        return 0

    while True:  # 循环直到找到最小值或者达到最大值。
        c = abs((a % b) - (a // b))...  # 取模运算符并以绝对形式表示输出结果...

        if c > 0:  # 若当前结果大于0，则递归到此处；否则继续执行下一步程序。
            break

        else:  # 当前结果小于等于0时，将当前结果与原变量a相加即可。
```

13B:

```python
def gcd(a, b):  # This function returns GCF of two numbers in Python
    while True:   # Loop until either number becomes 0 or they are equal to each other
        if (b == 0 and a != 0) or \
           ((a % b) == 0 and (not((a // b)))):      # Check for divisibility by `b` with remainder as zero
            return abs(b);                                # Return result when found

        elif (a == 0 and b !=0) or\
             ((b%a)===0 and not((b//a))):       # Same logic but reversed order
            return abs(a);

    else:
        pass;              # If no common factor was found then just loop forever
```





**5、请你用python实现计算斐波那契数列的函数**

7B:

```python
def fibonacci(n):
    if n <= 1:
        return 0

    elif (n % 2 ===0) or (n >=1 and n < 4): # 如果是偶或小于4，则直接返回前两个项。

        yield 0,1

    else:
        for i in range(1,n-1):
            yield sum([fibonacci(i) + fibonacci(3-i) for j in range(0,i-1)]),2*j+3
```

13B:

```python
def fib(n):
    if n <= 1:
        return 0,1

    else:
        r = (fib(2-1) + fib(0)) / 2

        print("The Fibonacci number at index " , n,"is",r)

        return r+1,r2=((fib(1)+2)/3)+(1-r1)/(1-r1)*r2   # The next two numbers in the series are calculated using this formula

for i in range(100):
    print("%d %s"%(i,fib(i)))
```





**6、使用python写一个二分查找的代码**

7B:

```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2

        if arr[mid] == target:
            return True # Found!

        elif target < arr[mid]:
            low += 1
        else:
            high -= 1

    return False # Not found yet
```

13B:

```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while True:
        mid = (high + low) // 2

        if arr[mid] == target:
            return mid #找到了目标元素，返回它的索引位置

        elif arr[mid] > target:
            high = mid - 1

        else:
            low = mid + 1



    assert False,"binary search failed"
```






| #    | Prompt                                                       | 中文7B                                                       | 中文13B                                                      |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 7    | 如何将PyTorch的FP32模型保存为半精度？                        | 要将 PyTorch 中的 FP32 模型存储到半精度中，可以使用以下步骤：<br/>1. 在训练时设置参数大小。在调用函数之前，需要先定义一个大小方参数集，以确保模型不会因为太小而导致性能下降的问题。例如，您可以在训练过程中设定参数的大小方数组，并指定其大小级数和单位（如K、M或G）。<br/>2. 通过导热度调整器来优化模型。为了使模型能够更好地适应不同的数据格式的尺寸限制，我们建议使用梯形损失函数作为最优解方法之一。此外，也可以考虑采用其他常见的优化算法，比如Adam或者SGD等。 | 使用 PyTorch 中的 `torch.save` API，选择 "fp16" 作为存储格式。 |
| 8    | 如果运行程序出现segmentation fault，可能的原因是什么？如何避免？ | 当一个程序在执行时发生内存域冲突或引用错误时，就会导致该程序产生名为" segmentation 的错别字号（Segmentation Fault）。为了避免此种问题，程序员应该使用正确的数据类型和结构来表示代码中的变量、数组等对象，并确保这些对象不会相互干扰。此外，可以尝试检查代码中是否有潜在的引用错误或者指针错误等问题，以帮助排除上述原因的可能性。 | 当一个程序在执行时遇到内存溢出或非法操作等异常情况时会引发 segmentation fault。这通常是由于代码中的错误或者不完整性导致的。为了避免这种情况发生，可以使用一些工具来检测和修复这些问题，例如静态分析器、单元测试等等。 |
| 9    | C语言下使用printf异常输出“烫烫烫烫烫”，其原因是？            | 这是一个常见的错误。这是因为在打印函数中使用了一个名为"<unnamed>"的字符串变量作为格式参数（即第三个引号后的参数）。由于这个字节是未定义的，所以每次运行时都会导致该问题发生。为了避免此种情况，可以将该字符串变全局引用为常量或将其替换成一些其他有意义的值来解决这个问题。 | 在C语言中，由于使用了错误的格式字符串（%d）来打印一个整数值时导致了此情况。 |
| 10   | 请你给出在linux下将两个文件合并，然后按行去重的bash命令      | cp -f <file1> <destination>  # cp用于移动文件<br/>mv <file2> <destination>/<filename>.txt   # mv为新名称并创建新的文件。 | cp file1_name file2_name \| sort -u > outputfile                                   |

