# 代码编程

## 分数表

|             | 平均分 |  #1  |  #2  |  #3  |  #4  |  #5  |  #6  |  #7  |  #8  |  #9  | #10  | #11  | #12  | #13  | #14  | #15  | #16  | #17  | #18  | #19  | #20  |
| ----------- | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **7B**      |  5.65  |  9   |  4   |  2   |  9   |  9   |  10  |  6   |  3   |  8   |  4   |  2   |  7   |  2   |  2   |  6   |  2   |  8   |  4   |  10  |  6   |
| **13B**     |  6.4   | 9.5  |  5   |  1   |  10  | 9.5  |  10  |  9   |  2   |  3   |  2   |  3   |  9   |  5   |  2   |  7   |  9   |  9   |  7   |  10  |  6   |
| **Plus-7B** | 5.875  |  10  |  3   |  3   |  10  |  9   | 9.5  |  1   |  1   |  7   |  7   |  1   |  7   |  6   |  2   |  10  |  2   |  8   |  5   |  9   |  7   |

## 测试样例（前10道题为编程题，后10道题为解答题）

**1、请用python实现快速排序的代码**

7B:

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    middle = [x for x in arr[1:] if x == pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

13B:

````python
以下是快速排序的Python实现：

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    middle = [x for x in arr[1:] if x == pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

该函数接受一个列表作为参数，并返回排序后的列表。
````

Plus-7B:

```
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```



**2、请你用python写一下Dijkstra算法的代码**

7B:

```python
def dijkstra_shortest_path(graph, source):
    # 定义一个空图，用于记录所有未探索过的节点
    visited = set()

    # 初始化距离矩阵
    dist = [[float('inf')] * len(graph[source]) for _ in range(len(graph))]

    # 遍历每个节点
    for node in graph:
        if node == source:
            continue

        # 如果该节点已经探索过，则直接返回其距离
        if node in visited:
            return dist[node]

        # 计算该节点的最短路径
        min_distance = float('inf')
        neighbor = None
        for neighbor in graph[node]:
            if neighbor not in visited:
                distance = dijkstra_shortest_path(graph, neighbor)
                if distance < min_distance:
                    min_distance = distance
                    neighbor_distance = distance

            # 将该节点加入到未探索过的节点列表中
            visited.add(neighbor)

        dist[node] = min_distance + neighbor_distance

    return dist
```

13B:

````python
以下是一个简单的Dijkstra算法的Python代码：

```python
def dijkstra(graph, source):
    # 初始化距离矩阵
    dist = [[float('inf')] * len(graph)]
    # 初始化起点节点的距离为0
    dist[source][source] = 0

    # 遍历所有节点
    for i in range(len(graph)):
        # 计算当前节点到起点的距离
        for j in range(len(graph)):
            if graph[i][j] == 1:
                # 如果有边连接，则更新距离矩阵
                dist[i][j] = min(dist[i][j], dist[i][source] + graph[i][j])
    # 返回距离最小的节点
    return min(list(dist[source]))
```

该代码使用了一个二维数组来存储距离矩阵，并使用遍历遍历所有节点来计算起点到每个节点的距离。最后返回距离最小的节点。
````

Plus-7B:

```
import heapq

def dijkstra(graph, s):
    queue = []
    heap = [s]
    while len(heap) > 0:
        current = heap.pop()
        for i in graph[current]:
            if i not in heap and dist(i, s) < dist(s, current):
                heap.append(i)
    return queue
```





**3、请你给出在PyTorch下定义一个带ReLU激活函数的全连接层的代码**

7B:

```python
class ReLUFullConvLayer(nn.Module):
    def __init__(self, input_shape, kernel_size, num_filters, stride=None, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], kernel_size[0], kernel_size[1], kernel_size[2], padding=padding, activation='relu')
        self.pool1 = nn.MaxPool2d(kernel_size[3], kernel_size[4])
        self.conv2 = nn.Conv2d(input_shape[0], kernel_size[5], kernel_size[6], kernel_size[7], padding=padding, activation='relu')
        self.pool2 = nn.MaxPool2d(kernel_size[8], kernel_size[9])
        self.fc1 = nn.Linear(num_filters * (3 * 3), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1)(x)
        x = F.relu(self.pool1)(x)
        x = F.relu(self.conv2)(x)
        x = F.relu(self.pool2)(x)
        x = F.relu(self.fc1)(x)
        return F.softmax(x, axis=0)
```

13B:

```python
class MyFullConvNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], output_shape[0], kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.conv1)
        self.relu1 = nn.ReLU(2)
        self.conv2 = nn.Conv2d(output_shape[0], output_shape[1], kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(self.conv2)
        self.relu2 = nn.ReLU(2)
        self.fc1 = nn.Linear(self.conv1.in_shape[0], output_shape[2])
        self.fc2 = nn.Linear(self.conv2.in_shape[0], output_shape[3])
        self.fc3 = nn.Linear(self.bn2.in_shape[0], output_shape[4])
        self.fc4 = nn.Linear(self.bn1.in_shape[0], output_shape[5])

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out3 = self.fc1(out2)
        out4 = self.fc2(out3)
        out5 = self.fc3(out4)
        out6 = self.fc4(out5)
        return F.relu(out6)
```

Plus-7B:

```
import torch
from torch import nn

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, activation='relu')
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1/8, 1024, activation='relu')
        self.fc2 = nn.Linear(1024, 1, activation='sigmoid')

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```





**4、请写出计算两个数的最大公约数(GCD)的Python代码**

7B:

```python
def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)

print(gcd(3, 2))  # 输出 5
```

13B:

````python
```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
```

该函数使用while循环，将b从a除以b得到的商值b除以a，直到b为0为止。最后返回a。
````

Plus-7B:

```
def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)
```





**5、请你用python实现计算斐波那契数列的函数**

7B:

```python
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return (fibonacci(n-1) + fibonacci(n-2))

其中，`fibonacci(n)` 返回斐波那契数列的第n项。
```

13B:

````python
以下是一个简单的Python函数，可以计算斐波那契数列：

```python
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return (fibonacci(n-1) + fibonacci(n-2))
```

该函数使用递归的方式计算斐波那契数列，当n小于等于1时返回n，否则返回前两个数的和。
````

Plus-7B:

```
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return (fibonacci(n-1) + fibonacci(n-2))
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
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```

13B:

````python
```python
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

该函数接受一个数组和一个目标值，使用二分查找算法来查找目标值在数组中的位置。如果目标值不在数组中，则返回-1。
````

Plus-7B:

```
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```



**7、已知全班所有同学的姓名存储于列表names中，所有同学的身高存储于列表heights中，它们的关系是一一对应的。请写一段Python代码，将所有同学的姓名按身高由低到高排序。**

7B:

````python
可以使用zip函数将两个列表合并成一个新列表，然后使用列表推导式将新的列表转换为一个字典，最后使用字典推导式将字典转换为一个阵列即可：

```python
names = ["张三", "李四", "王五"]
heights = [155, 170, 180]

# 使用zip函数将两个列表合并成一个新列表
new_list = zip(names, heights)

# 使用列表推导式将新的列表转换为一个字典
dict_list = dict(new_list)

# 使用字典推导式将字典转换为一个阵列
arr_list = list(dict_list.items())
```

现在，我们有一个字典arr_list，其中每个键对应于两个列表中的对应值。我们可以使用这个字典来访问每个同学的姓名和身高：

```python
print("张三的身高是：", arr_list["张三"][1])
print("李四的身高是：", arr_list["李四"][1])
print("王五的身高是：", arr_list["王五"][1])
```
> 可以使用sorted函数对列表进行排序：

```python
names = ["张三", "李四", "王五"]
heights = [155, 170, 180]

sorted_names = sorted(zip(names, heights), key=lambda x: x[1])
print(sorted_names)
```

输出：

```python
[('李四', 155), ('王五', 180), ('张三', 170)]
```
````

13B:

````python
```python
names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "George", "Harry"]
heights = [55, 60, 58, 56, 54, 52, 48, 46]

names.sort(key=lambda x: heights.index(heights[names.index(x)]))
print(names)
```

输出：

```
['Alice', 'Charlie', 'David', 'Eve', 'Frank', 'George', 'Harry']
```
````

Plus-7B:

```
names = ['小明', '小红', '小黄', '小李', '小王']
heights = [54, 52, 48, 46, 40]

sorted_names = sorted(names, key=lambda x: heights[x])
```





**8、输入一个字符串，找出其中出现的所有数字，并输出它们的位置。请用python实现**

7B:

````python
```python
string = "1234567890123456"
numbers = []
for char in string:
    if char.isdigit():
        numbers.append(char)
print(numbers)
```

输出：

```
['1', '2', '3', '4', '5', '6', '7', '8', '9']
```
````

13B:

````python
```python
string = "Hello World!"
numbers = []
for i in range(len(string)):
    if string[i].isdigit():
        numbers.append((i, string[i]))
print(numbers)
```

输出：

```
[(1, 'H'), (2, 'e'), (4, 'l'), (6, 'l'), (8, 'o'), (9, 'W'), (10, 'r'), (11, '!')]
```
````

Plus-7B:

```
def find_digits(s):
    digits = []
    for c in s:
        if isdigit(c):
            digits.append(c)
    return digits

s = "hello world"
print(find_digits(s)) # [2, 7, 5]
```







**9、已知有一个数组A，如何随机打乱其中的元素顺序？请用C语言实现**

7B:

````python
以下是使用C语言实现将数组A随机打乱顺序的代码：

```
#include <stdio.h>

void shuffle(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int j = rand() % (n-i-1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    shuffle(arr, 4);
    for (int i = 0; i < 4; i++) {
        printf("%d ", arr[i]);
    }
    return 0;
}
```

该程序首先定义了一个数组A，然后使用C语言的rand()函数生成一个随机整数，用于选择要交换位置的元素。最后，通过循环遍历数组A中的每个元素，将需要交换的位置的元素交换位置，从而实现数组A的随机打乱顺序。
````

13B:

````python
以下是使用C语言实现随机打乱数组A的代码：

```
#include <stdio.h>
#include <stdlib.h>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    int i, j;

    for (i = 0; i < 5; i++) {
        for (j = 0; j < 5 - i; j++) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    printf("随机打乱后的数组：\n");
    for (i = 0; i < 5; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
```

该代码使用for循环将数组中的元素随机打乱，并输出结果。
````

Plus-7B:

```
可以使用以下代码实现：
void shuffle(int arr[], int n) {
    for (int i = 0; i < n - 1; ++i) {
        int j = rand() % (n-i-1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```







**10、输入一个正整数，输出其中每个数字出现的次数。请用python实现**

7B:

````python
```python
num = int(input())
counts = {}
for i in range(len(str(num))):
    if num[i] not in counts:
        counts[num[i]] = 0
    else:
        counts[num[i]] += 1
print(counts)
```

输出：

```
{'1': 2, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1}
```
````

13B:

```python
num = int(input("请输入一个正整数："))

counts = {}
for i in range(1, num+1):
    if i in counts:
        counts[i] += 1
    else:
        counts[i] = 1

for i in counts:
    print(f"数字{i}出现了{counts[i]}次")
```

Plus-7B:

````
以下是使用Python实现该功能的代码：

```python
def count_digits(num):
    # 将输入的数转换为字符串并去掉逗号和下划线
    num_str = str(num).replace(",", "").replace("-", "")

    # 将字符串拆分成一个列表，每个数字作为一个元素
    digits = [int(x) for x in re.findall('\\d+', num_str)]

    # 遍历列表并统计每个数字出现的次数
    count = {}
    for digit in digits:
        if digit not in count:
            count[digit] = 1
        else:
            count[digit] += 1

    return count
```

使用示例：

```python
>>> count_digits(1234)
{'1': 2, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1}
```
````






| #    | Prompt                                                       | 中文7B                                                       | 中文13B                                                      | 中文Plus-7B                                                  |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 11   | 如何将PyTorch的FP32模型保存为半精度？                        | 您可以使用PyTorch的torch.save()函数，并指定要使用的格式为"npy"。然后在导出的文件中使用np.load(filename)函数读取该模型，并将其转换为FP32格式。 | 使用PyTorch中的torch.save()函数，将模型保存为半精度。在保存前，需要将模型的激活函数设置为ReLU或Tanh，然后使用torch.save(model, 'example.pt', fp=16)命令保存模型。 | 可以使用torch.nn.FloatTensor()函数将浮点数转换为半精度，然后使用torch.save_hdf5()函数将模型保存到HDF5文件中。 |
| 12   | 如果运行程序出现segmentation fault，可能的原因是什么？如何避免？ | 当运行程序时遇到 segmentation fault 时，可能是由于内存管理或代码逻辑错误导致的。为了避免这个问题，可以尝试检查代码中的内存分配和引用，并确保代码正确且没有语法错误。此外，还可以使用调试工具来查找问题所在，例如在运行时查看日志文件或使用调试器来跟踪程序执行过程中的变量值。 | 运行程序出现 segmentation fault 可能是由于代码中引用了未定义的内存地址或超出了程序的内存范围。避免这种情况的方法包括：<br/><br/>1. 检查代码中的变量和函数是否正确定义。<br/>2. 确保程序在运行时有足够的内存来存储所有数据。<br/>3. 使用正确的数据类型来存储和处理数据。<br/>4. 避免在程序中使用未定义的变量或函数。<br/>5. 使用正确的调用顺序和参数传递方式。 | Segmentation fault通常是由于访问了无效内存地址或越界操作导致的。可以使用调试工具来定位问题，例如GDB、Valgrind等。另外，可以通过检查代码逻辑和数据结构是否正确，以及对程序进行单元测试来避免出现此类错误。 |
| 13   | C语言下使用printf异常输出“烫烫烫烫烫”，其原因是？            | 该错误是由于printf函数未正确地处理输入字符串导致的。在C语言中，printf函数会将输入的字符串转换为ASCII码并打印出来。如果输入的字符串包含超过7个字符时，则会导致printf函数无法正确处理字符串，导致输出异常。 | C语言下使用printf异常输出“烫烫烫烫烫”，其原因是？<br/>由于printf函数的参数类型与printf格式字符串不匹配，导致了异常输出。 | 该错误可能是由于printf函数的参数类型不正确导致的。在C语言中，printf函数需要一个字符串作为参数，而您提供的是一个整数。因此，您应该将整数转换为字符串并传递给printf函数。例如：<br/><br/>```c<br/>char str[] = "烫";<br/>printf("%s", str);<br/>``` |
| 14   | 请你给出在linux下将两个文件合并，然后按行去重的bash命令      | cat file1 file2 \|sort -k \|uniq \> output_file.new          | cat file1.txt file2.txt \| sort \> output.txt                | cat file1.txt file2.txt > combined_file.txt                  |
| 15   | 如何使用git命令提交对README.md文件的修改，更新日志为“update hyperparams” | git add README.md<br/>git commit -m "更新 hyperparams"       | 使用以下命令提交对README.md文件的修改，并更新日志为“update hyperparams”：<br/>git add README.md<br/>git commit -m "update hyperparams" | 可以使用以下命令来提交对README.md文件的修改并更新日志：<br/><br/>```bash<br/>git commit -m "update hyperparams" README.md<br/>```<br/><br/>其中-m选项用于指定提交信息，即“update hyperparams”。 |
| 16   | 请将字符串“helloworld”中的奇数位改为大写并输出修改后的字符串 | HELLO WORLD                                                  | HEllO wORlD                                                  | HWERDL                                                       |
| 17   | 在Python中如何使用pandas库读取大量数据并进行数据清洗？       | 使用pandas库中的read_csv函数可以轻松地读取大量数据，然后使用pandas中的各种数据清洗方法（例如缺失值填充、异常值处理、重复项删除等）来进行数据清洗。 | 1. 使用read_csv()函数读取CSV文件。<br/>2. 使用dropna()函数删除缺失值。<br/>3. 使用fillna()函数填充缺失值。<br/>4. 使用replace()函数替换特定的值。<br/>5. 使用drop_duplicates()函数删除重复值。<br/>6. 使用describe()函数查看数据的统计信息。 | 可以使用pandas的read_csv()函数来读取CSV文件，然后使用dropna()和fillna()方法对缺失值进行填充。还可以使用drop_duplicates()方法去除重复行，使用merge()方法合并多个数据集，使用groupby()方法对数据进行分组分析等操作。 |
| 18   | 运行python程序出现错误：TypeError: ‘list’ object cannot be interpreted as an integer，如何修正 | 您可以尝试将列表类型转换为整数类型，例如使用int()函数。      | 可以检查代码中的变量类型和数据类型是否匹配，或者使用try-except语句来捕获并处理错误。 | 这个错误通常是由于在代码中使用了一个列表而不是整数导致的。请检查您的代码并确保您正确地将整数转换为列表。 |
| 19   | 请列举python的基本数据类型及其作用                           | 以下是Python的基本数据类型及其作用：<br/><br/>1. 整数（int）：用于存储整型数值，例如1、2、3等。<br/>2. 浮点数（float）：用于存储小数型数值，例如0.5、10.0等。<br/>3. 布尔值（bool）：用于表示真假值，例如True和False。<br/>4. 字符串（str）：用于存储文本数据，例如“Hello World”等。<br/>5. 列表（list）：用于存储有序的数字序列，例如[1, 2, 3]等。<br/>6. 元组（tuple）：类似于列表，但不能修改元素。<br/>7. 字典（dict）：用于存储无序的键值对，例如{'name': 'John', 'age': 25}等。 | 基本数据类型包括：<br/><br/>- 整数（int）：用于存储整型数值，例如1, 2, 3等。<br/>- 浮点数（float）：用于存储小数值，例如0.1, 0.25等。<br/>- 字符串（str）：用于存储文本数据，例如“Hello, world!”等。<br/>- 布尔类型（bool）：用于存储真/假值，例如True/False等。<br/>- 列表（list）：用于存储多个元素的序列，例如[1, 2, 3]等。<br/>- 元组（tuple）：类似于列表，但不可修改，例如(1, 2, 3)等。<br/>- 字典（dict）：用于存储键值对，例如{'name': 'John', 'age': 25}等。 | Python的基本数据类型包括整数、浮点数、字符串、列表、元组、字典等。其中，整数用于表示数字，浮点数用于表示小数和科学计数法，字符串用于存储文本信息，列表用于存储一组相同类型的元素，元组用于存储一组不同类型的元素，字典用于存储键值对。 |
| 20   | 在机器学习中，如果学习率设置过大会出现什么问题？如何避免？   | 如果学习率设置过高，可能会导致模型过度拟合，从而导致性能下降。为了避免这种情况，可以尝试调整学习率的阈值，或者使用正则化技术来限制模型的泛化能力。此外，还可以通过增加数据量或采用不同的损失函数来提高模型的泛化能力。 | 如果学习率设置过大会导致模型过拟合，即模型过于依赖训练数据，无法准确预测新数据。为了避免这种情况，可以采用正则化技术，如L1和L2正则化、Dropout等方法，来减少过拟合的风险。此外，还可以通过增加训练数据量、调整超参数等方式来优化模型的泛化能力。 | 当学习率过高时，模型可能会陷入局部最优解，导致训练效果下降。为了避免这种情况，可以采用以下方法：1. 使用更小的学习率；2. 增加训练数据量以提高泛化能力；3. 尝试不同的优化器和超参数组合；4. 使用正则化技术来防止过拟合。 |

