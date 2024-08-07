# 使用说明书
### 此代码实现的功能如下：
+ 读取data文件夹下所有的文件
+ 数据预处理：将Cal文件内数据读取并进行处理
+ 计算响应电阻、响应大小、区分度、响应稳定性、恢复程度、基线偏差
+ 从响应电阻、响应大小、区分度、响应稳定性、恢复程度、基线偏差对数据进行分组
+ 绘制响应电阻、响应大小、区分度 曲线图像
+ 筛选出满足6参数或是7参数的部件

---

### 注意事项：
1. ~~生成的报告文件批次目前仍然要手动填写~~
2. 测试的文件一定要先进行预处理，包括文件内容要包含0 200 100 80 50 51 53 0 或 0 80 50 51 53 0 ,否则要在ETH文件定位并抓取缺失点
3.  如因环境原因导致基线偏移需手动修改两个基线（在ETH文件抓取，第一基线和第二基线相差120行）
4. 目前仍未知空载位置是否会对分析结果造成影响（如良品率，测试总数）
5. 将cal文件放入data文件夹， ETH文件放入ETH_data文件夹
6. 一次只能处理同一批次的 ***测试文件*** 

```mermaid
操作流程
    [1]获取同一批次数据文件 
    --> [2]查看数据情况
    --> [3]有无测试点丢失点 
      --> [4_1]有 --> [5]抓取数据，在Cal文件里补充 --> [6]
      --> [4_2]无 --> [6]
    [6]将同一批次的Cal文件放到 data文件夹， 总数据文件（如ETH.txt）放到batch文件夹 
    --> [7]运行代码
    --> [8]在report文件夹查看报告
```
