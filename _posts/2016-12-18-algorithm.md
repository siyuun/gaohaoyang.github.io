---
layout: post
title:  "algorithm实验"
date:   2016-12-18 20:00:54
categories: algorithm
tags: 分治 动态规划 贪心 图 网络流
---

* content
{:toc}

# 实验一 分治法 #

实验目的与要求：理解分治法的基本思想和设计方法。

实验题目：

>1.实现基于分治法的归并排序算法.

		#include <iostream>
		using namespace std;

		void merge(int *data, int p, int q, int r)
		{
			int n1, n2, i, j, k;  
			int *left=NULL, *right=NULL;  
			n1 =q-p+1;   
			n2 =r-q;  
			
			left = (int *)malloc(sizeof(int)*(n1));   
			right = (int *)malloc(sizeof(int)*(n2));  
			for(i=0; i<n1; i++)  //对左数组赋值  
				left[i] = data[p+i];  
			for(j=0; j<n2; j++)  //对右数组赋值  
				right[j] = data[q+1+j];  
			i =j=0;   
			k=p;  
			while(i<n1&&j<n2) //将数组元素值两两比较，并合并到data数组  
			{  
				if(left[i]<=right[j])  
					data[k++]=left[i++];  
				else  
					data[k++] = right[j++];  
			}  
			for(;i<n1;i++) //如果左数组有元素剩余，则将剩余元素合并到data数组  
				data[k++] = left[i];  
			for(;j<n2;j++) //如果右数组有元素剩余，则将剩余元素合并到data数组  
				data[k++] = right[j];  
		}


		void merge_Sort(int *data, int p, int r)
		{
			int q;  
			if(p<r) //只有一个或无记录时不须排序   
			{  
				q = (int)((p+r)/2);      //将data数组分成两半     
				merge_Sort(data, p, q);   //递归拆分左数组  
				merge_Sort(data, q+1, r); //递归拆分右数组  
				merge(data, p, q, r);    //合并数组  
			}  
			
		}

		void merge_sort(int a[],const int size)
		{
			merge_Sort(a,0,size-1);
		}



		void main()
		{
			int count=7;
			int a[]={1,3,2,4,5,1,2};
			merge_sort(a,count); //排序
			for(int i=0;i<count;i++)
				cout<<a[i]<<" ";
		}

>2.实现快速排序的算法,并尝试采用不同的方法实现线性的划分过程.

		#include <iostream>
		using namespace std;

		void quick_sort(int a[], int left, int right)
		{
			if(left<right)  
			{  
				int i = left;  
				int j = right;  
				int x = a[i];  
				
				while(i<j)  
				{  
					while(i<j&&a[j]>x)  
						j--;  
					if(i<j){  
						a[i] = a[j];  
						i++;  
					}  
					while(i<j&&a[i]<x)  
						i++;  
					if(i<j){  
						a[j] = a[i];  
						j--;  
					}  
				}  
				a[i] = x;  
				quick_sort(a, left, i-1);  
				quick_sort(a, i+1, right);  
			} 
		}
		void main()
		{
			int a[]={1,2,3,4,5,8,6,1};
			int size=8;
			quick_sort(a,0,size-1);
			for(int i=0;i<size;i++)
				cout<<a[i]<<" ";
		}

>3.有一个数的序列A[1]、A[2] 、A[3] 、…… 、A[n]，若i<j，并且A[i]>A[j]，则称A[i]与A[j]构成了一个逆序对，设计算法求数列A中逆序对的个数.


		#include <iostream>
		using namespace std;

		int merge(int *data, int p, int q, int r)
		{
			int n1, n2, i, j, k;  
			int *left=NULL, *right=NULL;  
			int count=0;
			n1 =q-p+1;   
			n2 =r-q;  
			
			left = (int *)malloc(sizeof(int)*(n1));   
			right = (int *)malloc(sizeof(int)*(n2));  
			for(i=0; i<n1; i++)  //对左数组赋值  
				left[i] = data[p+i];  
			for(j=0; j<n2; j++)  //对右数组赋值  
				right[j] = data[q+1+j];  
			i =j=0;   
			k=p;  
			while(i<n1&&j<n2) //将数组元素值两两比较，并合并到data数组  
			{  
				if(left[i]<=right[j])  
					data[k++]=left[i++]; 
					
				else  
				{
					count=count+n1-i;
					data[k++] = right[j++];  
				}
			}  
			for(;i<n1;i++) //如果左数组有元素剩余，则将剩余元素合并到data数组  
			{
				data[k++] = left[i]; 	
			}
			for(;j<n2;j++) //如果右数组有元素剩余，则将剩余元素合并到data数组  
				data[k++] = right[j];  
			return count;
		}

		int merge_Sort(int *data, int p, int r)
		{
			int count1,count2,count3,count=0;
			int q;  
			if(p<r) //只有一个或无记录时不须排序   
			{  
				
				q = (int)((p+r)/2);      //将data数组分成两半     
				count1=merge_Sort(data, p, q);   //递归拆分左数组  
				count2=merge_Sort(data, q+1, r); //递归拆分右数组
				count3=merge(data, p, q, r);    //合并数组
				count=count1+count2+count3;
			}  
			return count;
		}

		int merge_sort(int a[],const int size)
		{
			int count = merge_Sort(a,0,size-1);
			return count;
		}



		void main()
		{
			int size=7;
			int a[]={1,3,7,8,5,1,7};
			int count=merge_sort(a,size); //排序
			for(int i=0;i<size;i++)
				cout<<a[i]<<" ";
			cout<<endl<<"逆序对数："<<count<<endl;
		}

>4.(选做题) 引入逆序计数问题作为考察两个序列有多大差别的一个好的度量指标。但是人们可能感觉这个量度太敏感了。如果i<j，并且A[i]>2A[j]，我们把这对i,j叫做重要的逆序。设计一个O(nlogn) 的算法计数在两个序列中的重要逆序个数。

# 实验二 分治法 #

实验目的与要求：理解分治法的基本思想和设计方法。

实验题目：

>1.k-路合并操作问题：假定有k个有序数组，每个数组中含有n个元素，您的任务是将它们合并为单独的一个有序数组，该数组共有kn个元素。设计和实现 一个有效的分治算法解决k-路合并操作问题，并分析时间复杂度。

```
#include <iostream>  
using namespace std;  
  
#define LEN 10          //最大归并段长  
#define MINKEY -1     //默认全为正数  
#define MAXKEY 100    //最大值,当一个段全部输出后的赋值  
  
struct Array  
{  
    int arr[LEN];  
    int num;  
    int pos;  
}*A;  
  
    int k,count;  
    int *LoserTree,*External;  
  
void Adjust(int s)  
{  
    int t=(s+k)/2;  
    int temp;  
    while(t>0)  
    {  
        if(External[s] > External[LoserTree[t]])  
        {  
            temp = s;  
            s = LoserTree[t];  
            LoserTree[t]=temp;  
        }  
        t=t/2;  
    }  
    LoserTree[0]=s;  
}  
  
void CreateLoserTree()  
{  
    External[k]=MINKEY;  
    int i;  
    for(i=0;i<k;i++)LoserTree[i]=k;  
    for(i=k-1;i>=0;i--)Adjust(i);  
}  
  
void K_Merge()  
{  
    int i,p;  
    for(i=0;i<k;i++)  
    {  
        p = A[i].pos;  
        External[i]=A[i].arr[p];  
        //cout<<External[i]<<",";  
        A[i].pos++;  
    }  
    CreateLoserTree();  
    int NO = 0;  
    while(NO<count)  
    {  
        p=LoserTree[0];  
        cout<<External[p]<<",";  
        NO++;  
        if(A[p].pos>=A[p].num)External[p]=MAXKEY;  
        else   
        {  
            External[p]=A[p].arr[A[p].pos];  
            A[p].pos++;  
        }  
        Adjust(p);  
    }  
    cout<<endl;  
}  
  
int main()  
{  
    freopen("in.txt","r",stdin);  
  
    int i,j;  
    count=0; 
    cin>>k;  
    A=(Array *)malloc(sizeof(Array)*k);  
    for(i=0;i<k;i++)  
    {  
        cin>>A[i].num;  
        count=count+A[i].num;  
        for(j=0;j<A[i].num;j++)  
        {  
            cin>>A[i].arr[j];  
        }  
        A[i].pos=0;  
    }  
    LoserTree=(int *)malloc(sizeof(int)*k);  
    External=(int *)malloc(sizeof(int)*(k+1));  
  
    K_Merge();  
  
    return 0;  
}  
```
附件-文件in.txt
```
5
1 5 6 8 25
6
2 6 9 25 30 32
3
5 9 16
6
6 9 15 24 30 36
2
8 34
```

>2.Split操作的要求是：以一个数组s和一个值v为输入，将数组s划分成3个子集：比v小的元素组成的集合，等于v的元素组成的集合以及比v大的元素组成的集合。设计和实现一种O(n)的就地split算法，即该算法不额外分配新的内存。

```
#include <iostream>
#include <VECTOR>
using namespace std;

void Split(std::vector< int > &array,const int value)
{
	int length=array.size();
	int i=0,j=length-1;
	while(i<j)
	{
		while(array[i]<=value)
			++i;
        if((i<j))
            std::swap(array[i],array[j--]);
	}
	int max_pot=i;
	i=0;
	cout<<endl<<max_pot<<endl;
	for(int i=0;i < array.size();++i)
        cout<<array[i]<<" ";
	while(i<j)
	{
		while(array[i]<value)
			++i;
		while(array[j]>=value)
			--j;
		if((array[i]==value)&&(i<j))
			std::swap(array[i++],array[j--]);
	}
	int min_pot=i;
}
int main()
{
	int a[20]={1,3,4,2,5,5,5,7,9,19,4,6,5,7,23,3,4,5,8,12};
    vector<int> array(a,a+20);
	cout<<"Testing Data:\n";
	for(int i=0;i < array.size();++i)
        cout<<array[i]<<" ";
	Split(array,5);
    cout<<"\nOutput:\n";
	for(int i=0;i < array.size();++i)
        cout<<array[i]<<" ";
	return 0;
}
```
# 实验三 分治法 #

实验目的与要求：理解分治法的基本思想和设计方法。

实验题目：

>1.寻找中项
【问题描述】
对于长度为n的整型数组A，随机生成其数组元素值，然后实现一个线性时间的算法，在该数组中查找其中项。

```

#include<iostream>
#include<stdio.h>
#include <vector>
#include <time.h>
#include <stdlib.h>

int searchMed(std::vector<int>& arr);
int* split(std::vector<int>& arr_s,int m_begin,int m_end);
int select(std::vector<int>& arr_s,int k,int m_begin,int m_end);

int main()
{
    std::vector<int> arr={2,3,1,4,1,5,5,6};
    int r=searchMed(arr);
    std::cout<<r;
    return 0;
}


int searchMed(std::vector<int>& arr){
    //选出v  s split
    //选出k 刚开始是size/2 后来是(size-...)/2

    int r=select(arr,arr.size()/2,0,arr.size()-1);
    return r;
}

int* split(std::vector<int>& arr_s,int m_begin,int m_end){
    // TODO 把数组段第一个数划分，并返回中间值的下标
    int a[2];
    int k=arr_s[m_begin];
    while (m_begin < m_end)
    {
        while (m_begin < m_end && arr_s[m_end] > k)
            m_end--;
        if (m_begin < m_end)
            arr_s[m_begin++] = arr_s[m_end];
        while (m_begin < m_end && arr_s[m_begin] < k)
            m_begin++;
        if (m_begin < m_end)
            arr_s[m_end--] = arr_s[m_begin];
    }
    arr_s[m_begin] = k;

    a[0]= m_begin; a[1] = m_end;

    while (arr_s[a[0] - 1] == k)
    {
        a[0]--;
    }
    while (arr_s[a[1] + 1] ==k)
    {
        a[1]++;
    }
    return a;//==的两边位置

}
int select(std::vector<int>& arr_s,int k,int m_begin,int m_end){
    //S< S=
    int item;

    if (m_end - m_begin < 2)
    {
        if (arr_s[m_begin] <arr_s[m_end])
            item = k == m_begin ? arr_s[m_begin] : arr_s[m_end];
        else
            item = k == m_end ? arr_s[m_end] : arr_s[m_begin];
        return item;
    }
    int s1,s2;//s1是＜v的数的最后一个位置 s2是等于v的数的最后一个位置
    //srand( (unsigned)time( NULL ) );
   // int v= arr_s[rand()%arr_s.size()];
    int* a;

    a=split(arr_s,m_begin,m_end);//return s1,s2
    s1=a[0];
    s2=a[1];
    if(k<s1)
        item=select(arr_s,k,m_begin,s1-1);
        else
            if (k>=s1&&k<=s2)
                item=arr_s[s2];
            else item=select(arr_s,k,s2+1,m_end);
    return item;
}
```
>2.寻找最邻近的点对
【问题描述】
设p1=(x1,y1), p2=(x2,y2), … , pn=(xn,yn) 是平面上n个点构成的集合S，设计和实现找出集合S中距离最近点对的算法。

```
//二维最邻近点对问题
//#include "stdafx.h"
#include<time.h>
#include<iostream>
#include<cmath>
#include <stdlib.h>

using namespace std;
const int M = 50;

//用类PointX和PointY表示依x坐标和y坐标排好序的点
class PointX {
public:
    int operator<=(PointX a)const
    {
        return (x <= a.x);
    }
    int ID; //点编号
    float x, y; //点坐标
};

class PointY {
public:
    int operator<=(PointY a)const
    {
        return(y <= a.y);
    }
    int p; //同一点在数组x中的坐标
    float x, y; //点坐标
};

float Random();
template <class Type>
float dis(const Type&u, const Type&v);

bool Cpair2(PointX X[], int n, PointX& a, PointX& b, float& d);
void closest(PointX X[], PointY Y[], PointY Z[], int l, int r, PointX& a, PointX& b, float& d);

template <typename Type>
void Copy(Type a[], Type b[], int left, int right);

template <class Type>
void Merge(Type c[], Type d[], int l, int m, int r);

template <class Type>
void MergeSort(Type a[], Type b[], int left, int right);

int main()
{
    srand((unsigned)time(NULL));
    int length;

    cout << "请输入点对数：";
    cin >> length;

    PointX X[M];
    cout << "随机生成的二维点对为：" << endl;

    for (int i = 0; i<length; i++)
    {
        X[i].ID = i;
        X[i].x = Random();
        X[i].y = Random();
        cout << "(" << X[i].x << "," << X[i].y << ") ";
    }

    PointX a;
    PointX b;
    float d;

    Cpair2(X, length, a, b, d);

    cout << endl;
    cout << "最邻近点对为：(" << a.x << "," << a.y << ")和(" << b.x << "," << b.y << ") " << endl;
    cout << "最邻近距离为： " << d << endl;

    return 0;
}

float Random()
{
    float result = rand() % 10000;
    return result*0.01;
}

//平面上任意两点u和v之间的距离可计算如下
template <class Type>
inline float dis(const Type& u, const Type& v)
{
    float dx = u.x - v.x;
    float dy = u.y - v.y;
    return sqrt(dx*dx + dy*dy);
}

bool Cpair2(PointX X[], int n, PointX& a, PointX& b, float& d)
{
    if (n<2) return false;

    PointX* tmpX = new PointX[n];
    MergeSort(X, tmpX, 0, n - 1);

    PointY* Y = new PointY[n];
    for (int i = 0; i<n; i++) //将数组X中的点复制到数组Y中
    {
        Y[i].p = i;
        Y[i].x = X[i].x;
        Y[i].y = X[i].y;
    }

    PointY* tmpY = new PointY[n];
    MergeSort(Y, tmpY, 0, n - 1);

    PointY* Z = new PointY[n];
    closest(X, Y, Z, 0, n - 1, a, b, d);

    delete[]Y;
    delete[]Z;
    delete[]tmpX;
    delete[]tmpY;
    return true;
}
void closest(PointX X[], PointY Y[], PointY Z[], int l, int r, PointX& a, PointX& b, float& d)
{
    if (r - l == 1) //两点的情形
    {
        a = X[l];
        b = X[r];
        d = dis(X[l], X[r]);
        return;
    }

    if (r - l == 2) //3点的情形
    {
        float d1 = dis(X[l], X[l + 1]);
        float d2 = dis(X[l + 1], X[r]);
        float d3 = dis(X[l], X[r]);

        if (d1 <= d2 && d1 <= d3)
        {
            a = X[l];
            b = X[l + 1];
            d = d1;
            return;
        }

        if (d2 <= d3)
        {
            a = X[l + 1];
            b = X[r];
            d = d2;
        }
        else {
            a = X[l];
            b = X[r];
            d = d3;
        }
        return;
    }

    //多于3点的情形，用分治法
    int m = (l + r) / 2;
    int f = l, g = m + 1;

    //在算法预处理阶段，将数组X中的点依x坐标排序，将数组Y中的点依y坐标排序
    //算法分割阶段，将子数组X[l:r]均匀划分成两个不想交的子集，取m=(l+r)/2
    //X[l:m]和X[m+1:r]就是满足要求的分割。
    for (int i = l; i <= r; i++)
    {
        if (Y[i].p>m) Z[g++] = Y[i];
        else Z[f++] = Y[i];
    }

    closest(X, Z, Y, l, m, a, b, d);
    float dr;

    PointX ar, br;
    closest(X, Z, Y, m + 1, r, ar, br, dr);

    if (dr<d)
    {
        a = ar;
        b = br;
        d = dr;
    }

    Merge(Z, Y, l, m, r);//重构数组Y

    //d矩形条内的点置于Z中
    int k = l;
    for ( int i = l; i <= r; i++)
    {
        if (fabs(X[m].x - Y[i].x)<d)
        {
            Z[k++] = Y[i];
        }
    }

    //搜索Z[l:k-1]
    for (int i = l; i<k; i++)
    {
        for (int j = i + 1; j<k && Z[j].y - Z[i].y<d; j++)
        {
            float dp = dis(Z[i], Z[j]);
            if (dp<d)
            {
                d = dp;
                a = X[Z[i].p];
                b = X[Z[j].p];
            }
        }
    }
}

template <class Type>
void Merge(Type c[], Type d[], int l, int m, int r)
{
    int i = l, j = m + 1, k = l;
    while ((i <= m) && (j <= r))
    {
        if (c[i] <= c[j])
        {
            d[k++] = c[i++];
        }
        else
        {
            d[k++] = c[j++];
        }
    }

    if (i>m)
    {
        for (int q = j; q <= r; q++)
        {
            d[k++] = c[q];
        }
    }
    else
    {
        for (int q = i; q <= m; q++)
        {
            d[k++] = c[q];
        }
    }
}

template <class Type>
void MergeSort(Type a[], Type b[], int left, int right)
{
    if (left<right)
    {
        int i = (left + right) / 2;
        MergeSort(a, b, left, i);
        MergeSort(a, b, i + 1, right);
        Merge(a, b, left, i, right);//合并到数组b
        Copy(a, b, left, right);//复制回数组a
    }
}

template <typename Type>
void Copy(Type a[], Type b[], int left, int right)
{
    for (int i = left; i <= right; i++)
        a[i] = b[i];
}
```
# 实验四 动态规划 #
实验目的与要求：掌握动态规划方法的基本思想与设计策略。
>1.多段图中的最短路径问题
【问题描述】
建立一个从源点S到终点T的多段图，设计一个动态规划算法求出从S到T的最短路径值，并输出相应的最短路径。

```
#include<iostream>
#include<cmath>
#include<string.h>
#include<stdio.h>
using namespace std;
int  dp[100000],alone[100000],a[100000];
int main()
{
    int i,j,n,m;
    while(~scanf("%d",&m))
    {
        scanf("%d",&n);
        memset(dp,0,sizeof(dp));
        memset(alone ,0,sizeof(alone));
        for(i=1;i<=n;i++)scanf("%d",&a[i]);
        int tmax;
        for(i=1;i<=m;i++)//★分i段
        {
            tmax=-(1<<30);

            for(j=i;j<=n;j++)
            {
                dp[j]=max(dp[j-1],alone[j-1])+a[j];


                printf("%2d %2d %2d\n",a[j],alone[j-1],dp[j]);
                if(j>i)alone[j-1]=tmax;

                if(tmax<dp[j])tmax=dp[j];

            }

        }
        printf("%d\n",tmax);
    }
    return 0;
}
```
>2.有向无环图中的最短路径问题
【问题描述】
建立一个从源点S到终点E的有向无环图，设计一个动态规划算法求出从S到E的最短路径值，并输出相应的最短路径。

```
/*Dijkstra求单源最短路径 2010.8.26*/

#include <iostream>
#include <stack>
#include <stdlib.h>
#define M 100
#define N 100
using namespace std;

typedef struct node
{
    int matrix[N][M];      //邻接矩阵
    int n;                 //顶点数
    int e;                 //边数
}MGraph;

void DijkstraPath(MGraph g,int *dist,int *path,int v0)   //v0表示源顶点
{
    int i,j,k;
    bool *visited=(bool *)malloc(sizeof(bool)*g.n);
    for(i=0;i<g.n;i++)     //初始化
    {
        if(g.matrix[v0][i]>0&&i!=v0)
        {
            dist[i]=g.matrix[v0][i];
            path[i]=v0;     //path记录最短路径上从v0到i的前一个顶点
        }
        else
        {
            dist[i]=INT_MAX;    //若i不与v0直接相邻，则权值置为无穷大
            path[i]=-1;
        }
        visited[i]=false;
        path[v0]=v0;
        dist[v0]=0;
    }
    visited[v0]=true;
    for(i=1;i<g.n;i++)     //循环扩展n-1次
    {
        int min=INT_MAX;
        int u;
        for(j=0;j<g.n;j++)    //寻找未被扩展的权值最小的顶点
        {
            if(visited[j]==false&&dist[j]<min)
            {
                min=dist[j];
                u=j;
            }
        }
        visited[u]=true;
        for(k=0;k<g.n;k++)   //更新dist数组的值和路径的值
        {
            if(visited[k]==false&&g.matrix[u][k]>0&&min+g.matrix[u][k]<dist[k])
            {
                dist[k]=min+g.matrix[u][k];
                path[k]=u;
            }
        }
    }
}

void showPath(int *path,int v,int v0)   //打印最短路径上的各个顶点
{
    stack<int> s;
    int u=v;
    while(v!=v0)
    {
        s.push(v);
        v=path[v];
    }
    s.push(v);
    while(!s.empty())
    {
        cout<<s.top()<<" ";
        s.pop();
    }
}

int main(int argc, char *argv[])
{
    int n,e;     //表示输入的顶点数和边数
    cout<<"输入的顶点数和边数:";
    while(cin>>n>>e&&e!=0)
    {
        int i,j;
        int s,t,w;      //表示存在一条边s->t,权值为w
        MGraph g;
        int v0;
        int *dist=(int *)malloc(sizeof(int)*n);
        int *path=(int *)malloc(sizeof(int)*n);
        for(i=0;i<N;i++)
            for(j=0;j<M;j++)
                g.matrix[i][j]=0;
        g.n=n;
        g.e=e;
        cout<<"输入起始点，终点，权值"<<endl;
        for(i=0;i<e;i++)
        {
            cin>>s>>t>>w;
            g.matrix[s][t]=w;
        }
        cout<<"输入源顶点";
        cin>>v0;        //输入源顶点
        DijkstraPath(g,dist,path,v0);
        for(i=0;i<n;i++)
        {
            if(i!=v0)
            {
                showPath(path,i,v0);
                cout<<"dist"<<":"<<dist[i]<<endl;
            }
        }
    }
    return 0;
}
```
>3.最长递增子序列问题
【问题描述】
给定一个整数数组，设计一个动态规划算法求出该数组中的最长递增子序列。

```
#include <iostream>
using namespace std;


// 输出LIS  序列
void outputLIS(int * arr, int index, int lis, int *L)
{
    //终止条件
    if (lis == 0 || index < 0)
        return;

    //找到第一个L[index]==lis
    while (L[index]!=lis && index>0)
        index--;

    //反序输出
    if (index >= 0  && L[index]==lis)
    {
        outputLIS(arr, index - 1, lis - 1, L);
        cout << arr[index] << " ";
    }
}


int LIS(int *a, int n)
{
    //定义一个存取结果的数组
    int *L = new int[n];

    //填写次序 0 to n-1
    for (int j = 0; j < n;j++)
    {
        L[j] = 1;//BaseCase
        //find max L[i]
        for (int i = 0; i < j;i++)
        {
            if (a[i]<a[j] && L[i]+1 > L[j])
            {
                L[j] = L[i] + 1;
            }
        }
    }

    //return the max of L[0~n-1]
    int max = L[0];
    for (int i = 0; i < n; i++)
    {
        //cout << L[i] << "  ";
        if (L[i]>max)
        {
            max = L[i];
        }
    }

    //回溯输出
    cout << "最长递增子序列如下：";
    outputLIS(a, n,max, L);

    return max;
}

int main()
{
    int a[] = { 5, 2, 3, 2 ,8, 6, 2, 4, 5, 7};
    for(int i=0;i<10;i++)
        cout<<a[i]<<" ";
    cout<<endl;
    int n = sizeof(a) / sizeof(int);
    cout<<endl<<"长度为：" << LIS(a, n) << endl;
    return 0;
}
```
>4.矩阵连乘问题
【问题描述】
给定n个矩阵{A1，A2，…,An},其中AiAi+1是可乘的，i=1，2，…，n-1,考察这n个矩阵的连乘积A1A2…An，设计一个动态规划算法，求出这个矩阵连乘积问题的最优计算顺序。
实现要求：随机生成n个合法的可连乘的矩阵，以完全加括号的方式输出其最优计算顺序。

```
#include <iostream>
using namespace std;

const int L = 7;

int RecurMatrixChain(int i,int j,int **s,int *p);//递归求最优解
void Traceback(int i,int j,int **s);//构造最优解

int main()
{
    int p[L]={30,35,15,5,10,20,25};

    int **s = new int *[L];
    for(int i=0;i<L;i++)
    {
        s[i] = new int[L];
    }

    cout<<"矩阵的最少计算次数为："<<RecurMatrixChain(1,6,s,p)<<endl;
    cout<<"矩阵最优计算次序为："<<endl;
    Traceback(1,6,s);
    return 0;
}

int RecurMatrixChain(int i,int j,int **s,int *p)
{
    if(i==j) return 0;
    int u = RecurMatrixChain(i,i,s,p)+RecurMatrixChain(i+1,j,s,p)+p[i-1]*p[i]*p[j];
    s[i][j] = i;

    for(int k=i+1; k<j; k++)
    {
        int t = RecurMatrixChain(i,k,s,p) + RecurMatrixChain(k+1,j,s,p) + p[i-1]*p[k]*p[j];
        if(t<u)
        {
            u=t;
            s[i][j]=k;
        }
    }
    return u;
}

void Traceback(int i,int j,int **s)
{
    if(i==j) return;
    Traceback(i,s[i][j],s);
    Traceback(s[i][j]+1,j,s);
    cout<<"("<<i<<","<<s[i][j]<<")";
    cout<<"*("<<(s[i][j]+1)<<","<<j<<")"<<endl;
}
```

# 实验五 动态规划 #

实验目的与要求：掌握动态规划方法的基本思想与设计策略。

>1.最长公共子序列问题
【问题描述】
⑴ 给定两个字符串X和Y，设计一个动态规划算法，求出这两个字符串的最长公共子序列，并输出该子序列。

>⑵ 若仅要求求出两个字符串的最长公共子序列的长度值，为节省存储空间，采用“滚动数组”方式实现动态规划算法。

```
#include<iostream>
#include<vector>
#include<cstring>
using namespace std;

bool isodd(int i)
{
    if(i%2==0)
        return false;
    else
        return true;
}

int main()
{
    vector<char> a={'a','c','b','b','a'};
    vector<char> b={'a','c','d','b','c','d','b'};
    int bsize=b.size();
    int dp[2][bsize];
    memset(dp,0,sizeof(dp));
    for(int i=0;i<a.size();i++)
        for(int j=0;j<b.size();j++)
            if(!isodd(i))//i是偶数
                {
                if(a[i]==b[j])
                    if(i==0||j==0) dp[1][j]=1;
                        else dp[1][j]=dp[0][j-1]+1;
                else
                    if(i==0||j==0) dp[1][j]=dp[0][j];
                        else dp[1][j]=max(dp[0][j],dp[1][j-1]);
                }
            else
            {
                if(a[i]==b[j])
                    if(i==0||j==0) dp[0][j]=1;
                        else dp[0][j]=dp[1][j-1]+1;
                else
                    if(i==0||j==0) dp[0][j]=dp[1][j];
                        else dp[0][j]=max(dp[1][j],dp[0][j-1]);
            }
    cout<<"最长公共子序列:";
    if(isodd(a.size()))
        cout<<dp[1][b.size()-1];
    else
        cout<<dp[0][b.size()-1];
        return 0;
}
```
>2.0-1背包问题
【问题描述】
给定n种物品和一背包。物品i的重量是wi，其价值为vi，背包的容量为W（假定物品重量与背包容量值均为整数），应如何选择装入背包中的物品，使得装入背包中物品的总价值最大？设计一个动态规划算法，求解背包问题。
K(w,j)=max{k(w-wj,j-1)+vj,k(w,j-1)}
 
```
#include <iostream>
#include <cstring>
using namespace std;

#define W 50

void Trackback(int *weight, int n, int w,bool *p,int **a)
{
    if (n==0 || w==0)
        return;
    if (a[w][n]==a[w][n-1])//若和左边的一致，说明没有选最后一个
    {
        p[n - 1] = false;
        Trackback(weight, n - 1, w, p, a);
    }
    else
    {
        p[n - 1] = true;
        Trackback(weight, n - 1, w-weight[n-1], p, a);
    }
}
int getMaxValue(int w, int n, int *price, int * weight )
{
    //创建一个 w+1 * n+1 的二维表
    int **a = new int *[w + 1];
    for (int i = 0; i < w + 1;i++)
    {
        a[i] = new int[n + 1];
    }

    //创建一个数组 记录货物是否取的状态
    bool  *p = new bool[w];
    memset(p, false, sizeof(p));

    //base case
    for (int i = 0; i < w + 1; i++)
        a[i][0] = 0;
    for (int i = 0; i < n + 1; i++)
        a[0][i] = 0;

    //for
    for (int i = 1; i < w + 1;i++)
    {
        for (int j = 1; j < n + 1;j++)
        {
            if (i<weight[j-1])//填写a[i][j]，若当前背包重量小于物品，则不装
            {
                a[i][j] = a[i][j - 1];
            }
            else
            {
                if (a[i][j-1] <= a[i-weight[j-1]][j-1] + price[j-1])
                {
                    a[i][j] = a[i - weight[j - 1]][j - 1] + price[j - 1] ;
                }
                else
                    a[i][j] = a[i][j - 1];
            }
        }
    }

    Trackback(weight, n, w, p, a);
    cout << "从左到右是否取件为：";
    for (int i = 0; i < n; i++)
        cout << p[i] << " ";
    cout << endl;
    return a[w][n];
}

int main()
{
    int price[] = { 300, 100, 30 ,200};
    int weight[] = { 10, 20, 30 ,5};
    cout << "背包问题的解是："<<getMaxValue(W, 4, price, weight) << endl;
    return 0;
}
```
# 实验六 动态规划 贪心法#
实验目的与要求：
（1）	掌握树型动态规划方法的基本思想与设计策略；
（2）	掌握贪心法的基本思想和设计方法。


>1.树中的最大独立集问题
【问题描述】
给定一个无回路的无向图（即树），设计一个动态规划算法，求出该图的最大独立集，并输出该集合中的各个顶点值。

```
#include <iostream>
#include <vector>
#include <algorithm>
#include <string.h>
using namespace std;

const int MAXN=100;
vector<int> G[MAXN]; //G[i]表示顶点i的邻接点
int l[MAXN]; //结点层次
int p[MAXN]; //根树
int dp[MAXN]; //dp数组
int sumC[MAXN]; //孩子DP和
int sumS[MAXN]; //孙子DP和
int maxL; //最大层次
int n;



void readTree()
{
    int u,v;
    cin>>n;
    for(int i=0;i<n-1;++i)
    {
        cin>>u>>v;
        G[u].push_back(v);
        G[v].push_back(u);
    }
}

void dfs(int u,int fa)
{
    int d=G[u].size();
     l[u]= (fa==-1)? 0: (l[fa]+1);
     if(l[u]>maxL)
     {
         maxL=l[u];
     }
    for(int i=0;i<d;++i)
    {
        int v=G[u][i];
        if(v!=fa)
        {
            dfs(v,p[v]=u);
        }
    }
}

int rootDp(int u)
{
    //构造u根树
    p[u]=-1;
    maxL=-1;
    dfs(u,p[u]);
    for(int i=maxL;i>=0;--i)
    {
        for(int j=0;j<n;++j)
        {
            if(l[j]==i)
            {
                if (sumS[j]+1>sumC[j])
                {
                    dp[j]=sumS[j]+1;
                }
                else
                    dp[j] = sumC[j];
                if(i-1>=0)
                {
                    sumC[p[j]]+=dp[j];
                }
                if(i-2>=0)
                {
                    sumS[p[p[j]]]+=dp[j];
                }
            }
        }
    }
    return dp[u];
}

int main()
{
    readTree();
    int res=-1;
    //分别以每个顶点为根
    for(int i=0;i<n;++i)
    {
        memset(sumS,0,sizeof(sumS));
        memset(sumC,0,sizeof(sumC));
        int tmp;
        if((tmp=rootDp(i))>res)
            res=tmp;
    }
    cout<<res<<endl;
    return 0;
}
```
>2.区间调度问题
【问题描述】
给定n个活动，其中的每个活动ai包含一个起始时间si与结束时间fi。设计与实现算法从n个活动中找出一个最大的相互兼容的活动子集S。

要求：分别设计动态规划与贪心算法求解该问题。其中，对贪心算法分别给出递归与迭代两个版本的实现。

贪心算法

```
#include <iostream>
using namespace std;

//i是上一个符合条件的id，为了完整性，在第一列加上-1，n是总数目
void GetSet(int *si, int *fi, int i, int n)
{
    int m = i + 1;
    while (m <= n && si[m] < fi[i])//找第一个符合的
        m = m + 1;
    if (m <= n)
    {
        cout <<m<<" ";
        GetSet(si, fi, m, n);
    }
}

int main()
{
    int si[] = { -1,1, 3, 0, 5, 3, 5, 6, 8, 8, 2};
    int fi[] = { -1,4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    int n = 11;
    cout<<"兼容的活动为：";
    GetSet(si, fi, 0, 10);
}
```

```
#include<iostream>
using namespace std;

//动态规划实现
int GetSet(int *start, int *finish, int n)
{
    //c[i][j]表示第i个工作结束后到第j个工作开始前之间存在的可兼容的工作个数
    //new c[i][j]
    int **c = new int *[n];
    for(int i=0; i<n; i++)
        c[i] = new int[n];

    //c[i][j]初始赋值
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            c[i][j] = 0;

    for(int j=0; j<n; j++)
    {
        for(int i=0; i<n; i++)
        {
            if(i<j)
            {
                int s = finish[i];
                int f = start[j];
                for(int k=i+1; k<j; k++)
                    if(start[k]>=s && finish[k]<=f)
                    {
                        if(c[i][j]<(c[i][k]+c[k][j]+1))
                            c[i][j] = c[i][k]+c[k][j]+1; //分解成更小子问题
                    }
            }
        }
    }
    return c[0][n-1];   //最终目标
}

int main()
{
    //已经按完成时间排好序
    int start[] = {-1,3,0,5,3,5,6,8,8,2,1000};
    int finish[] = {0,5,6,7,8,10,10,11,12,13,10000};
    int n = 11; //活动只有9个
    cout<<"最大兼容活动子集的大小为："<<GetSet(start, finish, n)<<endl;
    return 0;
}
```
# 实验七 贪心法 #

实验目的与要求：
（1）	掌握贪心法的基本思想和设计方法。

>1.区间划分问题
【问题描述】
给定一组报告，其中的每个报告设置了一个开始时间si和结束时间fi。设计与实现一个算法，对这组报告分配最少数量的教室，使得这些报告能无冲突的举行。

```
#include <iostream>
#include <stdlib.h> 
using namespace std;

#define N 100

struct Report
{
    int num;//报告编号
    int begin;//开始时间
    int end;//结束时间
    bool flag;//是否已分配教室
    int classroom;//教室号
};

void QuickSort(Report* rep,int f,int t)//一开始将所有报告按结束时间排序
{
    if(f<t)
    {
        int i=f-1,j=f;
        Report r=rep[t];
        while(j<t)
        {
            if(rep[j].end<=r.end)
            {
                i++;
                Report tmp=rep[i];
                rep[i]=rep[j];
                rep[j]=tmp;
            }
            j++;
        }
        Report tmp1=rep[t];
        rep[t]=rep[i+1];
        rep[i+1]=tmp1;
        QuickSort(rep,f,i);
        QuickSort(rep,i+2,t);
    }
}

int select_room(Report *rep,int *time,int n)
{
    //第一个报告分给第一个教室
    int i=1,j=1;//i报告，j教室
    int sumRoom=1;
    int sumRep=1;//教室已分配的报告数
    time[1]=rep[0].end;
    rep[0].classroom=1;

    for(i=1;i<n;i++)
    {
        for(j=1;j<=sumRoom;j++)
        {
            if((rep[i].begin>=time[j])&&(!rep[i].flag))
            {
                rep[i].classroom=j;
                rep[i].flag=true;
                time[j]=rep[i].end;
                sumRep++;
            }
        }
        if(sumRep<n&&i==n-1)//报告没有分配完
        {
            i=0;
            sumRoom++;
        }
    }
    return sumRoom;
}

int main()
{
    int n;
    Report rep[N];
    int time[N];//每个教室最后一个报告的结束时间
    cout<<"请输入报告数量:"<<endl;
    cin>>n;
    int i;
    for(i=0;i<n;i++)
    {
        //初始化
        time[i+1]=0;//time[1]~time[10]
        rep[i].num=i+1;//编号1~10
        rep[i].flag=false;
        rep[i].classroom=0;

        cout<<"报告"<<i+1<<"开始时间:";
        cin>>rep[i].begin;
        cout<<"报告"<<i+1<<"结束时间:";
        cin>>rep[i].end;
    }
    QuickSort(rep,0,n-1);
    int roomNum=select_room(rep,time,n);
    cout<<"所用教室总数为:"<<roomNum<<endl;
    for(i=0;i<n;i++)
    {
        cout<<"活动"<<rep[i].num<<"在教室"<<rep[i].classroom<<"中"<<endl;
    }
    system("pause");
    return 0;
}
```
>2.最小延迟调度问题
【问题描述】
    假定有一单个的资源在一个时刻只能处理一个任务。现给定一组任务，其中的每个任务i包含一个持续时间ti和截止时间di。设计与实现一个算法，对这组任务给出一个最优调度方案，使其对所有任务的最大延迟最小化。

```
#include<iostream>
using namespace std;

#define N 100

struct Mission
{
    int num;
    int last;
    int end;
};

void QuickSort(Mission *mi,int f,int t)
{
    if(f<t)
    {
        int i=f-1,j=f;
        Mission m=mi[t];
        while(j<t)
        {
            if(mi[j].end<=m.end)
            {
                i++;
                Mission tmp=mi[i];
                mi[i]=mi[j];
                mi[j]=tmp;
            }
            j++;
        }
        Mission tmp1=mi[t];
        mi[t]=mi[i+1];
        mi[i+1]=tmp1;
        QuickSort(mi,f,i);
        QuickSort(mi,i+2,t);
    }
}

int main()
{
    int n;
    cout<<"请输入任务总数:"<<endl;
    cin>>n;
    Mission mi[N];//Mission[0]~Mission[n-1]
    int start[N+1];//排好序的任务的开始时间，start[1]~start[n]
    for(int i=0;i<n;i++)
    {
        mi[i].num=i+1;
        cout<<"任务"<<i+1<<"的持续时间为:";
        cin>>mi[i].last;
        cout<<"任务"<<i+1<<"的截止时间为:";
        cin>>mi[i].end;
    }
    QuickSort(mi,0,n-1);
    int delay=0;
    start[1]=0;

    if(start[1]+mi[0].last>mi[0].end)
    {
        delay+=start[1]+mi[0].last-mi[0].end;//如果开始时间+持续时间>截止时间，累计延迟
    }
    for(int i=1;i<n;i++)
    {
        start[i+1]=start[i]+mi[i-1].last;
        if(start[i+1]+mi[i].last>mi[i].end)
        {
            delay+=start[i+1]+mi[i].last-mi[i].end;
        }
    }
    cout<<"延迟最小为:"<<delay<<endl;
    for(int i=0;i<n;i++)
    {
        cout<<"任务"<<mi[i].num<<"的执行时间:["<<start[i+1]<<","<<mi[i].last+start[i+1]<<"]"<<endl;
    }
}
```
# 实验八 图 #


实验目的与要求：
（2）	掌握使用图的深度优先搜索算法实现对有向图中是否包含环的判断；
（3）	掌握使用图的深度优先搜索算法实现对有向图的强连通分量的划分。


>1.有向图中环的判断问题
【问题描述】
给定一个有向图，要求使用深度优先搜索策略，判断图中是否存在环。

```
#include<iostream>
#include<malloc.h>
using namespace std;
#define maxNum 100 //定义邻接举证的最大定点数
int pre[maxNum];
int post[maxNum];
int point=0;//pre和post的值
bool is_DAG=true;//标识位，表示有向无环图
/*
0 白色，未被访问过的节点标白色
-1 灰色，已经被访问过一次的节点标灰色
1 黑色，当该节点的所有后代都被访问过标黑色
时间复杂度：O(n+e)
*/
int color[maxNum];//顶点颜色表 color[u]
//图的邻接矩阵表示结构
typedef struct
{
    char v[maxNum];//图的顶点信息
    int e[maxNum][maxNum];//图的顶点信息
    int vNum;//顶点个数
    int eNum;//边的个数
}graph;
void createGraph(graph *g);//创建图g
void DFS(graph *g);//深度优先遍历图g
void dfs(graph *g,int i);//从顶点i开始深度优先遍历与其相邻的点
void dfs(graph *g,int i)
{
    //cout<<"顶点"<<g->v[i]<<"已经被访问"<<endl;
    cout<<"顶点"<<i<<"已经被访问"<<endl;
    color[i]=-1;
    pre[i]=++point;
    for(int j=1;j<=g->vNum;j++)
    {
        if(g->e[i][j]!=0)
        {
            if(color[j]==-1)//探索到回边,存在环
            {
                is_DAG=false;//不是有向无环图
            }
            else if(color[j]==0)
                dfs(g,j);
        }
    }
    post[i]=++point;
    color[i]=1;//表示i的后裔节点都被访问过
}
void DFS(graph *g)
{
    int i;
    //初始化color数组，表示一开始所有顶点都未被访问过，//初始化pre和post
    for(i=1;i<=g->vNum;i++)
    {
        color[i]=0;
        pre[i]=0;
        post[i]=0;
    }
    //深度优先搜索
    for(i=1;i<=g->vNum;i++)
    {
        if(color[i]==0)//如果这个顶点为被访问过，则从i顶点出发进行深度优先遍历
        {
            dfs(g,i);

        }
    }
}
void createGraph(graph *g)//创建图g
{
    cout<<"请输入顶点个数:";
    cin>>g->vNum;
    cout<<"请输入边的个数:";
    cin>>g->eNum;
    int i,j;
    //初始画图g
    for(i=1;i<=g->vNum;i++)
        for(j=1;j<=g->vNum;j++)
            g->e[i][j]=0;
    //输入边的情况
    cout<<"请输入边的头和尾"<<endl;
    for(int k=1;k<=g->eNum;k++)
    {
        cin>>i>>j;
        g->e[i][j]=1;
    }
}
int main()
{
    graph *g;
    g=(graph*)malloc(sizeof(graph));
    createGraph(g);//创建图g
    DFS(g);//深度优先遍历
    //各顶点的pre和post值
   // for(int i=1;i<=g->vNum;i++)
      //  cout<<"顶点"<<i<<"的pre和post分别为："<<pre[i]<<" "<<post[i]<<endl;
    //判断是否是有向无环图
    if(is_DAG)
        cout<<"有向无环图"<<endl;
    else
        cout<<"存在环"<<endl;
    int k;
    cin>>k;
    return 0;
}
```
>2.有向图的强连通分量问题
【问题描述】
给定一个有向图，设计一个算法，求解并输出该图的各个强连通分量。

```
#include <iostream>
#include <stack>
using namespace std;

#define MAX_VERTEX_SIZE 10001
struct EdgeNode{
    int vertex;
    EdgeNode *nextArc;
};

struct VerTexNode{
    EdgeNode* firstArc;
};

struct Graph{
    int n,e;
    VerTexNode vNode[MAX_VERTEX_SIZE];
};

int time = 0;
int low[MAX_VERTEX_SIZE];
int dfn[MAX_VERTEX_SIZE];
int visited[MAX_VERTEX_SIZE];
int inStack[MAX_VERTEX_SIZE];
stack<int> st;
Graph graph;

void initeGraph(int n,int m)
{
    for(int i = 1;i<=n;i++)
    {
        graph.vNode[i].firstArc = NULL;
    }
    graph.n = n;
    graph.e = m;

}

//头插法建立图
void creatGraph(int s,int v)
{
    EdgeNode *edgeNode = new EdgeNode;
    edgeNode->vertex = v;
    edgeNode->nextArc = graph.vNode[s].firstArc;
    graph.vNode[s].firstArc = edgeNode;
}

int min(int a,int b)
{
    if(a>b)
        return b;
    else
        return a;
}

void trajan(int u)
{
    dfn[u] = low[u] = time++;
    st.push(u);
    visited[u] = 1;
    inStack[u] = 1;
    EdgeNode *edgePtr = graph.vNode[u].firstArc;
    while(edgePtr !=NULL)
    {
        int v = edgePtr->vertex;
        if(visited[v] == 0)
        {
            trajan(v);
            low[u] = min(low[u],low[v]);
        }
        else
        {
            low[u] = min(low[u],dfn[v]);
        }
        edgePtr = edgePtr->nextArc;
    }

    if(dfn[u] == low[u])
    {
        int vtx;
        cout<<"set is: ";
        do{
            vtx = st.top();
            st.pop();
            inStack[vtx] = 0;//表示已经出栈
            cout<<vtx<<' ';
        }while(vtx !=u );
    }

}

int main()
{
    int n,m;
    int s,a;
    cout<<"vexs and edges:"<<endl;
    cin>>n>>m;
    initeGraph(n,m);
    for(int i = 1;i<=n;i++)
    {
        visited[i] = 0;
        inStack[i] = 0;
        dfn[i] = 0;
        low[i] = 0;
    }
    cout<<"the begin and the end of each edge:"<<endl;
    for(int j = 1;j<=m;j++)
    {
        cin>>s>>a;
        creatGraph(s,a);
    }

    for(int i =1;i<=n;i++)
        if(visited[i] == 0)
            trajan(i);
    return 0;
}
```
# 实验九 最大流 #

实验目的与要求：
（4）	理解与掌握求解最大流与最小割的基本算法。
（5）	学会应用最大流与最小割算法解决实际问题。

>1.实现Ford-Fulkerson算法，求出给定图中从源点s到汇点t的最大流，并输出最小割。

```
#include<queue>
#include<iostream>
#include<string.h>
using namespace std;
#define MAX 1024
int nodes,edges;

int capacity[MAX][MAX];//记录边的当前还可以通过的最大流量
int maxflow=0;
bool isVisited[MAX];//在BFS或DFS找增广路的时候记录该元素是否访问过
int pre[MAX];//记录节点的前一个节点

/*
    我最疑惑的地方是capacity[i][pre[i]]+=increase;这个地方。
    我们一开始以为这只是一个简单的有向图，其实不是，这个有向图会根据它的两个节点之间的通过的流量自动改变
        我们可以把它看成是最原始的有向图中有箭头的两个节点可以相互通过流，而不仅仅是沿箭头的方向通过流（通过判断两个节点之间的最大
        流量来判断。）

    表达能力实在有限。。我自己都觉得没说清楚..
*/
inline int min(int a,int b)
{
    return a>b?b:a;
}

bool DFS(int src)
{
    if(!src)
        pre[src]=-1;
    if(src==nodes-1)
        return true;
    isVisited[src]=true;
    for(int i=0;i<nodes;i++)
    {
        if(!isVisited[i]&&capacity[src][i])
        {
            isVisited[i]=true;
            pre[i]=src;
            if(DFS(i))
                return true;
        }
    }
    return false;
}

bool BFS()
{
    queue<int> myQueue;
    myQueue.push(0);
    isVisited[0]=true;
    pre[0]=-1;
    while(!myQueue.empty())
    {
        int current=myQueue.front();
        myQueue.pop();
        for(int i=0;i<nodes;i++)
        {
            if(!isVisited[i]&&capacity[current][i])
            {
                myQueue.push(i);
                pre[i]=current;
                isVisited[i]=true;
            }
        }
    }

    return isVisited[nodes-1];
}

void MaxFlow()
{
    while(1)
    {
        memset(isVisited,false,nodes);
        memset(pre,0xff,4*nodes);

    //  if(!DFS(0))
    //      break;
        if(!BFS())
            break;

        int increase=MAX;
        int i;
        for(i=nodes-1;pre[i]>=0;i=pre[i])
        {
            increase=min(increase,capacity[pre[i]][i]);
        }
        for(i=nodes-1;pre[i]>=0;i=pre[i])
        {
            capacity[pre[i]][i]-=increase;
            capacity[i][pre[i]]+=increase;
        }
        maxflow+=increase;


    }
}
int main()
{
    while(1)
    {
        cout<<"vexs edges:"<<endl;
        cin>>nodes>>edges;
        int firstnode,secondenode,capa;
        for(int i=0;i<edges;i++)
        {
            cin>>firstnode>>secondenode>>capa;
            capacity[firstnode][secondenode]=capa;
        }
        MaxFlow();
        cout<<"最大流："<<maxflow<<endl;
        maxflow=0;

    }
    return 0;
}
```
>2.设计与实现二部图匹配（Bipartite Matching）问题的算法。


```
//void *memset(void *s,int c,size_t n)将已开辟内存空间 s 的首 n 个字节的值设为值 c
//#include <Ford_Fulkerson>
#include <algorithm>
#include <cstdio>
#include <list>
#include <queue>
#include <iostream>
using namespace std;
#define INFI 1000
typedef struct _mark
{
    int pre_suc;
    int max_incr;
}MARK;
int iteration = 0;//增光路径数目
const int N = 100;
list<int> setS;
bool isMark[N], isCheck[N], isDone;
MARK markList[N];
int c[N][N], f[N][N];
int n; //顶点数
int Maxflow()
{
    int flow = 0;
    for (int i = 0; i<n; i++)
    {
        flow += f[0][i];
    }
    return flow;
}
void Mincut()//isMark的点就是最小割
{
    int i = 0;
    while (i<n)
    {
        if (isMark[i])
            setS.push_back(i);
        i++;
    }
}
int IncrFlowAuxi(int index)//计算增广路径中的最大可增量
{
    if (index == 0)
        return markList[index].max_incr;
    int prev = markList[index].pre_suc;
    int maxIncr = markList[index].max_incr;
    return min(maxIncr, IncrFlowAuxi(prev));//递归求瓶颈值为最大增量
}
void IncrFlow()//增广路径的增加
{
    iteration++;
    int incr = IncrFlowAuxi(n - 1); //最大可增量
    int index = n - 1;
    int prev;
    while (index != 0)
    {
        if (index != n - 1)
            cout << index << " ";
        prev = markList[index].pre_suc;
        f[prev][index] += incr; //增广路径增加后，相应的流量进行更新
        index = prev;
    }
    cout << endl;
}
void Mark(int index, int pre_suc, int max_incr)//被标记表示可能被纳入新路径
{
    isMark[index] = true;
    markList[index].pre_suc = pre_suc;//前驱
    markList[index].max_incr = max_incr;//当前路径的流值
}
void Check(int i)//被mark且被check的点表示已经被纳入新路径
{
    isCheck[i] = true;
    for (int j = 0; j<n; j++)
    {
        if (c[i][j]>0 && !isMark[j] && c[i][j]>f[i][j])//forward 边
            Mark(j, i, min(markList[i].max_incr, c[i][j] - f[i][j]));
        if (c[j][i]>0 && !isMark[j] && f[j][i]>0)//reverse 边
            Mark(j, i, min(markList[i].max_incr, f[j][i]));
    }
}
//ford_fulkerson算法
int ford_fulkerson()
{
    int i;
    while (1)//一次循环找到一个新路径
    {
        isDone = true;
        i = 0;
        while (i<n)//一次循环判断上次循环是否有找到新路径，若无则表明没有新路径，终止算法
        {
            if (isMark[i] && !isCheck[i])  //判断是否所有标记的点都已被检查：若是，结束整个算法
            {
                isDone = false;
                break;
            }
            i++;
        }
        if (isDone) //算法结束，则计算最小割和最大流
        {
            Mincut();
            return Maxflow();
        }
        while (i<n)//贪心法构建新路径
        {
            if (isMark[i] && !isCheck[i]) {
                Check(i);
                i = 0;
            }
            if (isMark[n - 1]) //如果汇t被标记，说明找到了一条增广路径，则增加该条路径的最大可增加量
            {
                IncrFlow();
                memset(isMark + 1, false, n - 1); //增加该增广路径后，除了源s，其余标记抹去
                memset(isCheck, false, n);
            }
            else i++;
        }
    }
}
int main()
{
    //测试数据为ppt第40页的图，只实现了二部图的最大匹配
    n = 12;
    for (int k = 0; k < n; ++k)
    {
        memset(c[k], 0, sizeof(c[0][0])*n);
        memset(f[k], 0, sizeof(f[0][0])*n);  //初始各分支流量为0
        memset(isMark, false, n);
        memset(isCheck, false, n);
    }
    isMark[0] = true; //给源做永久标记
    markList[0].max_incr = INFI;
    markList[0].pre_suc = INFI;
    c[1][6] = INFI;
    c[1][7] = INFI;
    c[2][7] = INFI;
    c[3][6] = INFI;
    c[3][8] = INFI;
    c[3][9] = INFI;
    c[4][7] = INFI;
    c[4][10] = INFI;
    c[5][7] = INFI;
    c[5][10] = INFI;
    for (int i = 1; i < n / 2; i++)
        c[0][i] = 1;
    for (int i = n/2; i < n-1; i++)
        c[i][n-1] = 1;
    cout << "最大匹配结果为：" << endl;
    int result= ford_fulkerson();
    cout << "匹配边个数为：" << result << endl;
    system("PAUSE");
}
```
>3.设计与实现项目选择（Project Selection）问题的算法。