#include <iostream>  
#define MAX 20  
using namespace std;  
int main()  
{  
    cout << "请输入数塔行数" << endl;   
    int n,i;  
    cin >> n;  
  
    int a[MAX+1][MAX+1][3]; //[0]用来存数，[1]参与运算，[2]表示向左(0)，还是向右(1)  
      
    for(i = 1; i <= n; ++i)  //输入数塔
    {  
        for(int j = 1; j <= i; ++j)       
        {  
            cin >> a[i][j][0];  
            a[i][j][1] = a[i][j][0];  
            a[i][j][2] = 0;  
        }  
    }   
    cout << endl;  
	
	//下面开始动态规划求解
    for( i = n-1; i >= 1; --i)   //从倒数第二行开始  
    {  
        for(int j=1; j <= i; j++)  
        {  
            if (a[i+1][j][1] > a[i+1][j+1][1])   //左边大  
            {  
                a[i][j][2] = 0;     //选择左边  
                //子父节点相加  
                a[i][j][1] += a[i+1][j][1];   
            }  
            else    //右边大  
            {  
                a[i][j][2] = 1;     //选择右边  
                a[i][j][1] += a[i+1][j+1][1];  
            }  
        }  
    }  
    for( i = 1; i <= n; ++i)       //输出数塔   
    {  
        for(int j = 1; j <= i; ++j)  
        {  
            cout << a[i][j][0] << " ";  
        }  
        cout << endl;  
    }  
      
    cout << a[1][1][1] << endl;  //输出最大值
     
    int j;  
	cout << "最优解路径为:"<< endl;
    for(i = 1, j = 1; i<= n; ++i)  //输出路径 
    {  
        cout << "[" << i << "," << j << "]" << " -> ";  
        j += a[i][j][2];  
    }  
    return 0;  
}  