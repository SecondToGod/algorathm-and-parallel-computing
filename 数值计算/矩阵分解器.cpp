#include<iostream>
#include<iomanip>
#include<math.h>
using namespace std;


//行交换函数
void rowChange(double a[][20], int d, int i, int j)//d为数组维数，i,j要交换的两行
{
	double t; int k;
	for (k = 0; k < d; k++)
	{
		t = a[i][k];
		a[i][k] = a[j][k];
		a[j][k] = t;
	}
}

//输出矩阵函数
void printMatrix(double a[][20], int d)
{
	int i, j;
	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
			cout << setw(10) << a[i][j] << " ";
		cout << endl;
	}
	cout << endl;
}
//定义矩阵乘法
void multiply(double a[][20], double b[][20], int d,int y)
{
	int i, j;
	double c[20][20];
	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
		{
			c[i][j] = 0;
			for (int k = 0; k < d;k++)
				c[i][j] += a[i][k] * b[k][j];	
		}
	}
	for (i = 0; i <d; i++)
	{
		for (j = 0; j < d; j++)
		{
			if (y == 1)
				a[i][j] = fabs(c[i][j])>1e-12?c[i][j]:0;
			else
				b[i][j] = fabs(c[i][j])>1e-12? c[i][j] : 0;
		}
	}
}


//LU分解函数
void LU(double A[][20], int d)
{
	int i,j,t;
	double k;
	double l[20][20] = { 0 }, u[20][20] = { 0 }, p[20][20] = { 0 },a[20][20];//初始化
	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
			a[i][j] = A[i][j];
	}
	for (i = 0; i < d; i++)
		p[i][i] = 1;//P矩阵单位化
	for (i = 0; i < d - 1; i++)
	{
		//找主元所在行
		k = fabs(a[i][i]); int m = 0;
		for (t = i + 1; t < d; t++)
		{
			if (k < fabs(a[t][i]))
			{
				k = a[t][i];
				j = t;
				m = 1;
			}
		}
		if (m)
		{
			rowChange(a, d, i, j);
			rowChange(p, d, i, j);
		}

		//进行消元运算
		for (j = i + 1; j < d; j++)//j是行数，t是列数
		{
			a[j][i] = a[j][i] / a[i][i];
			for (t = i + 1; t < d; t++)
				a[j][t] -= a[j][i] * a[i][t];
		}
	}
	//L矩阵和U矩阵的获取
	for (i = 0; i < d; i++)
	{
		l[i][i] = 1;
		for (j = 0; j < i; j++)
			l[i][j] = a[i][j];
		for (j = i; j < d; j++)
			u[i][j] = a[i][j];
	}

	//输出结果
	cout << "LU分解结果为" << endl;
	cout << "L矩阵为:" << endl<<endl;
	printMatrix(l, d);
	cout << "U矩阵为：" << endl;
	printMatrix(u, d);
	cout << "P矩阵为：" << endl;
	printMatrix(p, d);

}


//Gram-Schmidt分解
void schmidt(double a[][20], int d)//a[][]是待分解矩阵,d 表示维数
{
	int i, j, k;
	double q[20][20], r[20][20] = { 0 };

	//将q初始化为与a相同的矩阵
	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
			q[i][j] = a[i][j];
	}
	for (i = 0; i < d; i++)
	{
		for (j = 0; j < i; j++)
		{
			//计算r[j]][i]
			for (k = 0; k < d; k++)
				r[j][i] += q[k][j] * a[k][i];
			//计算矩阵q的第i列
			for (k = 0; k < d; k++)
				q[k][i] -= q[k][j]*r[j][i]  ;
		}
		//计算r[i][i]
		for (k = 0; k < d; k++)
			r[i][i] += q[k][i] * q[k][i];
		r[i][i] = sqrt(r[i][i]);
		//归一化q矩阵的第i列
		for (k = 0; k < d; k++)
			q[k][i] /= r[i][i];
	}
	cout << "Gram-Schmidt分解结果为:" << endl;
	cout << "Q矩阵为:" << endl<<endl;
	printMatrix(q, d);
	cout << "R矩阵为:" << endl;
	printMatrix(r, d);
}

//householder分解
void householder(double A[][20], int d)//待分解矩阵A[][]，d表示维数
{
	int k,l;
	//分解后Q矩阵是q[][]，分解后R矩阵是运算后的a[][]；
	double u[20], p[20][20], q[20][20] = {0}, a[20][20];//p[][]是中间矩阵；
	double norm,s,sum;
	for (k = 0; k < d; k++)
	{
		for (l = 0; l < d; l++)
			a[k][l] = A[k][l];
	}
	//q矩阵单位初始化
	for (k = 0; k < d; k++)
		q[k][k] = 1;
	
	for (int i = 0; i < d-1; i++)
	{
		norm = 0; s = 0,sum=0;
		//计算u[]矩阵，u[i]=a[][i]-||a[][i]||*e1
		for ( k = i; k < d; k++)
		{
			u[k - i] = a[k][i];
			norm += u[k-i] * u[k-i];
		}
		norm = sqrt(norm);
		u[0] -= norm;
		for (k = i; k < d; k++)
			sum += u[k]*u[k];
		//计算p[][]矩阵
		for (k = 0; k < d; k++)
		{//p矩阵单位初始化
			for (l = 0; l < d; l++)
			{
				if (k == l)
					p[k][l] = 1;
				else
					p[k][l] = 0;
			}
		}
		if (sum)
		{
			for (k = 0; k < d - i; k++)
				s += u[k] * u[k];
			for (k = i; k < d; k++)
			{
				for (l = i; l < d; l++)
				{
					if (k == l)
						p[k][l] = 1 - 2 / s*u[k - i] * u[l - i];//计算p[][i]=a[][i]-2*u*ut/(ut*u)
					else
						p[k][l] = 0 - 2 / s*u[k - i] * u[l - i];
				}
			}
		}	//q=p1*p2*p3,r=p3*p2*p1*a
		multiply(q, p, d, 1);//q=q*p;q=p1*p2*p3
		multiply(p, a, d, 0);//a=p*a
	}
	//结果输出
	cout << "Householder分解结果为：" << endl;
	cout << "Q矩阵为：" << endl;
	printMatrix(q, d);
	cout << "R矩阵为：" << endl;
	printMatrix(a, d);
}

//Givens 分解
void givens(double A[][20], int d)
{
	//a[][]是待分解矩阵。(Q=T12*T13*T23;R=T23*T13*T12*A)
	double t[20][20] = { 0 }, q[20][20] = { 0 },a[20][20];
	int i, j;
	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
			a[i][j] = A[i][j];
	}
	for (i = 0; i < d; i++)
	{//q矩阵单位初始化
		q[i][i] = 1;
	}
	for (int i = 0; i < d; i++)
	{
		for (int j = i+1; j < d; j++)
		{
			//t矩阵单位初始化
			for (int k = 0; k < d; k++)
			{
				for (int l = 0; l < d; l++)
				{
					if (k == l)
						t[k][l] = 1;
					else
						t[k][l] = 0;
				}
			}
			//计算t矩阵
			t[i][i] = a[i][i] / sqrt(a[i][i] * a[i][i] + a[j][i] * a[j][i]);
			t[i][j] = a[j][i] / sqrt(a[i][i] * a[i][i] + a[j][i] * a[j][i]);
			t[j][j] = t[i][i];
			t[j][i] = -1 * t[i][j];

			multiply(q, t, d, 1);//q=q*t
			multiply(t, a, d, 0);//a=t*a
		}
	}
	//输出最终结果
	cout << "Givens分解结果为：:" << endl;
	cout << "Q矩阵为:" << endl;
	printMatrix(q, d);
	cout << "R矩阵为:" << endl;
	printMatrix(a, d);
}
int main()
{
	double a[20][20];
	int i,j,d;
	cout << "请输入矩阵维数:" << endl;
	cin >> d;
	cout << "请输入矩阵元素:" << endl;
	for (i = 0; i < d; i++)
	for (j = 0; j < d; j++)
	cin >> a[i][j];
	cout << "输入矩阵为:" << endl;
	printMatrix(a, d);
	do
	{
		cout << "请选择分解方法:" << endl << "1:LU分解" << endl << "2:Gram-schmidt分解" << endl << "3:Householder分解" << endl << "4:Givens分解" << endl;
		cin >> i;
		switch (i)
		{
		case 1:LU(a, d); break;
		case 2:schmidt(a, d); break;
		case 3:householder(a, d); break;
		case 4:givens(a, d); break;
		default:cout << "非合法输入！" << endl;
		}
		cout << "选择其他方法?       1:是  0:否" << endl;
		cin >> j;
	} while (j);
}
