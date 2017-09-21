/*
author: Joe Zhu
date  : Sep,2017
from  : UCAS
*/
#include<iostream>
#include<cstdlib>
#include<cmath>
#include<time.h>
using namespace std;

//size          矩阵维数
//delta         存储主对角元素
//alfa          存储次对角元素
//a,b			储存区间端点
//lamda			储存特征值
#define N 10
#define TOL 0.000001 //终止精度

//统计大于x的特征值个数
int s(float x,float delta[N],float alfa[N-1],int size){
	float p[N];
	p[0]=1;p[1]=delta[0]-x;
	for(int i=2;i<size+1;++i){//循环求特征多项式序列
		p[i] = (delta[i-1]-x)*p[i-1]-alfa[i-2]*alfa[i-2]*p[i-2];
	}
	int count = 0;
	for(int i=1;i<size+1;i++){
		if(p[i]==0) p[i] = -p[i-1];
		else if(p[i]>0 && p[i-1]>0) count++;
		else if(p[i]<0 && p[i-1]<0) count++;
	}
	return count;
}
//二分法递归求解
void half_eigen(float a,float b,float delta[N],float alfa[N-1],float lamda[N],int size){
	if( b-a < TOL ){
		lamda[s(a,delta,alfa,size)-1] = (a+b)/2;//保存特征值
	}
	else{//递归
		if( s((a+b)/2,delta,alfa,size) < s(a,delta,alfa,size)){
			 half_eigen(a,(a+b)/2,delta,alfa,lamda,size);
		}//左递归
		else if( s((a+b)/2,delta,alfa,size) > s(a,delta,alfa,size)){
			half_eigen((a+b)/2,b,delta,alfa,lamda,size);//右递归
		} 
	}
}
int main(){
	/*
	初始化矩阵或读入矩阵数据
	*/
	clock_t start,finish;
	double total_time;
	int size = N;
	float delta[N]={1,1,1,1,1,1,1,1,1,1},
			alfa[N-1]={1,1,1,1,1,1,1,1,1};
	float lamda[N];
	//float delta[N],alfa[N-1];
	float Tinf;//谱半径
	start = clock();
	Tinf = fabs(delta[size-1])+fabs(alfa[size-2]);
	if(Tinf < (fabs(delta[0])+fabs(alfa[0]))){
		Tinf = fabs(delta[0])+fabs(alfa[0]);
	}
	for(int i=1;i<size-1;++i){
		if(Tinf < (fabs(alfa[i])+fabs(delta[i])+fabs(alfa[i-1]))){
			Tinf = fabs(alfa[i])+fabs(delta[i])+fabs(alfa[i-1]);
		}
	}
	half_eigen(-Tinf, Tinf, delta, alfa, lamda, size); //调用递归函数
	for(int i=0;i<size;++i){
		cout<<lamda[i]<<endl;
	}
	finish = clock();
	total_time = double(finish - start)/CLOCKS_PER_SEC*1000;
	cout<<endl<<"time:"<<total_time;
	system("pause");
	return 0;
}