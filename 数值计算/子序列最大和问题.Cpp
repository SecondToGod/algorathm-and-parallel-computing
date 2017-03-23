#include <iostream>
using namespace std;
#define MAX_SIZE 200


bool checkNeg(int *a,int size){
	int i;
	for(i=0;i<size;++i){
		if (a[i]>=0) return true;
   }
}
void subMax(int *a,int size){
	int i,l,m;//l,m分别代表最大子列下标
   int max=0,temp=0;
   if(!checkNeg(a,size)){
		cout<<"序列全为负，最大子序列的和为0"<<endl;
		return;
   }
   for(i=0;i<size;++i){
			l=1;m=1;
			temp += a[i];
			if(temp>max) {
			 max=temp;
			 m=i+1;
		   }//递增则继续更新max
			else if(temp<0) {
					temp=0;l=i+2;
		   } //小于0时则置0
	}
	cout<<"最大子序列为子段:"<<l<<"到"<<m<<endl;
   cout<<"最大子序列的和为:"<<max;
}
int main()
{
	int len,i;
   int a[MAX_SIZE];
	cout << "请输入序列元素的个数:" << endl;
   cin>>len;
	cout << "请输入序列的各个元素:" << endl;
   for(i=0;i<len;++i){
		cin>>a[i];
   }
   subMax(a,len);
	return 0;
}
