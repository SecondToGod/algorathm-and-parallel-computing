#include<stdio.h>
#include<string.h>
#define MAX 200

//删去a中首个递减元素i
void del(char *a,int i){
	int j,len = strlen(a);
	for( j = i;j<len-1;++j){
		a[j] = a[j+1];
	}
	a[j]='\0';
}
//找到首个递减位置
void delNum(char * a,int k){
	int i,len = strlen(a);
	while(k>0){
	for(i =0;i<len-1 && a[i]<=a[i+1];++i);
	del(a,i);
	k--;
	}
}
int main(){
	char a[MAX];
   int i,j,k;
   printf("请输入一个整数：\n");
   scanf("%s",a);
   printf("请输入要截去的位数：\n");
   scanf("%d",&k);
   if(k<strlen(a)) delNum(a,k);
   printf("最终结果为：\n");
   if (strlen(a)==1) printf("%c",a[0]);
   for(j=0;j<strlen(a)-1 && a[j]=='0';++j);//除去结果首部的0
	for(j;a[j]!='\0';++j){
		printf("%c",a[j]);
	}
	return 0;
}