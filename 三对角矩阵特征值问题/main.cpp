/*
author: JoeZhu
time: Oct 2nd,2017
institution: UCAS
license：GPL 3.0
*/

#include <iostream>
#include <time.h>
#include <cstdlib>
#include <fstream>
#include "func.h"

int main(){
    cout.precision(6);
    cout.setf(ios::fixed);
    SparseMatrix A;
    A.rows = 10;
    A.cols = 10;
    int rank = 10;
	int tmp;
    //读入矩阵数据
	FILE *fp;
	fp = fopen("./data.txt","r");
    for(int i=0;i<A.rows;i++){
        for(int j=0;j<A.cols;j++){
			if(fscanf(fp,"%d",&tmp)== EOF) break; 
            A.cells.push_back(Cell(i,j,tmp));
        }
		if(feof(fp)) break;
    }
	fclose(fp);
	clock_t start,finish;//计时
    vector<vector<double> >U,V;
    vector<double> s;
	start = clock();
	//char* method = "DC";//定义分解方法
    svd(A,rank,U,s,V,"DC");
    cout<<"[U,S,V]=svd(A,r)"<<endl;
    cout<<"r = "<<rank<<endl;
    cout<<"A = "<<endl;
    A.moveFirst();
    for(int i=0;i<A.rows;i++){
        for(int j=0;j<A.cols;j++){
            cout<<A.next().value<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

    cout<<"U = "<<endl;
    print(U);

    cout<<"s = "<<endl;
    for(int i=0;i<s.size();i++){
        for(int j=0;j<s.size();j++){
            if(i == j) cout<<s[i]<<' ';
            else cout<<0.0<<' ';
        }
        cout<<endl;
    }
    cout<<endl;
    cout<<"V = "<<endl;
    print(V);
	finish = clock();
	cout<<"计算过程用时："<<(double)(finish-start)/CLOCKS_PER_SEC<<"秒"<<endl;
	system("pause");
    return 0;
}